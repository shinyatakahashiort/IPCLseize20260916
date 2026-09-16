"""
IPCL Vault / サイズ予測ロジック
================================
Streamlit アプリ（app.py）と学習スクリプト（train_model.py）の共通部品です。
Streamlit がなくても、コマンドラインから予測できます。

    python ipcl_predict.py --age 35 --LV -0.12 --ACD 3.25 --ACW 11.75

モデル
  - Vault 予測（回帰）       : LV, ACD, size, ACW, age → Vault (mm)
  - 正しいサイズの確率（分類）: ACW より大きい候補サイズごとに「正しいサイズ」である確率
    目的変数を「ACW より大きい最小サイズから何段階(0.25mm)上か」に変換して学習しています
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, 'models', 'ipcl_bundle.pkl')

SIZE_GRID = np.round(np.arange(11.00, 14.01, 0.25), 2)   # ICL サイズ候補
STEP_MM = 0.25
N_STEPS = 5                      # ACW より大きい最小サイズ 〜 その +1.00mm
VAULT_RANGE = (0.25, 0.75)       # 広義の適正 Vault (mm)
VAULT_NARROW = (0.40, 0.60)      # 狭義の適正 Vault (mm)


def lower_size(acw):
    """ACW より大きい最小の候補サイズ"""
    acw = np.atleast_1d(np.asarray(acw, dtype=float))
    return np.array([SIZE_GRID[SIZE_GRID > a].min() if (SIZE_GRID > a).any() else SIZE_GRID.max()
                     for a in acw])


def add_gap(frame):
    """gap = ACW より大きい最小サイズ − ACW を列として追加する"""
    frame = frame.copy()
    frame['gap'] = lower_size(frame['ACW']) - frame['ACW'].to_numpy(dtype=float)
    return frame


# ============================================================
# モデル部品（pickle に保存されるため、クラス名とこのファイル名を変えないこと）
# ============================================================
class ScaledRegressor:
    """RobustScaler + 回帰モデル。列の順番は features で固定する"""
    def __init__(self, estimator, features):
        self.estimator = estimator
        self.features = list(features)

    def _x(self, X):
        return X[self.features].to_numpy(dtype=float)

    def fit(self, X, y):
        self.scaler_ = RobustScaler().fit(self._x(X))
        self.model_ = clone(self.estimator).fit(self.scaler_.transform(self._x(X)), y)
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(self._x(X)))


class StepClassifier:
    """RobustScaler + 分類モデル。学習データに無い段階の確率は 0 として N_STEPS 列を返す"""
    def __init__(self, estimator, features):
        self.estimator = estimator
        self.features = list(features)

    def _x(self, X):
        return X[self.features].to_numpy(dtype=float)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.scaler_ = RobustScaler().fit(self._x(X))
        if len(self.classes_) == 1:
            self.model_ = None
            return self
        remap = {c: i for i, c in enumerate(self.classes_)}
        self.model_ = clone(self.estimator).fit(self.scaler_.transform(self._x(X)),
                                                np.array([remap[c] for c in y]))
        return self

    def predict_proba(self, X):
        out = np.zeros((len(X), N_STEPS))
        if self.model_ is None:
            out[:, self.classes_[0]] = 1.0
            return out
        pr = self.model_.predict_proba(self.scaler_.transform(self._x(X)))
        for j, local in enumerate(self.model_.classes_):
            out[:, self.classes_[int(local)]] = pr[:, j]
        return out


class StepPrior:
    """比較基準: 学習データでの各段階の出現割合をそのまま確率にする"""
    def fit(self, X, y):
        self.p_ = np.bincount(np.asarray(y, dtype=int), minlength=N_STEPS) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


class AverageRegressor:
    """複数の回帰モデルの予測の平均"""
    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


class AverageClassifier:
    """複数の分類モデルの確率の平均"""
    def __init__(self, models):
        self.models = list(models)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


# ============================================================
# 予測
# ============================================================
def load_bundle(path=DEFAULT_BUNDLE):
    if not os.path.exists(path):
        raise FileNotFoundError(f'モデルファイルが見つかりません: {path}\n'
                                f'train_model.py（または retrain.bat）を実行して作成してください。')
    with open(path, 'rb') as f:
        return pickle.load(f)


def vault_label(v_mm):
    if VAULT_NARROW[0] <= v_mm <= VAULT_NARROW[1]:
        return '◎ 400-600µm'
    if VAULT_RANGE[0] <= v_mm <= VAULT_RANGE[1]:
        return '○ 250-750µm'
    return '△ 低い (<250µm)' if v_mm < VAULT_RANGE[0] else '△ 高い (>750µm)'


def check_inputs(bundle, age, LV, ACD, ACW):
    """学習データの範囲外の入力を知らせる"""
    msgs = []
    for name, v in [('age', age), ('LV', LV), ('ACD', ACD), ('ACW', ACW)]:
        r = bundle['feature_ranges'][name]
        if not (r['min'] <= float(v) <= r['max']):
            msgs.append(f'{name} = {float(v):g} は学習データの範囲（{r["min"]:g}〜{r["max"]:g}）の外です。'
                        f'予測の信頼性が下がります。')
    return msgs


def predict_eye(bundle, age, LV, ACD, ACW):
    """1眼分の予測。ACW より大きい候補サイズごとに確率と予測Vaultを返す"""
    x = add_gap(pd.DataFrame([{'age': float(age), 'LV': float(LV), 'ACD': float(ACD), 'ACW': float(ACW)}]))
    lower = float(lower_size(ACW)[0])
    sizes = np.round(lower + STEP_MM * np.arange(N_STEPS), 2)
    probs = bundle['size_prob']['model'].predict_proba(x)[0]

    keep = sizes <= SIZE_GRID.max()
    sizes, probs = sizes[keep], probs[keep]
    probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(sizes), 1 / len(sizes))

    grid = pd.DataFrame({'LV': LV, 'ACD': ACD, 'ACW': ACW, 'age': age, 'size': sizes}).astype(float)
    vaults = np.asarray(bundle['vault']['model'].predict(grid), dtype=float)

    rec = int(np.argmax(probs))
    table = pd.DataFrame({
        'ICLサイズ (mm)': [f'{s:.2f}' for s in sizes],
        '正しいサイズの確率 (%)': np.round(probs * 100, 1),
        '予測Vault (µm)': np.round(vaults * 1000).astype(int),
        'Vault判定': [vault_label(v) for v in vaults],
        '推奨': ['★' if i == rec else '' for i in range(len(sizes))],
    })

    warnings_ = check_inputs(bundle, age, LV, ACD, ACW)
    s_lo, s_hi = bundle['size_range_train']
    outside = [f'{s:.2f}' for s in sizes if s < s_lo or s > s_hi]
    if outside:
        warnings_.append(f'サイズ {", ".join(outside)}mm は学習データに無いサイズです。'
                         f'そのサイズの予測Vaultは参考程度にしてください。')

    return {'table': table, 'sizes': sizes, 'probs': probs, 'vaults': vaults,
            'recommended_size': float(sizes[rec]), 'recommended_prob': float(probs[rec]),
            'recommended_vault': float(vaults[rec]), 'lower_size': lower, 'warnings': warnings_}


def make_figure(res, title=''):
    """確率と予測Vaultの棒グラフ（文字化けを避けるためラベルは英語）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sizes, probs, vaults = res['sizes'], res['probs'], res['vaults']
    labels = [f'{s:.2f}' for s in sizes]
    rec = list(sizes).index(res['recommended_size'])
    colors = ['#E91E63' if i == rec else '#90CAF9' for i in range(len(sizes))]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))

    a1.bar(labels, probs * 100, color=colors, edgecolor='black', linewidth=0.5)
    for i, p in enumerate(probs):
        a1.text(i, p * 100 + 1.5, f'{p * 100:.0f}%', ha='center', fontsize=9)
    a1.set_ylim(0, probs.max() * 100 + 15)
    a1.set_xlabel('ICL size (mm)', fontweight='bold')
    a1.set_ylabel('Probability (%)', fontweight='bold')
    a1.set_title('Probability of correct size', fontweight='bold')

    a2.axhspan(VAULT_RANGE[0] * 1000, VAULT_RANGE[1] * 1000, color='#4CAF50', alpha=0.12, label='250-750 um')
    a2.axhspan(VAULT_NARROW[0] * 1000, VAULT_NARROW[1] * 1000, color='#4CAF50', alpha=0.22, label='400-600 um')
    a2.bar(labels, vaults * 1000, color=colors, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(vaults):
        a2.text(i, v * 1000 + 15, f'{v * 1000:.0f}', ha='center', fontsize=9)
    a2.set_ylim(0, max(1100, vaults.max() * 1000 + 150))
    a2.set_xlabel('ICL size (mm)', fontweight='bold')
    a2.set_ylabel('Predicted vault (um)', fontweight='bold')
    a2.set_title('Predicted vault', fontweight='bold')
    a2.legend(fontsize=7, loc='upper left')

    if title:
        fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    return fig


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description='IPCL Vault / 正しいサイズ予測')
    ap.add_argument('--age', type=float, required=True)
    ap.add_argument('--LV', type=float, required=True)
    ap.add_argument('--ACD', type=float, required=True)
    ap.add_argument('--ACW', type=float, required=True)
    ap.add_argument('--bundle', default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    res = predict_eye(load_bundle(a.bundle), a.age, a.LV, a.ACD, a.ACW)
    print(f"推奨サイズ: {res['recommended_size']:.2f}mm"
          f"（確率 {res['recommended_prob'] * 100:.0f}% / 予測Vault {res['recommended_vault'] * 1000:.0f}µm）")
    print(res['table'].to_string(index=False))
    for w in res['warnings']:
        print('注意:', w)


if __name__ == '__main__':
    main()
