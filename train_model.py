"""
Windows アプリ用の最終モデルを学習し、models/ipcl_bundle.pkl を作成する
=====================================================================
データ : IPCL8.xlsx（既定はこのフォルダ。--data で指定可）
評価   : 患者単位 LOOCV（Leave-One-Patient-Out）
         両眼データのため、同じ患者の左右眼は必ず同じ側（学習 or 検証）に入れる
選択   : ★ アンサンブル（上位モデルの平均）を第一選択とする
         順位は Vault 予測 = CV RMSE / 正しいサイズ確率 = CV LogLoss
         上位3モデルの平均を採用し、そこに決定木系が1つも入らない場合は
         最も良い決定木系モデルを加えて4モデルの平均にする
最終   : 全眼で学習し直して保存（pkl に個票データは保存しない）

使い方:
    python train_model.py
    python train_model.py --data "C:\\Users\\...\\IPCL8.xlsx"
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from ipcl_predict import (DEFAULT_BUNDLE, HERE, N_STEPS, STEP_MM, AverageClassifier, AverageRegressor,
                          ScaledRegressor, StepClassifier, StepPrior, add_gap, lower_size)

SEED = 2005
ENSEMBLE_N = 3                                                # アンサンブルに入れるモデル数
TREE_NAMES = {'RF', 'GBM', 'XGBoost', 'LightGBM', 'CatBoost'}  # 決定木系モデル
VAULT_FEATURES = ['LV', 'ACD', 'size', 'ACW', 'age']
PROB_FEATURES = ['ACW', 'LV', 'ACD', 'age', 'gap']
NUMERIC_COLS = ['age', 'LV', 'ACD', 'ACW', 'size', 'Vault2', '正しいサイズ']
REQUIRED_COLS = ['ID', 'RL'] + NUMERIC_COLS


# ============================================================
# データ
# ============================================================
def build_patient_id(frame):
    """患者ID。ID欠測の区間は「連続2行 = 同一患者の両眼」として復元する"""
    pid = [('' if pd.isna(v) else f'P{int(v)}') for v in frame['ID']]
    miss = [i for i, v in enumerate(pid) if v == '']
    for k in range(0, len(miss), 2):
        for j in miss[k:k + 2]:
            pid[j] = f'X{k // 2}'
    return pid


def load_data(path):
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f'データに必要な列がありません: {missing}')

    problems = []
    for c in NUMERIC_COLS:
        bad = df[c].map(lambda v: pd.notna(v) and not isinstance(v, (int, float, np.integer, np.floating)))
        if bad.any():
            problems.append(f'  {c}: 数値でない値が {int(bad.sum())}件（Excel行 {[i + 2 for i in df.index[bad][:5]]}）')
    if problems:
        raise SystemExit('データに異常があります（Excelの列ズレ・貼り付けミスの可能性）:\n' + '\n'.join(problems))

    df['pid'] = build_patient_id(df)
    n_before = len(df)
    df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
    return df, n_before - len(df)


# ============================================================
# モデル（ノートブック セル20・セル25 と同じ設定）
# ============================================================
def regressors():
    return {
        'MLR': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'SVR (Linear)': SVR(C=1.0, epsilon=0.05, kernel='linear'),
        'RF': RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                random_state=SEED, verbosity=0, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=15,
                                  subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                  verbose=-1, n_jobs=-1),
        'CatBoost': CatBoostRegressor(iterations=100, depth=3, learning_rate=0.05, l2_leaf_reg=3.0,
                                      subsample=0.8, random_seed=SEED, verbose=0,
                                      allow_writing_files=False),
    }


def classifiers():
    return {
        'LogReg': LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                     random_state=SEED, n_jobs=-1),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, eval_metric='mlogloss',
                                 random_state=SEED, verbosity=0, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, num_leaves=4,
                                   min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                                   random_state=SEED, verbose=-1, n_jobs=-1),
        'CatBoost': CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=3.0,
                                       random_seed=SEED, verbose=0, allow_writing_files=False),
    }


# ============================================================
# 評価指標
# ============================================================
def reg_metrics(y, p):
    err = y - p
    return {'R2': float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            'RMSE': float(np.sqrt((err ** 2).mean())),
            'MAE': float(np.abs(err).mean()),
            'within_100um': float((np.abs(err) <= 0.100 + 1e-9).mean() * 100)}


def prob_metrics(y, P):
    P = np.clip(P, 1e-6, 1)
    P = P / P.sum(axis=1, keepdims=True)
    n = len(y)
    p_true = P[np.arange(n), y]
    order = np.argsort(-P, axis=1)
    return {'LogLoss': float(-np.log(p_true).mean()),
            'Brier': float(np.sum((P - np.eye(N_STEPS)[y]) ** 2, axis=1).mean()),
            'Top1': float((order[:, 0] == y).mean() * 100),
            'Top2': float(((order[:, 0] == y) | (order[:, 1] == y)).mean() * 100),
            'p_true': float(p_true.mean() * 100)}


def ensemble_members(ranked):
    """アンサンブルに入れるモデルを決める。

    交差検証の上位 ENSEMBLE_N モデルを使う。ただし、そこに決定木系が
    1つも入らない場合は、最も成績の良い決定木系モデルを1つ加える。
    """
    members = list(ranked[:ENSEMBLE_N])
    if not any(m in TREE_NAMES for m in members):
        tree = next((m for m in ranked if m in TREE_NAMES), None)
        if tree is not None:
            members.append(tree)
            print(f'  （上位{ENSEMBLE_N}に決定木系がないため {tree} を追加）')
    return members


def print_table(results, cols, sort_key, ascending=True):
    tab = pd.DataFrame(results).T[cols].sort_values(sort_key, ascending=ascending)
    with pd.option_context('display.width', 200):
        print(tab.round(4).to_string())


# ============================================================
# Vault 予測
# ============================================================
def train_vault(df, splits):
    print('\n' + '=' * 70)
    print('【1】Vault 予測（回帰）  特徴量:', VAULT_FEATURES)
    print('=' * 70)
    X, y = df[VAULT_FEATURES], df['Vault2'].to_numpy(dtype=float)
    models = regressors()

    oof = {}
    for name, est in models.items():
        t = time.time()
        p = np.zeros(len(y))
        for tr, te in splits:
            p[te] = ScaledRegressor(est, VAULT_FEATURES).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        oof[name] = p
        print(f'  {name:<13} RMSE={reg_metrics(y, p)["RMSE"] * 1000:.0f}µm  ({time.time() - t:.1f}s)')

    ranked = sorted(models, key=lambda k: reg_metrics(y, oof[k])['RMSE'])
    members = ensemble_members(ranked)
    ens_name = f'Ensemble ({"+".join(members)})'
    oof[ens_name] = np.mean([oof[k] for k in members], axis=0)

    results = {k: reg_metrics(y, p) for k, p in oof.items()}
    best_single = ranked[0]
    print('\n  交差検証（患者単位 LOOCV）')
    print_table(results, ['R2', 'RMSE', 'MAE', 'within_100um'], 'RMSE')
    print(f'\n  ★ 採用: {ens_name}（アンサンブルを第一選択）')
    print(f'     単独最良 {best_single}: RMSE={results[best_single]["RMSE"] * 1000:.0f}µm  '
          f'／ アンサンブル: RMSE={results[ens_name]["RMSE"] * 1000:.0f}µm')

    final = AverageRegressor([ScaledRegressor(models[k], VAULT_FEATURES).fit(X, y) for k in members])
    return {'model': final, 'best_name': ens_name, 'features': VAULT_FEATURES,
            'cv': results[ens_name], 'cv_all': results, 'members': members,
            'best_single': best_single, 'cv_best_single': results[best_single]}


# ============================================================
# 正しいサイズの確率
# ============================================================
def train_size_prob(df, splits):
    print('\n' + '=' * 70)
    print('【2】正しいサイズの確率（分類）  特徴量:', PROB_FEATURES)
    print('=' * 70)
    d = add_gap(df)
    step_raw = np.round((d['正しいサイズ'].to_numpy(dtype=float) - lower_size(d['ACW'])) / STEP_MM).astype(int)
    y = np.clip(step_raw, 0, N_STEPS - 1)
    n_clip = int((step_raw != y).sum())
    X = d[PROB_FEATURES]

    print('  段階の分布（ACW より大きい最小サイズから何段階上か）: '
          + '  '.join(f'+{s * STEP_MM:.2f}mm={int((y == s).sum())}' for s in range(N_STEPS)))
    if n_clip:
        print(f'  ※ 候補範囲外の {n_clip}眼は最も近い候補として扱いました')

    models = classifiers()
    oof = {}
    baseline = 'Baseline (出現頻度)'
    P = np.zeros((len(y), N_STEPS))
    for tr, te in splits:
        P[te] = StepPrior().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
    oof[baseline] = P
    for name, est in models.items():
        t = time.time()
        P = np.zeros((len(y), N_STEPS))
        for tr, te in splits:
            P[te] = StepClassifier(est, PROB_FEATURES).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
        oof[name] = P
        print(f'  {name:<13} LogLoss={prob_metrics(y, P)["LogLoss"]:.3f}  ({time.time() - t:.1f}s)')

    ranked = sorted(models, key=lambda k: prob_metrics(y, oof[k])['LogLoss'])
    members = ensemble_members(ranked)
    ens_name = f'Ensemble ({"+".join(members)})'
    oof[ens_name] = np.mean([oof[k] for k in members], axis=0)

    results = {k: prob_metrics(y, P) for k, P in oof.items()}
    best_single = ranked[0]
    print('\n  交差検証（患者単位 LOOCV）  Top1=最高確率サイズが正解, Top2=上位2サイズに正解, p_true=正解サイズの平均確率')
    print_table(results, ['LogLoss', 'Brier', 'Top1', 'Top2', 'p_true'], 'LogLoss')
    print(f'\n  ★ 採用: {ens_name}（アンサンブルを第一選択）')
    print(f'     単独最良 {best_single}: LogLoss={results[best_single]["LogLoss"]:.3f}  '
          f'／ アンサンブル: LogLoss={results[ens_name]["LogLoss"]:.3f}  '
          f'／ 比較基準: LogLoss={results[baseline]["LogLoss"]:.3f}')

    final = AverageClassifier([StepClassifier(models[k], PROB_FEATURES).fit(X, y) for k in members])
    return {'model': final, 'best_name': ens_name, 'features': PROB_FEATURES,
            'cv': results[ens_name], 'cv_all': results, 'baseline': results[baseline],
            'members': members, 'best_single': best_single, 'cv_best_single': results[best_single],
            'n_steps': N_STEPS, 'n_clipped': n_clip}


# ============================================================
# main
# ============================================================
def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description='IPCL Windows アプリ用モデルの学習')
    ap.add_argument('--data', default=os.path.join(HERE, 'IPCL8.xlsx'))
    ap.add_argument('--out', default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        raise SystemExit(f'データファイルが見つかりません: {a.data}\n'
                         f'IPCL8.xlsx をこのフォルダに置くか、--data で場所を指定してください。')

    df, n_drop = load_data(a.data)
    groups = df['pid'].to_numpy()
    splits = list(LeaveOneGroupOut().split(np.zeros(len(df)), groups=groups))
    print('=' * 70)
    print('IPCL Windows アプリ用モデルの学習')
    print('=' * 70)
    print(f'データ: {os.path.basename(a.data)}  {len(df)}眼 / {df["pid"].nunique()}人'
          f'（欠測で除外 {n_drop}眼）')
    print(f'評価: 患者単位 LOOCV（Leave-One-Patient-Out, {len(splits)} fold）')

    vault = train_vault(df, splits)
    size_prob = train_size_prob(df, splits)

    import catboost
    import lightgbm
    import sklearn
    import xgboost
    bundle = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': os.path.basename(a.data),
        'n_eyes': int(len(df)),
        'n_patients': int(df['pid'].nunique()),
        'seed': SEED,
        'cv_method': '患者単位 LOOCV（Leave-One-Patient-Out）',
        'versions': {'python': '.'.join(map(str, sys.version_info[:3])), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                     'xgboost': xgboost.__version__, 'lightgbm': lightgbm.__version__,
                     'catboost': catboost.__version__},
        'vault': vault,
        'size_prob': size_prob,
        # 入力チェック用の要約統計のみ（個票データは保存しない）
        'feature_ranges': {c: {'min': float(df[c].min()), 'max': float(df[c].max()),
                               'median': float(df[c].median())} for c in ['age', 'LV', 'ACD', 'ACW']},
        'size_range_train': [float(df['size'].min()), float(df['size'].max())],
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(bundle, f)

    metrics = {k: v for k, v in bundle.items() if k not in ('vault', 'size_prob')}
    metrics['ensemble_rule'] = (f'交差検証の上位{ENSEMBLE_N}モデルの平均を第一選択。'
                                f'決定木系が入らない場合は最良の決定木系モデルを追加')
    metrics['vault'] = {'model': vault['best_name'], 'members': vault['members'],
                        'best_single': vault['best_single'], 'cv': vault['cv'],
                        'cv_best_single': vault['cv_best_single'], 'cv_all': vault['cv_all']}
    metrics['size_prob'] = {'model': size_prob['best_name'], 'members': size_prob['members'],
                            'best_single': size_prob['best_single'], 'cv': size_prob['cv'],
                            'cv_best_single': size_prob['cv_best_single'],
                            'cv_all': size_prob['cv_all'], 'baseline': size_prob['baseline']}
    with open(os.path.join(os.path.dirname(a.out), 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print(f'保存: {a.out}  ({os.path.getsize(a.out) / 1024:.0f} KB)')
    print(f'  Vault 予測       : {vault["best_name"]}  RMSE={vault["cv"]["RMSE"] * 1000:.0f}µm  '
          f'R²={vault["cv"]["R2"]:.3f}')
    print(f'  正しいサイズ確率 : {size_prob["best_name"]}  最高確率=正解 {size_prob["cv"]["Top1"]:.1f}%  '
          f'上位2 {size_prob["cv"]["Top2"]:.1f}%')
    print('=' * 70)


if __name__ == '__main__':
    main()
