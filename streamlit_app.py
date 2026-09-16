"""
IPCL Vault / サイズ予測アプリ（Windows ローカル実行用）
=====================================================
起動: start.bat をダブルクリック
      （手動なら  .venv\\Scripts\\python -m streamlit run app.py ）
モデル: models/ipcl_bundle.pkl（train_model.py で作成）
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ipcl_predict import DEFAULT_BUNDLE, VAULT_NARROW, VAULT_RANGE, load_bundle, make_figure, predict_eye

st.set_page_config(page_title='IPCL Vault / Size Predictor', page_icon='👁️', layout='wide')


@st.cache_resource
def get_bundle():
    return load_bundle(DEFAULT_BUNDLE)


try:
    B = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

V, P = B['vault'], B['size_prob']

# ============================================================
# ヘッダー
# ============================================================
st.title('👁️ IPCL Vault / Size Predictor')
st.caption('術前の測定値から、ACW より大きい候補サイズごとに「正しいサイズ」である確率と予測Vaultを表示します')
st.markdown(
    f"**学習データ**: {B['n_eyes']}眼 / {B['n_patients']}人 ・ "
    f"**Vault予測**: {V['best_name']}（誤差の目安 RMSE {V['cv']['RMSE'] * 1000:.0f}µm） ・ "
    f"**正しいサイズ**: {P['best_name']}（最高確率サイズが正解 {P['cv']['Top1']:.0f}% / "
    f"上位2サイズに正解 {P['cv']['Top2']:.0f}%）"
)

# 学習時と実行時のライブラリのバージョン違いを検出（pickle の互換性の問題を切り分けやすくする）
try:
    import catboost
    import lightgbm
    import sklearn
    import xgboost
    _now = {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
            'xgboost': xgboost.__version__, 'lightgbm': lightgbm.__version__,
            'catboost': catboost.__version__}
    _diff = {k: (v, _now[k]) for k, v in B['versions'].items() if k in _now and v != _now[k]}
    if _diff:
        st.warning('学習時とバージョンが異なるライブラリがあります: '
                   + ', '.join(f'{k} {a} → {b}' for k, (a, b) in _diff.items())
                   + '。予測結果が変わる可能性があります。requirements.txt の版で入れ直してください。')
except ImportError:
    pass

st.markdown('---')

# ============================================================
# 入力
# ============================================================
RNG = B['feature_ranges']


def num_input(label, name, key, step, fmt):
    r = RNG[name]
    pad = (r['max'] - r['min']) * 0.5
    return st.number_input(label, min_value=float(round(r['min'] - pad, 3)),
                           max_value=float(round(r['max'] + pad, 3)),
                           value=float(round(r['median'], 3)), step=step, format=fmt, key=key)


with st.sidebar:
    st.header('患者情報')
    age = st.number_input('年齢', min_value=10, max_value=90, value=int(round(RNG['age']['median'])), step=1)
    both = st.checkbox('両眼を入力する', value=True)
    st.markdown('---')
    st.caption(f"学習データの範囲: 年齢 {RNG['age']['min']:.0f}〜{RNG['age']['max']:.0f}歳 / "
               f"LV {RNG['LV']['min']:.2f}〜{RNG['LV']['max']:.2f} / "
               f"ACD {RNG['ACD']['min']:.2f}〜{RNG['ACD']['max']:.2f} / "
               f"ACW {RNG['ACW']['min']:.2f}〜{RNG['ACW']['max']:.2f}")

eyes = [('右眼 (OD)', 'R', 'Right eye (OD)'), ('左眼 (OS)', 'L', 'Left eye (OS)')] if both \
    else [('右眼 (OD)', 'R', 'Right eye (OD)')]
inputs = {}
for col, (name, k, _) in zip(st.columns(len(eyes)), eyes):
    with col:
        st.subheader(name)
        inputs[name] = {
            'LV': num_input('LV — Lens Vault (mm)', 'LV', f'lv{k}', 0.01, '%.3f'),
            'ACD': num_input('ACD — Anterior Chamber Depth (mm)', 'ACD', f'acd{k}', 0.01, '%.2f'),
            'ACW': num_input('ACW — Anterior Chamber Width (mm)', 'ACW', f'acw{k}', 0.01, '%.2f'),
        }

st.markdown('---')

# ============================================================
# 結果
# ============================================================
if st.button('予測する', type='primary'):
    for col, (name, _, name_en) in zip(st.columns(len(eyes)), eyes):
        with col:
            st.subheader(f'{name} の結果')
            v = inputs[name]
            res = predict_eye(B, age, v['LV'], v['ACD'], v['ACW'])
            rv = res['recommended_vault']

            # 両眼表示だと幅が狭く数値が切れるため、metric は1つだけにして残りは文章で出す
            st.metric('推奨サイズ', f"{res['recommended_size']:.2f} mm")
            st.markdown(f"**正しいサイズの確率**: {res['recommended_prob'] * 100:.0f} %　／　"
                        f"**そのときの予測Vault**: {rv * 1000:.0f} µm")

            if VAULT_NARROW[0] <= rv <= VAULT_NARROW[1]:
                st.success('予測Vaultは狭義の適正域（400–600µm）です。')
            elif VAULT_RANGE[0] <= rv <= VAULT_RANGE[1]:
                st.info('予測Vaultは広義の適正域（250–750µm）です。')
            elif rv < VAULT_RANGE[0]:
                st.warning('予測Vaultが低めです（<250µm）。')
            else:
                st.warning('予測Vaultが高めです（>750µm）。')

            st.caption(f"ACW {v['ACW']:.2f} mm → 候補は **{res['lower_size']:.2f} mm 以上** のサイズ")
            for w in res['warnings']:
                st.warning(w)

            fig = make_figure(res, name_en)
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(res['table'], hide_index=True)
else:
    st.info('左のサイドバーと上の入力欄を埋めて「予測する」を押してください。')

# ============================================================
# モデルの詳細
# ============================================================
st.markdown('---')
with st.expander('モデルの詳細と注意事項'):
    st.markdown(f"""
**学習データ**: {B['data_file']} — {B['n_eyes']}眼 / {B['n_patients']}人（作成: {B['created']}）

**性能の推定方法**: {B['cv_method']}。両眼データのため、同じ患者の左右眼は必ず同じ側に入れて評価しています。

**モデルの選び方**: 線形モデルと決定木系モデルの計7種類を比較し、
**アンサンブル（上位3モデルの平均）を第一選択**として採用しています。
上位3つに決定木系が入らない場合は、最も良い決定木系モデルを加えて4モデルの平均にします。

**Vault予測**: 特徴量 {', '.join(V['features'])} → 採用 **{V['best_name']}**
（単独最良は {V['best_single']}: RMSE {V['cv_best_single']['RMSE'] * 1000:.0f}µm ／
アンサンブル: RMSE {V['cv']['RMSE'] * 1000:.0f}µm）
""")
    st.dataframe(pd.DataFrame(V['cv_all']).T.sort_values('RMSE').rename(columns={
        'R2': 'R²', 'RMSE': 'RMSE (mm)', 'MAE': 'MAE (mm)', 'within_100um': '±100µm以内 (%)'}).round(3))
    st.markdown(f"""
**正しいサイズの確率**: 「ACW より大きい最小サイズから何段階（0.25mm）上が正しいサイズか」を分類で予測。
特徴量 {', '.join(P['features'])}（gap = ACW より大きい最小サイズ − ACW）→ 採用 **{P['best_name']}**
（単独最良は {P['best_single']}: LogLoss {P['cv_best_single']['LogLoss']:.3f} ／
アンサンブル: LogLoss {P['cv']['LogLoss']:.3f} ／ 比較基準: LogLoss {P['baseline']['LogLoss']:.3f}）
""")
    st.dataframe(pd.DataFrame(P['cv_all']).T.sort_values('LogLoss').rename(columns={
        'Top1': '最高確率サイズ=正解 (%)', 'Top2': '上位2サイズに正解 (%)',
        'p_true': '正解サイズの平均確率 (%)'}).round(3))
    st.markdown(f"""
---
⚠️ **本アプリは研究目的の参考情報です。** 予測Vaultの誤差は RMSE で {V['cv']['RMSE'] * 1000:.0f}µm 程度あり、
最も確率の高いサイズが正しいサイズと一致するのは {P['cv']['Top1']:.0f}% です。
最終的なレンズ選択は必ず医師の判断で行ってください。
""")
