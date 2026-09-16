# IPCL Vault / Size Predictor（Streamlit Cloud 版）

術前の測定値（年齢・LV・ACD・ACW）から、**ACW より大きい候補サイズごと**に

- 「正しいサイズ」である**確率**
- そのサイズを入れたときの**予測Vault**

を表示し、最も確率の高いサイズを**推奨サイズ**として示すウェブアプリです。

---

## ⚠️ 公開範囲について（先にお読みください）

Streamlit Community Cloud にデプロイすると、**URL を知っている人が誰でも開ける状態**が既定です。
院内だけで使う場合は、次のどちらかを必ず設定してください。

- デプロイ後に **Manage app → Settings → Sharing** で閲覧できる人を限定する（招待したメールアドレスのみ）
- または GitHub リポジトリを **Private** にしたうえで、上記の Sharing も限定する

このリポジトリに**患者データ（xlsx）は含めません**（`.gitignore` で除外済み）。
`models/ipcl_bundle.pkl` には学習済みモデルと要約統計（各変数の最小・最大・中央値）だけが入っており、
患者ごとのデータは含まれていません。

---

## デプロイ手順

1. **GitHub にリポジトリを作る**（院内利用なら Private を推奨）
2. **このフォルダの中身をすべてアップロードする**
   - `models/ipcl_bundle.pkl` も必ず含めてください（アプリの動作に必要です）
   - GitHub の Web UI からアップロードする場合は、`models` フォルダを作ってからその中に入れてください
3. **https://share.streamlit.io** を開き、GitHub アカウントで連携する
4. 「Create app」→ リポジトリを選び、次を指定する
   - **Main file path**: `streamlit_app.py`
   - **Advanced settings → Python version**: `3.11` ← **必ず指定してください**
     （Python のバージョンはこの画面でしか指定できません。`runtime.txt` のようなファイルを
     置いても読まれません）
5. 「Deploy」を押す（初回は5〜10分ほどかかります）
6. 起動したら **Manage app → Settings → Sharing** で公開範囲を設定する

### デプロイが進まない・いつまでも起動しないとき

エラーが出ないまま止まる場合は、インストールするライブラリが重すぎて無料枠の上限に
達していることがあります。次を確認してください。

1. `requirements.txt` に **xgboost / lightgbm が入っていないこと**
   （この2つはモデルの学習にだけ必要で、アプリの動作には不要です。合計で約135MBあります）
2. `requirements.txt` で **Streamlit のバージョンを固定していないこと**
   （Streamlit Cloud は Streamlit を自前で管理しているため、`streamlit==1.63.0` のように
   固定すると起動に失敗することがあります。`streamlit` とだけ書きます）
3. Advanced settings の **Python version が 3.11** になっていること
4. それでも進まないときは **Manage app → Reboot app**、
   それでもだめなら一度アプリを削除して作り直す（Delete → Create app）

---

## ファイル構成

```
├── streamlit_app.py       画面（Streamlit Cloud の Main file path に指定）
├── ipcl_predict.py        予測の計算
├── train_model.py         モデルの学習（手元で実行するスクリプト）
├── requirements.txt       アプリの実行に必要なライブラリ（モデル読み込みに関わる分だけ固定）
├── requirements-train.txt 学習し直すときだけ必要なライブラリ（クラウドでは未使用）
├── packages.txt           Linux 側の追加パッケージ（CatBoost 用）
├── .gitignore             患者データを誤って上げないための設定
├── .streamlit/config.toml 画面の配色
└── models/
    ├── ipcl_bundle.pkl    学習済みモデル（全症例で学習）
    └── metrics.json       交差検証の結果
```

---

## モデルについて

- 学習データ: IPCL8.xlsx（**全194眼 / 98人**）で最終モデルを構築
- 性能の推定: **患者単位 LOOCV**（同じ患者の左右眼は必ず同じ側に入れて評価）
- **アンサンブル（上位3モデルの平均）を第一選択**として採用。上位3つに決定木系（RF・XGBoost・LightGBM・CatBoost）が
  入らない場合は、最も良い決定木系モデルを加えて4モデルの平均にします

| | 採用モデル | 交差検証での精度 |
|---|---|---|
| Vault予測 | Ridge + MLR + SVR (Linear) + CatBoost | RMSE 120µm / R² 0.263 |
| 正しいサイズの確率 | CatBoost + RF + LogReg | 最高確率サイズが正解 60.3% / 上位2サイズに正解 87.1% |

詳しい数値はアプリ下部の「モデルの詳細と注意事項」と `models/metrics.json` にあります。

### データを更新してモデルを作り直す場合

クラウド上では学習できません（患者データを置かないため）。手元のパソコンで作り直してください。

```bash
pip install -r requirements-train.txt
python train_model.py --data /path/to/IPCL8.xlsx
```

`models/ipcl_bundle.pkl` と `models/metrics.json` が更新されるので、その2ファイルを GitHub に push すると
Streamlit Cloud が自動で再起動して新しいモデルになります。

---

## うまく動かないとき

| 症状 | 対処 |
|---|---|
| デプロイが進まない・エラーも出ない | 上の「デプロイが進まない・いつまでも起動しないとき」を参照 |
| 「モデルファイルが見つかりません」 | `models/ipcl_bundle.pkl` がリポジトリに含まれているか確認（GitHub 上でフォルダ構成を確認） |
| インストールでエラー | Advanced settings の Python version が `3.11` になっているか確認 |
| `.gitignore` や `.streamlit` が GitHub に無い | 先頭がドットのファイルは Finder やブラウザで見えないため、アップロード時に漏れがちです。`.gitignore` は患者データを誤って上げないための設定なので、追加をおすすめします |
| 「Oh no. Error running app.」と表示される | `requirements.txt` で Streamlit のバージョンを固定していないか確認。固定している場合は `streamlit` とだけ書いて Commit し、Reboot app |
| 「バージョンが異なるライブラリがあります」と表示される | モデル読み込みに関わる行（numpy〜catboost）は変更せずにそのまま使ってください |

---

⚠️ **本アプリは研究目的の参考情報です。** 予測Vaultの誤差は RMSE で約120µm あり、
最も確率の高いサイズが正しいサイズと一致するのは約60%です。
最終的なレンズ選択は必ず医師の判断で行ってください。
