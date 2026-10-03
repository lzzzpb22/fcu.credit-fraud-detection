#Credit Card Fraud Risk Decision Prototype
#本專題使用 Kaggle Credit Card Fraud Detection 歷史交易資料，建立 Logistic Regression、Isolation Forest 與 XGBoost 基準比較，並以 Validation Set 進行 FN:FP 成本敏感門檻最佳化，再用 Test Set 做最終驗證。Streamlit App 讀取離線訓練產生的模型與研究結果，提供交易回放、四級風控處置、成本情境、Top-K 人力限制與 SHAP 全域解釋。
#專案檔案
- train_pipeline.py：離線訓練、驗證、成本門檻搜尋、Test 評估、SHAP、輸出 artifacts。
- app.py：Streamlit 展示與實際載入 XGBoost 模型推論。
- requirements.txt：Python 套件。
- creditcard_part1.csv、creditcard_part2.csv：切分後資料集。
- artifacts/：執行訓練程式後自動產生。
#第一次執行
pip install -r requirements.txt
python train_pipeline.py
streamlit run app.py
#artifacts 自動產生
#執行 python train_pipeline.py 後會產生：
- artifacts/xgb_model.json
- artifacts/lr_model.pkl
- artifacts/iso_model.pkl
- artifacts/scaler.pkl
- artifacts/feature_cols.json
- artifacts/thresholds.json
- artifacts/metrics.json
- artifacts/shap_importance.json
#若部署到 Streamlit Community Cloud，請先在本機或 Codespaces 執行訓練，再把 artifacts/ 一起推到 GitHub。
#研究方法提醒
- 資料先依 Time 排序，再依 70% / 15% / 15% 做 Train / Validation / Test 時間切分。
- SMOTE 只用於 Logistic Regression 的 Train Set；Test Set 不做過抽樣。
- XGBoost 以 scale_pos_weight 處理類別不平衡。
- 5:1、10:1、20:1 的門檻只在 Validation Set 搜尋；Test Set 只做最終驗證。
- App 的模型績效與門檻從 artifacts/*.json 讀取，不手動寫死。
- Kaggle V1–V28 是 PCA 去識別化特徵，不應直接命名成真實客戶屬性。
