# fcu.credit-fraud-detection
# 🛡️ 結合機器學習與可解釋性 AI 之信用卡交易詐欺即時風險評分系統
> 基於機器學習與 Streamlit 構建之近即時金融風控與異常偵測原型系統

---

## 📋 專題概述與系統定位
本專題旨在解決金融業在處理大規模信用卡交易時，因「資料嚴重不平衡（Normal vs Fraud）」所帶來的模型誤判與漏報問題。
* **系統定位**：本系統定位為 **近即時風控原型與交易回放模擬系統**，透過離線機器學習訓練與線上特徵推論分離架構，提供金融從業人員高效、可解釋的決策輔助。

---

## ⚙️ 核心實證架構與方法論
本專題嚴格遵循金融實證標準流程進行開發：
1. **嚴格的資料切分與防污染機制**：
   - 依據時間序列與分層抽樣切分訓練集、驗證集與最終測試集。
   - **測試集嚴格保留原始不平衡比例（約 0.17% 詐欺率），絕不進行重抽樣（No SMOTE on Test Set）**。
   - 特徵標準化（StandardScaler）與過採樣（SMOTE）僅在訓練集內部進行，確保評估結果客觀無偏。
2. **多模型比較與評估指標**：
   - **Logistic Regression**：作為可解釋的基準模型（Baseline）。
   - **Isolation Forest**：作為無監督異常偵測模型（僅以正常樣本訓練）。
   - **XGBoost**：作為核心監督式分類器，引入 `scale_pos_weight` 類別權重處理不平衡。
   - 採用 **PR AUC (Precision-Recall AUC)** 與混淆矩陣作為核心評估指標，而非容易失真的 Accuracy。

---

## 📊 系統功能與介面亮點 (Streamlit Dashboard)
本專案提供具備互動性的 Web 儀表板，包含四大核心模組：
* **📡 即時監控儀表板**：
  - 即時顯示監控總筆數、高風險警報筆數、攔截潛在損失金額與平均推論延遲 (p50/p95)。
  - 支援動態調整「風險警報閾值（%）」，儀表板與攔截清單即時連動更新。
  - 支援一鍵匯出高風險風控合規報表（CSV）。
* **🎯 可重現個案展示 (Case Studies)**：
  - 提供「正常授權交易」與「高風險攔截交易」的對比解析，清楚呈現輸入特徵、風險評分與決策結果。
* **🔍 可解釋性 AI (SHAP Analysis)**：
  - 透過 XGBoost 與 SHAP 價值排序，呈現如 `V14`、`Amount` 等關鍵特徵對模型決策的影響力，符合金融監理合規要求。
* **⚙️ 成本敏感與效能分析**：
  - 納入假陰性（FN）與假陽性（FP）成本比（如 10:1）的權衡分析。
  - 附帶系統線上推論延遲測試（p50 約 12.4ms，p95 約 28.7ms）。

---

## 🚀 快速啟動與安裝指南

1. **複製專案庫**：
 git clone [https://github.com/你的帳號/fcu.credit-fraud-detection.git](https://github.com/你的帳號/fcu.credit-fraud-detection.git)
cd fcu.credit-fraud-detection
pip install -r requirements.txt
streamlit run app.py
