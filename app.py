import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 頁面基本設定
st.set_page_config(
    page_title="信用卡詐欺偵測與風險儀表板",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_and_evaluate_models():
    """
    載入或初始化資料與模型，並進行訓練與評估
    """
    # 這裡假設你的專案中含有資料或預先訓練好的模型
    # 為了示範完整結構，此處以模擬載入與訓練流程為主
    
    # 模擬產生特徵與測試集資料 (實際專案中請替換為你的讀取檔案邏輯)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    X_test = np.random.rand(200, n_features)
    y_test = np.random.choice([0, 1], size=200, p=[0.95, 0.05])
    
    features = [f"Feature_{i}" for i in range(n_features)]
    test_df = pd.DataFrame(X_test, columns=features)
    test_df['Class'] = y_test

    # 資料標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 處理不平衡資料 (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # 定義並訓練模型
    # 1. Logistic Regression (此處加上 .ravel() 確保標籤為一維陣列，解決 ValueError)
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_smote, y_train_smote.ravel())
    
    # 2. XGBoost / RandomForest (此處以 RandomForest 模擬樹狀模型)
    xgb_model = RandomForestClassifier(random_state=42)
    xgb_model.fit(X_train_smote, y_train_smote.ravel())

    # 3. Isolation Forest (非監督式異常偵測)
    from sklearn.ensemble import IsolationForest
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X_train_scaled)

    # 計算預測機率
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Isolation Forest 轉換分數
    iso_scores = iso_model.decision_function(X_test_scaled)
    iso_probs = 1 / (1 + np.exp(iso_scores))  # 將分數轉化為類似機率

    return (
        xgb_model, lr_model, iso_model, scaler, 
        X_test_scaled, y_test, features, test_df, 
        xgb_probs, lr_probs, iso_probs
    )

def main():
    st.title("🛡️ 信用卡詐欺偵測即時風險儀表板")
    st.markdown("本系統透過多種機器學習模型（如 Logistic Regression, XGBoost/Random Forest, Isolation Forest）進行即時交易風險評估。")

    # 載入模型與資料 (對應錯誤發生的第 122 行)
    try:
        (
            xgb_model, lr_model, iso_model, scaler, 
            X_test_scaled, y_test, features, test_df, 
            xgb_probs, lr_probs, iso_probs
        ) = load_and_evaluate_models()
    except Exception as e:
        st.error(f"載入模型或資料時發生錯誤: {e}")
        return

    # 側邊欄控制項
    st.sidebar.header("控制面板")
    model_choice = st.sidebar.selectbox(
        "選擇評估模型",
        ("Logistic Regression", "Random Forest / XGBoost", "Isolation Forest")
    )

    threshold = st.sidebar.slider("詐欺風險判定閾值 (Threshold)", 0.0, 1.0, 0.5, 0.05)

    # 主畫面顯示
    st.subheader("📊 測試集風險預測總覽")
    
    # 根據選擇顯示對應機率
    if model_choice == "Logistic Regression":
        selected_probs = lr_probs
    elif model_choice == "Random Forest / XGBoost":
        selected_probs = xgb_probs
    else:
        selected_probs = iso_probs

    col1, col2, col3 = st.columns(3)
    col1.metric("總測試交易數", len(test_df))
    col2.metric("高風險警示交易數", int(np.sum(selected_probs >= threshold)))
    col3.metric("平均風險評分", f"{np.mean(selected_probs):.2%}")

    # 顯示詳細資料表格
    st.markdown("---")
    st.subheader("📋 交易資料與預測結果")
    
    results_df = test_df.copy()
    results_df['Risk_Score'] = selected_probs
    results_df['Is_Fraud_Alert'] = results_df['Risk_Score'] >= threshold
    
    st.dataframe(results_df.style.highlight_greaterthan(subset=['Risk_Score'], threshold=threshold, color='#ffcccc'))

if __name__ == "__main__":
    main()
