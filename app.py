# -*- coding: utf-8 -*-
"""Streamlit 信用卡詐欺風控決策原型。模型與研究結果由 train_pipeline.py 產生。"""
from __future__ import annotations

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="信用卡交易詐欺風控決策原型", page_icon="🛡️", layout="wide")
ARTIFACT_DIR = Path("artifacts")
DATA_FILES = ["creditcard_part1.csv", "creditcard_part2.csv"]


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_artifacts():
    required = [
        ARTIFACT_DIR / "scaler.pkl",
        ARTIFACT_DIR / "iso_model.pkl",
        ARTIFACT_DIR / "xgb_model.json",
        ARTIFACT_DIR / "feature_cols.json",
        ARTIFACT_DIR / "thresholds.json",
        ARTIFACT_DIR / "metrics.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少模型產物：" + ", ".join(missing) + "。請先執行 python train_pipeline.py")

    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    iso_model = joblib.load(ARTIFACT_DIR / "iso_model.pkl")
    xgb_model = XGBClassifier()
    xgb_model.load_model(ARTIFACT_DIR / "xgb_model.json")
    feature_cols = read_json(ARTIFACT_DIR / "feature_cols.json")
    thresholds = read_json(ARTIFACT_DIR / "thresholds.json")
    metrics = read_json(ARTIFACT_DIR / "metrics.json")
    shap_imp = read_json(ARTIFACT_DIR / "shap_importance.json") if (ARTIFACT_DIR / "shap_importance.json").exists() else {}
    return scaler, iso_model, xgb_model, feature_cols, thresholds, metrics, shap_imp


@st.cache_data
def load_data():
    frames = []
    for filename in DATA_FILES:
        path = Path(filename)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).dropna(subset=["Class"]).copy()
    df["Class"] = df["Class"].astype(int)
    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)
    return df


try:
    scaler, iso_model, xgb_model, feature_cols, thresholds, metrics, shap_imp = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

df = load_data()
if df is None:
    st.error("找不到 creditcard_part1.csv / creditcard_part2.csv。")
    st.stop()

# 與離線研究一致：App 展示最後 15% Test Set，避免把 Train 結果當成測試績效。
test_start = int(len(df) * 0.85)
df_eval = df.iloc[test_start:].copy()
X_eval = df_eval[feature_cols]
X_eval_s = scaler.transform(X_eval)
df_eval["xgb_prob"] = xgb_model.predict_proba(X_eval_s)[:, 1]
df_eval["iso_score"] = -iso_model.decision_function(X_eval_s)

st.title("🛡️ 結合機器學習與成本敏感決策之信用卡交易詐欺風控原型")
st.caption("逢甲大學財金專題｜離線模型實證 + Streamlit 近即時交易回放與決策支援")

cost_label = st.sidebar.selectbox("FN : FP 成本權重", ["5:1", "10:1", "20:1"], index=1)
recommended = float(thresholds[cost_label])
threshold = st.sidebar.slider(
    "XGBoost 決策門檻",
    min_value=0.01,
    max_value=0.99,
    value=float(round(recommended, 2)),
    step=0.01,
    help=f"{cost_label} 情境下，由 Validation Set 最小化加權成本得到的推薦門檻約為 {recommended:.3f}",
)
st.sidebar.caption(f"Validation 推薦值：{recommended:.3f}。目前可手動調整以觀察風控取捨。")

df_eval["is_alert"] = (df_eval["xgb_prob"] >= threshold).astype(int)
tp = int(((df_eval.is_alert == 1) & (df_eval.Class == 1)).sum())
fp = int(((df_eval.is_alert == 1) & (df_eval.Class == 0)).sum())
fn = int(((df_eval.is_alert == 0) & (df_eval.Class == 1)).sum())
normal_n = int((df_eval.Class == 0).sum())
avoided_amount = float(df_eval.loc[(df_eval.is_alert == 1) & (df_eval.Class == 1), "Amount"].sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Test 交易數", f"{len(df_eval):,}")
c2.metric("真實詐欺率", f"{df_eval.Class.mean()*100:.4f}%")
c3.metric("警報率", f"{df_eval.is_alert.mean()*100:.3f}%")
c4.metric("每萬筆正常交易誤報", f"{fp/max(normal_n,1)*10000:.2f}")
c5.metric("資料集內攔截詐欺金額", f"${avoided_amount:,.2f}")
st.caption("『攔截詐欺金額』為資料集內 TP 交易 Amount 加總，用於研究比較，不代表銀行實際已挽回損失。")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 交易回放",
    "📊 模型比較",
    "💰 成本門檻",
    "👥 Top-K 人力限制",
    "🧠 XAI 與研究限制",
])


def action_from_prob(p: float, block_t: float):
    # Block 優先。若成本情境使 block_t < 0.70，Review 區間會自然縮小。
    if p >= block_t:
        return "🚨 Block"
    if p >= 0.70:
        return "⚠️ Manual Review"
    if p >= 0.50:
        return "📱 OTP/3DS"
    return "✅ Pass"


with tab1:
    st.subheader("近即時交易回放與四級處置")
    n = st.slider("抽樣交易筆數", 5, 50, 10)
    if st.button("執行交易回放"):
        sample = df_eval.sample(n=min(n, len(df_eval))).copy()
        sample["風險機率"] = sample["xgb_prob"]
        sample["處置"] = sample["xgb_prob"].apply(lambda p: action_from_prob(float(p), threshold))
        cols = [c for c in ["Time", "Amount", "風險機率", "iso_score", "處置", "Class"] if c in sample.columns]
        st.dataframe(sample[cols], use_container_width=True)

with tab2:
    st.subheader("共同 Test Set 模型比較")
    rows = []
    for model_name, m in metrics["models"].items():
        rows.append({
            "模型": model_name,
            "PR-AUC": m["pr_auc"],
            "ROC-AUC": m["roc_auc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "FP": m["fp"],
            "FN": m["fn"],
            "門檻": m["threshold"],
        })
    st.dataframe(pd.DataFrame(rows).style.format({
        "PR-AUC": "{:.4f}", "ROC-AUC": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1": "{:.4f}", "門檻": "{:.3f}"
    }), use_container_width=True)
    st.caption("所有數字由 train_pipeline.py 實際產生，不在 App 中手動寫死。")

with tab3:
    st.subheader("Validation 成本最佳化 → Test 無偏驗證")
    scenario_rows = []
    for key, s in metrics["cost_scenarios"].items():
        scenario_rows.append({
            "FN:FP": key,
            "Validation 最佳門檻": s["validation_best_threshold"],
            "Validation 加權成本": s["validation_weighted_cost"],
            "Test TP": s["test_tp"],
            "Test FP": s["test_fp"],
            "Test FN": s["test_fn"],
            "Test 加權成本": s["test_weighted_cost"],
            "每萬筆正常交易誤報": s["false_positives_per_10k_normal"],
        })
    st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True)
    st.info("這裡的『成本』是研究設定的相對權重，不等同真實新台幣損失。研究所可再加入交易金額、人工審核成本與客戶摩擦成本。")

with tab4:
    st.subheader("有限人工審核能力：Top-K")
    top_rows = []
    for _, r in metrics["top_k"].items():
        top_rows.append({
            "每日/每批審核 K 筆": r["k"],
            "命中詐欺": r["hits"],
            "Precision@K": r["precision_at_k"],
            "Recall@K": r["recall_at_k"],
        })
    st.dataframe(pd.DataFrame(top_rows), use_container_width=True)
    st.caption("Top-K 用來模擬風控人力有限時，優先審核最高風險交易的效果。")

with tab5:
    st.subheader("XGBoost 全域特徵重要性（SHAP）")
    if shap_imp:
        imp = pd.DataFrame(list(shap_imp.items()), columns=["Feature", "Mean |SHAP|"]).head(12).set_index("Feature")
        st.bar_chart(imp)
    else:
        st.info("本次訓練未產生 SHAP 檔案。")

    st.subheader("單筆模型推論延遲")
    if st.button("量測 100 次 XGBoost 推論"):
        one = X_eval_s[0:1]
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = xgb_model.predict_proba(one)[:, 1]
            times.append((time.perf_counter() - t0) * 1000)
        st.metric("p50", f"{np.percentile(times,50):.3f} ms")
        st.metric("p95", f"{np.percentile(times,95):.3f} ms")

    st.warning("Kaggle V1–V28 為 PCA 去識別化特徵，不能直接解讀成持卡人年齡、消費類別等真實業務欄位。")
    st.markdown("**研究定位：** 本系統以歷史離線資料做模型驗證與交易回放，是風控決策原型，不是直接串接銀行核心系統的正式線上生產系統。")
