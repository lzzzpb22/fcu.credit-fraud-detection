# -*- coding: utf-8 -*-
"""信用卡交易詐欺偵測：離線訓練、成本門檻最佳化與研究成果輸出。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
DATA_FILES = ["creditcard_part1.csv", "creditcard_part2.csv"]
COST_RATIOS = [5, 10, 20]


def load_data() -> pd.DataFrame:
    frames = []
    for filename in DATA_FILES:
        path = Path(filename)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            "找不到 creditcard_part1.csv / creditcard_part2.csv。請把兩個 CSV 放在專案根目錄。"
        )
    df = pd.concat(frames, ignore_index=True)
    if "Class" not in df.columns:
        raise ValueError("資料缺少 Class 欄位。")
    df = df.dropna(subset=["Class"]).copy()
    df["Class"] = df["Class"].astype(int)
    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame):
    """依時間順序切成 70% train / 15% validation / 15% test。"""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def cm_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def classification_metrics(y_true, probs, threshold: float):
    pred = (np.asarray(probs) >= threshold).astype(int)
    tn, fp, fn, tp = cm_counts(y_true, pred)
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def find_best_threshold_by_cost(y_true, probs, cost_fn=10, cost_fp=1):
    """只用 validation set 搜尋使 FN*cost_fn + FP*cost_fp 最小的門檻。"""
    probs = np.asarray(probs)
    thresholds = np.linspace(0.001, 0.999, 999)
    best = None
    for threshold in thresholds:
        pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = cm_counts(y_true, pred)
        weighted_cost = fn * cost_fn + fp * cost_fp
        row = {
            "threshold": float(threshold),
            "weighted_cost": int(weighted_cost),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        if best is None or row["weighted_cost"] < best["weighted_cost"]:
            best = row
    return best


def top_k_metrics(y_true, probs, k: int):
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(probs)
    k = min(k, len(y_arr))
    order = np.argsort(-p_arr)[:k]
    hits = int(y_arr[order].sum())
    positives = int(y_arr.sum())
    return {
        "k": int(k),
        "hits": hits,
        "precision_at_k": float(hits / k if k else 0),
        "recall_at_k": float(hits / positives if positives else 0),
    }


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    print("1) 載入資料")
    df = load_data()
    print(f"總筆數={len(df):,}；詐欺={int(df['Class'].sum()):,}；詐欺率={df['Class'].mean()*100:.4f}%")

    train_df, val_df, test_df = chronological_split(df)
    feature_cols = [c for c in df.columns if c not in ["Class", "Time"]]

    def xy(part):
        return part[feature_cols], part["Class"]

    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)

    print("2) 70/15/15 時間序列切分")
    for name, y_part in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        print(f"{name}: n={len(y_part):,}, fraud={int(y_part.sum())}, rate={y_part.mean()*100:.4f}%")

    print("3) 標準化；SMOTE 僅套用 Train")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)

    print("4) 訓練 Logistic Regression / XGBoost / Isolation Forest")
    lr_model = LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)
    lr_model.fit(X_train_res, y_train_res)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=neg / max(pos, 1),
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    # XGBoost 使用原始不平衡 Train；scale_pos_weight 處理類別不平衡。
    xgb_model.fit(X_train_s, y_train)

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=max(float(y_train.mean()), 0.001),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_model.fit(X_train_s[np.asarray(y_train) == 0])

    lr_val = lr_model.predict_proba(X_val_s)[:, 1]
    lr_test = lr_model.predict_proba(X_test_s)[:, 1]
    xgb_val = xgb_model.predict_proba(X_val_s)[:, 1]
    xgb_test = xgb_model.predict_proba(X_test_s)[:, 1]
    # Isolation Forest：decision_function 越大越正常，因此取負號作為風險分數。
    iso_val = -iso_model.decision_function(X_val_s)
    iso_test = -iso_model.decision_function(X_test_s)

    print("5) Validation Set 成本門檻最佳化")
    thresholds = {}
    scenarios = {}
    for cost_fn in COST_RATIOS:
        best = find_best_threshold_by_cost(y_val, xgb_val, cost_fn=cost_fn, cost_fp=1)
        key = f"{cost_fn}:1"
        thresholds[key] = best["threshold"]
        test_pred = (xgb_test >= best["threshold"]).astype(int)
        tn, fp, fn, tp = cm_counts(y_test, test_pred)
        scenarios[key] = {
            "cost_fn": cost_fn,
            "cost_fp": 1,
            "validation_best_threshold": best["threshold"],
            "validation_weighted_cost": best["weighted_cost"],
            "test_tn": tn,
            "test_fp": fp,
            "test_fn": fn,
            "test_tp": tp,
            "test_weighted_cost": int(fn * cost_fn + fp),
            "false_positives_per_10k_normal": float(fp / max(1, int((y_test == 0).sum())) * 10000),
        }
        print(key, scenarios[key])

    base_threshold = thresholds["10:1"]
    # LR 使用固定 0.5；Isolation Forest 門檻只用 Validation F1 最大化作 benchmark。
    iso_candidates = np.unique(np.quantile(iso_val, np.linspace(0.80, 0.999, 300)))
    iso_best_t, iso_best_f1 = float(iso_candidates[0]), -1.0
    for t in iso_candidates:
        f1 = f1_score(y_val, (iso_val >= t).astype(int), zero_division=0)
        if f1 > iso_best_f1:
            iso_best_t, iso_best_f1 = float(t), float(f1)

    metrics = {
        "dataset": {
            "total_n": int(len(df)),
            "fraud_n": int(df["Class"].sum()),
            "fraud_rate": float(df["Class"].mean()),
            "train_n": int(len(train_df)),
            "validation_n": int(len(val_df)),
            "test_n": int(len(test_df)),
            "test_fraud_n": int(y_test.sum()),
            "split": "chronological_70_15_15",
        },
        "models": {
            "Logistic Regression": classification_metrics(y_test, lr_test, 0.5),
            "Isolation Forest": classification_metrics(y_test, iso_test, iso_best_t),
            "XGBoost": classification_metrics(y_test, xgb_test, base_threshold),
        },
        "cost_scenarios": scenarios,
        "top_k": {
            "100": top_k_metrics(y_test, xgb_test, 100),
            "300": top_k_metrics(y_test, xgb_test, 300),
        },
    }

    # 以 Test 中成功攔截 TP 的交易金額作「資料集內可避免金額」示意，不宣稱真實銀行挽回損失。
    base_pred = (xgb_test >= base_threshold).astype(int)
    test_amount = test_df["Amount"].to_numpy() if "Amount" in test_df.columns else np.zeros(len(test_df))
    metrics["base_scenario"] = {
        "cost_ratio": "10:1",
        "threshold": float(base_threshold),
        "captured_fraud_amount": float(test_amount[(base_pred == 1) & (np.asarray(y_test) == 1)].sum()),
    }

    print("6) 計算 XGBoost 全域 SHAP 重要性（抽樣，避免記憶體過大）")
    shap_importance = {}
    try:
        import shap
        sample_n = min(3000, len(X_test))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_test), size=sample_n, replace=False)
        X_sample = pd.DataFrame(X_test_s[idx], columns=feature_cols)
        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer(X_sample)
        mean_abs = np.abs(sv.values).mean(axis=0)
        shap_importance = {
            feature: float(value)
            for feature, value in sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)
        }
    except Exception as exc:
        print(f"SHAP 計算略過：{exc}")

    print("7) 儲存模型與研究結果")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
    joblib.dump(lr_model, ARTIFACT_DIR / "lr_model.pkl")
    joblib.dump(iso_model, ARTIFACT_DIR / "iso_model.pkl")
    xgb_model.save_model(ARTIFACT_DIR / "xgb_model.json")

    with open(ARTIFACT_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(ARTIFACT_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(ARTIFACT_DIR / "shap_importance.json", "w", encoding="utf-8") as f:
        json.dump(shap_importance, f, ensure_ascii=False, indent=2)

    print("完成。請將 artifacts/ 一起提交到 GitHub，Streamlit App 會直接讀取這些結果。")


if __name__ == "__main__":
    main()
