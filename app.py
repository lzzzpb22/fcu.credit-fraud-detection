import io
import os
import zipfile
from collections import Counter
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("1. 載入並檢查資料...")

# 自動偵測多種資料來源（支援 Colab zip、本機 CSV 或切分檔）
df = None
if os.path.exists('/content/sample_data/creditcard_fixed.zip'):
  with zipfile.ZipFile('/content/sample_data/creditcard_fixed.zip', 'r') as z:
    with z.open(z.namelist()[0]) as f:
      df = pd.read_csv(f)
elif os.path.exists('creditcard_part1.csv') and os.path.exists(
    'creditcard_part2.csv'
):
  df1 = pd.read_csv('creditcard_part1.csv')
  df2 = pd.read_csv('creditcard_part2.csv')
  df = pd.concat([df1, df2], ignore_index=True)
elif os.path.exists('creditcard.csv'):
  df = pd.read_csv('creditcard.csv')
else:
  raise FileNotFoundError(
      '找不到資料集檔案！請確認 creditcard_fixed.zip 或 CSV 檔案是否存在當前路徑。'
  )

# 清理空值並確保 Class 為整數型態
df = df.dropna(subset=['Class'])
df['Class'] = df['Class'].astype(int)

# 依時間排序 (Time 欄位)
if 'Time' in df.columns:
  df = df.sort_values('Time').reset_index(drop=True)

X = df.drop(columns=['Class', 'Time'] if 'Time' in df.columns else ['Class'])
y = df['Class']

total_fraud = int(y.sum())
print(
    f"資料總筆數: {len(df)}, 原始詐欺總數: {total_fraud} ({y.mean()*100:.4f}%)"
)

# 關鍵防呆檢查
if total_fraud == 0:
  raise ValueError(
      "【嚴重錯誤】載入的資料集裡 Class=1 (詐欺) 筆數為 0！請檢查讀取到的 CSV"
      " 是否為空或資料被過濾掉了。"
  )

# ==========================================
# 2. 時間序列/分層切分：嚴格保留未抽樣的測試集
# ==========================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(
    f"訓練集大小: {len(X_train)} (詐欺: {y_train.sum()} 筆), 驗證集大小:"
    f" {len(X_val)}, 測試集大小: {len(X_test)} (詐欺: {y_test.sum()} 筆)"
)

# ==========================================
# 3. 特徵標準化與僅在訓練集套用 SMOTE
# ==========================================
from imblearn.over_sampling import SMOTE

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 動態設定 k_neighbors 避免正例數量過少造成報錯
pos_train = int(y_train.sum())
k_neighbors = min(5, pos_train - 1) if pos_train > 1 else 1

smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"SMOTE 平衡後訓練集分佈: {Counter(y_train_res)}")

# ==========================================
# 4. 模型訓練 (Logistic Regression, XGBoost, Isolation Forest)
# ==========================================
# A. Logistic Regression (基準模型)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_res, y_train_res)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

# B. XGBoost (加入 scale_pos_weight 處理不平衡)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = neg_count / max(1, pos_count)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric='logloss',
)
xgb_model.fit(X_train_scaled, y_train)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# C. Isolation Forest (無監督異常偵測，僅用正常樣本訓練)
X_train_normal = X_train_scaled[y_train == 0]
iso_model = IsolationForest(
    n_estimators=100, contamination=0.002, random_state=42
)
iso_model.fit(X_train_normal)
iso_scores_raw = iso_model.decision_function(X_test_scaled)
iso_probs = 1 / (1 + np.exp(iso_scores_raw))

print("模型訓練與推論完成！")

# ==========================================
# 5. 驗證集門檻選擇與成本分析
# ==========================================
val_probs = xgb_model.predict_proba(X_val_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"驗證集選出的最佳決策門檻 (Threshold): {best_threshold:.4f}")


def calculate_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
  tn, fp, fn, tp = cm.ravel()
  total_cost = (fn * cost_fn) + (fp * cost_fp)
  return total_cost, tn, fp, fn, tp


# 在測試集套用最佳門檻進行最終評估
y_test_pred = (xgb_probs >= best_threshold).astype(int)
print("\n=== XGBoost 測試集最終評估報告 ===")
print(classification_report(y_test, y_test_pred, labels=[0, 1]))

cost_val, tn, fp, fn, tp = calculate_cost(
    y_test, y_test_pred, cost_fn=10, cost_fp=1
)
print(
    f"成本分析 (FN成本=10, FP成本=1) -> 總成本: {cost_val} (TN={tn}, FP={fp},"
    f" FN={fn}, TP={tp})"
)

# ==========================================
# 6. SHAP 可解釋性分析
# ==========================================
print("\n計算 XGBoost 的 SHAP 解釋...")
explainer = shap.TreeExplainer(xgb_model)
# 修正處：改傳入 X_test_scaled (NumPy 陣列)，避免 Pandas DataFrame 轉換報錯
shap_values = explainer(X_test_scaled)
print("SHAP 解釋計算完成！全部流程執行完畢。")
