import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier
import joblib

# ------------ CONFIGURATION ------------
PARQUET_FILE = "flattened_by_seaons.parquet"   # <-- Update if needed
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
N_DISHES = 15

season_months = {
    "Winter": ['Jan', 'Feb', 'Mar', 'Nov', 'Dec'],
    "Summer": ['Apr', 'May', 'Jun', 'Jul'],
    "Monsoon": ['Aug', 'Sep', 'Oct'],
}

# Create lookup: month -> season
month_to_season = {}
for season, months_in_season in season_months.items():
    for m in months_in_season:
        month_to_season[m] = season

def bitstring_to_array(s):
    return np.array([int(ch) for ch in str(s).zfill(N_DISHES)])

def pad_features(feats, max_len):
    if len(feats) < max_len:
        feats.extend([0] * (max_len - len(feats)))
    return feats

# ------------ CALCULATE FEATURE VECTOR MAX LENGTH ------------
# Each sample: (number of other months in the biggest season) × 30 + 12 one-hot
max_hist = max(len(v) for v in season_months.values()) - 1
max_len = max_hist * N_DISHES * 2 + len(MONTHS)  # (num_months-1)*30 + 12

# ------------ DATA LOAD & SEASONAL FEATURE ENGINEERING ------------
df = pd.read_parquet(PARQUET_FILE)
samples, targets = [], []

for _, row in df.iterrows():
    for t, pred_month in enumerate(MONTHS):
        season = month_to_season[pred_month]
        # Use all months in same season except the month being predicted
        season_month_list = [m for m in season_months[season] if m != pred_month]
        feats = []
        for m in season_month_list:
            feats.extend(bitstring_to_array(row[m]))
            feats.extend(bitstring_to_array(row[f"{m}_craving"]))
        # One-hot for prediction month
        month_onehot = [0]*len(MONTHS)
        month_onehot[t] = 1
        feats.extend(month_onehot)
        # Pad to fixed length
        feats = pad_features(feats, max_len)
        # Target: eating vector for prediction month
        target = bitstring_to_array(row[pred_month])
        samples.append(feats)
        targets.append(target)
X = np.array(samples)
y = np.array(targets)
print(f"Feature matrix shape: {X.shape}")

# ------------ TRAIN/TEST SPLIT ------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------ MODEL TRAINING ------------
base_lgbm = LGBMClassifier(
    num_leaves=64,
    n_estimators=150,
    class_weight='balanced',
    n_jobs=-1
)
model = MultiOutputClassifier(base_lgbm, n_jobs=-1)
print("Training LightGBM multilabel model...")
model.fit(X_train, y_train)
print("Training complete.")

# ------------ PER-DISH THRESHOLD TUNING ------------
threshold_grid = np.linspace(0.05, 0.95, 19)
proba_test = np.array([p[:,1] for p in model.predict_proba(X_test)]).T
best_thresholds = np.full(N_DISHES, 0.5)
y_pred = np.zeros_like(y_test)
for dish_i in range(N_DISHES):
    best_f1, best_thresh = 0, 0.5
    for thresh in threshold_grid:
        preds = (proba_test[:,dish_i] > thresh).astype(int)
        f1 = f1_score(y_test[:,dish_i], preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    best_thresholds[dish_i] = best_thresh
    y_pred[:,dish_i] = (proba_test[:,dish_i] > best_thresh).astype(int)
    print(f"Dish {dish_i+1} best F1={best_f1:.3f} at threshold={best_thresh:.2f}")

# ------------ EVALUATION: CONFUSION MATRIX & F1 ------------
for dish_i in range(N_DISHES):
    print(f"\nDish {dish_i+1} confusion matrix:")
    print(confusion_matrix(y_test[:,dish_i], y_pred[:,dish_i]))
    print(classification_report(y_test[:,dish_i], y_pred[:,dish_i], zero_division=0))

# Exact-match multilabel accuracy
exact_accuracy = np.mean(np.all(y_test == y_pred, axis=1))
print(f"\nOverall multilabel exact-match accuracy: {exact_accuracy:.4f}")

# ------------ MODEL SAVE ------------
joblib.dump({'model': model, 'thresholds': best_thresholds}, "model300test.pkl")
print("Model and thresholds saved as 'model300test.pkl'")
