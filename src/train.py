import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- Paths ---
MODEL_DIR = "models_v3/"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load combined data ---
print("📦 Loading combined data...")
loaded_data = np.load("combined.npy", allow_pickle=True)
manual_features_df = pd.read_csv("features_train_model_ready.csv")
sbert_embeddings = np.load("train_sbert.npy")
clip_embeddings = np.load("image_embeddings.npy")

manual_cols = list(manual_features_df.columns)
sbert_cols = [f"sbert_{i}" for i in range(sbert_embeddings.shape[1])]
clip_cols = [f"clip_{i}" for i in range(clip_embeddings.shape[1])]
final_cols = manual_cols + sbert_cols + clip_cols
final_train_df = pd.DataFrame(loaded_data, columns=final_cols)
print(f"✅ Final training data shape: {final_train_df.shape}")

# --- SMAPE metric ---
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / (denom + 1e-8)
    diff[denom == 0] = 0
    return 100 * np.mean(diff)

# --- Data cleaning ---
num_cols = [c for c in final_train_df.columns if c.startswith(("sbert_", "clip_", "ipq", "log_price"))]
for c in num_cols:
    final_train_df[c] = pd.to_numeric(final_train_df[c], errors="coerce")

for col in ["is_organic", "is_gluten_free", "is_kosher"]:
    if col in final_train_df.columns:
        final_train_df[col] = final_train_df[col].astype(int)

final_train_df["quantity"] = pd.to_numeric(final_train_df["quantity"], errors="coerce")

# --- Categorical encoding ---
final_train_df["brand"] = final_train_df["brand"].astype("category")
if "unit_cleaned" in final_train_df.columns:
    final_train_df["unit_cleaned"] = final_train_df["unit_cleaned"].astype("category")

TARGET = "log_price"
y = final_train_df[TARGET]

# --- Feature Engineering ---
# Brand statistics
brand_stats = final_train_df.groupby("brand")[TARGET].agg(["mean", "std"]).rename(
    columns={"mean": "brand_mean_price", "std": "brand_std_price"}
)
final_train_df = final_train_df.merge(brand_stats, on="brand", how="left")

# PCA on embeddings
embedding_cols = [c for c in final_train_df.columns if c.startswith(("sbert_", "clip_"))]
print("🧠 Applying PCA on embeddings (896 → 256 dims)...")
scaler = StandardScaler()
scaled_embs = scaler.fit_transform(final_train_df[embedding_cols].fillna(0))

pca = PCA(n_components=256, random_state=42)
pca_features = pca.fit_transform(scaled_embs)
pca_df = pd.DataFrame(pca_features, columns=[f"pca_{i}" for i in range(256)])
final_train_df = pd.concat([final_train_df.reset_index(drop=True), pca_df], axis=1)
print(f"✅ PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")

# Extra engineered features
final_train_df["quantity_log"] = np.log1p(final_train_df["quantity"])
final_train_df["ipq_log"] = np.log1p(final_train_df["ipq"])
final_train_df["price_per_quantity"] = final_train_df["ipq"] / (final_train_df["quantity"] + 1e-6)
final_train_df["organic_gluten"] = final_train_df["is_organic"] * final_train_df["is_gluten_free"]
final_train_df["brand_mean_diff"] = final_train_df[TARGET] - final_train_df["brand_mean_price"]
final_train_df["brand_zscore"] = final_train_df["brand_mean_diff"] / (final_train_df["brand_std_price"] + 1e-6)

# Scale numerical tabular features
num_features = ["ipq", "quantity", "quantity_log", "ipq_log", "price_per_quantity"]
scaler_tab = StandardScaler()
final_train_df[num_features] = scaler_tab.fit_transform(final_train_df[num_features].fillna(0))

# --- Final feature list ---
exclude = [TARGET, "sample_id"] + embedding_cols
X = final_train_df.drop(columns=exclude)
y = final_train_df[TARGET]

# Drop unused text columns
for col in ["Item Name"]:
    if col in X.columns:
        X = X.drop(columns=[col])
        print(f"Dropped column: {col}")

# Convert remaining object columns to category
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category")
        print(f"Converted {col} to category dtype.")

# --- Stratified K-Fold Cross-Validation ---
N_SPLITS = 5
# Use bins for stratification
bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof = np.zeros(X.shape[0])

params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 128,
    "learning_rate": 0.02,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 4,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
    "min_data_in_leaf": 40,
    "n_estimators": 6000,
    "early_stopping_rounds": 300,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}

print(f"\n🚀 Starting {N_SPLITS}-Fold Stratified Cross-Validation...\n")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, bins)):
    print(f"🧩 Fold {fold+1}/{N_SPLITS}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)],
    )

    preds = model.predict(X_val)
    oof[val_idx] = preds

    # Save model
    model_path = os.path.join(MODEL_DIR, f"lgbm_fold_{fold+1}.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Saved model to {model_path}")

# --- Final Evaluation ---
print("\n📊 Evaluating SMAPE...")
y_true = np.expm1(y.values)
y_pred = np.expm1(oof)
final_smape = smape(y_true, y_pred)
print(f"✅ Final Cross-Validation SMAPE: {final_smape:.3f}%")