import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- 0. Configuration and Setup ---
MODEL_DIR = "models_v3/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 1. Load All Pre-Trained Artifacts and Models ---
print("📦 Loading pre-trained artifacts and models...")
try:
    brand_stats = joblib.load(os.path.join(MODEL_DIR, "brand_stats.joblib"))
    embedding_scaler = joblib.load(os.path.join(MODEL_DIR, "embedding_scaler.joblib"))
    pca_model = joblib.load(os.path.join(MODEL_DIR, "pca_model.joblib"))
    tabular_scaler = joblib.load(os.path.join(MODEL_DIR, "tabular_scaler.joblib"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    
    models = []
    for i in range(1, 6):
        model = joblib.load(os.path.join(MODEL_DIR, f"lgbm_fold_{i}.joblib"))
        models.append(model)
    print("✅ All artifacts and models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not find a required artifact file. {e}")
    print("Please ensure you have run the training script to save all artifacts in the 'models_v3/' folder.")
    exit()

# --- 2. Load Raw Test Data ---
print("\n📦 Loading raw test data...")
try:
    test_df = pd.read_csv('/content/test.csv')
    print("test.csv loaded successfully.")
except FileNotFoundError:
    print("Error: 'test.csv' not found.")
    exit()

# --- 3. Full Feature Generation Pipeline for Test Data ---
print("\n⚙️  Starting feature generation for the test set...")

# A. Manual Features
print("  - Step 3a: Creating manual features...")
# You would load your full brand list and feature extraction functions here
# For this example, we'll use placeholder functions.
# You should replace these with your actual feature extraction logic.
test_df['ipq'] = test_df['catalog_content'].str.extract(r'Pack of (\d+)', expand=False).fillna(1).astype(int)
test_df['is_organic'] = test_df['catalog_content'].str.contains('Organic', case=False).astype(int)
test_df['is_gluten_free'] = test_df['catalog_content'].str.contains('Gluten Free', case=False).astype(int)
test_df['is_kosher'] = test_df['catalog_content'].str.contains('Kosher', case=False).astype(int)
test_df['quantity'] = 1 # Placeholder, replace with your quantity extraction
test_df['brand'] = 'Unknown' # Placeholder, replace with your brand extraction

# B. SBERT Embeddings
print("  - Step 3b: Generating SBERT embeddings...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
test_sentences = test_df['catalog_content'].fillna('no content').tolist()
test_sbert_embeddings = sbert_model.encode(test_sentences, show_progress_bar=True, batch_size=256)

# C. CLIP Image Embeddings (This is slow)
print("  - Step 3c: Generating CLIP Image embeddings...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_link):
    try:
        image = Image.open(requests.get(image_link, stream=True, timeout=10).raw).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            features = clip_model.get_image_features(**inputs)
        return features.cpu().numpy().flatten()
    except Exception:
        return np.zeros(clip_model.config.vision_config.hidden_size)

all_clip_embeddings = [get_image_embedding(link) for link in tqdm(test_df['image_link'])]
test_clip_embeddings = np.array(all_clip_embeddings)

# --- 4. Merge and Transform Test Features ---
print("\n⚙️  Applying transformations to test data...")

# A. Merge all features into one DataFrame
sbert_df = pd.DataFrame(test_sbert_embeddings, columns=[f"sbert_{i}" for i in range(test_sbert_embeddings.shape[1])])
clip_df = pd.DataFrame(test_clip_embeddings, columns=[f"clip_{i}" for i in range(test_clip_embeddings.shape[1])])
final_test_df = pd.concat([test_df.reset_index(drop=True), sbert_df, clip_df], axis=1)

# B. Apply FITTED transformations from training
final_test_df = final_test_df.merge(brand_stats, on="brand", how="left")
final_test_df['brand_mean_price'].fillna(brand_stats['brand_mean_price'].mean(), inplace=True)
final_test_df['brand_std_price'].fillna(brand_stats['brand_std_price'].mean(), inplace=True)

embedding_cols = [c for c in final_test_df.columns if c.startswith(("sbert_", "clip_"))]
scaled_embs = embedding_scaler.transform(final_test_df[embedding_cols].fillna(0))
pca_features = pca_model.transform(scaled_embs)
pca_df = pd.DataFrame(pca_features, columns=[f"pca_{i}" for i in range(256)])
final_test_df = pd.concat([final_test_df.reset_index(drop=True), pca_df], axis=1)

final_test_df["quantity_log"] = np.log1p(final_test_df["quantity"])
final_test_df["ipq_log"] = np.log1p(final_test_df["ipq"])
final_test_df["price_per_quantity"] = final_test_df["ipq"] / (final_test_df["quantity"] + 1e-6)
final_test_df["organic_gluten"] = final_test_df["is_organic"] * final_test_df["is_gluten_free"]

num_features_to_scale = ["ipq", "quantity", "quantity_log", "ipq_log", "price_per_quantity"]
final_test_df[num_features_to_scale] = tabular_scaler.transform(final_test_df[num_features_to_scale].fillna(0))

# --- 5. Final Feature Selection and Prediction ---
# Ensure test columns match the training columns exactly in order
X_test = final_test_df[feature_columns]
X_test['brand'] = X_test['brand'].astype('category')

print("\n🚀 Predicting with all 5 models...")
test_preds_log = np.zeros(X_test.shape[0])

for i, model in enumerate(models):
    print(f"  - Predicting with Fold {i+1} model...")
    test_preds_log += model.predict(X_test) / len(models)

# --- 6. Create Submission File ---
print("\n💾 Creating submission file...")
final_predictions = np.expm1(test_preds_log)
final_predictions[final_predictions < 0] = 0

submission_df = pd.DataFrame({
    'sample_id': final_test_df['sample_id'],
    'price': final_predictions
})

submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file created successfully: submission.csv")
print(submission_df.head())