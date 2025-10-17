import pandas as pd
import numpy as np
import os

# --- 1. Load ALL separate feature files ---

# File from the Feature Engineer (contains 'brand', 'IPQ', etc.)
try:
    manual_features_df = pd.read_csv('/content/features_train_model_ready.csv')
    print("structured_features_train.csv loaded successfully.")
except FileNotFoundError:
    print("Error: 'structured_features_train.csv' not found.")
    exit()

# Load the original train file to get sample_ids in the correct order
# User has indicated this file now has 74999 rows
try:
    train_df = pd.read_csv('/content/train.csv')
    print("train.csv loaded successfully.")
    # Assuming the extra row was removed from this specific file
    # You might need additional code here if the row removal logic is not in this script
    if len(train_df) != 74999:
        print(f"Warning: train.csv has {len(train_df)} rows, but 74999 rows were expected after removal.")
except FileNotFoundError:
    print("Error: 'train.csv' not found.")
    exit()

# Load the SBERT text embeddings from the .npy file
try:
    sbert_embeddings = np.load('/content/train_sbert.npy')
    print("train_sbert.npy loaded successfully.")
except FileNotFoundError:
    print("Error: 'train_sbert.npy' not found. Please generate SBERT embeddings first.")
    exit()

# Load the CLIP image embeddings from the .npy file
image_embeddings_path = '/content/image_embeddings.npy'
if not os.path.exists(image_embeddings_path):
    print(f"Error: '{image_embeddings_path}' not found. Please generate image embeddings first.")
    exit()

try:
    clip_embeddings = np.load(image_embeddings_path)
    print("image_embeddings.npy loaded successfully.")
except Exception as e:
    print(f"Error loading '{image_embeddings_path}': {e}")
    exit()

# --- 2. Validate shapes and Create DataFrames from the .npy files ---

# Updated expected number of samples based on user feedback
expected_samples = len(train_df) # This will now be 74999 if train_df was loaded correctly
expected_sbert_dim = 384 # Based on 'all-MiniLM-L6-v2'
expected_clip_dim = 512 # Assuming a standard CLIP model output dimension

if sbert_embeddings.shape[0] != expected_samples or sbert_embeddings.shape[1] != expected_sbert_dim:
    print(f"Error: SBERT embeddings shape mismatch. Expected ({expected_samples}, {expected_sbert_dim}), got {sbert_embeddings.shape}.")
    print("Please re-generate SBERT embeddings to match the corrected train_df size.")
    exit()

if clip_embeddings.shape[0] != expected_samples or clip_embeddings.shape[1] != expected_clip_dim:
    print(f"Error: CLIP embeddings shape mismatch. Expected ({expected_samples}, {expected_clip_dim}), got {clip_embeddings.shape}.")
    print("Please check the image embedding generation process to ensure it produces the correct shape (74999 samples).")
    exit()


# Create DataFrame for SBERT embeddings
sbert_df = pd.DataFrame(sbert_embeddings, columns=[f'sbert_{i}' for i in range(sbert_embeddings.shape[1])])
sbert_df['sample_id'] = train_df['sample_id'].reset_index(drop=True) # Use sample_id from the potentially modified train_df

# Create DataFrame for CLIP image embeddings
clip_df = pd.DataFrame(clip_embeddings, columns=[f'clip_{i}' for i in range(clip_embeddings.shape[1])])
clip_df['sample_id'] = train_df['sample_id'].reset_index(drop=True) # Use sample_id from the potentially modified train_df


# --- 3. Merge everything into one master DataFrame ---

print("Merging manual features with SBERT and CLIP embeddings...")
# Ensure manual_features_df also has the correct sample_ids and number of rows if it was modified
# Assuming manual_features_df should also align with the 74999 samples
# If not, you might need to filter manual_features_df based on the sample_ids in the modified train_df
manual_features_df = manual_features_df[manual_features_df['sample_id'].isin(train_df['sample_id'])].reset_index(drop=True)
if len(manual_features_df) != expected_samples:
     print(f"Warning: manual_features_df has {len(manual_features_df)} rows after filtering, but {expected_samples} rows were expected.")


# First, merge the manual features (which include 'brand') with the text embeddings
final_train_df = pd.merge(manual_features_df, sbert_df, on='sample_id', how='left')

# Next, merge the image embeddings
final_train_df = pd.merge(final_train_df, clip_df, on='sample_id', how='left')

output_path = '/content/combined.npy'
np.save(output_path, final_train_df)

print("\n--- Merge Complete ---")
print("Final training data shape:", final_train_df.shape)

# You can see the 'brand' column is now present alongside the embeddings
print("\nPreview of the final merged data:")
# Display a few columns to show the merge was successful
display(final_train_df[['sample_id', 'brand', 'ipq', 'sbert_0', 'sbert_1', 'clip_0', 'clip_1']].head()) 