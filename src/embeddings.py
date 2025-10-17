import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

print("--- Step 1: Loading Training Data ---")
try:
    train_df = pd.read_csv('/content/features_test_model_ready (1).csv')
    print("structured_features_train.csv loaded successfully.")
except FileNotFoundError:
    print("Error: Make sure 'train.csv' is in a 'dataset' folder.")
    exit()

print("\n--- Step 2: Generating SBERT Text Embeddings ---")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
sentences = train_df['Item Name'].fillna('no content').tolist()[:25000] # Modified to take only the first 25000 rows

print(f"Encoding {len(sentences)} sentences...")
text_embeddings = model.encode(sentences, show_progress_bar=True, batch_size=256)

# --- Step 3: Saving the Embeddings as .npy ---
output_path = '/content/train_sbert.npy' 
np.save(output_path, text_embeddings)

print(f"\nSuccessfully saved SBERT embeddings as a .npy file to: -> {output_path}")
print("Shape of saved array:", text_embeddings.shape)