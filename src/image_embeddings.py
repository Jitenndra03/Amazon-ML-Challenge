import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip

# 📁 Paths
IMAGE_DIR = "downloaded_images"
OUTPUT_EMB_PATH = "image_embeddings_test.npy"
OUTPUT_IDS_PATH = "embedding_test_ids.npy"  # Optional: Save corresponding sample_ids
BATCH_SIZE = 16  # adjust based on GPU memory

# 🧠 Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# 🖼️ Get all images (allow common formats)
image_files = sorted([
    os.path.join(IMAGE_DIR, f) 
    for f in os.listdir(IMAGE_DIR) 
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
])

# (Optional) Save sample_ids for reference
image_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

# 🧩 Create dataset
def load_image(path):
    try:
        image = Image.open(path).convert("RGB")
        return preprocess(image)
    except Exception as e:
        print(f"⚠️ Skipping {path} due to {e}")
        return None

# 🪣 Batch embedding extraction
embeddings = []
valid_ids = []

with torch.no_grad():
    for i in tqdm(range(0, len(image_files), BATCH_SIZE)):
        batch_files = image_files[i:i + BATCH_SIZE]
        batch_images = [load_image(f) for f in batch_files]
        batch_ids = [image_ids[j] for j in range(i, min(i+BATCH_SIZE, len(image_ids)))]
        # Only keep non-None images (and their IDs)
        valid_pairs = [(img, id_) for img, id_ in zip(batch_images, batch_ids) if img is not None]
        if not valid_pairs:
            continue
        batch_images, batch_ids_clean = zip(*valid_pairs)
        batch_tensor = torch.stack(batch_images).to(device)
        # 🎯 Extract embeddings
        image_features = model.encode_image(batch_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())
        valid_ids.extend(batch_ids_clean)

# 🔐 Combine all batches
embeddings = np.concatenate(embeddings, axis=0)

# 💾 Save to file (embeddings and ids)
np.save(OUTPUT_EMB_PATH, embeddings)
np.save(OUTPUT_IDS_PATH, np.array(valid_ids))
print(f"✅ Saved {embeddings.shape} embeddings to {OUTPUT_EMB_PATH}")
print(f"✅ Saved corresponding IDs to {OUTPUT_IDS_PATH}")