import os
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration

# ===============================================================
# CONFIGURATION
# ===============================================================
IMAGE_DIR = "data/images"              # Folder containing images (e.g., chest X-rays or COCO)
OUTPUT_CSV = "multimodal_clip_blip_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

# CLIP text prompts (can be domain-specific)
TEXT_PROMPTS = [
    "An X-ray showing pneumonia",
    "A normal chest X-ray",
    "An image with lung infiltration",
    "A healthy chest image",
    "An MRI scan of the brain",
    "An image of medical device in body"
]

# ===============================================================
# MODEL LOADING
# ===============================================================
print("🧠 Loading CLIP model (ViT-B/32)...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
text_tokens = clip.tokenize(TEXT_PROMPTS).to(DEVICE)
print(f"✅ CLIP Model loaded. {len(TEXT_PROMPTS)} text prompts prepared.\n")

# Optional BLIP model for caption generation
print("🧠 Loading BLIP model for caption generation...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
print("✅ BLIP Model loaded.\n")

# ===============================================================
# IMAGE LIST
# ===============================================================
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if len(image_files) == 0:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")
print(f"📁 Found {len(image_files)} images for multimodal processing.\n")

# ===============================================================
# INFERENCE LOOP
# ===============================================================
results = []

for i in tqdm(range(0, len(image_files), BATCH_SIZE)):
    batch_files = image_files[i:i + BATCH_SIZE]
    images, valid_files = [], []

    for f in batch_files:
        try:
            img_path = os.path.join(IMAGE_DIR, f)
            img = clip_preprocess(Image.open(img_path).convert("RGB"))
            images.append(img)
            valid_files.append(f)
        except Exception as e:
            print(f"⚠️ Skipping unreadable image: {f} ({e})")

    if not images:
        continue

    images = torch.stack(images).to(DEVICE)

    # CLIP: Text-Image similarity (zero-shot classification)
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

    # Process each image
    for j, fname in enumerate(valid_files):
        sim_scores = similarity[j]
        best_idx = np.argmax(sim_scores)
        matched_text = TEXT_PROMPTS[best_idx]
        confidence = float(sim_scores[best_idx])

        # BLIP: Generate caption
        try:
            raw_img = Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB")
            blip_inputs = blip_processor(raw_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                blip_out = blip_model.generate(**blip_inputs, max_new_tokens=30)
                caption = blip_processor.decode(blip_out[0], skip_special_tokens=True)
        except Exception as e:
            caption = f"Error generating caption: {e}"

        row = {
            "image_name": fname,
            "clip_matched_text": matched_text,
            "clip_confidence": confidence,
            "blip_caption": caption,
        }

        # Add all similarity scores for reference
        for k, t in enumerate(TEXT_PROMPTS):
            row[f"score_{t[:20]}"] = sim_scores[k]

        results.append(row)

# ===============================================================
# SAVE RESULTS
# ===============================================================
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved multimodal results to {OUTPUT_CSV}")

# ===============================================================
# VISUALIZATION
# ===============================================================
# Plot histogram of top-1 CLIP confidence scores
plt.figure(figsize=(8, 4))
plt.hist(df_out["clip_confidence"], bins=20, color="skyblue", edgecolor="black")
plt.title("CLIP Zero-Shot Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ===============================================================
# SUMMARY
# ===============================================================
avg_conf = df_out["clip_confidence"].mean()
print(f"🧩 Average CLIP Confidence: {avg_conf:.3f}")
print(f"📝 Sample Results:")
print(df_out.head(10))