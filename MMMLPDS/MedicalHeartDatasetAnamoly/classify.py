import os
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image

# ===============================================================
# CONFIGURATION
# ===============================================================
IMAGE_DIR = "data/images"  # path to your folder containing NIH images (.png)
DATA_ENTRY_CSV = "Data_Entry_2017.csv"  # NIH metadata file
OUTPUT_CSV = "multi_label_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5  # probability threshold for positive diagnosis
BATCH_SIZE = 8

# ===============================================================
# MODEL
# ===============================================================
MODEL_NAME = "microsoft/biovil-image-classification"
print(f"Loading pretrained model: {MODEL_NAME}")
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Get the label names from model config
label_names = list(model.config.id2label.values())
print(f"Model outputs {len(label_names)} disease labels:")
print(label_names)

# ===============================================================
# LOAD NIH GROUND TRUTH
# ===============================================================
df_labels = pd.read_csv(DATA_ENTRY_CSV)
df_labels = df_labels.rename(columns={"Image Index": "image_name", "Finding Labels": "ground_truth"})

# Simplify ground truth (split multiple findings)
df_labels["ground_truth_list"] = df_labels["ground_truth"].apply(lambda x: x.split("|"))
df_labels["ground_truth_primary"] = df_labels["ground_truth_list"].apply(lambda x: x[0] if len(x) > 0 else "No Finding")

# ===============================================================
# IMAGE FILES
# ===============================================================
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if len(image_files) == 0:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

# Keep only rows for images we have
df_labels = df_labels[df_labels["image_name"].isin(image_files)]

# ===============================================================
# INFERENCE
# ===============================================================
results = []

print(f"\n🩻 Running multi-label inference on {len(df_labels)} images...")

for i in tqdm(range(0, len(df_labels), BATCH_SIZE)):
    batch = df_labels.iloc[i:i+BATCH_SIZE]
    batch_images = []
    for fname in batch["image_name"]:
        try:
            img = Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB")
            batch_images.append(img)
        except:
            print(f"⚠️ Skipping unreadable image: {fname}")

    if not batch_images:
        continue

    inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()  # multi-label probabilities

    # Store results
    for j, fname in enumerate(batch["image_name"]):
        row_probs = {label_names[k]: float(probs[j][k]) for k in range(len(label_names))}
        predicted_labels = [lbl for lbl, p in row_probs.items() if p >= THRESHOLD]
        gt_list = batch.iloc[j]["ground_truth_list"]

        # compute match ratio (intersection / union)
        if len(gt_list) > 0:
            matches = len(set(predicted_labels) & set(gt_list))
            total = len(set(gt_list))
            match_ratio = matches / total
        else:
            match_ratio = 1.0 if len(predicted_labels) == 0 else 0.0

        row = {
            "image_name": fname,
            **row_probs,
            "predicted_labels": "|".join(predicted_labels) if predicted_labels else "None",
            "ground_truth": batch.iloc[j]["ground_truth"],
            "ground_truth_primary": batch.iloc[j]["ground_truth_primary"],
            "match_ratio": round(match_ratio, 2),
        }
        results.append(row)

# ===============================================================
# SAVE RESULTS
# ===============================================================
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

# summary
print(f"\n✅ Saved multi-label predictions to: {OUTPUT_CSV}")
print(f"Total images evaluated: {len(df_results)}")
print("\n=== Sample Rows ===")
print(df_results.head(5))
