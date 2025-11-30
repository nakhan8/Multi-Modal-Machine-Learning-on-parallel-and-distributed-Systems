import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchxrayvision as xrv
import matplotlib.pyplot as plt

# ===============================================================
# CONFIGURATION
# ===============================================================
IMAGE_DIR = "data/images"            # Folder containing NIH images
DATA_ENTRY_CSV = "Data_Entry_2017.csv"  # Optional ground truth labels
OUTPUT_CSV = "pneumonia_infiltration_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_THRESHOLD = 0.6
BATCH_SIZE = 8

# Only detect these
TARGET_FINDINGS = ["Pneumonia", "Infiltration"]

# ===============================================================
# MODEL LOADING
# ===============================================================
print("🧠 Loading TorchXRayVision DenseNet121 (NIH pre-trained)...")
model = xrv.models.DenseNet(weights="densenet121-res224-all").to(DEVICE)
model.eval()
labels = model.pathologies
print(f"✅ Model supports {len(labels)} diagnostic categories.")
print(f"🎯 Target classes: {TARGET_FINDINGS}\n")

# ===============================================================
# IMAGE TRANSFORM
# ===============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ===============================================================
# LOAD GROUND TRUTH (OPTIONAL)
# ===============================================================
df_gt = None
if os.path.exists(DATA_ENTRY_CSV):
    print(f"📄 Found ground truth: {DATA_ENTRY_CSV}")
    df_gt = pd.read_csv(DATA_ENTRY_CSV)
    df_gt = df_gt.rename(columns={"Image Index": "image_name", "Finding Labels": "ground_truth"})
    df_gt["ground_truth_list"] = df_gt["ground_truth"].apply(lambda x: x.split("|"))
    df_gt["ground_truth_primary"] = df_gt["ground_truth_list"].apply(lambda x: x[0] if len(x) > 0 else "No Finding")

# ===============================================================
# IMAGE LIST
# ===============================================================
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if len(image_files) == 0:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")
print(f"📁 Found {len(image_files)} X-rays for detection.\n")

# ===============================================================
# INFERENCE LOOP
# ===============================================================
results = []
for i in tqdm(range(0, len(image_files), BATCH_SIZE)):
    batch_files = image_files[i:i + BATCH_SIZE]
    imgs, valid_files = [], []

    for f in batch_files:
        try:
            img = Image.open(os.path.join(IMAGE_DIR, f)).convert("L")
            imgs.append(transform(img))
            valid_files.append(f)
        except Exception as e:
            print(f"⚠️ Skipping unreadable image: {f} ({e})")

    if not imgs:
        continue

    imgs = torch.stack(imgs).to(DEVICE)

    with torch.no_grad():
        preds = model(imgs)
        probs = torch.sigmoid(preds).cpu().numpy()

    # =========================
    # Post-processing
    # =========================
    for j, fname in enumerate(valid_files):
        prob_dict = {labels[k]: float(probs[j][k]) for k in range(len(labels))}
        target_probs = {lbl: prob_dict[lbl] for lbl in TARGET_FINDINGS if lbl in prob_dict}

        # Adaptive threshold based on image variance
        mean_p = np.mean(list(target_probs.values()))
        std_p = np.std(list(target_probs.values()))
        adaptive_threshold = max(BASE_THRESHOLD, mean_p + 0.25 * std_p)

        # Select only strong findings
        detected_labels = [lbl for lbl, p in target_probs.items() if p >= adaptive_threshold]

        # Determine anomaly logic
        if len(detected_labels) == 0:
            is_anomaly = False
            detected_labels = ["None"]
        else:
            is_anomaly = True

        # Dominant class and confidence
        dominant_label = max(target_probs, key=target_probs.get)
        dominant_conf = target_probs[dominant_label]

        row = {
            "image_name": fname,
            "detected_labels": "|".join(detected_labels),
            "dominant_label": dominant_label,
            "dominant_confidence": dominant_conf,
            "anomaly_detected": is_anomaly,
            **target_probs
        }

        if df_gt is not None and fname in set(df_gt["image_name"]):
            gt_row = df_gt[df_gt["image_name"] == fname].iloc[0]
            row["ground_truth"] = gt_row["ground_truth"]
            row["ground_truth_primary"] = gt_row["ground_truth_primary"]

        results.append(row)

# ===============================================================
# SAVE TO CSV
# ===============================================================
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)

# ===============================================================
# SUMMARY & VISUALIZATION
# ===============================================================
total = len(df_out)
anomalies = df_out["anomaly_detected"].sum()
print(f"\n✅ Saved refined results to {OUTPUT_CSV}")
print(f"🩻 Total Images: {total}")
print(f"🚨 Pneumonia / Infiltration Detected in: {anomalies} ({100 * anomalies / total:.1f}%)")

# Plot histograms
plt.figure(figsize=(8, 4))
df_out[["Pneumonia", "Infiltration"]].hist(bins=20, figsize=(8, 4))
plt.suptitle("Probability Distribution: Pneumonia & Infiltration")
plt.tight_layout()
plt.show()

print("\n=== Sample Output ===")
print(df_out.head(10))
