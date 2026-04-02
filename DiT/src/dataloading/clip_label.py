import os
import clip
import torch
import torch.nn.functional as F
from PIL import Image
import shutil
import pandas as pd

IMAGE_FOLDER      = "data"
OUTPUT_FOLDER     = "data_labeled"
UNCERTAIN_FOLDER  = "data_uncertain"
OUTPUT_CSV        = "labels.csv"
BATCH_SIZE        = 64
CONF_THRESHOLD    = 0.50
CATCHALL_CLASSES  = {"landscapes"}   

# Multiple prompts per class - averaged into one feature vector (prompt ensembling)
LABEL_PROMPTS = {
    "landscapes": [
        "a scenic landscape photo",
        "a wide open landscape",
        "a natural scenery photo",
    ],
    "mountain": [
        "a photo of a mountain",
        "a mountain landscape",
        "mountain peaks and rocky terrain",
        "snowy mountain peaks",
    ],
    "desert": [
        "a photo of a desert",
        "a sandy desert landscape",
        "arid desert with sand dunes",
        "a dry barren desert",
    ],
    "sea": [
        "a photo of the sea",
        "open ocean view",
        "seascape with waves",
        "the ocean horizon",
    ],
    "beach": [
        "a photo of a beach",
        "sandy beach with waves",
        "a tropical beach shoreline",
        "beach with sand and water",
    ],
    "island": [
        "a photo of an island",
        "a tropical island surrounded by water",
        "aerial view of an island",
        "a small island in the ocean",
    ],
    "japan": [
        "a Japanese landscape photo",
        "Japanese nature scenery",
        "cherry blossom trees in Japan",
        "Mount Fuji landscape",
        "a tranquil Japanese garden or countryside",
    ],
}
# ------------------------ #

LABELS = list(LABEL_PROMPTS.keys())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

print("Encoding text prompts...")
class_features = []
with torch.no_grad():
    for label in LABELS:
        tokens = clip.tokenize(LABEL_PROMPTS[label]).to(device)
        feats  = model.encode_text(tokens)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        class_features.append(feats.mean(dim=0))
text_features = torch.stack(class_features)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (C, d)

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print(f"Found {len(image_files)} images.\n")

os.makedirs(UNCERTAIN_FOLDER, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(OUTPUT_FOLDER, label), exist_ok=True)

# CSV columns: filename | label | p_landscapes | p_mountain | p_desert | p_sea | p_beach | p_island | p_japan
records = []

for start in range(0, len(image_files), BATCH_SIZE):
    batch_files = image_files[start:start + BATCH_SIZE]
    images, valid_files = [], []

    for fname in batch_files:
        try:
            img = preprocess(Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("RGB"))
            images.append(img)
            valid_files.append(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    if not images:
        continue

    batch_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        img_features = model.encode_image(batch_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        logits = img_features @ text_features.T            # (B, C)
        probs  = F.softmax(logits * 100.0, dim=-1)         # (B, C)  — all classes sum to 1
        top_probs, top_idx = probs.max(dim=-1)             # (B,)

    for fname, conf, idx, all_probs in zip(
        valid_files, top_probs.tolist(), top_idx.tolist(), probs.tolist()
    ):
        label    = LABELS[idx]
        assigned = label if (conf >= CONF_THRESHOLD and label not in CATCHALL_CLASSES) else "uncertain"
        src      = os.path.join(IMAGE_FOLDER, fname)

        if assigned == "uncertain":
            shutil.copy(src, os.path.join(UNCERTAIN_FOLDER, fname))
        else:
            shutil.copy(src, os.path.join(OUTPUT_FOLDER, label, fname))

        # One probability column per class for every image
        row = {"filename": fname, "label": assigned}
        for l, p in zip(LABELS, all_probs):
            row[f"p_{l}"] = round(p, 4)
        records.append(row)

    done = min(start + BATCH_SIZE, len(image_files))
    print(f"Processed {done}/{len(image_files)}")

df = pd.DataFrame(records)
# Sort by max-class probability ascending so uncertain/borderline rows are at the top
df["_max_p"] = df[[f"p_{l}" for l in LABELS]].max(axis=1)
df.sort_values("_max_p", ascending=True, inplace=True)
df.drop(columns="_max_p", inplace=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. '{OUTPUT_CSV}' saved (lowest-confidence rows first).")
print(f"  uncertain (< {CONF_THRESHOLD:.0%}): {(df['label'] == 'uncertain').sum()}")
for label in LABELS:
    print(f"  {label}: {(df['label'] == label).sum()}")
