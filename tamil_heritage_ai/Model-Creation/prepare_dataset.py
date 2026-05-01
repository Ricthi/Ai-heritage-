import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
prepare_dataset.py
------------------
Scans Labels/ directory, collects all character class folders that
contain JPG images, merges duplicates across Fig datasets, and
splits 70% → data/train / 30% → data/val.

Run once before training:
    python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
LABELS_ROOT = Path(__file__).parent.parent / "Labels"
DATA_ROOT   = Path(__file__).parent / "data"
TRAIN_DIR   = DATA_ROOT / "train"
VAL_DIR     = DATA_ROOT / "val"
TRAIN_RATIO = 0.70
RANDOM_SEED = 42

# Folders that are NOT character classes (dump / utility folders)
SKIP_FOLDERS = {
    "1 - Original Dataset",
    "1 - Multipart",
    "2 - Multipart",
    "2 - Unknown",
    "3 - Unidentified",
    "Labelled Dataset - Fig 6",
    "Labelled Dataset - Fig 11",
    "Labelled Dataset - Fig 51",
    "Multi-Part",
    "Class1-MultipartCharacter-po",
    "Class2-NonMultipartCharcaters",
}

# Minimum images a class must have to be included in training
MIN_IMAGES = 2

# ── MAPPING: folder name → Tamil Unicode display label ─────────────────────────
LABEL_MAP = {
    "ai"   : "ஐ",
    "c"    : "ச்",
    "e"    : "எ",
    "i"    : "இ",
    "k"    : "க்",
    "ku"   : "கு",
    "l"    : "ல்",
    "l2"   : "ழ்",
    "l5"   : "ள்",
    "l5u"  : "ளு",
    "l5u4" : "ளூ",
    "li"   : "லி",
    "m"    : "ம்",
    "n"    : "ந்",
    "n1"   : "ன்",
    "n1u"  : "னு",
    "n1u4" : "னூ",
    "n2"   : "ண்",
    "n2u4" : "ணூ",
    "n3"   : "ங்",
    "n5"   : "ம",
    "n5i"  : "மி",
    "n5u"  : "மு",
    "ni"   : "நி",
    "o"    : "ஒ",
    "p"    : "ப்",
    "pi"   : "பி",
    "pu"   : "பு",
    "pu4"  : "பூ",
    "r"    : "ர்",
    "r5"   : "ற்",
    "r5i"  : "றி",
    "ru"   : "ரு",
    "t"    : "த்",
    "t (maybe)": "த்",   # merge with t
    "t2"   : "ட்",
    "t2i"  : "டி",
    "ti"   : "தி",
    "tu"   : "து",
    "v"    : "வ்",
    "vi"   : "வி",
    "vu (maybe)": "வு",
    "y"    : "ய்",
    "yi"   : "யி",
}

# ── COLLECT IMAGES PER CLASS ───────────────────────────────────────────────────
random.seed(RANDOM_SEED)
class_images: dict[str, list[Path]] = {}

for folder in LABELS_ROOT.rglob("*"):
    if not folder.is_dir():
        continue
    if folder.name in SKIP_FOLDERS:
        continue

    jpgs = list(folder.glob("*.JPG")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    if not jpgs:
        continue

    raw_name = folder.name
    # Normalise "t (maybe)" → "t", "vu (maybe)" → "vu (maybe)" etc.
    class_key = raw_name  # keep as-is for mapping lookup

    if class_key not in LABEL_MAP:
        print(f"[SKIP] Unknown class '{class_key}' ({len(jpgs)} images) — add to LABEL_MAP to include")
        continue

    if class_key not in class_images:
        class_images[class_key] = []
    class_images[class_key].extend(jpgs)

# ── SPLIT & COPY ───────────────────────────────────────────────────────────────
print(f"\n{'Class':<20} {'Tamil':>8} {'Total':>7} {'Train':>7} {'Val':>5}")
print("-" * 55)

total_train = 0
total_val   = 0

for class_key, images in sorted(class_images.items()):
    if len(images) < MIN_IMAGES:
        print(f"[SKIP] '{class_key}' has only {len(images)} image(s) — need ≥{MIN_IMAGES}")
        continue

    random.shuffle(images)
    split = max(1, int(len(images) * TRAIN_RATIO))
    train_imgs = images[:split]
    val_imgs   = images[split:] if len(images) > split else images[:1]  # at least 1 in val

    # Use ASCII folder name (class_key) for ImageFolder compatibility
    for img in train_imgs:
        dest = TRAIN_DIR / class_key / img.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest)

    for img in val_imgs:
        dest = VAL_DIR / class_key / img.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest)

    tamil = LABEL_MAP[class_key]
    print(f"  {class_key:<18} {tamil:>8} {len(images):>7} {len(train_imgs):>7} {len(val_imgs):>5}")
    total_train += len(train_imgs)
    total_val   += len(val_imgs)

print("─" * 55)
print(f"  {'TOTAL':<18} {'':>8} {total_train+total_val:>7} {total_train:>7} {total_val:>5}")
print(f"\n✅ Dataset ready at: {DATA_ROOT}")
print(f"   Train: {TRAIN_DIR}")
print(f"   Val:   {VAL_DIR}")

# ── SAVE LABEL MAP JSON (used by inference) ────────────────────────────────────
import json
label_map_path = Path(__file__).parent / "label_map.json"
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)
print(f"   Label map saved: {label_map_path}")
