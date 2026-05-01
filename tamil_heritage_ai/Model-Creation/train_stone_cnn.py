"""
train_stone_cnn.py
------------------
Train a CNN on the prepared stone-inscription character dataset.
Saves best model as:  model_stone.pth
  checkpoint keys:
    - state_dict      : model weights
    - idx_to_label    : list of ASCII folder names (class_00, class_01, ...)
    - label_to_tamil  : dict  folder_name → Tamil Unicode string
    - class_to_idx    : dict  folder_name → int index
    - num_classes     : int

Run:
    python train_stone_cnn.py
"""

import os, json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATA_ROOT    = SCRIPT_DIR / "data"
TRAIN_DIR    = DATA_ROOT / "train"
VAL_DIR      = DATA_ROOT / "val"
OUT_MODEL    = SCRIPT_DIR / "model_stone.pth"
LABEL_MAP_F  = SCRIPT_DIR / "label_map.json"

IMG_SIZE     = 64
BATCH_SIZE   = 32
EPOCHS       = 40
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── VALIDATE DATASET ───────────────────────────────────────────────────────────
if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError(
        "data/train or data/val not found.\n"
        "Run  python prepare_dataset.py  first."
    )

# ── LOAD LABEL MAP ─────────────────────────────────────────────────────────────
label_to_tamil: dict[str, str] = {}
if LABEL_MAP_F.exists():
    with open(LABEL_MAP_F, encoding="utf-8") as f:
        label_to_tamil = json.load(f)

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.1),          # mild augmentation
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),                            # [0, 1]
])

val_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ── DATASETS ───────────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

num_classes = len(train_ds.classes)
idx_to_label = train_ds.classes          # e.g. ['ai', 'c', 'e', ...]

print(f"\n{'='*55}")
print(f"  Device      : {DEVICE}")
print(f"  Classes     : {num_classes}")
print(f"  Train images: {len(train_ds)}")
print(f"  Val   images: {len(val_ds)}")
print(f"  Image size  : {IMG_SIZE}×{IMG_SIZE}")
print(f"  Epochs      : {EPOCHS}")
print(f"{'='*55}")
print("\n  Class list:")
for cls in idx_to_label:
    tamil = label_to_tamil.get(cls, cls)
    print(f"    {cls:<20} → {tamil}")
print()

# ── MODEL ──────────────────────────────────────────────────────────────────────
class StoneTamilCNN(nn.Module):
    """3-block CNN suited for small, grayscale stone-character crops."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),           # 64→32

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),           # 32→16

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),           # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = StoneTamilCNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── TRAINING LOOP ──────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # — Train —
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_loss = total_loss / total
    train_acc  = correct / total

    # — Validate —
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step()
    elapsed = time.time() - t0

    saved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "state_dict"     : model.state_dict(),
            "idx_to_label"   : idx_to_label,
            "label_to_tamil" : label_to_tamil,
            "class_to_idx"   : train_ds.class_to_idx,
            "num_classes"    : num_classes,
        }, str(OUT_MODEL))
        saved = "  ✅ saved"

    print(
        f"Epoch {epoch:>3}/{EPOCHS}  "
        f"loss={train_loss:.4f}  "
        f"train={train_acc:.3f}  "
        f"val={val_acc:.3f}  "
        f"[{elapsed:.1f}s]{saved}"
    )

print(f"\n🏆 Best val accuracy : {best_val_acc:.3f}")
print(f"   Model saved at   : {OUT_MODEL}")
