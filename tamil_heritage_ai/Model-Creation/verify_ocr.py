
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate

# --- Configuration ---
BASE_DIR = r"d:\Ancient-Tamil-Script-Recognition-master\Ancient-Tamil-Script-Recognition-master"
MODEL_PATH = os.path.join(BASE_DIR, "Model-Creation", "model_torch.pth")
DATADIR = os.path.join(BASE_DIR, "Labels", "Labelled Dataset - Fig 51", "Labelled Dataset - Fig 51")
IGNORE_FOLDERS = ['1 - Multipart', '2 - Unknown']
IMG_SIZE = 50

# --- Helper Functions ---
def get_categories():
    return [d for d in sorted(os.listdir(DATADIR)) if d not in IGNORE_FOLDERS]

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    print("-" * 50)
    print(" VERIFYING OCR OUTPUT - SAMPLE RECOGNITION")
    print("-" * 50)

    # 1. Load Categories
    CATEGORIES = get_categories()
    print(f"Categories loaded: {len(CATEGORIES)}")

    # 2. Load Model
    model = CNN(len(CATEGORIES))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # 3. Load Sample Image
    # Notebooks use 'Original.jpg' or images from the dataset
    # Let's use an image that has multiple characters if possible, or just a sample from the dataset.
    # Fig 51 labeled dataset has individual character folders.
    # Let's pick a few images and pretend it's a sequence.
    
    sample_images = [
        (os.path.join(DATADIR, 'k', '1.JPG'), 'k'),
        (os.path.join(DATADIR, 'l', '11.JPG'), 'l'),
        (os.path.join(DATADIR, 'r', '106.JPG'), 'r')
    ]

    print("\nStarting OCR Sequence Run...")
    print(f"{'Image Path':<60} | {'True':<5} | {'Predicted':<10} | {'Confidence':<10}")
    print("-" * 95)

    recognized_text = []

    for img_path, true_label in sample_images:
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0
        tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            pred_label = CATEGORIES[idx.item()]
            
        recognized_text.append(pred_label)
        path_short = os.path.basename(img_path)
        print(f"{path_short:<60} | {true_label:<5} | {pred_label:<10} | {conf.item()*100:>8.2f}%")

    print("\n" + "=" * 50)
    print(f" FINAL OCR STRING: {' '.join(recognized_text)}")
    print("=" * 50)

if __name__ == '__main__':
    main()
