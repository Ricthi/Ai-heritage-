import torch
import torch.nn as nn
import os
import numpy as np

# Define the architecture (MUST match core_pipeline.py)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

def diagnostic():
    model_path = "c:/Users/ASUS/OneDrive/Desktop/site/Ai-heritage--main/Ai-heritage--main/tamil_heritage_ai/Model-Creation/model_torch.pth"
    
    print(f"--- Step 3: File Integrity ---")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return
        
    size = os.path.getsize(model_path)
    print(f"File size: {size / 1024:.2f} KB")
    
    with open(model_path, 'rb') as f:
        header = f.read(100).decode('utf-8', 'ignore')
        if "version https://git-lfs" in header:
            print("⚠️ WARNING: This is a Git LFS pointer file, not the actual weights!")
        else:
            print("✅ File header looks like a binary torch save.")

    print(f"\n--- Step 1: Weight Norms ---")
    try:
        # Load for 26 classes (legacy)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        for name, param in state_dict.items():
            norm = torch.norm(param).item()
            print(f"{name:20} | Shape: {str(list(param.shape)):15} | Norm: {norm:.4f}")
            if norm < 1e-8:
                print(f"  ⚠️ ALERT: Dead weights detected in {name}!")
        
        print("\n✅ Weight inspection complete.")
    except Exception as e:
        print(f"❌ ERROR: Could not load model: {e}")

if __name__ == "__main__":
    diagnostic()
