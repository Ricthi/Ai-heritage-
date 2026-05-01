import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import os

# --- Model Architecture (Matching main_app.py) ---
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        # The input size to fc1 depends on the image size (50x50)
        # After 3 rounds of Conv+Pool:
        # 50 -> (50-2)/2 = 24
        # 24 -> (24-2)/2 = 11
        # 11 -> (11-2)/2 = 4
        # So 32 * 4 * 4 = 512
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train():
    # Load data
    try:
        with open("data/processed/X.pickle", "rb") as f:
            X = pickle.load(f)
        with open("data/processed/y.pickle", "rb") as f:
            y = pickle.load(f)
        with open("data/processed/categories.pickle", "rb") as f:
            categories = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found. Run prepare_labels.py first.")
        return

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(len(categories)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    epochs = 50 # Increased from 5 to 50 for proper learning
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

    # Save model
    os.makedirs("Model-Creation", exist_ok=True)
    torch.save(model.state_dict(), "Model-Creation/model_torch.pth")
    print("Model saved to Model-Creation/model_torch.pth")

if __name__ == "__main__":
    train()
