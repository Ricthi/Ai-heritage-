# engine.py
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from dataclasses import dataclass
from tamil_charset import TAMIL_CHARS

@dataclass
class EngineConfig:
    img_size: int = 50
    model_path: str = os.path.join(os.path.dirname(__file__), "model_torch.pth")
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
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

def load_cnn(model_path: str | None = None):
    cfg = EngineConfig()
    if model_path is None:
        model_path = cfg.model_path

    abs_path = os.path.abspath(model_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file not found at {abs_path}")

    num_classes = len(TAMIL_CHARS)
    model = CNN(num_classes).to(cfg.device)

    sd = torch.load(abs_path, map_location=cfg.device)
    # Allow loading full model or state_dict
    if isinstance(sd, nn.Module):
        sd = sd.state_dict()
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg

@torch.inference_mode()
def predict_char(model: nn.Module,
                 cfg: EngineConfig,
                 roi_binary: np.ndarray):
    """
    roi_binary: grayscale or binary ROI, HxW
    returns (char, conf_float)
    """
    img = roi_binary.astype(np.float32)
    img = cv2.resize(img, (cfg.img_size, cfg.img_size))  # type: ignore
    img = img / 255.0
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(cfg.device)
    logits = model(t)
    prob = torch.softmax(logits, 1)
    conf, idx = torch.max(prob, 1)
    return TAMIL_CHARS[idx.item()], float(conf.item())
