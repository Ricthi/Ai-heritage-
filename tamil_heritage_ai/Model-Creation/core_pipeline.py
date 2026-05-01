import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import sys

# Ensure Model-Creation is in path for preprocessing
sys.path.append(os.path.join(os.getcwd(), "Model-Creation"))
from preprocessing import Preprocessor, PreprocessConfig

# --- Constants ---
IMG_SIZE = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

from decoding import decode_predictions
from tamil_charset import TAMIL_CHARS
from ocr_postprocess import group_chars_into_words

def run_full_pipeline(image_bgr, model, preprocessor, min_area=150, do_gradcam=True):
    # 1. Preprocessing
    bgr_resized, gray, denoised, binary = preprocessor.run(image_bgr)
    
    # 2. Segmentation
    dilate = cv2.dilate(binary, None, iterations=2)
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort boxes in reading order
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1] // 40 * 1000 + cv2.boundingRect(ctr)[0])
    
    results = []
    annotated_bgr = bgr_resized.copy()
    recognized_chars = []
    total_conf = 0
    
    # Prepare for batch decoding if needed, but here we do it per crop for now
    for idx, cnt in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < min_area:
            continue
            
        roi = binary[y:y+h, x:x+w]
        # Preprocessing Sanity: Ensure 0..1 range and correct shape
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(roi_resized).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1) # Ensure dim=1 for class probabilities
            conf, class_idx = torch.max(probs, dim=1)
            
            # Use dynamic mapping if available
            cid = int(class_idx.item())
            if hasattr(model, 'category_mapping'):
                label_name = model.category_mapping[cid]
                # Map 'ai' -> 'ஐ', etc.
                from tamil_charset import TAMIL_MAP
                tamil = TAMIL_MAP.get(label_name, label_name)
                
                # For top 5
                top5_probs, top5_idxs = torch.topk(probs, 5)
                top5_labels = [TAMIL_MAP.get(model.category_mapping[int(i)], model.category_mapping[int(i)]) for i in top5_idxs[0]]
                top5 = list(zip(top5_labels, top5_probs[0].tolist()))
            else:
                # Fallback to legacy decoding
                conf_val = float(conf.item())
                labels, scores = decode_predictions([cid], [conf_val])
                tamil = labels[0]
                top5_probs, top5_idxs = torch.topk(probs, 5)
                top5_labels, _ = decode_predictions(top5_idxs[0].tolist(), top5_probs[0].tolist())
                top5 = list(zip(top5_labels, top5_probs[0].tolist()))

            conf_val = float(conf.item())

            total_conf += conf_val
            recognized_chars.append(tamil)
            
            # Character-level color coding (legacy)
            if conf_val >= 0.9:
                color = (32, 211, 74)       # green
            elif conf_val >= 0.7:
                color = (240, 180, 41)      # yellow
            else:
                color = (0, 0, 255)         # red
            cv2.rectangle(annotated_bgr, (x, y), (x+w, y+h), color, 1)
            
            results.append({
                "index": idx,
                "label": tamil,
                "confidence": conf_val,
                "status": "ok" if conf_val >= 0.7 else "low_conf",
                "x1": int(x), "y1": int(y), "x2": int(x+w), "y2": int(y+h),
                "width": int(w), "height": int(h),
                "top5": [(str(a), float(b)) for a, b in top5],
                "crop": roi,
                "heatmap": None, 
                "overlay": None,
            })

    # 3. Word Grouping & Filtering
    raw_chars_for_grouping = [
        {"char": r["label"], "confidence": r["confidence"], "bbox": [r["x1"], r["y1"], r["x2"], r["y2"]]}
        for r in results
    ]
    words = group_chars_into_words(raw_chars_for_grouping)
    word_text = " ".join([w.text for w in words])

    avg_conf = float(total_conf / len(results)) if results else 0.0
    text = "".join(recognized_chars)
    suspicious = bool(avg_conf < 0.6 or len(text) == 0)

    json_spec = {
        "source_name": "uploaded_image",
        "model_path": "Model-Creation/model_torch.pth",
        "preprocess": {
            "angle": 0.0,
            "shape": list(bgr_resized.shape),
        },
        "character_count": len(results),
        "word_count": len(words),
        "avg_confidence": avg_conf,
        "suspicious": suspicious,
        "recognized_text": word_text,
        "words": [
            {
                "text": w.text,
                "confidence": w.avg_confidence,
                "bbox": w.bbox,
                "is_suspicious": w.is_suspicious
            } for w in words
        ],
        "characters": [
            {
                "index": r["index"],
                "label": r["label"],
                "confidence": r["confidence"],
                "status": r["status"],
                "bbox": [r["x1"], r["y1"], r["x2"], r["y2"]],
                "top5": r["top5"],
            }
            for r in results
        ],
    }

    analysis_result = {
        "source_name": "uploaded_image",
        "original_bgr": image_bgr,
        "bgr_resized": bgr_resized,
        "preprocess": {
            "gray": gray,
            "denoised": denoised,
            "binary": binary,
            "clean": binary,
            "angle": 0.0,
        },
        "boxes": [[r["x1"], r["y1"], r["x2"], r["y2"]] for r in results],
        "results": results,
        "words": words,
        "annotated_bgr": annotated_bgr,
        "recognized_text": word_text,
        "raw_text": text,
        "avg_confidence": avg_conf,
        "gradcam_count": sum(1 for r in results if r["heatmap"] is not None),
        "suspicious": suspicious,
        "json_spec": json_spec,
    }
    return analysis_result
