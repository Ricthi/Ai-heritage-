import streamlit as st
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import pandas as pd
import sys
import io
import json
import pytesseract
from ocr_fallback import run_ocr_fallback, TESSDATA_DIR, HAS_TESSERACT
from dataclasses import dataclass
import base64
from streamlit_drawable_canvas import st_canvas

# Pathing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tamil_charset import TAMIL_CHARS

# --- Data Structures ---
@dataclass
class EngineConfig:
    img_size: int = 50
    model_path: str = os.path.join(os.path.dirname(__file__), "model_torch.pth")
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class PreprocessConfig:
    denoise_h: int = 12
    template_window: int = 7
    search_window: int = 21
    block_size: int = 15
    c_value: int = 8

    def normalized(self):
        block = self.block_size if self.block_size % 2 == 1 else self.block_size + 1
        block = max(3, block)
        return PreprocessConfig(
            denoise_h=max(1, self.denoise_h),
            template_window=max(3, self.template_window),
            search_window=max(3, self.search_window),
            block_size=block,
            c_value=self.c_value,
        )

# --- Model Architecture ---
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

# --- Core Logic ---
@st.cache_resource
def get_model(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None, f"File not found: {abs_path}"
    
    num_classes = len(TAMIL_CHARS)
    model = CNN(num_classes).to(EngineConfig().device)
    
    # Attempt loading
    sd = None
    error_msg = ""
    
    # Try strict loading first
    try:
        sd = torch.load(abs_path, map_location=EngineConfig().device, weights_only=True)
    except Exception as e1:
        # Fallback to legacy loading if strict fails
        try:
            sd = torch.load(abs_path, map_location=EngineConfig().device, weights_only=False)
        except Exception as e2:
            error_msg = f"Strict Load Error: {e1} | Legacy Load Error: {e2}"

    if sd is None:
        return None, error_msg

    try:
        # If it's a full model object instead of state_dict, extract state_dict
        if isinstance(sd, nn.Module):
            sd = sd.state_dict()
            
        # Advanced Weight Bridging
        if 'fc3.weight' in sd:
            src_classes = sd['fc3.weight'].shape[0]
            if src_classes != num_classes:
                if src_classes == 26:
                    # Specific Legacy Mapping
                    from legacy_to_full_mapping import transfer_classifier_head
                    new_w, new_b = transfer_classifier_head(
                        sd['fc3.weight'].cpu().numpy(), sd['fc3.bias'].cpu().numpy(),
                        model.fc3.weight.detach().cpu().numpy(), model.fc3.bias.detach().cpu().numpy()
                    )
                    sd['fc3.weight'], sd['fc3.bias'] = torch.from_numpy(new_w), torch.from_numpy(new_b)
                else:
                    # Generic Best-Fit Bridging (for 50 or other class counts)
                    new_w = model.fc3.weight.clone().detach()
                    new_b = model.fc3.bias.clone().detach()
                    
                    # Copy matching parts
                    copy_size = min(src_classes, num_classes)
                    new_w[:copy_size, :] = sd['fc3.weight'][:copy_size, :]
                    new_b[:copy_size] = sd['fc3.bias'][:copy_size]
                    
                    sd['fc3.weight'], sd['fc3.bias'] = new_w, new_b
        
        model.load_state_dict(sd)
        model.eval()
        return model, "Success"
    except Exception as e:
        return None, f"State Dict Error: {str(e)}"

# --- Stone CNN loader (model_stone.pth) ---
class StoneTamilCNN(nn.Module):
    """Architecture matching train_stone_cnn.py (64×64 input, 44 classes)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def get_stone_model(path):
    """Load model_stone.pth and return (model, idx_to_label, label_to_tamil, 'stone')."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None, [], {}, f"File not found: {abs_path}"
    try:
        ckpt = torch.load(abs_path, map_location=EngineConfig().device, weights_only=False)
        idx_to_label  = ckpt["idx_to_label"]        # list[str]
        label_to_tamil = ckpt.get("label_to_tamil", {})
        num_classes   = ckpt["num_classes"]
        model = StoneTamilCNN(num_classes).to(EngineConfig().device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, idx_to_label, label_to_tamil, "ok"
    except Exception as e:
        return None, [], {}, str(e)

@st.cache_data(show_spinner=False)
def load_image_bytes(file_bytes):
    pil_img = Image.open(file_bytes).convert("RGB")
    return np.array(pil_img)

def preprocess_image(image_rgb: np.ndarray, config: PreprocessConfig):
    cfg = config.normalized()
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=cfg.denoise_h, templateWindowSize=cfg.template_window, searchWindowSize=cfg.search_window)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, cfg.block_size, cfg.c_value)
    return {"input": image_rgb, "gray": gray, "denoised": denoised, "binary": binary}

# --- Grad-CAM & Visualization Helpers ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if self.forward_handle:
            self.forward_handle.remove()
        if self.backward_handle:
            self.backward_handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        grads = self.gradients
        acts = self.activations
        if grads is None or acts is None:
            raise RuntimeError("Gradients or activations not captured for Grad-CAM.")
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            raise RuntimeError("Grad-CAM heatmap is flat. No meaningful activation map generated.")
        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam, class_idx

def find_last_conv_layer(model):
    candidate = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            candidate = module
    if candidate is None:
        raise ValueError("No Conv2d layer found for Grad-CAM target.")
    return candidate

def apply_colormap_on_image(rgb_img_np, cam_map, alpha=0.45):
    heatmap_uint8 = np.uint8(255 * cam_map)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if rgb_img_np.dtype != np.uint8:
        rgb_img_np = np.clip(rgb_img_np, 0, 255).astype(np.uint8)
    if heatmap_rgb.shape[:2] != rgb_img_np.shape[:2]:
        heatmap_rgb = cv2.resize(
            heatmap_rgb,
            (rgb_img_np.shape[1], rgb_img_np.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    blended = cv2.addWeighted(rgb_img_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return heatmap_rgb, blended

def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

def non_max_suppression(detections, iou_thresh=0.15):
    """
    Standard NMS: keep the highest-score box; suppress any box that
    overlaps it by >= iou_thresh. Works on dicts with 'bbox' and 'score'.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d.get("score", 0), reverse=True)
    kept = []
    for det in dets:
        suppress = False
        for k in kept:
            if compute_iou(det["bbox"], k["bbox"]) >= iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept

def suppress_duplicates(detections, iou_thresh=0.5):
    """
    Keeps highest-confidence detection when boxes overlap strongly.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    kept = []

    for det in dets:
        duplicate = False
        for existing in kept:
            if compute_iou(det["bbox"], existing["bbox"]) >= iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    return kept

def classic_fallback_detect(gray, min_area=300, max_area=9000,
                            aspect_min=0.20, aspect_max=4.5):
    """
    Robust character detection using connected components with tight filters.
    Avoids the 'too many tiny blobs' problem of vanilla adaptiveThreshold.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,   # larger block = less aggressive than 11-15
        12    # larger C = raises threshold = fewer spurious blobs
    )

    open_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thr,   cv2.MORPH_OPEN,  open_kernel,  iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        morph, connectivity=8
    )

    h, w = gray.shape[:2]
    image_area = h * w
    detections = []

    for i in range(1, num_labels):   # skip background label 0
        x, y, bw, bh, area = stats[i]

        if area < min_area or area > min(max_area, image_area * 0.03):
            continue
        if bw < 8 or bh < 8:
            continue

        aspect    = bw / float(bh)
        fill_ratio = area / float(max(bw * bh, 1))

        if not (aspect_min <= aspect <= aspect_max):
            continue
        if not (0.15 <= fill_ratio <= 0.75):
            continue

        # Refine bounding box via contours on the component ROI
        roi_mask = morph[y:y+bh, x:x+bw]
        cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        rx, ry, rw, rh = cv2.boundingRect(max(cnts, key=cv2.contourArea))

        x1, y1 = x + rx, y + ry
        x2, y2 = x1 + rw, y1 + rh

        if rw * rh < min_area:
            continue

        score = min(0.95, 0.72 + (rw * rh / float(max_area)) * 0.20)
        detections.append({
            "bbox":  [int(x1), int(y1), int(x2), int(y2)],
            "score": float(score),
        })

    return non_max_suppression(detections, iou_thresh=0.15)

def sort_reading_order(detections, y_tolerance=20):
    """
    Sort Tamil inscription detections top-to-bottom, then left-to-right.
    Groups nearby y-values into visual rows.
    """
    if not detections:
        return []

    dets = list(detections)
    dets = sorted(dets, key=lambda d: (d["bbox"][1], d["bbox"][0]))

    rows = []
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        cy = (y1 + y2) / 2

        placed = False
        for row in rows:
            row_cy = np.mean([(r["bbox"][1] + r["bbox"][3]) / 2 for r in row])
            if abs(cy - row_cy) <= y_tolerance:
                row.append(det)
                placed = True
                break

        if not placed:
            rows.append([det])

    ordered = []
    for row in rows:
        row_sorted = sorted(row, key=lambda d: d["bbox"][0])
        ordered.extend(row_sorted)

    for i, det in enumerate(ordered):
        det["order"] = i + 1

    return ordered

def draw_boxes(image_pil, detections, selected_idx=None, show_labels=True, show_order=True):
    img = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        conf = float(d.get("confidence", 0))

        if idx == selected_idx:
            color = "cyan"
            width = 5
        elif conf >= 90:
            color = "lime"
            width = 2
        elif conf >= 70:
            color = "yellow"
            width = 2
        else:
            color = "red"
            width = 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label_parts = []
        if show_order:
            label_parts.append(f"#{d.get('order', idx + 1)}")
        if show_labels:
            label_parts.append(f"{conf:.1f}%")
            preview = str(d.get("text", d.get("raw_text", "")))[:12]
            if preview:
                label_parts.append(preview)

        if label_parts:
            draw.text((x1, max(0, y1 - 16)), " ".join(label_parts), fill=color)

    return img

def crop_with_padding(image_pil, bbox, pad=20):
    x1, y1, x2, y2 = bbox
    w, h = image_pil.size
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    return image_pil.crop((x1, y1, x2, y2))

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="Heritage AI Dashboard")

# Initialize Session State
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'stages' not in st.session_state:
    st.session_state.stages = {}
if 'annotated' not in st.session_state:
    st.session_state.annotated = None
if 'up_name' not in st.session_state:
    st.session_state.up_name = ""
if 'image_pil' not in st.session_state:
    st.session_state.image_pil = None

# Parameter Initialization
if "denoise_h" not in st.session_state: st.session_state.denoise_h = 12
if "block_size" not in st.session_state: st.session_state.block_size = 15
if "c_value" not in st.session_state: st.session_state.c_value = 8
if "min_area" not in st.session_state: st.session_state.min_area = 150

# Custom Styling
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1e1e3f 0%, #0d0d1a 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
    .success-text { color: #2ea043; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("Model")
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    model_choice = st.selectbox("Select Model", options=model_files if model_files else ["model_torch.pth"], label_visibility="collapsed")
    
    st.markdown("Min segmentation area")
    st.slider("Min Area", 50, 1000, key="min_area", label_visibility="collapsed")
    st.caption(f"{st.session_state.min_area}")
    
    st.markdown("---")
    st.markdown("🛠️ Preprocessing")
    st.slider("Denoise Strength", 1, 50, key="denoise_h")
    st.slider("Adaptive Block Size", 3, 51, step=2, key="block_size")
    st.slider("Adaptive C", -20, 20, key="c_value")

    gradcam_enabled = st.checkbox("Enable Grad-CAM heatmaps", value=True)
    
    st.markdown("Export format")
    export_fmt = st.radio("Export Format", ["PNG", "PDF", "SVG"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("Confidence colour policy")
    st.markdown("🟢 ≥ 90% High")
    st.markdown("🟡 70-89% Medium")
    st.markdown("🔴 < 70% Low")
    
    st.markdown("---")
    st.markdown("### 🔍 OCR Fallback")
    tess_path_input = st.text_input("Tesseract EXE Path", value=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if os.path.exists(tess_path_input):
        pytesseract.pytesseract.tesseract_cmd = tess_path_input
        st.success("✅ Tesseract Ready")
        tess_ready = True
    else:
        st.error("❌ Tesseract Path Invalid")
        tess_ready = False

    if TESSDATA_DIR:
        st.success(f"✅ Tamil.traineddata found:\n`{TESSDATA_DIR}`")
    else:
        st.warning("⚠️ Tamil.traineddata not found. Using built-in 'tam' model.")

# Main Header
st.markdown("""
<div class='header-box'>
    <h1 style='color: white; margin: 0;'>🏛️ Tamil Heritage AI</h1>
    <p style='color: #8b949e; margin: 5px 0 0 0;'>Research-grade recognition · Grad-CAM explainability · Multi-format export</p>
</div>
""", unsafe_allow_html=True)

# File Uploader
up = st.file_uploader("📂 Upload a palm-leaf / inscription image", type=['png','jpg','jpeg','tif'])

if up:
    if up.name != st.session_state.up_name:
        st.session_state.analyzed = False
        st.session_state.results = []
        st.session_state.up_name = up.name

    st.markdown(f"📄 **{up.name}** {up.size/1024/1024:.2f}MB")
    
    if st.button("Run Analysis", use_container_width=False):
        st.session_state.up_name = up.name
        
        image_rgb = load_image_bytes(io.BytesIO(up.getvalue()))
        p_cfg = PreprocessConfig(
            denoise_h=st.session_state.denoise_h, 
            block_size=st.session_state.block_size, 
            c_value=st.session_state.c_value
        )
        stages = preprocess_image(image_rgb, p_cfg)
        
        # Neural Analysis
        model_dir  = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, model_choice)

        # --- Decide which loader to use ---
        is_stone_model = (model_choice == "model_stone.pth")

        if is_stone_model:
            stone_model, stone_idx_to_label, stone_label_to_tamil, stone_msg = get_stone_model(model_path)
            model_ok = stone_model is not None
            if not model_ok:
                st.error(f"Stone model failed to load: {stone_msg}")
        else:
            model, model_msg = get_model(model_path)
            model_ok = model is not None
            if not model_ok:
                st.error(f"Neural engine failed to load: {model_msg}")
                st.info("Check if the model file is correctly placed in the 'Model-Creation' folder.")

        if model_ok:
            with st.spinner("Analyzing Script..."):
                gray_for_detect = stages["denoised"]  # NLM-denoised gray

                # Use tight connected-component detection (avoids stone texture blobs)
                candidate_boxes = classic_fallback_detect(
                    gray_for_detect,
                    min_area=max(st.session_state.min_area, 300),
                    max_area=9000,
                )

                binary = stages["binary"]
                raw_detections = []
                image_pil = Image.fromarray(image_rgb)



                for box_info in candidate_boxes:
                    x1, y1, x2, y2 = box_info["bbox"]
                    w, h = x2 - x1, y2 - y1

                    roi = binary[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    if is_stone_model:
                        # Stone model: 64×64 grayscale, real label mapping
                        inp = cv2.resize(roi, (64, 64)).astype(np.float32) / 255.0
                        t   = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(EngineConfig().device)
                        with torch.no_grad():
                            logits = stone_model(t)
                            prob   = torch.softmax(logits, 1)
                            top5_probs, top5_idxs = torch.topk(prob, min(5, prob.shape[1]))
                            conf_t, idx_t = top5_probs[0][0], top5_idxs[0][0]
                        folder_name = stone_idx_to_label[idx_t.item()]
                        tamil_char = stone_label_to_tamil.get(folder_name, folder_name)
                        char = f"{tamil_char} [{folder_name}]"
                        conf_val = conf_t.item() * 100
                        
                        top5 = []
                        for i, s in zip(top5_idxs[0].tolist(), top5_probs[0].tolist()):
                            fname = stone_idx_to_label[int(i)]
                            tchar = stone_label_to_tamil.get(fname, fname)
                            top5.append((f"{tchar} [{fname}]", float(s)))
                            
                        active_model = stone_model
                    else:
                        # Original model: 50×50, TAMIL_CHARS mapping
                        inp = cv2.resize(roi, (50, 50)).astype(np.float32) / 255.0
                        t   = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(EngineConfig().device)
                        with torch.no_grad():
                            logits = model(t)
                            prob   = torch.softmax(logits, 1)
                            top5_probs, top5_idxs = torch.topk(prob, min(5, prob.shape[1]))
                            conf_t, idx_t = top5_probs[0][0], top5_idxs[0][0]
                        char = TAMIL_CHARS[idx_t.item()]
                        conf_val = conf_t.item() * 100
                        top5 = [(TAMIL_CHARS[int(i)], float(s))
                                for i, s in zip(top5_idxs[0].tolist(), top5_probs[0].tolist())]
                        active_model = model

                    ocr_char = ""
                    if conf_val < 70 and tess_ready:
                        ocr_result = run_ocr_fallback(roi, lang="Tamil")
                        if ocr_result.get("ok"):
                            ocr_char = ocr_result.get("text", "")

                    crop_pil = image_pil.crop((x1, y1, x2, y2))

                    det = {
                        "bbox": (x1, y1, x2, y2),
                        "crop_pil": crop_pil,
                        "input_tensor": t,
                        "pred_idx": idx_t.item(),
                        "text": char,
                        "char": char,
                        "confidence": conf_val,
                        "ocr_fallback": ocr_char,
                        "uncertain": 0,
                        "top5": top5,
                    }
                    raw_detections.append(det)

                successful_gradcams = 0
                for det in raw_detections:
                    if gradcam_enabled:
                        try:
                            _active = stone_model if is_stone_model else model
                            _target = find_last_conv_layer(_active)
                            _gcam   = GradCAM(_active, _target)
                            cam_map, class_idx = _gcam.generate(det["input_tensor"], class_idx=det["pred_idx"])
                            crop_rgb = np.array(det["crop_pil"].convert("RGB"))
                            cam_map_resized = cv2.resize(
                                cam_map,
                                (crop_rgb.shape[1], crop_rgb.shape[0]),
                                interpolation=cv2.INTER_LINEAR
                            )
                            heatmap_rgb, blended_rgb = apply_colormap_on_image(crop_rgb, cam_map_resized, alpha=0.45)
                            det["gradcam_heatmap"] = heatmap_rgb
                            det["gradcam_blend"]   = blended_rgb
                            det["gradcam_ok"]      = True
                            successful_gradcams   += 1
                            _gcam.remove_hooks()
                        except Exception as e:
                            det["gradcam_heatmap"] = None
                            det["gradcam_blend"]   = None
                            det["gradcam_ok"]      = False
                            det["gradcam_error"]   = str(e)
                    else:
                        det["gradcam_ok"] = False

                work_dets = suppress_duplicates(raw_detections, iou_thresh=0.5)
                work_dets = sort_reading_order(work_dets)
                
                annotated = draw_boxes(image_pil, work_dets)
                
                st.session_state.stages = stages
                st.session_state.results = work_dets
                st.session_state.annotated = annotated
                st.session_state.analyzed = True
                st.session_state.image_pil = image_pil
        else:
            if not is_stone_model:
                st.error(f"Neural engine failed to load: {model_msg}")
                st.info("Check if the model file is correctly placed in the 'Model-Creation' folder.")

    if 'analyzed' in st.session_state and st.session_state.analyzed:
        st.markdown("✅ **Analysis Complete!**")

        stages = st.session_state.stages if isinstance(st.session_state.stages, dict) else {}
        results = st.session_state.results if isinstance(st.session_state.results, list) else []
        annotated = st.session_state.annotated

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{len(results)}</div>"
                f"<div class='metric-label'>Characters Detected</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with m2:
            avg_conf = float(np.mean([r.get('confidence', 0) for r in results])) if results else 0.0
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{avg_conf:.1f}%</div>"
                f"<div class='metric-label'>Avg. Confidence</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with m3:
            gradcam_count = sum(1 for r in results if r.get("gradcam_ok"))
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{gradcam_count}</div>"
                f"<div class='metric-label'>Grad-CAM Maps</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with m4:
            st.markdown(
                "<div class='metric-card'>"
                "<div class='metric-value success-text'>SUCCESS</div>"
                "<div class='metric-label'>Pipeline Status</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        t_orig, t_annot, t_zoom, t_text, t_table, t_export, t_json = st.tabs(
            [
                "🖼️ Original",
                "🖼️ Annotated",
                "🔍 Interactive Zoom",
                "📝 Recognized Text",
                "📊 Per Character Table",
                "💾 Export",
                "📄 JSON Spec",
            ]
        )

        with t_orig:
            st.markdown("### Original & Preprocessed")
            required_keys = ["input", "gray", "denoised", "binary"]
            if not stages or any(k not in stages for k in required_keys):
                st.warning("Preprocessing data missing. Please run analysis again.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(stages["input"], caption="Input", use_container_width=True)
                with col2:
                    st.image(stages["gray"], caption="Grayscale", use_container_width=True)

                col3, col4 = st.columns(2)
                with col3:
                    st.image(stages["denoised"], caption="Denoised (NL-Means)", use_container_width=True)
                with col4:
                    st.image(stages["binary"], caption="Adaptive Threshold", use_container_width=True)

                st.markdown(
                    "<p style='text-align: center; color: gray;'>"
                    "Fig. 2. Preprocessing pipeline output: (a) original input, "
                    "(b) greyscale conversion, (c) NL-Means denoising, and (d) adaptive thresholding"
                    "</p>",
                    unsafe_allow_html=True,
                )

        with t_annot:
            st.markdown("### Annotated with Confidence Colour Coding")
            if annotated is not None:
                # Legend and metrics
                l_col1, l_col2, l_col3, l_col4 = st.columns(4)
                avg_conf = float(np.mean([r.get('confidence', 0) for r in results])) if results else 0.0
                
                with l_col1:
                    st.metric("Total Detections", len(results))
                with l_col2:
                    st.metric("Average Confidence", f"{avg_conf:.1f}%")
                with l_col3:
                    st.markdown("**Legend**")
                    st.markdown("🟢 ≥ 90% (High)")
                with l_col4:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("🟡 70-89% (Med) &nbsp;&nbsp; 🔴 < 70% (Low)")
                
                st.image(annotated, use_container_width=True)
            else:
                st.info("Annotated view not available. Run analysis again.")

        with t_zoom:
            st.markdown("## 🔍 Interactive Character Zoom")
            st.markdown(
                "<p style='color:#8b949e; font-size:0.9rem; margin-top:-8px;'>Select a character index to zoom in and inspect its Grad-CAM heatmap and Top-5 predictions.</p>",
                unsafe_allow_html=True,
            )

            if not results:
                st.warning("No detections available.")
            else:
                work_dets = list(results)
                total = len(work_dets)

                # --- Single clean index selector (matches reference UI) ---
                idx_input = st.number_input(
                    "Character index",
                    min_value=0,
                    max_value=total - 1,
                    value=int(st.session_state.get("zoom_idx", 0)),
                    step=1,
                    key="zoom_idx_input",
                )
                st.session_state["zoom_idx"] = int(idx_input)
                selected_idx = int(idx_input)
                selected_det = work_dets[selected_idx]

                # --- 3-column inspection panel (matches reference exactly) ---
                col_crop, col_heat, col_blend = st.columns(3)

                with col_crop:
                    st.markdown("**Original crop**")
                    crop = crop_with_padding(
                        st.session_state.image_pil, selected_det["bbox"], pad=20
                    )
                    st.image(crop, use_container_width=True)

                with col_heat:
                    st.markdown("**Grad-CAM heatmap**")
                    if selected_det.get("gradcam_heatmap") is not None:
                        st.image(selected_det["gradcam_heatmap"], use_container_width=True)
                    else:
                        st.markdown(
                            """
                            <div style='background:#0d1117; border:1px solid #30363d;
                                        border-radius:8px; padding:48px 16px;
                                        text-align:center; color:#58a6ff;
                                        font-size:0.95rem; font-weight:500;'>
                                No heatmap available
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        err = selected_det.get("gradcam_error", "")
                        if err:
                            st.caption(f"Reason: {err}")

                with col_blend:
                    st.markdown("**Blended overlay**")
                    if selected_det.get("gradcam_blend") is not None:
                        st.image(selected_det["gradcam_blend"], use_container_width=True)
                    else:
                        st.markdown(
                            """
                            <div style='background:#0d1117; border:1px solid #30363d;
                                        border-radius:8px; padding:48px 16px;
                                        text-align:center; color:#58a6ff;
                                        font-size:0.95rem; font-weight:500;'>
                                No blend available
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # --- Metadata panel ---
                conf_val = float(selected_det.get("confidence", 0))
                conf_color = (
                    "#2ea043" if conf_val >= 90 else ("#e3b341" if conf_val >= 70 else "#cf222e")
                )
                meta1, meta2, meta3, meta4 = st.columns(4)
                with meta1:
                    st.markdown(
                        f"""<div class='metric-card'>
                            <div class='metric-value'>{selected_det.get('text','?')}</div>
                            <div class='metric-label'>Character</div></div>""",
                        unsafe_allow_html=True,
                    )
                with meta2:
                    st.markdown(
                        f"""<div class='metric-card'>
                            <div class='metric-value' style='color:{conf_color}'>{conf_val:.1f}%</div>
                            <div class='metric-label'>Confidence</div></div>""",
                        unsafe_allow_html=True,
                    )
                with meta3:
                    x1, y1, x2, y2 = selected_det["bbox"]
                    st.markdown(
                        f"""<div class='metric-card'>
                            <div class='metric-value' style='font-size:1rem'>({x1},{y1})</div>
                            <div class='metric-label'>BBox origin</div></div>""",
                        unsafe_allow_html=True,
                    )
                with meta4:
                    ocr_fb = selected_det.get("ocr_fallback") or "—"
                    st.markdown(
                        f"""<div class='metric-card'>
                            <div class='metric-value' style='font-size:1rem'>{ocr_fb}</div>
                            <div class='metric-label'>OCR Fallback</div></div>""",
                        unsafe_allow_html=True,
                    )

                # --- Top-5 predictions ---
                if selected_det.get("top5"):
                    st.markdown("##### 🏆 Top-5 Predictions")
                    for rank, (ch, sc) in enumerate(selected_det["top5"], 1):
                        st.markdown(f"**#{rank}** `{ch}` — {sc*100:.1f}%")
                        st.progress(float(sc))

        with t_text:
            st.markdown("#### 📝 Neural Transcription")
            full_text = "".join([r.get('text', "") for r in results])
            st.text_area("Neural Output", value=full_text, height=120, label_visibility="collapsed")

            # Word-level grouping
            words_list = []
            for r in results:
                t = r.get("text", "")
                if t:
                    words_list.append(t)
            if words_list:
                st.markdown(f"**Total characters recognised:** {len(words_list)}")
                st.markdown("**Full reading (spaced):** " + " ".join(words_list))

            ocr_fallback_text = "".join([r.get('ocr_fallback', "") for r in results if r.get('ocr_fallback')])
            if ocr_fallback_text:
                st.markdown("---")
                st.markdown("#### 🔍 Tesseract Fallback Verification")
                st.caption("Characters below 70% neural confidence were verified with Tamil.traineddata")
                st.text_area("OCR Fallback Output", value=ocr_fallback_text, height=120, label_visibility="collapsed")

        with t_table:
            if results:
                df = pd.DataFrame(results)
                # Drop non-serialisable object columns
                for col in ["crop_pil", "input_tensor", "gradcam_heatmap", "gradcam_blend", "char"]:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                # Convert top5 (list of tuples containing floats) → human-readable string
                if "top5" in df.columns:
                    df["top5"] = df["top5"].apply(
                        lambda v: ", ".join(f"{ch}:{sc*100:.1f}%" for ch, sc in v)
                        if isinstance(v, (list, tuple)) else (str(v) if v is not None else "")
                    )
                # Convert any remaining object columns that may hold mixed types
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = df[col].apply(
                        lambda x: str(x) if not isinstance(x, (str, type(None))) else (x or "")
                    )
                st.dataframe(df, use_container_width=True)
                st.caption("Note: 'ocr_fallback' is populated when neural confidence is below 70%.")
            else:
                st.info("No character-level results to display yet.")

        with t_export:
            st.markdown("### 💾 Export Results")
            if results:
                if export_fmt == "PNG" and annotated is not None:
                    import io as _io
                    buf = _io.BytesIO()
                    annotated.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Annotated PNG",
                        data=buf.getvalue(),
                        file_name="annotated_result.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                export_data = json.dumps([
                    {"char": r.get("text"), "conf": float(r.get("confidence", 0)),
                     "ocr_fallback": r.get("ocr_fallback"), "bbox": list(r.get("bbox", []))}
                    for r in results
                ], ensure_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ Download JSON Results",
                    data=export_data.encode("utf-8"),
                    file_name="recognition_results.json",
                    mime="application/json",
                    use_container_width=True,
                )
                full_text_export = "".join([r.get("text", "") for r in results])
                st.download_button(
                    label="⬇️ Download Text Transcript",
                    data=full_text_export.encode("utf-8"),
                    file_name="transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.info("Run analysis first to enable export.")

        with t_json:
            json_res = [
                {
                    "char": r.get("text"),
                    "conf": float(r.get("confidence", 0)),
                    "ocr_fallback": r.get("ocr_fallback"),
                    "bbox": r.get("bbox"),
                }
                for r in results
            ]
            st.json(json_res)
