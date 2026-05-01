# 🏛️ Tamil Heritage AI

A research-grade Python application designed to digitize, segment, and recognize ancient Tamil script from degraded stone inscriptions and palm-leaf manuscripts. This project leverages advanced image preprocessing techniques and a custom-trained Convolutional Neural Network (CNN) to achieve high-accuracy character recognition, complete with explainable AI (Grad-CAM) visualization.

## 🌟 Key Features

* **Advanced Image Preprocessing:** Combines Non-Local Means (NLM) denoising, Contrast Limited Adaptive Histogram Equalization (CLAHE), and Adaptive Thresholding to clean heavily degraded heritage images.
* **Smart Character Segmentation:** Uses connected-component analysis with tight bounding box filtering to ignore stone textures and isolate valid characters in their natural reading order.
* **Deep Learning Inference:** Powered by a custom 3-block PyTorch CNN (`model_stone.pth`), trained specifically on ancient Tamil character shapes (44 classes), achieving ~96% validation accuracy.
* **Explainable AI (Grad-CAM):** Generates heatmaps indicating exactly which parts of the character stroke the neural network focused on to make its prediction, crucial for archaeological transparency.
* **Interactive Dashboard:** A rich Streamlit UI that allows users to upload images, tweak preprocessing sliders in real-time, inspect individual characters in an interactive zoom panel, and export the transcribed text.
* **Multi-Format Export:** Download the recognized script and coordinates as JSON, Text, or PDF files.

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Optional, for fallback recognition)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ricthi/Ai-heritage-.git
   cd Ai-heritage-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `streamlit`, `torch`, `torchvision`, `opencv-python`, `numpy`, `pillow`, `pandas`, and `streamlit-drawable-canvas` installed).*

### Running the Application

Launch the Streamlit dashboard:

```bash
cd tamil_heritage_ai/Model-Creation
streamlit run main_app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

## 🧠 Training Your Own Model

If you have new labeled datasets of ancient characters, you can easily retrain the CNN.

1. **Organize Labels:** Place your class folders inside `tamil_heritage_ai/Labels/`.
2. **Prepare Dataset:** Run the dataset prep script to automatically split your images into `train` and `val` sets:
   ```bash
   python prepare_dataset.py
   ```
3. **Train the Model:** Run the training script to generate a new `model_stone.pth`:
   ```bash
   python train_stone_cnn.py
   ```
4. **Use It:** The Streamlit app will automatically detect `model_stone.pth`. Select it from the sidebar dropdown!

## 📁 Repository Structure

* `tamil_heritage_ai/Labels/` — Raw dataset containing folders of labeled ancient characters.
* `tamil_heritage_ai/Model-Creation/data/` — The split 70/30 training and validation datasets.
* `tamil_heritage_ai/Model-Creation/train_stone_cnn.py` — PyTorch training script for the CNN.
* `tamil_heritage_ai/Model-Creation/prepare_dataset.py` — Data aggregation and splitting script.
* `tamil_heritage_ai/Model-Creation/main_app.py` — The primary Streamlit frontend application.
* `tamil_heritage_ai/Model-Creation/model_stone.pth` — The compiled weights and label mappings for the neural network.

---
*Built to preserve and digitize the rich epigraphical history of the Tamil language.*
