import os
import sys

# Absolute path to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to Model-Creation folder
MODEL_CREATION_DIR = os.path.join(CURRENT_DIR, "tamil_heritage_ai", "Model-Creation")

# Add to sys.path so Python can find main_app.py
if MODEL_CREATION_DIR not in sys.path:
    sys.path.insert(0, MODEL_CREATION_DIR)

# This imports your full Streamlit UI from main_app.py
import main_app
