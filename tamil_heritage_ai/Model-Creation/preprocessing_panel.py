import cv2
import streamlit as st

# Environment Status: Verified (Packages installed in .venv)


def bgr_to_rgb(image):
    """Convert BGR numpy image to RGB (or passthrough grayscale)."""
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_png_bytes(image):
    """Encode numpy image as PNG bytes for download buttons."""
    if image is None:
        return b""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return b""
    return buf.tobytes()


def inject_preprocess_css():
    """Inject dark card styling for preprocessing grid."""
    st.markdown(
        """
        <style>
        .pp-card {
            background: #111315;
            border: 1px solid #191c1f;
            border-radius: 18px;
            padding: 10px 10px 14px 10px;
            margin-bottom: 18px;
        }
        .pp-label {
            color: #b8b0a3;
            font-size: 12px;
            letter-spacing: 1.3px;
            text-transform: uppercase;
            margin: 6px 4px 12px 4px;
            font-weight: 600;
        }
        .pp-download-spacer {
            height: 8px;
        }
        .stDownloadButton > button {
            width: 100%;
            border-radius: 10px;
            background: #131625;
            color: #ece7de;
            border: 1px solid #1a1f31;
            min-height: 42px;
        }
        .stDownloadButton > button:hover {
            border-color: #2c3355;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_preprocessing_panel(analysis_result):
    """
    Render 2x2 preprocessing grid with download buttons.

    Expects analysis_result structure:
    {
        "original_bgr": np.ndarray,
        "preprocess": {
            "gray": np.ndarray,
            "denoised": np.ndarray,
            "binary": np.ndarray,
        }
    }
    """
    if analysis_result is None:
        st.info("Upload an image to see preprocessing outputs.")
        return

    inject_preprocess_css()

    try:
        original = analysis_result["original_bgr"]
        preprocess = analysis_result.get("preprocess", {})
        gray = preprocess.get("gray")
        denoised = preprocess.get("denoised")
        binary = preprocess.get("binary")
    except Exception as exc:
        st.error(f"Preprocessing data missing: {exc}")
        return

    grid = [
        ("Input", bgr_to_rgb(original), image_to_png_bytes(original), "input.png"),
        ("Grayscale", gray, image_to_png_bytes(gray), "grayscale.png"),
        ("Denoised (NL-Means)", denoised, image_to_png_bytes(denoised), "denoised_nlmeans.png"),
        ("Adaptive Threshold", binary, image_to_png_bytes(binary), "adaptive_threshold.png"),
    ]

    row1 = st.columns(2)
    row2 = st.columns(2)

    for col, item in zip(row1 + row2, grid):
        label, img, file_bytes, file_name = item
        with col:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="pp-label">{label}</div>', unsafe_allow_html=True)
            if img is not None:
                st.image(img, use_container_width=True)
                st.markdown('<div class="pp-download-spacer"></div>', unsafe_allow_html=True)
                st.download_button(
                    label=f"Download {label.lower()} PNG",
                    data=file_bytes,
                    file_name=file_name,
                    mime="image/png",
                    use_container_width=True,
                    key=f"download_{file_name}",
                )
            else:
                st.info("Stage not available for this image.")
            st.markdown("</div>", unsafe_allow_html=True)
