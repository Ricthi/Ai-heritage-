from pathlib import Path

import cv2

from preprocessing import Preprocessor, PreprocessConfig


def main():
    input_path = Path("samples/palm_leaf_01.jpg")
    out_dir = Path("outputs/preprocessing_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessConfig(
        nlm_h=18.0,
        nlm_template_window_size=7,
        nlm_search_window_size=21,
        thresh_block_size=31,
        thresh_C=10,
        thresh_invert=True,
        resize_long_edge=1024 
    )
    pre = Preprocessor(cfg)

    if not input_path.exists():
        print(f"ERROR: Input image not found: {input_path}")
        print("Please place a sample image at that location or update the script.")
        return

    bgr, gray, denoised, binary = pre.run_on_path(input_path)

    cv2.imwrite(str(out_dir / "01_original_bgr.png"), bgr)
    cv2.imwrite(str(out_dir / "02_gray.png"), gray)
    cv2.imwrite(str(out_dir / "03_denoised_nlm.png"), denoised)
    cv2.imwrite(str(out_dir / "04_adaptive_thresh.png"), binary)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
