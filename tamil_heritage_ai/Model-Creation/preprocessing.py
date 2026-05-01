from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    # NLM denoising parameters
    nlm_h: float = 15.0
    nlm_template_window_size: int = 7
    nlm_search_window_size: int = 21

    # Adaptive threshold parameters
    thresh_block_size: int = 31  # must be odd >= 3
    thresh_C: int = 10
    thresh_invert: bool = True  # True -> characters white, background black

    # Optional resize to normalize scale
    resize_long_edge: Optional[int] = 1024


class Preprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._validate_config()

    def _validate_config(self):
        cfg = self.config
        if cfg.thresh_block_size % 2 == 0 or cfg.thresh_block_size < 3:
            raise ValueError("thresh_block_size must be an odd integer >= 3")
        if cfg.nlm_template_window_size % 2 == 0:
            raise ValueError("nlm_template_window_size must be odd")
        if cfg.nlm_search_window_size % 2 == 0:
            raise ValueError("nlm_search_window_size must be odd")

    def load_bgr(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img

    def maybe_resize(self, img: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.resize_long_edge is None:
            return img

        h, w = img.shape[:2]
        long_edge = max(h, w)

        if long_edge <= cfg.resize_long_edge:
            return img

        scale = cfg.resize_long_edge / float(long_edge)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def to_gray(bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected BGR image with 3 channels")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def denoise_nlm(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.config
        if gray.ndim != 2:
            raise ValueError("Expected single-channel grayscale image")

        denoised = cv2.fastNlMeansDenoising(
            src=gray,
            h=cfg.nlm_h,
            templateWindowSize=cfg.nlm_template_window_size,
            searchWindowSize=cfg.nlm_search_window_size,
        )
        return denoised

    def adaptive_threshold(self, gray_or_denoised: np.ndarray) -> np.ndarray:
        cfg = self.config
        if gray_or_denoised.ndim != 2:
            raise ValueError("Expected single-channel grayscale image")

        thresh_type = cv2.THRESH_BINARY_INV if cfg.thresh_invert else cv2.THRESH_BINARY

        binary = cv2.adaptiveThreshold(
            gray_or_denoised,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=thresh_type,
            blockSize=cfg.thresh_block_size,
            C=cfg.thresh_C,
        )
        return binary

    def run(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full pipeline:
        1) optional resize
        2) grayscale
        3) NLM denoising
        4) adaptive threshold

        Returns: (bgr_resized, gray, denoised, binary)
        """
        bgr_resized = self.maybe_resize(img_bgr)
        gray = self.to_gray(bgr_resized)
        denoised = self.denoise_nlm(gray)
        binary = self.adaptive_threshold(denoised)
        return bgr_resized, gray, denoised, binary

    def run_on_path(self, path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bgr = self.load_bgr(path)
        return self.run(bgr)
