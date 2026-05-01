"""
Bridge between the legacy 26-class Tamil charset and the new 247-class charset.

Use cases
---------
1. Re-index legacy training labels into the 247-class space (for fine-tuning).
2. Translate legacy model predictions (class idx 0..25) into the new 247-class
   index space, so downstream code can speak one language.
3. Build a weight-transfer mask when initializing a new 247-class classifier
   head from old 26-class weights.

Legacy charset (26 classes) was, in order:
  ['ai', 'c', 'e', 'i', 'k', 'l', 'l5', 'l5u', 'l5u4',
   'n', 'n1', 'n1u4', 'n2', 'n2u4', 'n3', 'n5',
   'o', 'p', 'pu4', 'r', 'r5', 'r5i', 'ru',
   't', 'ti', 'y']
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

from tamil_charset import (
    TAMIL_CHARS,
    CHAR_TO_IDX,
    NUM_CLASSES,           # 247
)

# ---------------------------------------------------------------------------
# 1. Legacy charset definition (frozen — do NOT reorder)
# ---------------------------------------------------------------------------
LEGACY_LABELS: List[str] = [
    "ai", "c", "e", "i", "k", "l", "l5", "l5u", "l5u4",
    "n", "n1", "n1u4", "n2", "n2u4", "n3", "n5",
    "o", "p", "pu4", "r", "r5", "r5i", "ru",
    "t", "ti", "y",
]

# Legacy label -> Tamil character (from your original TAMIL_MAP)
LEGACY_LABEL_TO_CHAR: Dict[str, str] = {
    "ai":   "ஐ",
    "c":    "ச்",
    "e":    "எ",
    "i":    "இ",
    "k":    "க்",
    "l":    "ல்",
    "l5":   "ள்",
    "l5u":  "லு",
    "l5u4": "லூ",
    "n":    "ந்",
    "n1":   "ன்",
    "n1u4": "னூ",
    "n2":   "ண்",
    "n2u4": "ணூ",
    "n3":   "ங்",
    "n5":   "ம்",
    "o":    "ஒ",
    "p":    "ப்",
    "pu4":  "பூ",
    "r":    "ர்",
    "r5":   "ற்",
    "r5i":  "றி",
    "ru":   "ரு",
    "t":    "த்",
    "ti":   "தி",
    "y":    "ய்",
}

NUM_LEGACY_CLASSES: int = len(LEGACY_LABELS)   # 26
assert NUM_LEGACY_CLASSES == 26
assert len(LEGACY_LABEL_TO_CHAR) == 26


# ---------------------------------------------------------------------------
# 2. Build the index mapping  legacy_idx -> full_idx
# ---------------------------------------------------------------------------
def _build_legacy_to_full_index_map() -> Dict[int, int]:
    """For each legacy class index (0..25), find its index in the 247-class list."""
    mapping: Dict[int, int] = {}
    missing: List[str] = []

    for legacy_idx, label in enumerate(LEGACY_LABELS):
        ch = LEGACY_LABEL_TO_CHAR[label]
        if ch not in CHAR_TO_IDX:
            missing.append(f"{label!r} ({ch})")
            continue
        mapping[legacy_idx] = CHAR_TO_IDX[ch]

    if missing:
        raise RuntimeError(
            "These legacy characters were not found in the 247-class charset:\n  "
            + "\n  ".join(missing)
        )
    return mapping


LEGACY_TO_FULL_IDX: Dict[int, int] = _build_legacy_to_full_index_map()
FULL_TO_LEGACY_IDX: Dict[int, int] = {v: k for k, v in LEGACY_TO_FULL_IDX.items()}

assert len(LEGACY_TO_FULL_IDX) == 26


# ---------------------------------------------------------------------------
# 3. Vectorized lookup tables (for fast batched conversion)
# ---------------------------------------------------------------------------
# Shape (26,) — element i is the full-charset index for legacy class i
LEGACY_TO_FULL_LUT: np.ndarray = np.array(
    [LEGACY_TO_FULL_IDX[i] for i in range(NUM_LEGACY_CLASSES)],
    dtype=np.int64,
)

# Shape (247,) — element i is the legacy index, or -1 if not in legacy set
FULL_TO_LEGACY_LUT: np.ndarray = np.full(NUM_CLASSES, fill_value=-1, dtype=np.int64)
for full_idx, legacy_idx in FULL_TO_LEGACY_IDX.items():
    FULL_TO_LEGACY_LUT[full_idx] = legacy_idx


# ---------------------------------------------------------------------------
# 4. Public conversion API
# ---------------------------------------------------------------------------
def legacy_idx_to_full_idx(legacy_idx: int) -> int:
    if not 0 <= legacy_idx < NUM_LEGACY_CLASSES:
        raise IndexError(f"legacy_idx {legacy_idx} out of range [0, 26)")
    return LEGACY_TO_FULL_IDX[legacy_idx]


def full_idx_to_legacy_idx(full_idx: int) -> int:
    """Returns -1 if the full-charset class is not part of the legacy set."""
    if not 0 <= full_idx < NUM_CLASSES:
        raise IndexError(f"full_idx {full_idx} out of range [0, {NUM_CLASSES})")
    return int(FULL_TO_LEGACY_LUT[full_idx])


def remap_legacy_labels_to_full(labels: np.ndarray) -> np.ndarray:
    """
    Vectorized: convert an array of legacy class indices (any shape, dtype int)
    into the equivalent indices in the 247-class space.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels.copy()
    if labels.min() < 0 or labels.max() >= NUM_LEGACY_CLASSES:
        raise ValueError("Found legacy labels outside [0, 26)")
    return LEGACY_TO_FULL_LUT[labels]


def remap_full_labels_to_legacy(labels: np.ndarray) -> np.ndarray:
    """
    Vectorized inverse. Indices not present in the legacy set become -1.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels.copy()
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(f"Found full labels outside [0, {NUM_CLASSES})")
    return FULL_TO_LEGACY_LUT[labels]


# ---------------------------------------------------------------------------
# 5. Logits / probability remapping
# ---------------------------------------------------------------------------
def expand_legacy_logits_to_full(
    legacy_logits: np.ndarray,
    fill_value: float = -1e9,
) -> np.ndarray:
    """
    Convert logits with shape (..., 26) into shape (..., 247).
    Classes not covered by the legacy model are filled with `fill_value`
    (default = very negative, so softmax assigns ~0 probability).

    Useful when you want the legacy model to behave as a 247-class classifier
    that simply abstains on unseen classes.
    """
    legacy_logits = np.asarray(legacy_logits)
    if legacy_logits.shape[-1] != NUM_LEGACY_CLASSES:
        raise ValueError(
            f"Last dim must be {NUM_LEGACY_CLASSES}, got {legacy_logits.shape[-1]}"
        )

    out_shape = legacy_logits.shape[:-1] + (NUM_CLASSES,)
    full_logits = np.full(out_shape, fill_value=fill_value, dtype=legacy_logits.dtype)
    full_logits[..., LEGACY_TO_FULL_LUT] = legacy_logits
    return full_logits


def reduce_full_logits_to_legacy(full_logits: np.ndarray) -> np.ndarray:
    """
    Inverse: take a 247-class logit/probability tensor and slice out only
    the 26 legacy classes (preserving order). Shape (..., 247) -> (..., 26).
    """
    full_logits = np.asarray(full_logits)
    if full_logits.shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Last dim must be {NUM_CLASSES}, got {full_logits.shape[-1]}"
        )
    return full_logits[..., LEGACY_TO_FULL_LUT]


# ---------------------------------------------------------------------------
# 6. Weight-transfer helper (for fine-tuning a new 247-class head)
# ---------------------------------------------------------------------------
def transfer_classifier_head(
    legacy_weight: np.ndarray,           # shape (26, D)
    legacy_bias:   np.ndarray | None,    # shape (26,)  or None
    new_weight_init: np.ndarray,         # shape (247, D), pre-initialized (e.g. random)
    new_bias_init:   np.ndarray | None,  # shape (247,) or None
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Copy the 26 legacy classifier rows into the correct positions of the new
    247-class head. Other rows keep their pre-initialized values.

    Returns (new_weight, new_bias).
    """
    if legacy_weight.shape[0] != NUM_LEGACY_CLASSES:
        raise ValueError("legacy_weight first dim must be 26")
    if new_weight_init.shape[0] != NUM_CLASSES:
        raise ValueError(f"new_weight_init first dim must be {NUM_CLASSES}")
    if legacy_weight.shape[1] != new_weight_init.shape[1]:
        raise ValueError("Feature dimensions D must match")

    new_w = new_weight_init.copy()
    new_w[LEGACY_TO_FULL_LUT, :] = legacy_weight

    new_b = None
    if legacy_bias is not None:
        if new_bias_init is None:
            raise ValueError("new_bias_init required when legacy_bias is given")
        if legacy_bias.shape[0] != NUM_LEGACY_CLASSES:
            raise ValueError("legacy_bias must have shape (26,)")
        if new_bias_init.shape[0] != NUM_CLASSES:
            raise ValueError(f"new_bias_init must have shape ({NUM_CLASSES},)")
        new_b = new_bias_init.copy()
        new_b[LEGACY_TO_FULL_LUT] = legacy_bias

    return new_w, new_b


# ---------------------------------------------------------------------------
# 7. Self-test / inspection
# ---------------------------------------------------------------------------
def _print_mapping_table() -> None:
    print(f"{'legacy_idx':>10} {'label':>6} {'char':>4} {'full_idx':>10}")
    print("-" * 36)
    for i, label in enumerate(LEGACY_LABELS):
        ch = LEGACY_LABEL_TO_CHAR[label]
        print(f"{i:>10} {label:>6} {ch:>4} {LEGACY_TO_FULL_IDX[i]:>10}")


if __name__ == "__main__":
    print("Legacy classes :", NUM_LEGACY_CLASSES)
    print("Full classes   :", NUM_CLASSES)
    print()
    _print_mapping_table()

    # Quick sanity demo
    print("\nDemo: remap legacy labels [0, 4, 25] ->",
          remap_legacy_labels_to_full(np.array([0, 4, 25])).tolist())

    # Demo logit expansion
    rng = np.random.default_rng(0)
    fake_legacy_logits = rng.standard_normal((3, 26)).astype(np.float32)
    expanded = expand_legacy_logits_to_full(fake_legacy_logits)
    print("Expanded logits shape:", expanded.shape)   # (3, 247)
    recovered = reduce_full_logits_to_legacy(expanded)
    assert np.allclose(recovered, fake_legacy_logits)
    print("Round-trip OK ✅")
