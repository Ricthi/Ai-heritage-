from typing import List, Tuple
from tamil_charset import TAMIL_CHARS

def decode_predictions(
    class_ids: List[int],
    confidences: List[float],
) -> Tuple[List[str], List[float]]:
    """Map model class indices to Tamil characters + confidences."""

    labels: List[str] = []
    scores: List[float] = []

    num_classes = len(TAMIL_CHARS)

    for cid, conf in zip(class_ids, confidences):
        if 0 <= cid < num_classes:
            labels.append(TAMIL_CHARS[cid])
        else:
            # Out of range index, mark as unknown
            labels.append("[UNK]")
        scores.append(float(conf))

    return labels, scores
