from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class OCRChar:
    char: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)

@dataclass
class OCRWord:
    text: str
    avg_confidence: float
    chars: List[OCRChar]
    bbox: Tuple[float, float, float, float]
    is_suspicious: bool

def _iou_y(a: OCRChar, b: OCRChar) -> float:
    ay1, ay2 = a.bbox[1], a.bbox[3]
    by1, by2 = b.bbox[1], b.bbox[3]
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    union = max(ay2, by2) - min(ay1, by1)
    if union == 0:
        return 0.0
    return inter / union

def _build_word(chars: List[OCRChar], suspicious_word_threshold: float) -> OCRWord:
    if not chars:
        return None
    text = "".join([c.char for c in chars])
    avg_conf = sum(c.confidence for c in chars) / max(len(chars), 1)

    x_min = min(c.bbox[0] for c in chars)
    y_min = min(c.bbox[1] for c in chars)
    x_max = max(c.bbox[2] for c in chars)
    y_max = max(c.bbox[3] for c in chars)

    is_suspicious = avg_conf < suspicious_word_threshold

    return OCRWord(
        text=text,
        avg_confidence=avg_conf,
        chars=chars,
        bbox=(x_min, y_min, x_max, y_max),
        is_suspicious=is_suspicious,
    )

def group_chars_into_words(
    raw_chars: List[Dict[str, Any]],
    char_conf_threshold: float = 0.35,
    max_gap_factor: float = 1.5,
    line_overlap_threshold: float = 0.5,
    suspicious_word_threshold: float = 0.6,
) -> List[OCRWord]:
    """
    Convert noisy per-character predictions into filtered word-level objects.
    `raw_chars` should be a list of dicts containing: char, confidence, bbox.
    """

    # 1) Convert and filter by confidence
    chars: List[OCRChar] = []
    for c in raw_chars:
        conf = float(c.get("confidence", 0.0))
        if conf < char_conf_threshold:
            continue
        if "bbox" not in c or "char" not in c:
            continue
        ch = str(c["char"])
        if not ch:
            continue
        bbox = tuple(c["bbox"])  # [x_min, y_min, x_max, y_max]
        chars.append(OCRChar(char=ch, confidence=conf, bbox=bbox))

    if not chars:
        return []

    # 2) Sort chars top-to-bottom, then left-to-right
    chars.sort(key=lambda c: (c.bbox[1], c.bbox[0]))

    # 3) Group into lines using vertical overlap
    lines: List[List[OCRChar]] = []
    for ch in chars:
        placed = False
        for line in lines:
            # Check overlap with the first char of the line
            if _iou_y(ch, line[0]) >= line_overlap_threshold:
                line.append(ch)
                placed = True
                break
        if not placed:
            lines.append([ch])

    # 4) Within each line, sort by x and group into words using horizontal gap
    words: List[OCRWord] = []
    for line in lines:
        line.sort(key=lambda c: c.bbox[0])
        if not line:
            continue

        widths = [c.bbox[2] - c.bbox[0] for c in line]
        avg_width = sum(widths) / len(widths)
        max_gap = max_gap_factor * avg_width

        current_chars: List[OCRChar] = [line[0]]

        for i in range(1, len(line)):
            prev = line[i-1]
            cur = line[i]
            gap = cur.bbox[0] - prev.bbox[2]
            if gap > max_gap:
                word = _build_word(current_chars, suspicious_word_threshold)
                if word: words.append(word)
                current_chars = [cur]
            else:
                current_chars.append(cur)

        if current_chars:
            word = _build_word(current_chars, suspicious_word_threshold)
            if word: words.append(word)

    return words
