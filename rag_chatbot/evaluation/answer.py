from __future__ import annotations

import re
import unicodedata
from collections import Counter


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def char_f1(prediction: str, reference: str) -> float:
    pred = list(normalize(prediction))
    ref = list(normalize(reference))
    if not pred or not ref:
        return float(pred == ref)

    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def reference_substring_match(prediction: str, reference: str) -> bool:
    return normalize(reference) in normalize(prediction)
