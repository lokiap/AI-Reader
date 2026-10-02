"""Propositions de régions candidates sur le plan (avant classification par le CNN)."""

import cv2
import numpy as np

from .imaging import blue_mask
from .recognition import Detection

MAX_SIDE = 170  # le plus grand symbole du catalogue fait 145 px


def blue_proposals(bgr, min_side=8, max_side=MAX_SIDE, min_pixels=40, dilation=5):
    """Boîtes (x, y, w, h) des amas de pixels bleus du plan.

    Les traits bleus proches sont fusionnés par dilatation, puis on écarte ce qui est trop
    petit (bruit) ou trop grand (gaines de ventilation, cadres).
    """
    mask = (blue_mask(bgr) > 60).astype(np.uint8)
    merged = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged)
    boxes = []
    for i in range(1, count):
        x, y, w, h, _ = (int(v) for v in stats[i])
        if max(w, h) > max_side or max(w, h) < min_side:
            continue
        if int(mask[y:y + h, x:x + w].sum()) < min_pixels:
            continue
        boxes.append((x, y, w, h))
    return boxes
