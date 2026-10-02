"""Reconnaissance de symboles par template matching.

Chaque symbole du catalogue est cherché dans l'image avec une corrélation normalisée
(TM_CCOEFF_NORMED), dans ses quatre rotations (0/90/180/270°). Les détections qui se
recouvrent sont ensuite fusionnées (non-maximum suppression) : le meilleur score gagne,
même s'il vient d'un autre symbole.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .imaging import blue_mask, fixed_threshold, load_bgr, load_gray


@dataclass
class Detection:
    label: str
    x: int
    y: int
    w: int
    h: int
    score: float
    angle: int = 0

    def to_dict(self):
        return {
            "label": self.label, "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "score": round(self.score, 4), "angle": self.angle,
        }


# Seuils par symbole, choisis en inspectant visuellement les candidats (voir README).
PLAN_THRESHOLDS = {"luminaire_07": 0.75, "luminaire_11": 0.80}
PLAN_DEFAULT_THRESHOLD = 0.85
PAGE_DEFAULT_THRESHOLD = 0.90


def _rotations(template):
    """Les rotations distinctes d'un gabarit (un symbole symétrique n'est testé qu'une fois)."""
    seen = []
    for k in range(4):
        rotated = np.ascontiguousarray(np.rot90(template, k))
        if not any(r.shape == rotated.shape and np.array_equal(r, rotated) for _, r in seen):
            seen.append((90 * k, rotated))
    return seen


def match_template(image, template, threshold, label, rotate=True, peak_window=15):
    """Candidats pour un gabarit : maxima locaux de la corrélation au-dessus du seuil."""
    variants = _rotations(template) if rotate else [(0, template)]
    kernel = np.ones((peak_window, peak_window), np.uint8)
    found = []
    for angle, tpl in variants:
        if tpl.shape[0] > image.shape[0] or tpl.shape[1] > image.shape[1]:
            continue
        scores = cv2.matchTemplate(image, tpl, cv2.TM_CCOEFF_NORMED)
        scores = np.nan_to_num(scores, nan=-1.0)  # gabarit ou zone uniforme : variance nulle
        peaks = (scores >= cv2.dilate(scores, kernel)) & (scores >= threshold)
        for y, x in zip(*np.nonzero(peaks)):
            found.append(Detection(label, int(x), int(y), tpl.shape[1], tpl.shape[0], float(scores[y, x]), angle))
    return found


def non_max_suppression(detections, iou_threshold=0.3):
    """Garde, parmi des détections qui se recouvrent, celle qui a le meilleur score."""
    kept = []
    for d in sorted(detections, key=lambda d: -d.score):
        if all(_iou(d, k) < iou_threshold for k in kept):
            kept.append(d)
    return kept


def _iou(a, b):
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def _catalogue(catalogue_dir):
    paths = sorted(Path(catalogue_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"Catalogue vide ou introuvable : {catalogue_dir}")
    return paths


def detect_plan(plan_path, catalogue_dir, thresholds=None, default_threshold=PLAN_DEFAULT_THRESHOLD):
    """Luminaires d'un plan. Le matching se fait sur le masque « bleu » du plan et des gabarits."""
    thresholds = PLAN_THRESHOLDS if thresholds is None else thresholds
    image = blue_mask(load_bgr(plan_path))
    detections = []
    for path in _catalogue(catalogue_dir):
        threshold = thresholds.get(path.stem, default_threshold)
        template = blue_mask(load_bgr(path))
        detections += match_template(image, template, threshold, path.stem)
    return non_max_suppression(detections)


def detect_page(page_path, catalogue_dir, threshold=PAGE_DEFAULT_THRESHOLD, size_tolerance=0.15):
    """Caractères d'une page. Les gabarits sont de simples recadrages de la page : pas de rotation.

    Un gabarit très fin (comme « I ») a une forte corrélation avec le fût de « P », « L » ou « T ».
    On ne garde donc une détection que si la composante connexe sous la boîte a, à peu près,
    la taille du gabarit (segmentation + reconnaissance).
    """
    gray = load_gray(page_path)
    _, labels = cv2.connectedComponents(fixed_threshold(gray))
    stats = _component_boxes(labels)
    inverted = 255 - gray
    detections = []
    for path in _catalogue(catalogue_dir):
        template = 255 - load_gray(path)
        for d in match_template(inverted, template, threshold, path.stem, rotate=False, peak_window=9):
            if _same_size(_dominant_component(labels, stats, d), d, size_tolerance):
                detections.append(d)
    return non_max_suppression(detections)


def _component_boxes(labels):
    """Boîte englobante (x, y, w, h) de chaque étiquette de composante."""
    boxes = {}
    for label, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is not None:
            boxes[label] = (sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start)
    return boxes


def _dominant_component(labels, boxes, d):
    """Boîte de la composante qui couvre le plus de pixels dans la détection (None si fond seul)."""
    region = labels[d.y:d.y + d.h, d.x:d.x + d.w]
    counts = np.bincount(region[region > 0])
    return boxes[int(counts.argmax())] if counts.size else None


def _same_size(box, d, tolerance):
    if box is None:
        return False
    _, _, w, h = box
    return abs(w - d.w) <= max(2, tolerance * d.w) and abs(h - d.h) <= max(2, tolerance * d.h)


def draw_detections(bgr, detections):
    """Encadre et nomme chaque détection."""
    canvas = bgr.copy()
    scale = max(1.0, max(canvas.shape[:2]) / 4000)
    for d in detections:
        cv2.rectangle(canvas, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 0, 255), max(2, int(2 * scale)))
        cv2.putText(canvas, d.label.replace("luminaire_", "L"), (d.x, max(d.y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 200), max(1, int(2 * scale)))
    return canvas
