"""Détection de composantes connexes (étape de segmentation)."""

from dataclasses import dataclass

import cv2


@dataclass
class Component:
    x: int
    y: int
    w: int
    h: int
    area: int


def find_components(binary, min_area=50, max_area=None, min_side=1, max_side=None):
    """Composantes connexes d'une image binaire, filtrées par aire et par taille.

    Le fond (étiquette 0) est ignoré. Les composantes trop petites (bruit) ou trop
    grandes (murs, cadres) sont écartées.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    components = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or (max_area is not None and area > max_area):
            continue
        if min(w, h) < min_side or (max_side is not None and max(w, h) > max_side):
            continue
        components.append(Component(x, y, w, h, area))
    return components


def draw_components(gray, components, color=(0, 0, 255)):
    """Image couleur sur laquelle chaque composante est encadrée et numérotée."""
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, c in enumerate(components):
        cv2.rectangle(canvas, (c.x, c.y), (c.x + c.w, c.y + c.h), color, 2)
        cv2.putText(canvas, str(index), (c.x, max(c.y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    return canvas
