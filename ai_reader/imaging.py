"""Chargement des images et binarisation."""

import cv2
import numpy as np

BLOCK_SIZE = 11  # voisinage (en pixels) utilisé par les seuils adaptatifs
ADAPTIVE_C = 2  # constante soustraite à la moyenne locale


def load_bgr(path):
    """Charge une image en BGR (le canal alpha éventuel est ignoré)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image illisible : {path}")
    return image


def load_gray(path):
    """Charge une image en niveaux de gris."""
    return cv2.cvtColor(load_bgr(path), cv2.COLOR_BGR2GRAY)


def fixed_threshold(gray, value=125):
    """Seuil global : les pixels sombres (< value) deviennent blancs."""
    return cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)[1]


def mean_threshold(gray):
    """Seuil adaptatif sur la moyenne locale."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, BLOCK_SIZE, ADAPTIVE_C
    )


def gaussian_threshold(gray):
    """Seuil adaptatif sur la moyenne locale pondérée par une gaussienne."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, BLOCK_SIZE, ADAPTIVE_C
    )


THRESHOLDS = {
    "fixed": fixed_threshold,
    "mean": mean_threshold,
    "gaussian": gaussian_threshold,
}


def blue_mask(bgr):
    """Intensité de « bleu pur » (0-255) : les symboles du plan sont tracés en bleu,
    alors que le fond, les cotes et le mobilier sont gris, noirs ou d'autres couleurs."""
    b, g, r = cv2.split(bgr.astype(np.int16))
    return np.clip((b - np.maximum(g, r)) * 2, 0, 255).astype(np.uint8)
