import cv2
import numpy as np

from ai_reader import evaluation, imaging, recognition, segmentation
from ai_reader.recognition import Detection


def _canvas():
    image = np.zeros((200, 300), np.uint8)
    cv2.rectangle(image, (40, 50), (70, 90), 255, -1)  # gabarit à retrouver
    cv2.circle(image, (200, 120), 15, 255, -1)  # autre forme
    return image


def test_blue_mask_keeps_only_blue():
    bgr = np.zeros((1, 3, 3), np.uint8)
    bgr[0, 0] = (255, 0, 0)  # bleu pur (BGR)
    bgr[0, 1] = (128, 128, 128)  # gris
    bgr[0, 2] = (0, 0, 255)  # rouge
    assert imaging.blue_mask(bgr).tolist() == [[255, 0, 0]]


def test_find_components_filters_by_area():
    binary = _canvas()
    assert len(segmentation.find_components(binary, min_area=50)) == 2
    assert len(segmentation.find_components(binary, min_area=1000)) == 1  # seul le disque (~700) est écarté
    assert segmentation.find_components(binary, min_area=50, max_side=35)[0].w <= 35


def test_match_template_finds_shape_at_right_place():
    image = _canvas()
    template = image[45:95, 35:75].copy()
    found = recognition.match_template(image, template, 0.9, "carre", rotate=False)
    best = max(found, key=lambda d: d.score)
    assert (best.x, best.y) == (35, 45) and best.score > 0.99


def test_rotations_skip_symmetric_templates():
    square = np.full((10, 10), 255, np.uint8)
    assert len(recognition._rotations(square)) == 1
    bar = np.zeros((4, 12), np.uint8)
    bar[:, :6] = 255
    assert [a for a, _ in recognition._rotations(bar)] == [0, 90, 180, 270]


def test_non_max_suppression_keeps_best_overlapping():
    a = Detection("a", 0, 0, 10, 10, 0.9)
    b = Detection("b", 1, 1, 10, 10, 0.95)
    far = Detection("a", 50, 50, 10, 10, 0.8)
    assert recognition.non_max_suppression([a, b, far]) == [b, far]


def test_evaluate_precision_recall():
    truth = [Detection("a", 0, 0, 10, 10, 1), Detection("a", 50, 50, 10, 10, 1)]
    predicted = [Detection("a", 1, 1, 10, 10, 0.9), Detection("a", 100, 100, 10, 10, 0.8)]
    total = evaluation.evaluate(predicted, truth)["TOTAL"]
    assert (total["tp"], total["fp"], total["fn"]) == (1, 1, 1)
    assert total["precision"] == total["recall"] == 0.5


def test_detect_page_on_repository_data():
    detections = recognition.detect_page("data/caracteres/page.png", "data/caracteres/catalogue")
    labels = {d.label for d in detections}
    assert {"d", "o", "u"} <= labels
    assert sum(d.label == "2" for d in detections) == 3
