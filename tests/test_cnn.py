import numpy as np
import pytest

pytest.importorskip("torch")

from ai_reader import cnn, evaluation  # noqa: E402


def test_to_patch_keeps_aspect_and_centres():
    bar = np.full((14, 145), 255, np.uint8)
    patch = cnn.to_patch(bar)
    assert patch.shape == (cnn.SIZE, cnn.SIZE)
    rows = np.nonzero(patch.any(axis=1))[0]
    assert rows.max() - rows.min() < 10  # la barre reste fine
    assert abs((rows.min() + rows.max()) / 2 - cnn.SIZE / 2) < 3  # et centrée


def test_augmentation_returns_uint8_image():
    rng = np.random.default_rng(0)
    template = np.zeros((30, 40), np.uint8)
    template[5:25, 5:35] = 255
    for _ in range(20):
        out = cnn._augment_template(template, rng)
        assert out.dtype == np.uint8 and out.ndim == 2 and min(out.shape) >= 4


def test_synthetic_negatives_are_not_empty():
    rng = np.random.default_rng(1)
    templates = [np.full((20, 20), 255, np.uint8)]
    assert all(cnn._synthetic_negative(templates, rng).size > 0 for _ in range(50))


def test_cnn_on_repository_plan_matches_reviewed_truth():
    """Garde-fou : le modèle versionné retrouve les luminaires rares et ne dépasse pas 10 faux positifs."""
    detections = cnn.detect_plan_cnn("data/plans/plan.png", "models/plan_cnn.pt")
    truth = evaluation.load_detections("data/plans/verite_terrain.json")
    total = evaluation.evaluate(detections, truth)
    assert total["luminaire_11"]["tp"] == 6 and total["luminaire_03"]["tp"] == 1
    assert total["TOTAL"]["recall"] > 0.85 and total["TOTAL"]["fp"] <= 10
