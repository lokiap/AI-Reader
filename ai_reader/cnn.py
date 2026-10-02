"""Classifieur CNN des symboles du plan.

Le CNN classe une région candidate (amas de pixels bleus, voir `proposals`) en 7 classes :
les 6 luminaires du catalogue ou « autre ». Il est entraîné **uniquement** sur :

- des versions augmentées des gabarits du catalogue (rotations, échelle, flou, bruit, traits
  parasites, occlusions partielles) pour les 6 classes de luminaires ;
- des régions candidates réelles de la moitié gauche du plan (hors luminaires connus) et des
  négatifs synthétiques (fragments de gabarits, traits, cercles) pour la classe « autre ».

Aucun luminaire du plan n'est vu à l'entraînement, et l'évaluation se fait sur la moitié droite.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .imaging import blue_mask, load_bgr
from .proposals import blue_proposals
from .recognition import Detection, _catalogue

SIZE = 64  # côté de la vignette en entrée du réseau
OTHER = "autre"
SPLIT_X = 6800  # x < SPLIT_X : zone d'entraînement des négatifs ; x >= SPLIT_X : zone de test


class SymbolNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def to_patch(mask):
    """Vignette SIZE×SIZE : le masque est mis à l'échelle (proportions conservées) et centré sur fond noir."""
    h, w = mask.shape
    scale = (SIZE - 4) / max(h, w)
    resized = cv2.resize(mask, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    y, x = (SIZE - resized.shape[0]) // 2, (SIZE - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def crop_box(mask, box, margin=3):
    x, y, w, h = box
    return mask[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]


# --------------------------------------------------------------------------- données synthétiques

def _augment_template(template, rng):
    """Une variation réaliste d'un gabarit tel qu'il peut apparaître sur le plan."""
    img = np.rot90(template, rng.integers(4)).astype(np.float32)
    h, w = img.shape
    sx, sy = rng.uniform(0.9, 1.1, 2)
    img = cv2.resize(img, (max(4, round(w * sx)), max(4, round(h * sy))), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.9))
    pad = int(rng.integers(0, 7))
    canvas = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad), np.float32)
    canvas[pad:pad + img.shape[0], pad:pad + img.shape[1]] = img * rng.uniform(0.7, 1.0)
    H, W = canvas.shape
    if rng.random() < 0.4:  # traits bleus parasites qui touchent le symbole (étiquettes, cotes)
        for _ in range(int(rng.integers(1, 4))):
            p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            p2 = (p1[0] + int(rng.integers(-W // 2, W // 2 + 1)), p1[1] + int(rng.integers(-H // 2, H // 2 + 1)))
            cv2.line(canvas, p1, p2, float(rng.uniform(120, 255)), 1)
    if rng.random() < 0.3:  # petite occlusion (texte gris par-dessus)
        ow, oh = max(2, int(W * rng.uniform(0.1, 0.3))), max(2, int(H * rng.uniform(0.1, 0.3)))
        ox, oy = int(rng.integers(0, W - ow + 1)), int(rng.integers(0, H - oh + 1))
        canvas[oy:oy + oh, ox:ox + ow] = 0
    canvas += rng.normal(0, rng.uniform(0, 8), canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _synthetic_negative(templates, rng):
    kind = rng.integers(3)
    if kind == 0:  # fragment d'un gabarit : on en garde moins de 65 %
        t = np.rot90(templates[int(rng.integers(len(templates)))], rng.integers(4))
        h, w = t.shape
        fh, fw = max(3, int(h * rng.uniform(0.25, 0.65))), max(3, int(w * rng.uniform(0.25, 1.0)))
        if fh * fw > 0.65 * h * w:
            fw = max(3, int(0.6 * w))
        y, x = int(rng.integers(0, h - fh + 1)), int(rng.integers(0, w - fw + 1))
        return t[y:y + fh, x:x + fw].copy()
    canvas = np.zeros((int(rng.integers(12, 90)), int(rng.integers(12, 90))), np.uint8)
    H, W = canvas.shape
    for _ in range(int(rng.integers(1, 5))):
        if kind == 1:
            cv2.line(canvas, (int(rng.integers(W)), int(rng.integers(H))), (int(rng.integers(W)), int(rng.integers(H))), 255, int(rng.integers(1, 3)))
        else:
            c = (int(rng.integers(W)), int(rng.integers(H)))
            if rng.random() < 0.5:
                cv2.circle(canvas, c, int(rng.integers(3, max(4, min(W, H) // 2 + 1))), 255, int(rng.integers(1, 3)))
            else:
                cv2.rectangle(canvas, c, (c[0] + int(rng.integers(3, W)), c[1] + int(rng.integers(3, H))), 255, int(rng.integers(1, 3)))
    return canvas


def _real_negatives(plan_mask, boxes, known, split_x=SPLIT_X):
    """Vignettes des candidats de la zone d'entraînement qui ne sont pas des luminaires connus."""
    patches = []
    for (x, y, w, h) in boxes:
        if x + w >= split_x:
            continue
        if any(_overlap(((x, y, w, h)), k) for k in known):
            continue
        patches.append(crop_box(plan_mask, (x, y, w, h)))
    return patches


def _overlap(a, b, threshold=0.1):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) > threshold


def make_dataset(catalogue_dir, plan_path, known_detections, n_per_class, seed=0):
    """Tableaux (X, y, noms de classes) d'entraînement. `known_detections` : luminaires à exclure des négatifs."""
    rng = np.random.default_rng(seed)
    paths = _catalogue(catalogue_dir)
    classes = [p.stem for p in paths] + [OTHER]
    templates = [blue_mask(load_bgr(p)) for p in paths]

    bgr = load_bgr(plan_path)
    plan_mask = blue_mask(bgr)
    known = [(d.x, d.y, d.w, d.h) for d in known_detections]
    real_neg = _real_negatives(plan_mask, blue_proposals(bgr), known)

    X, y = [], []
    for label, template in enumerate(templates):
        for _ in range(n_per_class):
            X.append(to_patch(_augment_template(template, rng)))
            y.append(label)
    for _ in range(n_per_class):  # classe « autre » : 1/2 négatifs réels, 1/2 synthétiques
        if real_neg and rng.random() < 0.5:
            patch = np.rot90(real_neg[int(rng.integers(len(real_neg)))], rng.integers(4))
            if rng.random() < 0.5:
                patch = patch[:, ::-1]
        else:
            patch = _synthetic_negative(templates, rng)
        X.append(to_patch(np.ascontiguousarray(patch)))
        y.append(len(templates))
    X = np.stack(X).astype(np.float32)[:, None] / 255.0
    return X, np.array(y, np.int64), classes, len(real_neg)


# --------------------------------------------------------------------------- entraînement / inférence

def train(catalogue_dir, plan_path, known_detections, out_path, epochs=25, n_per_class=800, seed=0, verbose=True):
    torch.manual_seed(seed)
    X, y, classes, n_real = make_dataset(catalogue_dir, plan_path, known_detections, n_per_class, seed)
    if verbose:
        print(f"{len(X)} vignettes ({n_real} négatifs réels de la zone d'entraînement), classes : {classes}")
    perm = np.random.default_rng(seed).permutation(len(X))
    n_val = len(X) // 10
    val, tr = perm[:n_val], perm[n_val:]
    model = SymbolNet(len(classes))
    optimiser = torch.optim.Adam(model.parameters(), 2e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for epoch in range(epochs):
        model.train()
        order = tr[np.random.default_rng(seed + epoch).permutation(len(tr))]
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            optimiser.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
        if verbose and (epoch % 5 == 4 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                acc = (model(Xt[val]).argmax(1) == yt[val]).float().mean().item()
            print(f"époque {epoch + 1}/{epochs} : précision de validation synthétique {acc:.3f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "classes": classes}, out_path)
    return out_path


def load_model(path):
    data = torch.load(path, map_location="cpu")
    model = SymbolNet(len(data["classes"]))
    model.load_state_dict(data["state"])
    return model.eval(), data["classes"]


def detect_plan_cnn(plan_path, model_path, threshold=0.5):
    """Luminaires du plan : le CNN classe chaque candidat bleu ; « autre » et les scores < seuil sont écartés."""
    model, classes = load_model(model_path)
    bgr = load_bgr(plan_path)
    mask = blue_mask(bgr)
    boxes = blue_proposals(bgr)
    if not boxes:
        return []
    batch = np.stack([to_patch(crop_box(mask, b)) for b in boxes]).astype(np.float32)[:, None] / 255.0
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(batch)), 1).numpy()
    detections = []
    for (x, y, w, h), p in zip(boxes, probs):
        k = int(p.argmax())
        if classes[k] != OTHER and p[k] >= threshold:
            # on rogne la marge ajoutée par la dilatation des candidats
            detections.append(Detection(classes[k], x + 2, y + 2, max(1, w - 4), max(1, h - 4), float(p[k])))
    return detections
