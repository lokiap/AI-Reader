# AI-Reader

[🇫🇷 Français](README.fr.md) · 🇬🇧 English

Symbol detection and recognition in images from a small catalogue of templates, built with OpenCV.
School project ("Projet IA", 2024-2025), picked up again and completed.

Two use cases:

| Source | Image | Catalogue | What we look for |
|---|---|---|---|
| `plan` | building floor plan (13,624 × 5,061 px) | 6 light fixtures | the light fixtures, whatever their orientation |
| `page` | page of text (2,480 × 3,507 px) | 6 characters | the characters `2 I d n o u` |

![Light fixtures detected on the plan](docs/plan_detections.jpg)

## Method

1. **Segmentation** (`ai_reader/segmentation.py`): binarisation (fixed, mean or gaussian threshold) followed by
   connected components. On the page it boxes every letter, but cannot say which one it is:

   ![Page segmentation](docs/page_segmentation.jpg)

2. **Recognition** (`ai_reader/recognition.py`): template matching (normalised correlation) of every symbol in the
   catalogue, then merging of overlapping detections (the best score wins).
   - *Plan*: correlation is computed on a "pure blue" mask (`imaging.blue_mask`), because the symbols are drawn
     in blue while walls, dimensions and furniture are grey or black. Templates are tried at 0°, 90°, 180° and 270°.
   - *Page*: a very thin template such as `I` correlates strongly with the stem of `P`, `L` or `T`. A detection is
     therefore kept only if the connected component under it has about the size of the template (within 15%).

   ![Characters recognised on the page](docs/page_detections.jpg)

## Results

| Source | Detections |
|---|---|
| plan | 181: 173 × `luminaire_01` (light bars), 6 × `luminaire_11`, 1 × `luminaire_03`, 1 × `luminaire_07` |
| page | 132: 52 × `o`, 38 × `u`, 27 × `d`, 7 × `I`, 5 × `n`, 3 × `2` |

Thresholds (0.85 by default on the plan, 0.75 for `luminaire_07`, 0.80 for `luminaire_11`, 0.90 on the page) were
chosen by visually inspecting every candidate scoring ≥ 0.5.

**Honest limitations**
- There is no ground truth for these images: the counts above are **not** recall measurements.
  I visually reviewed the 60 lowest-scoring plan detections and all of them are correct. Recall is not measured.
- `luminaire_10` and `luminaire_15` are not found: there is no credible candidate, so they are probably absent from this plan.
- On the page, only the 6 catalogue characters are searched; every other letter is ignored by design. The `n` template
  is bold and only matches the `n` of headings, and letters that touch each other do not form a component of the right
  size, so they are missed.
- Template matching is sensitive to font and scale. A learned classifier (CNN) would be the next step.

To measure precision and recall, annotate a ground truth in the same format as `resultats/*_detections.json`
(a list of `{label, x, y, w, h, score, angle}`) and run `python -m ai_reader evaluate predictions.json truth.json`.

## Installation

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```bash
python -m ai_reader detect page                 # ~3 s
python -m ai_reader detect plan                 # ~1 min: 6 templates × 4 rotations on 69 Mpx
python -m ai_reader segment page --threshold gaussian --min-area 50
python -m ai_reader detect plan --threshold 0.9 --output out/
python -m ai_reader evaluate predictions.json truth.json
```

Outputs (annotated image and JSON of the detections) go to `resultats/` (ignored by git).

To search more characters, drop crops of them in `data/caracteres/catalogue/` (e.g. `a.png`); no code change needed.

## Tests

```bash
python -m pytest
```

## Layout

```
ai_reader/
  imaging.py       loading, thresholds, blue mask
  segmentation.py  connected components
  recognition.py   template matching, NMS, plan / page detection
  evaluation.py    precision / recall
  cli.py           command line
data/              plan, page and template catalogues
docs/              README images
tests/
```
