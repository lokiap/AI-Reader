# AI-Reader

[🇫🇷 Français](README.fr.md) · 🇬🇧 English

Finding and recognising symbols in large technical images: light fixtures on a building floor plan and characters on a
page of text. Two approaches are built and compared on the same data: a classical computer-vision baseline
(template matching) and a small CNN trained only from the symbol catalogue. Built with OpenCV and PyTorch.

![Light fixtures detected on the plan](docs/plan_detections.jpg)

| Source | Image | Catalogue | What we look for |
|---|---|---|---|
| `plan` | building floor plan (13,624 × 5,061 px) | 6 light fixtures | the light fixtures, whatever their orientation |
| `page` | page of text (2,480 × 3,507 px) | 6 characters | the characters `2 I d n o u` |

## Results on the plan

Evaluated against a hand-reviewed ground truth of 185 light fixtures (`data/plans/verite_terrain.json`).

| Method | Precision | Recall | F1 | Time |
|---|---|---|---|---|
| Template matching | 1.00 | 0.98 | 0.99 | ~70 s |
| CNN | 0.97 | 0.92 | 0.95 | ~11 s |

On the right half of the plan only (the CNN never saw this area): template matching 1.00 / 0.98, CNN 0.97 / 0.90.

**How to read these numbers**
- The ground truth was built by pooling the detections of both methods and reviewing each disagreement by eye. It is
  therefore favourable to template matching by construction (all its detections are in it), and anything *both*
  methods miss is not counted, so recall is an upper bound.
- The CNN is the interesting result: it is trained **without a single light fixture from the plan** (only augmented
  catalogue templates + non-fixture regions of the left half of the plan), yet it reaches 97% precision, finds 4 bars
  that template matching missed (partly hidden by text), and runs about 6× faster.
- Its errors are known: 14 bars are missed because the candidate region also swallows neighbouring blue lines, and
  5 false positives (3 "M" symbols with arcs labelled `luminaire_10`, a smiley, a solid dark bar).
- `luminaire_10` and `luminaire_15` are not present on this plan, so these classes are not really tested.

## Method

1. **Segmentation** (`ai_reader/segmentation.py`): binarisation (fixed, mean or gaussian threshold) followed by
   connected components. On the page it boxes every letter, but cannot say which one it is:

   ![Page segmentation](docs/page_segmentation.jpg)

2. **Template matching** (`ai_reader/recognition.py`): normalised correlation of every symbol in the catalogue, then
   merging of overlapping detections (the best score wins).
   - *Plan*: the correlation is computed on a "pure blue" mask (`imaging.blue_mask`), because the symbols are drawn
     in blue while walls, dimensions and furniture are grey or black. Templates are tried at 0°, 90°, 180° and 270°.
   - *Page*: a very thin template such as `I` correlates strongly with the stem of `P`, `L` or `T`. A detection is
     therefore kept only if the connected component under it has about the size of the template (within 15%).

   ![Characters recognised on the page](docs/page_detections.jpg)

3. **CNN** (`ai_reader/cnn.py`, plan only): candidate regions are blue clusters of the plan (`ai_reader/proposals.py`).
   A 4-block convolutional network (~130 k parameters) classifies each 64×64 patch as one of the 6 fixtures or "other".
   - *Positives*: only the 6 catalogue templates, augmented (90° rotations, scale, blur, noise, stray lines touching
     the symbol, small occlusions).
   - *Negatives*: real candidate regions from the left half of the plan (known fixtures excluded) and synthetic ones
     (template fragments, lines, circles).
   - *Evaluation*: the right half of the plan is held out for the negatives; no fixture of the plan is used for training.

Page thresholds: 0.90 on the page, and on the plan 0.85 by default, 0.75 for `luminaire_07`, 0.80 for `luminaire_11`,
chosen by visually inspecting every candidate scoring ≥ 0.5. These thresholds were tuned on this same plan, which is
another reason to read the template-matching numbers as optimistic. The CNN uses the default argmax threshold of 0.5.

**Limitations on the page**: only the 6 catalogue characters are searched; every other letter is ignored by design.
The `n` template is bold and only matches the `n` of headings, and letters that touch each other do not form a
component of the right size. There is no ground truth for the page, and no CNN on it.

## Installation

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```bash
python -m ai_reader detect page                          # template matching on the page, ~3 s
python -m ai_reader detect plan                          # template matching on the plan, ~1 min
python -m ai_reader detect plan --method cnn             # CNN on the plan, ~10 s
python -m ai_reader train                                # retrain the CNN (~2 min on CPU)
python -m ai_reader segment page --threshold gaussian --min-area 50

# precision / recall against the ground truth (--x-min 6800: right half only)
python -m ai_reader evaluate resultats/plan_cnn_detections.json data/plans/verite_terrain.json
```

Outputs (annotated image and JSON of the detections) go to `resultats/` (ignored by git). To search more characters,
drop crops of them in `data/caracteres/catalogue/` (e.g. `a.png`); no code change needed.

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
  proposals.py     candidate regions on the plan
  cnn.py           synthetic data, network, training, detection
  evaluation.py    precision / recall
  cli.py           command line
data/              plan, page, template catalogues, ground truth
models/            trained CNN weights
docs/              README images
tests/
```
