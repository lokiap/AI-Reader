# AI-Reader

🇫🇷 Français · [🇬🇧 English](README.md)

Détection et reconnaissance de symboles dans des images à partir d'un petit catalogue de gabarits, avec OpenCV.
Projet d'école « Projet IA » 2024-2025, repris et complété.

Deux cas d'usage :

| Source | Image | Catalogue | Ce qu'on cherche |
|---|---|---|---|
| `plan` | plan d'un bâtiment (13 624 × 5 061 px) | 6 luminaires | les luminaires, quelle que soit leur orientation |
| `page` | page de texte (2 480 × 3 507 px) | 6 caractères | les caractères `2 I d n o u` |

![Luminaires détectés sur le plan](docs/plan_detections.jpg)

## Méthode

1. **Segmentation** (`ai_reader/segmentation.py`) : binarisation (seuil fixe, moyenne ou gaussien) puis composantes
   connexes. Sur la page, elle encadre toutes les lettres, mais sans dire lesquelles :

   ![Segmentation de la page](docs/page_segmentation.jpg)

2. **Reconnaissance** (`ai_reader/recognition.py`) : template matching (corrélation normalisée) de chaque symbole du
   catalogue, puis fusion des détections qui se recouvrent (le meilleur score gagne).
   - *Plan* : la corrélation est faite sur un masque « bleu pur » (`imaging.blue_mask`), car les symboles sont
     tracés en bleu alors que murs, cotes et mobilier sont gris ou noirs. Les gabarits sont testés à 0°, 90°, 180° et 270°.
   - *Page* : un gabarit très fin comme `I` corrèle fortement avec le fût de `P`, `L` ou `T`. On ne garde donc une
     détection que si la composante connexe située dessous a la taille du gabarit (à 15 % près).

   ![Caractères reconnus sur la page](docs/page_detections.jpg)

## Résultats

| Source | Détections |
|---|---|
| plan | 181 : 173 × `luminaire_01` (barres lumineuses), 6 × `luminaire_11`, 1 × `luminaire_03`, 1 × `luminaire_07` |
| page | 132 : 52 × `o`, 38 × `u`, 27 × `d`, 7 × `I`, 5 × `n`, 3 × `2` |

Les seuils (0,85 par défaut sur le plan, 0,75 pour `luminaire_07`, 0,80 pour `luminaire_11`, 0,90 sur la page) ont été
choisis en inspectant visuellement tous les candidats avec un score ≥ 0,5.

**Limites honnêtes**
- Aucune vérité terrain n'existe pour ces images : les nombres ci-dessus ne sont **pas** des mesures de rappel.
  J'ai relu à l'œil les 60 détections de plus faible score du plan : toutes sont correctes. Le rappel n'est pas mesuré.
- `luminaire_10` et `luminaire_15` ne sont pas trouvés : aucun candidat crédible, ils sont probablement absents de ce plan.
- Sur la page, le gabarit `n` est en gras et ne reconnaît que les `n` de titre ; les lettres qui se touchent ne
  forment pas une composante de la bonne taille et sont manquées.
- Le template matching est sensible à la police et à l'échelle. Un classifieur appris (CNN) serait l'étape suivante.

Pour mesurer précision et rappel, annoter une vérité terrain au même format que `resultats/*_detections.json`
(liste de `{label, x, y, w, h, score, angle}`) puis lancer `python -m ai_reader evaluate prédictions.json vérité.json`.

## Installation

Python 3.10 ou plus.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation

```bash
python -m ai_reader detect page                 # ~3 s
python -m ai_reader detect plan                 # ~1 min : 6 gabarits × 4 rotations sur 69 Mpx
python -m ai_reader segment page --threshold gaussian --min-area 50
python -m ai_reader detect plan --threshold 0.9 --output sortie/
python -m ai_reader evaluate predictions.json verite.json
```

Les sorties (image annotée et JSON des détections) vont dans `resultats/` (ignoré par git).

## Tests

```bash
python -m pytest
```

## Organisation

```
ai_reader/
  imaging.py       chargement, seuils, masque bleu
  segmentation.py  composantes connexes
  recognition.py   template matching, NMS, détection plan / page
  evaluation.py    précision / rappel
  cli.py           ligne de commande
data/              plan, page et catalogues de gabarits
docs/              images du README
tests/
```
