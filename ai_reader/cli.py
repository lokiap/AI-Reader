"""Ligne de commande : python -m ai_reader <commande>."""

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import evaluation, imaging, recognition, segmentation

DATA = Path("data")
MODEL = Path("models") / "plan_cnn.pt"
TRUTH = DATA / "plans" / "verite_terrain.json"
SOURCES = {
    "plan": (DATA / "plans" / "plan.png", DATA / "plans" / "catalogue"),
    "page": (DATA / "caracteres" / "page.png", DATA / "caracteres" / "catalogue"),
}


def _save(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    print(f"Écrit : {path}")


def cmd_segment(args):
    image_path = args.image or SOURCES[args.source][0]
    gray = imaging.load_gray(image_path)
    binary = imaging.THRESHOLDS[args.threshold](gray)
    components = segmentation.find_components(binary, min_area=args.min_area)
    print(f"{len(components)} composantes (aire >= {args.min_area}, seuil « {args.threshold} »)")
    _save(args.output, segmentation.draw_components(gray, components))


def cmd_detect(args):
    if args.method == "cnn" and args.source != "plan":
        sys.exit("--method cnn n'existe que pour le plan")
    image_path, catalogue = SOURCES[args.source]
    image_path = args.image or image_path
    catalogue = args.catalogue or catalogue
    if args.source == "plan" and args.method == "cnn":
        from . import cnn  # PyTorch n'est nécessaire que pour cette méthode

        detections = cnn.detect_plan_cnn(image_path, args.model, threshold=args.threshold or 0.5)
    elif args.source == "plan":
        detections = recognition.detect_plan(image_path, catalogue, default_threshold=args.threshold or recognition.PLAN_DEFAULT_THRESHOLD)
    else:
        detections = recognition.detect_page(image_path, catalogue, threshold=args.threshold or recognition.PAGE_DEFAULT_THRESHOLD)
    counts = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    print(f"{len(detections)} détections : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_cnn" if args.method == "cnn" else ""
    (out / f"{args.source}{suffix}_detections.json").write_text(
        json.dumps([d.to_dict() for d in detections], indent=1), encoding="utf-8")
    _save(out / f"{args.source}{suffix}_detections.png", recognition.draw_detections(imaging.load_bgr(image_path), detections))


def cmd_train(args):
    from . import cnn

    known = evaluation.load_detections(args.truth)
    image_path, catalogue = SOURCES["plan"]
    cnn.train(catalogue, image_path, known, args.output, epochs=args.epochs, n_per_class=args.samples)
    print(f"Modèle écrit : {args.output}")


def cmd_evaluate(args):
    predicted = evaluation.load_detections(args.predictions)
    truth = evaluation.load_detections(args.truth)
    if args.x_min is not None:  # ne comparer que la zone x >= x_min (ex. la moitié droite du plan)
        predicted = [d for d in predicted if d.x >= args.x_min]
        truth = [d for d in truth if d.x >= args.x_min]
    print(evaluation.format_report(evaluation.evaluate(predicted, truth)))


def build_parser():
    parser = argparse.ArgumentParser(prog="ai_reader", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="binarise l'image et encadre les composantes connexes")
    seg.add_argument("source", choices=SOURCES)
    seg.add_argument("--image", help="image à analyser (par défaut celle de data/)")
    seg.add_argument("--threshold", choices=imaging.THRESHOLDS, default="mean")
    seg.add_argument("--min-area", type=int, default=50)
    seg.add_argument("--output", default="resultats/segmentation.png")
    seg.set_defaults(func=cmd_segment)

    det = sub.add_parser("detect", help="reconnaît les symboles du catalogue par template matching")
    det.add_argument("source", choices=SOURCES)
    det.add_argument("--image")
    det.add_argument("--catalogue")
    det.add_argument("--threshold", type=float, help="seuil de corrélation (défaut : propre à chaque source)")
    det.add_argument("--output", default="resultats")
    det.add_argument("--method", choices=["template", "cnn"], default="template", help="méthode pour le plan (défaut : template)")
    det.add_argument("--model", default=str(MODEL), help="poids du CNN (--method cnn)")
    det.set_defaults(func=cmd_detect)

    tr = sub.add_parser("train", help="entraîne le CNN de classification des luminaires du plan")
    tr.add_argument("--truth", default=str(TRUTH), help="luminaires connus, exclus des négatifs")
    tr.add_argument("--output", default=str(MODEL))
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--samples", type=int, default=800, help="vignettes générées par classe")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="précision / rappel par rapport à une vérité terrain")
    ev.add_argument("predictions")
    ev.add_argument("truth")
    ev.add_argument("--x-min", type=int, help="n'évalue que les objets dont x >= x-min")
    ev.set_defaults(func=cmd_evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
