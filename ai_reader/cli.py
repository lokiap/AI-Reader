"""Ligne de commande : python -m ai_reader <commande>."""

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import evaluation, imaging, recognition, segmentation

DATA = Path("data")
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
    image_path, catalogue = SOURCES[args.source]
    image_path = args.image or image_path
    catalogue = args.catalogue or catalogue
    if args.source == "plan":
        detections = recognition.detect_plan(image_path, catalogue, default_threshold=args.threshold or recognition.PLAN_DEFAULT_THRESHOLD)
    else:
        detections = recognition.detect_page(image_path, catalogue, threshold=args.threshold or recognition.PAGE_DEFAULT_THRESHOLD)
    counts = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    print(f"{len(detections)} détections : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.source}_detections.json").write_text(
        json.dumps([d.to_dict() for d in detections], indent=1), encoding="utf-8")
    _save(out / f"{args.source}_detections.png", recognition.draw_detections(imaging.load_bgr(image_path), detections))


def cmd_evaluate(args):
    report = evaluation.evaluate(evaluation.load_detections(args.predictions), evaluation.load_detections(args.truth))
    print(evaluation.format_report(report))


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
    det.set_defaults(func=cmd_detect)

    ev = sub.add_parser("evaluate", help="précision / rappel par rapport à une vérité terrain")
    ev.add_argument("predictions")
    ev.add_argument("truth")
    ev.set_defaults(func=cmd_evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
