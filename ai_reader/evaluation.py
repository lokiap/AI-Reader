"""Comparaison des détections à une vérité terrain (précision / rappel)."""

import json

from .recognition import Detection, _iou


def load_detections(path):
    with open(path, encoding="utf-8") as f:
        return [Detection(**d) for d in json.load(f)]


def evaluate(predicted, truth, iou_threshold=0.3):
    """Précision, rappel et F1, globaux et par symbole.

    Une détection est correcte si elle a le bon label et recouvre (IoU) une vérité
    pas encore appariée.
    """
    labels = sorted({d.label for d in predicted} | {d.label for d in truth})
    report = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for label in labels:
        preds = sorted((d for d in predicted if d.label == label), key=lambda d: -d.score)
        truths = [d for d in truth if d.label == label]
        matched = set()
        tp = 0
        for p in preds:
            best, best_iou = None, iou_threshold
            for i, t in enumerate(truths):
                if i not in matched and _iou(p, t) >= best_iou:
                    best, best_iou = i, _iou(p, t)
            if best is not None:
                matched.add(best)
                tp += 1
        fp, fn = len(preds) - tp, len(truths) - tp
        report[label] = _scores(tp, fp, fn)
        for key, value in (("tp", tp), ("fp", fp), ("fn", fn)):
            totals[key] += value
    report["TOTAL"] = _scores(**totals)
    return report


def _scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def format_report(report):
    def fmt(v):
        return "  -  " if v is None else f"{v:5.2f}"

    lines = [f"{'symbole':<14}{'TP':>5}{'FP':>5}{'FN':>5}  prec.  rappel   F1"]
    for label, r in report.items():
        lines.append(f"{label:<14}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}  {fmt(r['precision'])}  {fmt(r['recall'])}  {fmt(r['f1'])}")
    return "\n".join(lines)
