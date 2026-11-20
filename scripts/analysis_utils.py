"""Independent metrics for regenerated supplementary analyses."""

import csv
import hashlib
import statistics
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def independent_metrics(gold, pred, labels):
    cm = [[sum(g == a and p == b for g, p in zip(gold, pred)) for b in labels] for a in labels]
    fs, per_class = [], {}
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0
        fs.append(f1)
        per_class[lab] = dict(
            precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4), support=support
        )
    return dict(
        accuracy=round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
        macro_f1=round(statistics.mean(fs), 4),
        confusion_matrix=cm,
        per_class=per_class,
    )
