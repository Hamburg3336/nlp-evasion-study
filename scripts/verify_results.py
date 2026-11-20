"""Recompute published test metrics using only Python's standard library."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {value}")
    return path


def compute_metrics(gold, predictions, labels):
    if not gold or len(gold) != len(predictions):
        raise ValueError("Gold/prediction arrays must be nonempty and equally sized")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Labels must be nonempty and unique")
    if not set(gold).issubset(labels) or not set(predictions).issubset(labels):
        raise ValueError("Unknown gold or predicted label")
    lookup = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for expected, actual in zip(gold, predictions):
        matrix[lookup[expected]][lookup[actual]] += 1
    per_class = {}
    precision_values, recall_values, f1_values = [], [], []
    weighted = 0.0
    for i, label in enumerate(labels):
        correct = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted += support * f1
        per_class[label] = dict(
            precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4), support=support
        )
    return dict(
        accuracy=round(sum(matrix[i][i] for i in range(len(labels))) / len(gold), 4),
        macro_precision=round(sum(precision_values) / len(labels), 4),
        macro_recall=round(sum(recall_values) / len(labels), 4),
        macro_f1=round(sum(f1_values) / len(labels), 4),
        weighted_f1=round(weighted / len(gold), 4),
        per_class=per_class,
        confusion_matrix=matrix,
    )


def verify(root):
    index = load_json(root / "results/index.json")
    if not index:
        raise ValueError("Empty results index")
    verified = []
    for entry in index:
        with safe_path(root, entry["predictions"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != entry["n"]:
            raise ValueError(f"Unexpected row count: {entry['predictions']}")
        if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"Invalid sample order: {entry['predictions']}")
        gold = [row[entry["gold_column"]] for row in rows]
        predictions = [row["prediction"] for row in rows]
        expected = load_json(safe_path(root, entry["metrics"]))
        actual = compute_metrics(gold, predictions, entry["labels"])
        for key, value in actual.items():
            if key in expected and value != expected[key]:
                raise ValueError(f"Metric mismatch: {entry['metrics']} / {key}")
        probability_keys = [key for key in rows[0] if key.startswith("prob_")]
        for row in rows:
            probabilities = [float(row[key]) for key in probability_keys]
            if probabilities and (
                not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                or abs(sum(probabilities) - 1) > 1e-5
            ):
                raise ValueError(f"Invalid probabilities: {entry['predictions']}")
            # Nine-way argmax followed by taxonomy mapping is a different rule
            # from argmax of summed coarse probabilities. Do not conflate them.
            if probabilities and expected.get("decision_rule", "argmax") == "argmax":
                label = probability_keys[
                    max(range(len(probabilities)), key=probabilities.__getitem__)
                ][5:]
                if label != row["prediction"]:
                    raise ValueError(f"Probability/prediction mismatch: {entry['predictions']}")
        verified.append(
            dict(
                path=entry["predictions"],
                n=len(rows),
                accuracy=actual["accuracy"],
                macro_f1=actual["macro_f1"],
            )
        )
    manifest_path = root / "provenance/package_manifest.json"
    checked = 0
    if manifest_path.exists():
        for relative, expected in load_json(manifest_path)["file_sha256"].items():
            actual = hashlib.sha256(safe_path(root, relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Package checksum mismatch: {relative}")
            checked += 1
    return dict(
        passed=True,
        prediction_files=len(verified),
        metrics_recomputed=True,
        package_files_checked=checked,
        results=verified,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Optional path for a new verification record")
    args = parser.parse_args()
    result = verify(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
