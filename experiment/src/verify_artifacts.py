"""独立重算指标并检验修订产物；验证失败以非零码退出。"""

import csv
import json

import config as C
import numpy as np
from evaluate import compute_metrics
from protocol import write_json


def main():
    C.ensure_directories()
    rows = []
    for p in sorted(C.PRED.glob("*.csv")):
        with p.open(encoding="utf-8-sig", newline="") as f:
            data = list(csv.DictReader(f))
        saved = json.loads((C.METRICS / f"{p.stem}.json").read_text(encoding="utf-8"))
        labels = saved["labels"]
        gold = [r["gold"] for r in data]
        pred = [r["prediction"] for r in data]
        recalculated = compute_metrics(gold, pred, labels=labels)
        for key in [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "confusion_matrix",
            "per_class",
        ]:
            if saved[key] != recalculated[key]:
                raise ValueError(f"{p.name}: {key}不一致")
        probs = np.array([[float(r["prob_" + label]) for label in labels] for r in data])
        if (
            not np.isfinite(probs).all()
            or (probs < 0).any()
            or (probs > 1).any()
            or not np.allclose(probs.sum(1), 1, atol=1e-5)
        ):
            raise ValueError(f"{p.name}: 概率非法")
        disagreements = int((np.array(labels)[probs.argmax(1)] != np.array(pred)).sum())
        if saved.get("decision_rule", "argmax") == "argmax" and disagreements:
            raise ValueError(f"{p.name}: 概率与预测不同步")
        rows.append(
            dict(
                model=p.stem,
                n=len(data),
                macro_f1=saved["macro_f1"],
                metrics_match=True,
                decision_rule=saved.get("decision_rule", "argmax"),
                argmax_disagreements=disagreements,
            )
        )
    if not rows:
        raise ValueError("没有预测文件")
    write_json(C.METRICS / "artifact_verification.json", {"status": "passed", "predictions": rows})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
