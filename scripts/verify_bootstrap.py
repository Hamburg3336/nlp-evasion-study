"""Recompute five answer-cluster bootstrap contrasts from released predictions."""

import csv
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
SEEDS = [42, 2024, 3407]


def read(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def raw_f1(matrix):
    matrix = np.asarray(matrix).reshape(-1, 3, 3)
    denominator = matrix.sum(1) + matrix.sum(2)
    scores = np.divide(
        2 * np.diagonal(matrix, axis1=1, axis2=2),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return scores.mean(1)


def main():
    groups = read("results/analysis/qevasion_test_groups.csv")
    group_ids = [int(row["answer_group"]) for row in groups]
    count = len(set(group_ids))
    if len(groups) != 308 or set(group_ids) != set(range(count)) or count != 191:
        raise ValueError("Unexpected answer-group layout")
    weights = np.random.default_rng(20261116).multinomial(
        count, np.full(count, 1 / count), size=5000
    )
    boot, observed = {}, {}
    for target in ["clarity", "evasion_boost"]:
        for weighting in ["weighted", "unweighted"]:
            arm = target + "_" + weighting
            boot[arm], observed[arm] = [], []
            for seed in SEEDS:
                group = "primary" if weighting == "weighted" else "unweighted"
                suffix = "" if weighting == "weighted" else "_unweighted"
                rows = read(f"results/predictions/{group}/roberta_{target}{suffix}_s{seed}.csv")
                grouped = np.zeros((count, 9), dtype=np.int64)
                for i, row in enumerate(rows):
                    grouped[
                        group_ids[i],
                        LABELS.index(row["gold"]) * 3 + LABELS.index(row["prediction"]),
                    ] += 1
                boot[arm].append(raw_f1(weights @ grouped))
                observed[arm].append(float(raw_f1(grouped.sum(0))[0]))
            boot[arm] = np.mean(boot[arm], axis=0)
            observed[arm] = statistics.mean(observed[arm])
    for row in read("results/analysis/paired_contrasts_cluster_bootstrap.csv"):
        if row["contrast"] == "granularity_by_weighting_interaction":
            samples = (boot["evasion_boost_weighted"] - boot["clarity_weighted"]) - (
                boot["evasion_boost_unweighted"] - boot["clarity_unweighted"]
            )
            delta = (observed["evasion_boost_weighted"] - observed["clarity_weighted"]) - (
                observed["evasion_boost_unweighted"] - observed["clarity_unweighted"]
            )
        else:
            samples = boot[row["comparison"]] - boot[row["reference"]]
            delta = observed[row["comparison"]] - observed[row["reference"]]
        low, high = np.quantile(samples, [0.025, 0.975])
        if not np.allclose(
            [delta, low, high],
            [
                float(row[key])
                for key in ["delta_unrounded_mean", "bootstrap_low", "bootstrap_high"]
            ],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"Bootstrap mismatch: {row['contrast']}")
    print(
        json.dumps(
            dict(
                passed=True,
                contrasts=5,
                answer_clusters=count,
                repeats=5000,
                caveat="Conditional on saved checkpoints; does not include retraining or source-population uncertainty.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
