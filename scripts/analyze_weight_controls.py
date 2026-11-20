"""Verify the 2x2 weighting study, compute paired contrasts and clustered uncertainty."""

import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path

import numpy as np
from analysis_utils import digest, independent_metrics, read_csv


def normalize(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip().casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weighted-only", action="store_true")
    ap.add_argument(
        "--development",
        type=Path,
        help="Optional exported selected-checkpoint development diagnostics",
    )
    args = ap.parse_args()
    root, controls, audit, out = [
        p.resolve() for p in [args.experiment, args.controls, args.audit, args.output]
    ]
    out.mkdir(parents=True, exist_ok=False)
    labels = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
    seeds = [42, 2024, 3407]
    inputs = {}

    def track(path):
        inputs[str(path)] = digest(path)
        return path

    def load_json(path):
        return json.loads(track(path).read_text(encoding="utf-8"))

    def table(name, rows):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    state = None
    if not args.weighted_only:
        state = load_json(controls / "training_plan.json")
        assert state["status"] == "complete" and len(state["completed"]) == 6
        for files in state["completed_hashes"].values():
            for path, value in files.items():
                assert digest(controls / path) == value, path
        source_manifest = load_json(root / "data/source_manifest.json")
        for p, record in source_manifest.items():
            if p.startswith("models/roberta-base/"):
                assert state["protocol"]["pretrained_sha256"][Path(p).name] == record["sha256"]
        for p, h in state["protocol"]["source_code_sha256"].items():
            assert digest(root / "src" / p) == h
    test = read_csv(track(root / "data/processed/qevasion_test.csv"))
    audit_rows = read_csv(track(audit / "test_sample_audit.csv"))
    assert len(test) == len(audit_rows) == 308
    gold = np.array([labels.index(r["label"]) for r in test])
    clean = np.array([r["near_or_exact_seen"] == "False" for r in audit_rows])
    truncated = np.array([r["truncated"] == "True" for r in audit_rows])
    groups = [normalize(r["text_b"]) for r in test]
    group_keys = list(dict.fromkeys(groups))
    group_idx = np.array([group_keys.index(g) for g in groups])
    all_rows, class_rows, configs, predictions, training_rows = [], [], {}, {}, []
    for weight in ["weighted"] if args.weighted_only else ["weighted", "unweighted"]:
        for target in ["clarity", "evasion_boost"]:
            arm = f"{target}_{weight}"
            predictions[arm] = {}
            for seed in seeds:
                tag = f"roberta_{target}{'_unweighted' if weight == 'unweighted' else ''}_s{seed}"
                base = controls if weight == "unweighted" else root / "outputs/revised"
                cfg = load_json(base / f"metrics/{tag}_config.json")
                saved = load_json(base / f"metrics/{tag}.json")
                rows = read_csv(track(base / f"predictions/{tag}.csv"))
                assert cfg["class_weight"] == (weight == "weighted")
                if state:
                    assert cfg["data_sha256"] == state["protocol"]["data_sha256"]
                    if weight == "unweighted":
                        assert cfg["run_id"] == state["run_id"]
                assert (
                    cfg["seed"] == seed
                    and cfg["train_n"] == 3138
                    and cfg["val_n"] == 217
                    and cfg["test_n"] == 308
                )
                assert len(rows) == len(test)
                for i, r in enumerate(rows):
                    assert (
                        int(r["sample_index"]) == i
                        and r["gold"] == test[i]["label"]
                        and r["text_a"] == test[i]["text_a"]
                    )
                    probs = [float(r[f"prob_{c}"]) for c in labels]
                    assert (
                        all(np.isfinite(p) and 0 <= p <= 1 for p in probs)
                        and abs(sum(probs) - 1) < 1e-5
                    )
                    if target == "clarity":
                        assert labels[int(np.argmax(probs))] == r["prediction"]
                pred = np.array([labels.index(r["prediction"]) for r in rows])
                predictions[arm][seed] = pred
                configs[(target, weight, seed)] = cfg
                selected = max(cfg["train_log"], key=lambda r: r["val_macro_f1"])
                training_rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        epochs_run=cfg["epochs_run"],
                        selected_epoch=selected["epoch"],
                        best_val_macro_f1=cfg["best_val_macro_f1"],
                        selected_dev_fine_f1=selected.get("val_evasion_macro_f1", ""),
                        elapsed_sec=cfg["elapsed_sec"],
                        batch=cfg["batch"],
                        lr=cfg["lr"],
                        max_len=cfg["max_len"],
                        trunc=cfg["trunc"],
                        precision=cfg["prediction_precision"],
                        data_sha256=cfg["data_sha256"],
                        run_id=cfg["run_id"],
                    )
                )
                for subset, mask in [
                    ("all", np.ones(len(test), dtype=bool)),
                    ("exclude_seen", clean),
                    ("truncated", truncated),
                    ("not_truncated", ~truncated),
                ]:
                    g = np.array(labels)[gold[mask]].tolist()
                    p = np.array(labels)[pred[mask]].tolist()
                    metrics = independent_metrics(g, p, labels)
                    if subset == "all":
                        for key in metrics:
                            assert metrics[key] == saved[key], (tag, key)
                    all_rows.append(
                        dict(
                            arm=arm,
                            target=target,
                            weight=weight,
                            seed=seed,
                            subset=subset,
                            n=int(mask.sum()),
                            accuracy=metrics["accuracy"],
                            macro_f1=metrics["macro_f1"],
                            best_val_macro_f1=cfg["best_val_macro_f1"],
                            epochs_run=cfg["epochs_run"],
                        )
                    )
                    for label, values in metrics["per_class"].items():
                        class_rows.append(
                            dict(arm=arm, seed=seed, subset=subset, label=label, **values)
                        )
    if state:
        fields = [
            "data_sha256",
            "batch",
            "lr",
            "max_len",
            "trunc",
            "train_n",
            "val_n",
            "test_n",
            "prediction_precision",
            "frozen",
        ]
        for target in ["clarity", "evasion_boost"]:
            for seed in seeds:
                a, b = configs[(target, "weighted", seed)], configs[(target, "unweighted", seed)]
                for field in fields:
                    assert a[field] == b[field], (target, seed, field)
                assert a["labels"] == b["labels"]
    table("four_arm_seed_metrics.csv", all_rows)
    table("training_protocol_comparison.csv", training_rows)
    table("four_arm_class_metrics.csv", class_rows)
    summary = []
    for arm in predictions:
        for subset in ["all", "exclude_seen", "truncated", "not_truncated"]:
            rows = [r for r in all_rows if r["arm"] == arm and r["subset"] == subset]
            summary.append(
                dict(
                    arm=arm,
                    subset=subset,
                    n_per_seed=rows[0]["n"],
                    accuracy_mean=statistics.mean(r["accuracy"] for r in rows),
                    accuracy_sample_std=statistics.stdev(r["accuracy"] for r in rows),
                    macro_f1_mean=statistics.mean(r["macro_f1"] for r in rows),
                    macro_f1_sample_std=statistics.stdev(r["macro_f1"] for r in rows),
                )
            )
    table("four_arm_summary.csv", summary)
    contrasts = [("granularity_with_weights", "clarity_weighted", "evasion_boost_weighted")]
    if state:
        contrasts += [
            ("granularity_without_weights", "clarity_unweighted", "evasion_boost_unweighted"),
            ("weighting_direct", "clarity_unweighted", "clarity_weighted"),
            ("weighting_hierarchy", "evasion_boost_unweighted", "evasion_boost_weighted"),
        ]
    # Paired cluster bootstrap keeps questions sharing an answer together.
    # Conditional on existing checkpoints: it does not estimate retraining uncertainty.
    rng = np.random.default_rng(20261116)
    repeats = 5000
    weights = rng.multinomial(
        len(group_keys), np.full(len(group_keys), 1 / len(group_keys)), size=repeats
    )

    def raw_f1(cm):
        cm = np.asarray(cm).reshape(-1, 3, 3)
        den = cm.sum(1) + cm.sum(2)
        f = np.divide(
            2 * np.diagonal(cm, axis1=1, axis2=2),
            den,
            out=np.zeros_like(den, dtype=float),
            where=den != 0,
        )
        return f.mean(1)

    boot, observed = {}, {}
    for arm, pp in predictions.items():
        boot[arm], observed[arm] = [], []
        for seed in seeds:
            grouped = np.zeros((len(group_keys), 9), dtype=np.int64)
            for i in range(len(test)):
                grouped[group_idx[i], gold[i] * 3 + pp[seed][i]] += 1
            assert grouped.sum() == len(test)
            boot[arm].append(raw_f1(weights @ grouped))
            observed[arm].append(float(raw_f1(grouped.sum(0))[0]))
        boot[arm] = np.mean(boot[arm], axis=0)
        observed[arm] = statistics.mean(observed[arm])
    contrast_rows = []
    for name, a, b in contrasts:
        samples = boot[b] - boot[a]
        low, high = np.quantile(samples, [0.025, 0.975])
        delta = [
            next(
                r["macro_f1"]
                for r in all_rows
                if r["arm"] == b and r["seed"] == seed and r["subset"] == "all"
            )
            - next(
                r["macro_f1"]
                for r in all_rows
                if r["arm"] == a and r["seed"] == seed and r["subset"] == "all"
            )
            for seed in seeds
        ]
        contrast_rows.append(
            dict(
                contrast=name,
                reference=a,
                comparison=b,
                delta_unrounded_mean=observed[b] - observed[a],
                seed42_delta=round(delta[0], 4),
                seed2024_delta=round(delta[1], 4),
                seed3407_delta=round(delta[2], 4),
                bootstrap_low=float(low),
                bootstrap_high=float(high),
                zero_in_interval=bool(low <= 0 <= high),
            )
        )
    if state:
        interaction = (boot["evasion_boost_weighted"] - boot["clarity_weighted"]) - (
            boot["evasion_boost_unweighted"] - boot["clarity_unweighted"]
        )
        low, high = np.quantile(interaction, [0.025, 0.975])
        by_name = {r["contrast"]: r for r in contrast_rows}
        interaction_seed = {
            f"seed{s}_delta": round(
                by_name["granularity_with_weights"][f"seed{s}_delta"]
                - by_name["granularity_without_weights"][f"seed{s}_delta"],
                4,
            )
            for s in seeds
        }
        contrast_rows.append(
            dict(
                contrast="granularity_by_weighting_interaction",
                reference="unweighted granularity effect",
                comparison="weighted granularity effect",
                delta_unrounded_mean=(
                    observed["evasion_boost_weighted"] - observed["clarity_weighted"]
                )
                - (observed["evasion_boost_unweighted"] - observed["clarity_unweighted"]),
                **interaction_seed,
                bootstrap_low=float(low),
                bootstrap_high=float(high),
                zero_in_interval=bool(low <= 0 <= high),
            )
        )
    table("paired_contrasts_cluster_bootstrap.csv", contrast_rows)
    verified_development = 0
    if args.development:
        dev_out = args.development.resolve()
        dev_manifest = load_json(dev_out / "manifest.json")
        for path, h in dev_manifest["inputs"].items():
            assert digest(path) == h, path
        for path, h in dev_manifest["outputs"].items():
            assert digest(dev_out / path) == h, path
        data = read_csv(track(root / "data/processed/qevasion_train.csv"))
        dev = [dict(r, processed_row=str(i)) for i, r in enumerate(data) if r["role"] == "val"]
        mapping = {}
        for r in data:
            if r["evasion"] in mapping:
                assert mapping[r["evasion"]] == r["label"]
            mapping[r["evasion"]] = r["label"]
        for seed in seeds:
            for target in ["clarity", "evasion_boost"]:
                tag = f"roberta_{target}_unweighted_s{seed}"
                cfg = configs[(target, "unweighted", seed)]
                frame = read_csv(track(dev_out / f"{tag}.csv"))
                saved = load_json(dev_out / f"{tag}_metrics.json")
                assert len(frame) == len(dev)
                for r, original in zip(frame, dev):
                    for field in [
                        "processed_row",
                        "_raw_idx",
                        "label",
                        "evasion",
                        "group",
                        "text_a",
                        "text_b",
                    ]:
                        assert r[field] == original[field], (tag, field)
                    probs = [float(r[f"prob_{c}"]) for c in cfg["labels"]]
                    assert (
                        all(np.isfinite(p) and 0 <= p <= 1 for p in probs)
                        and abs(sum(probs) - 1) < 1e-5
                    )
                    assert cfg["labels"][int(np.argmax(probs))] == r["prediction"]
                    if target == "evasion_boost":
                        assert mapping[r["prediction"]] == r["mapped_prediction"]
                coarse = independent_metrics(
                    [r["label"] for r in frame],
                    [
                        r["mapped_prediction" if target == "evasion_boost" else "prediction"]
                        for r in frame
                    ],
                    labels,
                )
                assert coarse["macro_f1"] == cfg["best_val_macro_f1"]
                for key in coarse:
                    assert coarse[key] == saved["coarse"][key], (tag, key)
                if target == "evasion_boost":
                    fine = independent_metrics(
                        [r["evasion"] for r in frame],
                        [r["prediction"] for r in frame],
                        cfg["labels"],
                    )
                    for key in fine:
                        assert fine[key] == saved["fine"][key], (tag, key)
                    selected = max(cfg["train_log"], key=lambda r: r["val_macro_f1"])
                    assert fine["macro_f1"] == selected["val_evasion_macro_f1"]
                verified_development += 1
    for path, h in inputs.items():
        assert digest(path) == h
    result = dict(
        passed=True,
        verified_model_prediction_files=sum(len(v) for v in predictions.values()),
        independently_verified_development_files=verified_development,
        source_and_data_protocol_matched=bool(state),
        bootstrap=dict(
            seed=20261116,
            repeats=repeats,
            clusters=len(group_keys),
            cluster_unit="normalized complete answer, include all associated questions",
            interval="paired percentile 95%, mean of per-seed Macro-F1 differences",
            caveat="Conditional on selected checkpoints; no retraining uncertainty; answer clusters do not guarantee interview/source independence.",
        ),
        contrasts=contrast_rows,
        inputs=inputs,
        script_sha256=digest(Path(__file__)),
        outputs={p.name: digest(p) for p in out.iterdir() if p.is_file()},
    )
    (out / "verification_and_uncertainty.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                verified=result["verified_model_prediction_files"],
                summary=summary,
                contrasts=contrast_rows,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
