"""Length-only baseline and matched-protocol supplementary comparisons."""

import json
import statistics

import config as C
import numpy as np
import pandas as pd
from evaluate import compute_metrics, evaluate_and_save
from protocol import sha256, write_json
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(tag):
    return json.loads((C.METRICS / f"{tag}.json").read_text(encoding="utf-8"))


def length_baseline():
    train = pd.read_csv(C.DATA_PROC / "qevasion_train.csv")
    test = pd.read_csv(C.DATA_PROC / "qevasion_test.csv")
    fit = train[train.role == "train"]
    val = train[train.role == "val"]

    def features(frame):
        return np.log1p(np.column_stack([frame.text_a.str.len(), frame.text_b.str.len()]))

    choices = []
    for weight in [None, "balanced"]:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight=weight, random_state=42, max_iter=2000),
        )
        model.fit(features(fit), fit.label)
        score = compute_metrics(val.label, model.predict(features(val)))["macro_f1"]
        choices.append((score, weight, model))
    score, weight, model = max(choices, key=lambda item: item[0])
    order = [list(model[-1].classes_).index(label) for label in C.CLARITY_LABELS]
    evaluate_and_save(
        "length_only_s42",
        test.label,
        model.predict(features(test)),
        model.predict_proba(features(test))[:, order],
        texts=test.text_a,
    )
    write_json(
        C.METRICS / "length_only_s42_config.json",
        dict(
            features=["log1p(question_chars)", "log1p(answer_chars)"],
            class_weight=weight,
            val_macro_f1=score,
            validation_scores=[dict(class_weight=w, macro_f1=s) for s, w, _ in choices],
            train_n=len(fit),
            val_n=len(val),
            test_n=len(test),
            seed=42,
            data_sha256=sha256(C.DATA_PROC / "qevasion_train.csv"),
        ),
    )


def compare():
    seeds = [42, 2024, 3407]
    direct = [load(f"roberta_clarity_s{s}") for s in seeds]
    evasion = [load(f"roberta_evasion_boost_s{s}") for s in seeds]
    configs = [load(f"roberta_evasion_boost_s{s}_config") for s in seeds]
    for field in [
        "data_sha256",
        "batch",
        "max_len",
        "lr",
        "trunc",
        "train_n",
        "val_n",
        "prediction_precision",
    ]:
        values = [cfg[field] for cfg in configs]
        if any(v != values[0] for v in values):
            raise ValueError(f"evasion协议不一致: {field}")
    for seed, cfg in zip(seeds, configs):
        direct_cfg = load(f"roberta_clarity_s{seed}_config")
        if cfg["seed"] != seed or direct_cfg["seed"] != seed:
            raise ValueError("配对种子错误")
        for field in [
            "data_sha256",
            "batch",
            "max_len",
            "lr",
            "trunc",
            "train_n",
            "val_n",
            "prediction_precision",
        ]:
            if cfg[field] != direct_cfg[field]:
                raise ValueError(f"配对协议不一致: {field}")
    rows = []
    for mode in ["frozen", "finetuned"]:
        source = load(f"transfer_qnli_{mode}")
        control = load(f"control_qnli_{mode}")
        a = load(f"transfer_qnli_{mode}_config")
        b = load(f"control_qnli_{mode}_config")
        for field in [
            "data_sha256",
            "batch",
            "max_len",
            "lr",
            "trunc",
            "train_n",
            "val_n",
            "seed",
            "frozen",
            "class_weight",
            "prediction_precision",
        ]:
            if a[field] != b[field]:
                raise ValueError(f"{mode}对照协议不一致: {field}")
        if not a["init_from"] or b["init_from"] is not None:
            raise ValueError("对照初始化错误")
        for cfg in [a, b]:
            if cfg["val_n"] <= 0 or cfg["best_val_macro_f1"] < 0:
                raise ValueError("对照缺少独立开发集选轮次证据")
        rows.append(
            dict(
                mode=mode,
                original_macro_f1=control["macro_f1"],
                qevasion_macro_f1=source["macro_f1"],
                delta=round(source["macro_f1"] - control["macro_f1"], 4),
                original_config_run_id=b["run_id"],
                source_config_run_id=a["run_id"],
                data_sha256=a["data_sha256"],
            )
        )
    deltas = [round(e["macro_f1"] - d["macro_f1"], 4) for d, e in zip(direct, evasion)]
    tags = [
        *[f"roberta_clarity_s{s}" for s in seeds],
        *[f"roberta_evasion_boost_s{s}" for s in seeds],
        "transfer_qnli_frozen",
        "transfer_qnli_finetuned",
        "control_qnli_frozen",
        "control_qnli_finetuned",
        "length_only_s42",
    ]
    inventory = {
        tag: {
            "config_sha256": sha256(C.METRICS / f"{tag}_config.json"),
            "prediction_sha256": sha256(C.PRED / f"{tag}.csv"),
        }
        for tag in tags
    }
    write_json(
        C.METRICS / "supplement_comparison.json",
        dict(
            evasion=dict(
                seeds=seeds,
                macro_f1=[m["macro_f1"] for m in evasion],
                mean=statistics.mean(m["macro_f1"] for m in evasion),
                sample_std=statistics.stdev(m["macro_f1"] for m in evasion),
                paired_delta=deltas,
                paired_delta_mean=statistics.mean(deltas),
                run_ids=[cfg["run_id"] for cfg in configs],
            ),
            transfer_controls=rows,
            length_only=load("length_only_s42"),
            artifact_inventory=inventory,
            limits="Three seeds are descriptive; QNLI controls are single-seed; no significance or causal mechanism claim.",
        ),
    )
    pd.DataFrame(rows).to_csv(
        C.METRICS / "table_transfer_controls.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(load("supplement_comparison"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--length-only", action="store_true")
    args = ap.parse_args()
    length_baseline()
    if not args.length_only:
        compare()
