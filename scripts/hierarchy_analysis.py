"""Inference-only hierarchy diagnostics using previously selected checkpoints.

Run with the experiment's Python environment and compatible runtime packages.
No optimization, checkpoint selection, or modification of existing artifacts.
"""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.experiment.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "src"))
    import config as C
    from evaluate import compute_metrics
    from protocol import sha256, write_json
    from train_encoder import PadCollator, PairDataset, set_seed

    def save(name, obj):
        write_json(out / name, obj)

    def metric(gold, pred, labels):
        return {k: v for k, v in compute_metrics(gold, pred, labels).items() if k != "report"}

    def table(name, rows):
        pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")

    inputs = {}

    def track(path):
        inputs[str(path)] = sha256(path)
        return path

    data_path = track(root / "data/processed/qevasion_train.csv")
    test_path = track(root / "data/processed/qevasion_test.csv")
    data = pd.read_csv(data_path, keep_default_na=False)
    test = pd.read_csv(test_path, keep_default_na=False)
    data["processed_row"] = np.arange(len(data))
    val = data.loc[data.role == "val"].copy().reset_index(drop=True)
    assert len(val) == 217 and len(test) == 308
    assert not set(data.loc[data.role == "train", "group"]) & set(val.group)
    mapped = data.evasion.map(C.EVASION_TO_CLARITY)
    assert mapped.notna().all() and (mapped == data.label).all()
    assert (test.evasion == "").all()
    distribution = []
    for fine in C.EVASION_LABELS:
        distribution.append(
            dict(
                fine_label=fine,
                coarse_label=C.EVASION_TO_CLARITY[fine],
                train_n=int(((data.role == "train") & (data.evasion == fine)).sum()),
                dev_n=int((val.evasion == fine).sum()),
            )
        )
    table("label_hierarchy.csv", distribution)
    results, checks, val_predictions, seed_rows, class_rows, pair_rows, case_rows = (
        {},
        [],
        {},
        [],
        [],
        [],
        [],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA required to match the saved bf16 inference protocol")
    for seed in C.SEEDS:
        for target in ["clarity", "evasion_boost"]:
            tag = f"roberta_{target}_s{seed}"
            cfg_path = track(root / f"outputs/revised/metrics/{tag}_config.json")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert cfg["data_sha256"] == inputs[str(data_path)]
            assert cfg["seed"] == seed and cfg["batch"] == 8
            assert cfg["prediction_precision"] == "bf16"
            ck = root / "outputs/revised/models" / tag
            for name in [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ]:
                track(ck / name)
            set_seed(seed)
            tok = AutoTokenizer.from_pretrained(ck, local_files_only=True)
            model = (
                AutoModelForSequenceClassification.from_pretrained(ck, local_files_only=True)
                .to(device)
                .eval()
            )
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            assert labels == cfg["labels"]
            loader = DataLoader(
                PairDataset(val, tok, cfg["max_len"], trunc_mode=cfg["trunc"]),
                batch_size=cfg["batch"],
                shuffle=False,
                num_workers=0,
                collate_fn=PadCollator(tok.pad_token_id),
            )
            probs = []
            with torch.inference_mode():
                for batch, _ in loader:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                    probs.append(logits.float().softmax(-1).cpu().numpy())
            p = np.vstack(probs)
            assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
            prediction = np.array(labels)[p.argmax(1)]
            frame = val[
                ["processed_row", "_raw_idx", "group", "text_a", "text_b", "label", "evasion"]
            ].copy()
            frame["prediction"] = prediction
            for i, lab in enumerate(labels):
                frame[f"prob_{lab}"] = p[:, i]
            if target == "evasion_boost":
                frame["mapped_prediction"] = [C.EVASION_TO_CLARITY[x] for x in prediction]
                cp = np.column_stack(
                    [
                        p[
                            :,
                            [
                                i
                                for i, lab in enumerate(labels)
                                if C.EVASION_TO_CLARITY[lab] == coarse
                            ],
                        ].sum(1)
                        for coarse in C.CLARITY_LABELS
                    ]
                )
                frame["probability_sum_prediction"] = np.array(C.CLARITY_LABELS)[cp.argmax(1)]
                fine_ok = prediction == val.evasion.to_numpy()
                coarse_ok = frame.mapped_prediction.to_numpy() == val.label.to_numpy()
                assert not np.any(fine_ok & ~coarse_ok)
                counts = dict(
                    fine_correct=int(fine_ok.sum()),
                    fine_wrong_coarse_correct=int((~fine_ok & coarse_ok).sum()),
                    cross_coarse_error=int((~coarse_ok).sum()),
                )
                result = dict(
                    fine=metric(val.evasion, prediction, labels),
                    coarse=metric(val.label, frame.mapped_prediction, C.CLARITY_LABELS),
                    error_decomposition=counts,
                    probability_sum_diagnostic=dict(
                        changed_predictions=int(
                            (frame.mapped_prediction != frame.probability_sum_prediction).sum()
                        ),
                        metrics=metric(
                            val.label, frame.probability_sum_prediction, C.CLARITY_LABELS
                        ),
                    ),
                )
                # These are validation diagnostics at the checkpoint selected by coarse F1.
                selected = max(cfg["train_log"], key=lambda row: row["val_macro_f1"])
                checks.append(
                    dict(
                        tag=tag,
                        selected_epoch=selected["epoch"],
                        saved_fine_f1=selected["val_evasion_macro_f1"],
                        inferred_fine_f1=result["fine"]["macro_f1"],
                        saved_coarse_f1=cfg["best_val_macro_f1"],
                        inferred_coarse_f1=result["coarse"]["macro_f1"],
                    )
                )
                assert result["fine"]["macro_f1"] == selected["val_evasion_macro_f1"]
                coarse_result = result["coarse"]
            else:
                result = metric(val.label, prediction, labels)
                coarse_result = result
            assert coarse_result["macro_f1"] == cfg["best_val_macro_f1"], (
                tag,
                coarse_result["macro_f1"],
                cfg["best_val_macro_f1"],
            )
            frame.to_csv(out / f"dev_{tag}.csv", index=False, encoding="utf-8-sig")
            save(f"dev_{tag}_metrics.json", result)
            results[tag] = result
            val_predictions[tag] = frame
            print(f"{tag}: dev coarse F1={coarse_result['macro_f1']}", flush=True)
            del model, loader
            gc.collect()
            torch.cuda.empty_cache()

        for split in ["dev", "test"]:
            if split == "dev":
                a = val_predictions[f"roberta_clarity_s{seed}"]
                b = val_predictions[f"roberta_evasion_boost_s{seed}"]
                gold = a.label.to_numpy()
                pa, pb = a.prediction.to_numpy(), b.mapped_prediction.to_numpy()
                texts = val
            else:
                paths = [
                    track(root / f"outputs/revised/predictions/roberta_{target}_s{seed}.csv")
                    for target in ["clarity", "evasion_boost"]
                ]
                a, b = [pd.read_csv(path, keep_default_na=False) for path in paths]
                assert len(a) == len(b) == len(test)
                assert np.array_equal(a.sample_index, np.arange(len(test)))
                assert np.array_equal(a.sample_index, b.sample_index)
                assert np.array_equal(a.text_a, test.text_a) and np.array_equal(
                    b.text_a, test.text_a
                )
                assert np.array_equal(a.gold, test.label) and np.array_equal(b.gold, test.label)
                gold = a.gold.to_numpy()
                pa, pb = a.prediction.to_numpy(), b.prediction.to_numpy()
                texts = test
            ac, bc = pa == gold, pb == gold
            status = np.where(
                ac & bc,
                "both_correct",
                np.where(~ac & bc, "fixed", np.where(ac & ~bc, "regressed", "both_wrong")),
            )
            pair_rows.append(
                dict(
                    seed=seed,
                    split=split,
                    n=len(gold),
                    agreement=float(np.mean(pa == pb)),
                    **{
                        k: int((status == k).sum())
                        for k in ["both_correct", "both_wrong", "fixed", "regressed"]
                    },
                )
            )
            table(
                f"{split}_paired_s{seed}.csv",
                [
                    dict(
                        sample_index=i,
                        gold=gold[i],
                        direct_prediction=pa[i],
                        hierarchy_prediction=pb[i],
                        status=status[i],
                        agree=bool(pa[i] == pb[i]),
                    )
                    for i in range(len(gold))
                ],
            )
            ma, mb = metric(gold, pa, C.CLARITY_LABELS), metric(gold, pb, C.CLARITY_LABELS)
            if split == "test":
                for target, actual in [("clarity", ma), ("evasion_boost", mb)]:
                    saved_path = track(
                        root / f"outputs/revised/metrics/roberta_{target}_s{seed}.json"
                    )
                    saved = json.loads(saved_path.read_text(encoding="utf-8"))
                    assert actual["confusion_matrix"] == saved["confusion_matrix"]
                    assert actual["macro_f1"] == saved["macro_f1"]
            seed_rows.append(
                dict(
                    seed=seed,
                    split=split,
                    direct_accuracy=ma["accuracy"],
                    hierarchy_accuracy=mb["accuracy"],
                    direct_macro_f1=ma["macro_f1"],
                    hierarchy_macro_f1=mb["macro_f1"],
                    delta_macro_f1=round(mb["macro_f1"] - ma["macro_f1"], 4),
                )
            )
            for lab in C.CLARITY_LABELS:
                class_rows.append(
                    dict(
                        seed=seed,
                        split=split,
                        label=lab,
                        support=ma["per_class"][lab]["support"],
                        **{
                            f"direct_{m}": ma["per_class"][lab][m]
                            for m in ["precision", "recall", "f1"]
                        },
                        **{
                            f"hierarchy_{m}": mb["per_class"][lab][m]
                            for m in ["precision", "recall", "f1"]
                        },
                        delta_f1=round(mb["per_class"][lab]["f1"] - ma["per_class"][lab]["f1"], 4),
                    )
                )
            if split == "test":
                for i in np.flatnonzero((status == "fixed") | (status == "regressed")):
                    case_rows.append(
                        dict(
                            seed=seed,
                            sample_index=int(i),
                            status=status[i],
                            gold=gold[i],
                            direct_prediction=pa[i],
                            hierarchy_prediction=pb[i],
                            question=texts.iloc[i].text_a,
                            answer=texts.iloc[i].text_b,
                        )
                    )
    table("paired_summary.csv", pair_rows)
    table("seed_comparison.csv", seed_rows)
    table("per_class_comparison.csv", class_rows)
    table("paired_case_candidates.csv", case_rows)
    table("checkpoint_reproduction.csv", checks)
    # Simple inference-only charts, kept separate from the original figures.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for seed in C.SEEDS:
        m = results[f"roberta_evasion_boost_s{seed}"]["fine"]
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(9), C.EVASION_LABELS, rotation=40, ha="right")
        ax.set_yticks(range(9), C.EVASION_LABELS)
        for i in range(9):
            for j in range(9):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set(
            xlabel="Predicted strategy",
            ylabel="Gold strategy",
            title=f"Development diagnostics, seed {seed} (n=217)",
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"dev_fine_confusion_s{seed}.png", dpi=160)
        plt.close(fig)
    save(
        "analysis_summary.json",
        dict(
            label_agreement=dict(n=len(data), consistent=int((mapped == data.label).sum())),
            test_fine_labels_available=0,
            development_metrics=results,
            seed_comparison=seed_rows,
            paired_summary=pair_rows,
            checkpoint_checks=checks,
            limitations=[
                "Development data was used for checkpoint selection; diagnostics are not an independent fine-label test.",
                "Three seeds are descriptive, not evidence of statistical significance.",
                "Fine and coarse schemes use different inverse-frequency class weights.",
                "Probability-sum predictions are post-hoc diagnostics; the original decision rule is unchanged.",
            ],
        ),
    )
    for path, digest in inputs.items():
        assert sha256(path) == digest, f"Input changed during analysis: {path}"
    outputs = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    save(
        "manifest.json",
        dict(
            completed_utc=datetime.now(timezone.utc).isoformat(),
            script_sha256=sha256(Path(__file__)),
            experiment=str(root),
            inputs=inputs,
            outputs=outputs,
            runtime=dict(
                python=sys.version,
                torch=torch.__version__,
                device=torch.cuda.get_device_name(0),
                precision="bf16",
            ),
            all_input_hashes_unchanged=True,
            all_checkpoint_scores_reproduced=True,
        ),
    )
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
