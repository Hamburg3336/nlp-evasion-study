"""Reinfer selected ambiguity-exclusion models; compare primary and sensitivity arms."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    root = a.experiment.resolve()
    controls = a.controls.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NLP_OUTPUT_DIR"] = str(out / "runtime_unused")
    sys.path.insert(0, str(root / "src"))
    import config as C
    from analysis_utils import independent_metrics, read_csv
    from protocol import sha256, write_json
    from train_encoder import PadCollator, PairDataset, set_seed

    workspace = Path(__file__).resolve().parents[1]
    state = json.loads((controls / "training_plan.json").read_text(encoding="utf-8-sig"))
    assert state["status"] == "complete" and len(state["completed"]) == 6
    inputs = {}

    def track(p):
        inputs[str(p)] = sha256(p)
        return p

    track(controls / "training_plan.json")
    for n, h in state["outputs_sha256"].items():
        assert sha256(controls / n) == h, n
    original = pd.read_csv(track(root / "data/processed/qevasion_train.csv"), keep_default_na=False)
    reduced = pd.read_csv(track(controls / "data/qevasion_train.csv"), keep_default_na=False)
    dev = original.loc[original.role == "val"].copy().reset_index(drop=True)
    test = pd.read_csv(track(root / "data/processed/qevasion_test.csv"), keep_default_na=False)
    assert len(original) == 3355 and len(reduced) == 3332
    assert reduced.loc[reduced.role == "val"].reset_index(drop=True).equals(dev)
    removed = original.loc[
        (original.role == "train") & original._raw_idx.isin(state["excluded_fit_raw_indices"])
    ]
    assert len(removed) == 23 and original.drop(index=removed.index).reset_index(drop=True).equals(
        reduced.reset_index(drop=True)
    )
    rows = []
    for seed in [42, 2024, 3407]:
        for weighted in [True, False]:
            name = "weighted" if weighted else "unweighted"
            tag = f"roberta_evasion_ambiguity_{name}_s{seed}"
            cfg = json.loads(
                track(controls / f"metrics/{tag}_config.json").read_text(encoding="utf-8")
            )
            assert (
                cfg["class_weight"] == weighted
                and cfg["data_sha256"] == state["reduced_train_sha256"]
            )
            counts = reduced.loc[reduced.role == "train", "evasion"].value_counts()
            expected = {
                lab: 3115 / (9 * counts[lab]) if weighted else 1.0 for lab in C.EVASION_LABELS
            }
            assert all(abs(cfg["class_weights"][lab] - w) < 1e-12 for lab, w in expected.items())
            ck = controls / "models" / tag
            for n in [
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ]:
                track(ck / n)
            set_seed(seed)
            tok = AutoTokenizer.from_pretrained(ck, local_files_only=True)
            model = (
                AutoModelForSequenceClassification.from_pretrained(ck, local_files_only=True)
                .cuda()
                .eval()
            )
            assert [model.config.id2label[i] for i in range(9)] == C.EVASION_LABELS
            evaluated = {}
            for split, frame in [("val", dev), ("test", test)]:
                loader = DataLoader(
                    PairDataset(frame, tok, 512),
                    batch_size=16,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=PadCollator(tok.pad_token_id),
                )
                chunks = []
                with torch.inference_mode():
                    for batch, _ in loader:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(**{k: v.cuda() for k, v in batch.items()}).logits
                        chunks.append(logits.float().softmax(-1).cpu().numpy())
                prob = np.vstack(chunks)
                assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-5)
                fine = np.array(C.EVASION_LABELS)[prob.argmax(1)]
                coarse = [C.EVASION_TO_CLARITY[x] for x in fine]
                metrics = independent_metrics(frame.label.tolist(), coarse, C.CLARITY_LABELS)
                saved_frame = frame[["_raw_idx", "text_a", "text_b", "label", "evasion"]].copy()
                saved_frame.insert(0, "sample_index", np.arange(len(frame)))
                saved_frame["fine_prediction"] = fine
                saved_frame["prediction"] = coarse
                for i, lab in enumerate(C.EVASION_LABELS):
                    saved_frame[f"prob_{lab}"] = prob[:, i]
                saved_frame.to_csv(out / f"{tag}_{split}.csv", index=False, encoding="utf-8-sig")
                if split == "val":
                    assert metrics["macro_f1"] == cfg["best_val_macro_f1"]
                    fm = independent_metrics(frame.evasion.tolist(), fine, C.EVASION_LABELS)
                    selected = max(cfg["train_log"], key=lambda r: r["val_macro_f1"])
                    assert fm["macro_f1"] == selected["val_evasion_macro_f1"]
                    clean = frame._raw_idx.to_numpy() != 1485
                    cleanfine = independent_metrics(
                        frame.evasion.to_numpy()[clean], fine[clean], C.EVASION_LABELS
                    )["macro_f1"]
                    evaluated.update(
                        dev_coarse=metrics["macro_f1"],
                        dev_fine=fm["macro_f1"],
                        dev_fine_exclude_ambiguous=cleanfine,
                        selected_epoch=selected["epoch"],
                    )
                else:
                    saved = json.loads(
                        track(controls / f"metrics/{tag}.json").read_text(encoding="utf-8")
                    )
                    for key, value in metrics.items():
                        assert saved[key] == value, (tag, key)
                    pred = read_csv(track(controls / f"predictions/{tag}.csv"))
                    for i, row in enumerate(pred):
                        assert (
                            row["prediction"] == coarse[i]
                            and row["gold"] == frame.iloc[i].label
                            and row["text_a"] == frame.iloc[i].text_a
                        )
                        for lab in C.CLARITY_LABELS:
                            total = sum(
                                float(prob[i, j])
                                for j, f in enumerate(C.EVASION_LABELS)
                                if C.EVASION_TO_CLARITY[f] == lab
                            )
                            assert abs(float(row[f"prob_{lab}"]) - total) < 1e-6
                    evaluated.update(test_accuracy=metrics["accuracy"], test_f1=metrics["macro_f1"])
            if weighted:
                baseline = root / "outputs/revised"
                btag = f"roberta_evasion_boost_s{seed}"
                bdev = workspace / f"补充实验/层级分析_20261116/dev_{btag}.csv"
            else:
                baseline = workspace / "补充实验/权重消融_20261116"
                btag = f"roberta_evasion_boost_unweighted_s{seed}"
                bdev = workspace / f"补充实验/权重消融开发集_20261116/{btag}.csv"
            baseline_rows = read_csv(track(baseline / f"predictions/{btag}.csv"))
            bm = independent_metrics(
                [r["gold"] for r in baseline_rows],
                [r["prediction"] for r in baseline_rows],
                C.CLARITY_LABELS,
            )
            bp = read_csv(track(bdev))
            bf = independent_metrics(
                [r["evasion"] for r in bp], [r["prediction"] for r in bp], C.EVASION_LABELS
            )
            row = dict(
                seed=seed,
                weighting=name,
                **evaluated,
                baseline_test_f1=bm["macro_f1"],
                baseline_dev_fine=bf["macro_f1"],
                test_delta=round(evaluated["test_f1"] - bm["macro_f1"], 4),
            )
            rows.append(row)
            print(row, flush=True)
            del model, loader
            gc.collect()
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / "per_seed_comparison.csv", index=False, encoding="utf-8-sig")
    summary = []
    for name in ["weighted", "unweighted"]:
        group = [r for r in rows if r["weighting"] == name]
        summary.append(
            dict(
                weighting=name,
                **{
                    f"{key}_{stat}": float(
                        getattr(np, stat)(
                            [r[key] for r in group], **({"ddof": 1} if stat == "std" else {})
                        )
                    )
                    for key in [
                        "baseline_test_f1",
                        "test_f1",
                        "baseline_dev_fine",
                        "dev_fine",
                        "test_delta",
                    ]
                    for stat in ["mean", "std"]
                },
            )
        )
    write_json(
        out / "analysis.json",
        dict(
            passed=True,
            per_seed=rows,
            summary=summary,
            fit_removed=23,
            fit_retained=3115,
            development_unchanged=217,
            test_unchanged=308,
            caveats=[
                "Same development used for checkpoint selection; fine development scores are diagnostic.",
                "Removing records also changes input distribution, shuffle, step count and recomputed weights; not an isolated causal effect of label ambiguity.",
                "Three seeds are descriptive; do not use test scores to replace the primary analysis or choose the favorable protocol.",
            ],
        ),
    )
    for p, h in inputs.items():
        assert sha256(p) == h
    write_json(
        out / "verification.json",
        dict(
            passed=True,
            six_checkpoint_dev_scores_reproduced=True,
            six_test_predictions_and_probabilities_reproduced=True,
            original_data_unchanged=True,
            script_sha256=sha256(Path(__file__)),
            inputs_sha256=inputs,
            outputs_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
