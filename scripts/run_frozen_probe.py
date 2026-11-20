"""Fixed eval-mode representations and development-selected convergent LR heads."""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from artifact_io import sha256_file as sha
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-seeds",
        nargs="+",
        default=["42"],
        help="QEvasion source encoder seeds to compare; original RoBERTa is always included.",
    )
    args = parser.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "src"))
    from evaluate import compute_metrics
    from train_encoder import PadCollator, PairDataset, set_seed

    frames = {
        s: pd.read_csv(root / f"data/processed/transfer_qnli_revised_{s}.csv")
        for s in ["train", "val", "test"]
    }
    labels = ["entailment", "not_entailment"]
    sources = dict(original=root / "models/roberta-base")
    for source_seed in args.source_seeds:
        sources[f"qevasion_s{source_seed}"] = (
            root / f"outputs/revised/models/roberta_clarity_s{source_seed}"
        )
    plan = dict(
        status="planned",
        seed=42,
        source_seeds=[str(s) for s in args.source_seeds],
        c_candidates=[0.01, 0.1, 1.0, 10.0],
        feature="base_model.last_hidden_state[:,0,:]; encoder eval; no pooling or classifier",
        tokenizer="same roberta-base tokenizer; longest_first pair length 512",
        extraction_precision="CUDA bf16, cache float32",
        batch=8,
        classifier="train-only StandardScaler + L2 LogisticRegression(lbfgs, class_weight=balanced)",
        max_iter=5000,
        tol=1e-7,
        selection="development Macro-F1; first candidate on ties; no test selection",
        interpretation="Independent fixed-representation linear probe; not a one-factor ablation of the original neural head.",
        data_sha256={
            s: sha(root / f"data/processed/transfer_qnli_revised_{s}.csv") for s in frames
        },
        source_sha256={
            name: {
                p.name: sha(p)
                for p in folder.iterdir()
                if p.name in ["config.json", "model.safetensors"]
            }
            for name, folder in sources.items()
        },
        script_sha256=sha(Path(__file__)),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    save("plan.json", plan)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This registered run uses CUDA bf16; do not silently change precision."
    torch.set_num_threads(4)
    set_seed(42)
    candidates, selected, feature_paths = [], {}, {}
    for origin, folder in sources.items():
        print("EXTRACT", origin, flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(folder).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        tokenizer = AutoTokenizer.from_pretrained(root / "models/roberta-base")
        x = {}
        for split, frame in frames.items():
            loader = DataLoader(
                PairDataset(frame, tokenizer, 512),
                batch_size=8,
                shuffle=False,
                collate_fn=PadCollator(tokenizer.pad_token_id),
                num_workers=0,
            )
            chunks = []
            with torch.inference_mode():
                for i, (batch, _) in enumerate(loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        values = model.base_model(**batch).last_hidden_state[:, 0, :]
                    chunks.append(values.float().cpu().numpy())
                    if i and i % 1000 == 0:
                        print(origin, split, i * 8, "/", len(frame), flush=True)
            x[split] = np.concatenate(chunks)
            path = out / f"{origin}_{split}_features.npy"
            np.save(path, x[split])
            feature_paths[f"{origin}_{split}"] = sha(path)
            assert x[split].shape == (len(frame), 768) and np.isfinite(x[split]).all()
            print("CACHED", origin, split, x[split].shape, flush=True)
        del model
        torch.cuda.empty_cache()
        best = None
        for c in plan["c_candidates"]:
            pipe = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=c,
                            solver="lbfgs",
                            class_weight="balanced",
                            max_iter=5000,
                            tol=1e-7,
                            random_state=42,
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipe.fit(x["train"], frames["train"]["label"])
            assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), (origin, c)
            pred = pipe.predict(x["val"])
            metrics = compute_metrics(frames["val"]["label"], pred, labels=labels)
            record = dict(
                origin=origin,
                C=c,
                val_macro_f1=metrics["macro_f1"],
                iterations=pipe["clf"].n_iter_.tolist(),
                converged=True,
            )
            candidates.append(record)
            pd.DataFrame(
                dict(
                    sample_index=np.arange(len(pred)), gold=frames["val"]["label"], prediction=pred
                )
            ).to_csv(out / f"{origin}_C{c}_development.csv", index=False, encoding="utf-8-sig")
            print("DEVELOPMENT", record, flush=True)
            if best is None or record["val_macro_f1"] > best["val_macro_f1"]:
                best = dict(record, pipeline=pipe)
        # All encoder origins use the same candidate set and fitting procedure.
        selected[origin] = best
        np.savez(
            out / f"{origin}_selected_head.npz",
            mean=best["pipeline"]["scale"].mean_,
            scale=best["pipeline"]["scale"].scale_,
            coef=best["pipeline"]["clf"].coef_,
            intercept=best["pipeline"]["clf"].intercept_,
            classes=best["pipeline"]["clf"].classes_,
        )
        del x
    save(
        "development_selection.json",
        dict(
            candidates=candidates,
            selected={
                name: {k: v for k, v in record.items() if k != "pipeline"}
                for name, record in selected.items()
            },
            test_evaluated=False,
        ),
    )
    results = []
    # Evaluate final labels only after every encoder's selection has been saved.
    for origin, record in selected.items():
        features = np.load(out / f"{origin}_test_features.npy")
        pipe = record["pipeline"]
        pred, proba = pipe.predict(features), pipe.predict_proba(features)
        order = [list(pipe.classes_).index(label) for label in labels]
        proba = proba[:, order]
        assert np.array_equal(np.array(labels)[proba.argmax(1)], pred)
        metrics = compute_metrics(frames["test"]["label"], pred, labels=labels)
        save(f"{origin}_test_metrics.json", {k: v for k, v in metrics.items() if k != "report"})
        rows = pd.DataFrame(
            dict(
                sample_index=np.arange(len(pred)),
                text_a=frames["test"]["text_a"],
                gold=frames["test"]["label"],
                prediction=pred,
            )
        )
        for i, label in enumerate(labels):
            rows["prob_" + label] = proba[:, i]
        rows.to_csv(out / f"{origin}_test_predictions.csv", index=False, encoding="utf-8-sig")
        results.append(
            dict(
                origin=origin,
                C=record["C"],
                val_macro_f1=record["val_macro_f1"],
                test_accuracy=metrics["accuracy"],
                test_macro_f1=metrics["macro_f1"],
            )
        )
    plan.update(
        status="complete",
        feature_sha256=feature_paths,
        results=results,
        test_macro_f1_delta={
            r["origin"]: r["test_macro_f1"] - results[0]["test_macro_f1"] for r in results[1:]
        },
        outputs_sha256={
            p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "plan.json"
        },
    )
    save("plan.json", plan)
    print("COMPLETE", json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
