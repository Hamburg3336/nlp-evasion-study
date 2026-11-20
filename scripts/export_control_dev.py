"""Export development diagnostics for the six selected unweighted checkpoints."""

import argparse
import gc
import json
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
    args = ap.parse_args()
    root, controls, out = args.experiment.resolve(), args.controls.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "src"))
    import config as C
    from evaluate import compute_metrics
    from protocol import sha256, write_json
    from train_encoder import PadCollator, PairDataset, set_seed

    state = json.loads((controls / "training_plan.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    tr_path = root / "data/processed/qevasion_train.csv"
    assert sha256(tr_path) == state["protocol"]["data_sha256"]
    tr = pd.read_csv(tr_path, keep_default_na=False)
    tr["processed_row"] = np.arange(len(tr))
    val = tr.loc[tr.role == "val"].copy().reset_index(drop=True)
    inputs = {str(tr_path): sha256(tr_path)}
    rows = []
    for seed in C.SEEDS:
        for target in ["clarity", "evasion_boost"]:
            tag = f"roberta_{target}_unweighted_s{seed}"
            cfg_path = controls / f"metrics/{tag}_config.json"
            inputs[str(cfg_path)] = sha256(cfg_path)
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert cfg["run_id"] == state["run_id"] and cfg["class_weight"] is False
            ck = controls / "models" / tag
            for name in [
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ]:
                inputs[str(ck / name)] = sha256(ck / name)
            set_seed(seed)
            tok = AutoTokenizer.from_pretrained(ck, local_files_only=True)
            model = (
                AutoModelForSequenceClassification.from_pretrained(ck, local_files_only=True)
                .cuda()
                .eval()
            )
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            assert labels == cfg["labels"]
            loader = DataLoader(
                PairDataset(val, tok, cfg["max_len"], trunc_mode=cfg["trunc"]),
                batch_size=cfg["batch"] * 2,
                shuffle=False,
                num_workers=0,
                collate_fn=PadCollator(tok.pad_token_id),
            )
            probs = []
            with torch.inference_mode():
                for batch, _ in loader:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(**{k: v.cuda() for k, v in batch.items()}).logits
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
            fine = None
            if target == "evasion_boost":
                frame["mapped_prediction"] = [C.EVASION_TO_CLARITY[x] for x in prediction]
                coarse = compute_metrics(
                    val.label, frame.mapped_prediction, labels=C.CLARITY_LABELS
                )
                fine = compute_metrics(val.evasion, prediction, labels=C.EVASION_LABELS)
                selected = max(cfg["train_log"], key=lambda x: x["val_macro_f1"])
                assert fine["macro_f1"] == selected["val_evasion_macro_f1"]
            else:
                coarse = compute_metrics(val.label, prediction, labels=C.CLARITY_LABELS)
            assert coarse["macro_f1"] == cfg["best_val_macro_f1"], (
                tag,
                coarse["macro_f1"],
                cfg["best_val_macro_f1"],
            )
            frame.to_csv(out / f"{tag}.csv", index=False, encoding="utf-8-sig")
            write_json(
                out / f"{tag}_metrics.json",
                dict(
                    coarse={k: v for k, v in coarse.items() if k != "report"},
                    fine={k: v for k, v in fine.items() if k != "report"} if fine else None,
                ),
            )
            rows.append(
                dict(
                    tag=tag,
                    seed=seed,
                    target=target,
                    coarse_macro_f1=coarse["macro_f1"],
                    fine_macro_f1=fine["macro_f1"] if fine else None,
                    checkpoint_score_reproduced=True,
                )
            )
            print(rows[-1], flush=True)
            del model, loader
            gc.collect()
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / "development_summary.csv", index=False, encoding="utf-8-sig")
    for path, h in inputs.items():
        assert sha256(path) == h
    write_json(
        out / "manifest.json",
        dict(
            passed=True,
            inputs=inputs,
            script_sha256=sha256(Path(__file__)),
            outputs={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
            note="Development diagnostics at already selected coarse-F1 checkpoints; no new selection.",
        ),
    )


if __name__ == "__main__":
    main()
