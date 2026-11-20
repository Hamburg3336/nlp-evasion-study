"""Exclude 23 ambiguous fit records; keep development/test and original protocol."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NLP_OUTPUT_DIR"] = str(out)
    sys.path.insert(0, str(root / "src"))
    import pandas as pd
    import torch
    from protocol import sha256, write_json

    assert torch.cuda.is_available()
    raw = Path(__file__).resolve().parents[1] / "审核记录/原始细标签冲突审核_20261116.json"
    audit = json.loads(raw.read_text(encoding="utf-8-sig"))
    original = root / "data/processed/qevasion_train.csv"
    test = root / "data/processed/qevasion_test.csv"
    df = pd.read_csv(original, keep_default_na=False)
    ids = [r["_raw_idx"] for d in audit["details"] for r in d["retained"] if r["role"] == "train"]
    mask = (df.role == "train") & df._raw_idx.isin(ids)
    assert len(ids) == 23 and mask.sum() == 23
    reduced = df.loc[~mask].copy()
    assert (reduced.role == "train").sum() == 3115 and (reduced.role == "val").sum() == 217
    assert reduced.loc[reduced.role == "val"].equals(df.loc[df.role == "val"])
    data = out / "data"
    data.mkdir()
    reduced.to_csv(data / "qevasion_train.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(test, data / "qevasion_test.csv")
    df.loc[mask].to_csv(out / "excluded_fit_records.csv", index=False, encoding="utf-8-sig")
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    for p in (root / "src").glob("*.py"):
        shutil.copy2(p, snapshots / p.name)
    state = dict(
        status="planned",
        run_id=uuid.uuid4().hex,
        seeds=[42, 2024, 3407],
        purpose="Fit-only ambiguity exclusion sensitivity, not relabeling or replacement of the primary protocol.",
        excluded_fit_raw_indices=ids,
        fit_n=3115,
        val_n=217,
        test_n=308,
        development_unchanged=True,
        test_unchanged=True,
        refit_class_weights=True,
        protocol=dict(
            batch=8,
            max_len=512,
            lr=2e-5,
            epochs=5,
            patience=2,
            warmup_ratio=0.1,
            trunc="longest_first",
            workers=2,
            precision="bf16",
            selection="development coarse Macro-F1",
        ),
        original_train_sha256=sha256(original),
        test_sha256=sha256(test),
        reduced_train_sha256=sha256(data / "qevasion_train.csv"),
        audit_sha256=sha256(raw),
        script_sha256=sha256(Path(__file__)),
        source_sha256={p.name: sha256(p) for p in snapshots.glob("*.py")},
        pretrained_sha256={
            p.name: sha256(p) for p in (root / "models/roberta-base").iterdir() if p.is_file()
        },
        completed=[],
    )
    write_json(out / "training_plan.json", state)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for seed in state["seeds"]:
        for weighted in [True, False]:
            tag = f"roberta_evasion_ambiguity_{'weighted' if weighted else 'unweighted'}_s{seed}"
            cmd = [
                sys.executable,
                str(root / "src/train_encoder.py"),
                "--model",
                str(root / "models/roberta-base"),
                "--data-dir",
                str(data),
                "--target",
                "evasion",
                "--seed",
                str(seed),
                "--batch",
                "8",
                "--max-len",
                "512",
                "--lr",
                "2e-5",
                "--epochs",
                "5",
                "--trunc",
                "longest_first",
                "--workers",
                "2",
                "--tag",
                tag,
                "--run-id",
                state["run_id"],
            ]
            if not weighted:
                cmd += ["--no-class-weight"]
            state.update(status="running", active_model=tag, command=cmd)
            write_json(out / "training_plan.json", state)
            print("RUN", tag, flush=True)
            with (out / f"{tag}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode:
                state.update(status="failed", exit_code=result.returncode)
                write_json(out / "training_plan.json", state)
                return result.returncode
            cfg = json.loads((out / f"metrics/{tag}_config.json").read_text(encoding="utf-8"))
            assert (
                cfg["data_sha256"] == state["reduced_train_sha256"]
                and cfg["class_weight"] == weighted
            )
            assert cfg["train_n"] == 3115 and cfg["val_n"] == 217 and cfg["test_n"] == 308
            state["completed"].append(tag)
            write_json(out / "training_plan.json", state)
            print(
                "DONE",
                tag,
                "best_dev",
                cfg["best_val_macro_f1"],
                "seconds",
                cfg["elapsed_sec"],
                flush=True,
            )
    assert (
        sha256(original) == state["original_train_sha256"] and sha256(test) == state["test_sha256"]
    )
    state.update(status="complete", active_model=None)
    state["outputs_sha256"] = {
        p.relative_to(out).as_posix(): sha256(p)
        for p in out.rglob("*")
        if p.is_file() and p.name != "training_plan.json"
    }
    write_json(out / "training_plan.json", state)
    print("ALL SIX AMBIGUITY CONTROLS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
