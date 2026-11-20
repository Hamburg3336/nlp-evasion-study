"""Run six unweighted controls with the original data and training protocol."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    sys.path.insert(0, str(root / "src"))
    from protocol import sha256, write_json

    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "training_plan.json"
    data = root / "data/processed/qevasion_train.csv"
    model = root / "models/roberta-base"
    train_script = root / "src/train_encoder.py"
    protocol = dict(
        batch=8,
        max_len=512,
        lr=2e-5,
        epochs=5,
        patience=2,
        warmup_ratio=0.1,
        trunc="longest_first",
        workers=2,
        class_weight=False,
        precision="bf16",
        selection="development coarse Macro-F1",
        seeds=[42, 2024, 3407],
        data_sha256=sha256(data),
        test_sha256=sha256(data.with_name("qevasion_test.csv")),
        script_sha256=sha256(train_script),
        source_code_sha256={p.name: sha256(p) for p in (root / "src").glob("*.py")},
        pretrained_sha256={p.name: sha256(p) for p in model.iterdir() if p.is_file()},
    )
    if plan_path.exists():
        state = json.loads(plan_path.read_text(encoding="utf-8"))
        if state["protocol"] != protocol:
            raise ValueError("Protocol or source changed; use a new output directory")
    else:
        state = dict(
            run_id=uuid.uuid4().hex,
            created_utc=datetime.now(timezone.utc).isoformat(),
            purpose="2x2 target granularity by class weighting; reuse weighted arms, train unweighted arms.",
            protocol=protocol,
            completed=[],
            status="planned",
        )
    env = dict(os.environ, NLP_OUTPUT_DIR=str(out))
    for seed in protocol["seeds"]:
        for target in ["clarity", "evasion"]:
            tag = f"roberta_{'evasion_boost' if target == 'evasion' else target}_unweighted_s{seed}"
            config_path = out / f"metrics/{tag}_config.json"
            if tag in state["completed"]:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                if (
                    cfg["run_id"] != state["run_id"]
                    or cfg["data_sha256"] != protocol["data_sha256"]
                ):
                    raise ValueError(f"Invalid resumed artifact: {tag}")
                for name, value in state["completed_hashes"][tag].items():
                    if sha256(out / name) != value:
                        raise ValueError(f"Resumed artifact changed: {name}")
                continue
            cmd = [
                sys.executable,
                str(train_script),
                "--model",
                str(model),
                "--target",
                target,
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
                "--no-class-weight",
                "--tag",
                tag,
                "--run-id",
                state["run_id"],
            ]
            state.update(status="running", active_model=tag, command=cmd)
            write_json(plan_path, state)
            print("RUN", tag, flush=True)
            # Each model has a separate log and a fresh process/CUDA context.
            with (out / f"{tag}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode:
                state.update(status="failed", exit_code=result.returncode)
                write_json(plan_path, state)
                return result.returncode
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if cfg["run_id"] != state["run_id"] or cfg["data_sha256"] != protocol["data_sha256"]:
                raise ValueError(f"Invalid new artifact: {tag}")
            assert cfg["class_weight"] is False and all(
                w == 1 for w in cfg["class_weights"].values()
            )
            state["completed"].append(tag)
            paths = [
                f"metrics/{tag}_config.json",
                f"metrics/{tag}.json",
                f"predictions/{tag}.csv",
                f"models/{tag}/model.safetensors",
                f"models/{tag}/config.json",
            ]
            state.setdefault("completed_hashes", {})[tag] = {p: sha256(out / p) for p in paths}
            write_json(plan_path, state)
            print("DONE", tag, "best_val", cfg["best_val_macro_f1"], flush=True)
    state.update(
        status="complete", active_model=None, completed_utc=datetime.now(timezone.utc).isoformat()
    )
    write_json(plan_path, state)
    print("ALL SIX WEIGHT CONTROLS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
