"""分阶段复现。--stage all为修订实验；任何失败立即返回非零退出码。"""

import argparse
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
sys.path.insert(0, str(ROOT / "src"))
import config as C
from protocol import data_record, write_json


def commands(stage, seeds, run_id):
    if stage == "data":
        return [("clean_data.py", [])]
    if stage == "model":
        specs = [
            ("roberta-base", "roberta-base"),
            ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ]
        return [
            ("fetch_model.py", [repo])
            for folder, repo in specs
            if not (ROOT / "models" / folder / "model.safetensors").is_file()
        ]
    if stage == "encoder":
        jobs = [
            (
                "train_encoder.py",
                [
                    "--model",
                    str(ROOT / "models/roberta-base"),
                    "--seed",
                    str(seed),
                    "--tag",
                    f"roberta_clarity_s{seed}",
                    "--run-id",
                    run_id,
                ],
            )
            for seed in seeds
        ]
        jobs.append(
            (
                "train_encoder.py",
                [
                    "--model",
                    str(ROOT / "models/roberta-base"),
                    "--target",
                    "evasion",
                    "--tag",
                    "roberta_evasion_boost_s42",
                    "--run-id",
                    run_id,
                ],
            )
        )
        return jobs
    if stage == "tfidf":
        return [("train_tfidf.py", ["--seeds", "42"])]
    if stage == "st":
        return [("train_st.py", ["--model", str(ROOT / "models/all-MiniLM-L6-v2")])]
    if stage == "transfer":
        return [("transfer.py", ["--target", "qnli", "--with-encoder", "--run-id", run_id])]
    if stage == "supplement":
        return [("run_supplement.py", []), ("supplement_analysis.py", [])]
    if stage == "error":
        return [
            ("select_model.py", ["--seeds", ",".join(map(str, seeds))]),
            ("error_analysis.py", ["--model", "auto"]),
        ]
    if stage == "error_light":
        return [("error_analysis.py", ["--model", "tfidf_uni_bi_bal_s42"])]
    if stage == "summary":
        return [
            ("summarize.py", ["--seeds", ",".join(map(str, seeds))]),
            ("plot_training.py", []),
            ("verify_artifacts.py", []),
            ("plot_run_output.py", []),
        ]
    return [("eda.py", [])]


ALL = [
    "data",
    "eda",
    "tfidf",
    "model",
    "st",
    "encoder",
    "transfer",
    "supplement",
    "error",
    "summary",
]
LIGHT = ["data", "eda", "tfidf", "error_light"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="light", choices=["all", "light", *ALL, "error_light"])
    ap.add_argument("--seeds", default="42,2024,3407")
    ap.add_argument("--dry-run", action="store_true", help="列出命令，不下载、不训练")
    args = ap.parse_args(argv)
    if not args.dry_run:
        C.ensure_directories()
    seeds = list(dict.fromkeys(int(s) for s in args.seeds.split(",")))
    if 42 not in seeds:
        ap.error("完整流程需要seed42作为表4与迁移的固定源模型")
    stages = ALL if args.stage == "all" else LIGHT if args.stage == "light" else [args.stage]
    run_id = uuid.uuid4().hex
    manifest = {
        "run_id": run_id,
        "stage": args.stage,
        "seeds": seeds,
        "python": sys.executable,
        "status": "running",
        "commands": [],
    }
    path = C.OUT / f"run_{run_id}.json"
    started = time.time()
    for stage in stages:
        for script, flags in commands(stage, seeds, run_id):
            cmd = [sys.executable, str(ROOT / "src" / script), *flags]
            print(f"[{stage}] {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            write_json(path, manifest)
            result = subprocess.run(cmd, cwd=ROOT)
            manifest["commands"].append(
                {"stage": stage, "argv": cmd, "returncode": result.returncode}
            )
            if result.returncode:
                manifest.update(
                    status="failed", failed_stage=stage, elapsed_sec=round(time.time() - started, 1)
                )
                write_json(path, manifest)
                return result.returncode
    if not args.dry_run:
        manifest.update(
            status="complete",
            elapsed_sec=round(time.time() - started, 1),
            artifacts_sha256=data_record([*C.METRICS.glob("*.json"), *C.METRICS.glob("*.csv")]),
        )
        write_json(path, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
