"""补充对照：复用已审定协议，断点继续时只跳过来源校验通过的任务。"""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import config as C
from protocol import sha256, write_json


def main():
    C.ensure_directories()
    path = C.OUT / "supplement_run.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = dict(run_id=uuid.uuid4().hex, status="running", completed=[])
    jobs = []
    for mode in ["frozen", "finetuned"]:
        tag = f"control_qnli_{mode}"
        flags = ["--transfer-csv", str(C.DATA_PROC / "transfer_qnli_revised_unified.csv")]
        if mode == "frozen":
            flags += ["--frozen"]
        jobs.append((tag, flags))
    for seed in [2024, 3407]:
        jobs.append(
            (f"roberta_evasion_boost_s{seed}", ["--target", "evasion", "--seed", str(seed)])
        )
    state["status"] = "running"
    write_json(path, state)
    for tag, flags in jobs:
        if tag in state["completed"]:
            cfg = json.loads((C.METRICS / f"{tag}_config.json").read_text(encoding="utf-8"))
            if cfg["run_id"] != state["run_id"]:
                raise ValueError("补充运行来源不匹配")
            expected = C.DATA_PROC / (
                "transfer_qnli_revised_unified.csv"
                if tag.startswith("control")
                else "qevasion_train.csv"
            )
            if cfg["data_sha256"] != sha256(expected):
                raise ValueError("断点产物数据已改变")
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("train_encoder.py")),
            "--model",
            str(C.ROOT / "models/roberta-base"),
            "--tag",
            tag,
            "--run-id",
            state["run_id"],
            *flags,
        ]
        state["active_model"] = tag
        write_json(path, state)
        print("RUN", tag, flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode:
            state["status"] = "failed"
            write_json(path, state)
            return result.returncode
        cfg = json.loads((C.METRICS / f"{tag}_config.json").read_text(encoding="utf-8"))
        expected = C.DATA_PROC / (
            "transfer_qnli_revised_unified.csv"
            if tag.startswith("control")
            else "qevasion_train.csv"
        )
        if cfg["run_id"] != state["run_id"] or cfg["data_sha256"] != sha256(expected):
            raise ValueError("本次产物来源或数据不匹配")
        state["completed"].append(tag)
        write_json(path, state)
    state.update(status="complete", active_model=None)
    write_json(path, state)
    print("SUPPLEMENT TRAINING COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
