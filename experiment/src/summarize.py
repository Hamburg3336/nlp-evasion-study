"""仅汇总明确命名和指定seed的修订实验，不扫描混入历史结果。"""

import argparse
import json

import config as C
import numpy as np
import pandas as pd
from protocol import write_json


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,2024,3407")
    args = ap.parse_args()
    seeds = list(dict.fromkeys(map(int, args.seeds.split(","))))
    groups = [("多数类基线", ["majority"])]
    groups += [(cfg["name"], [cfg["name"] + "_s42"]) for cfg in C.TFIDF_CONFIGS]
    groups += [
        ("Sentence-Transformer + LR", ["st_all-MiniLM-L6-v2"]),
        ("RoBERTa-base", [f"roberta_clarity_s{s}" for s in seeds]),
        ("RoBERTa evasion-boost", [f"roberta_evasion_boost_s{s}" for s in seeds]),
    ]
    rows, inventory = [], {}
    for name, tags in groups:
        records = []
        for tag in tags:
            p = C.METRICS / f"{tag}.json"
            if not p.is_file():
                raise FileNotFoundError(f"缺少指定实验，拒绝少报种子: {p}")
            records.append(json.loads(p.read_text(encoding="utf-8")))
        if name == "RoBERTa-base":
            configs = [
                json.loads((C.METRICS / f"{tag}_config.json").read_text(encoding="utf-8"))
                for tag in tags
            ]
            for field in (
                "run_id",
                "data_sha256",
                "batch",
                "max_len",
                "lr",
                "trunc",
                "train_n",
                "val_n",
            ):
                values = [cfg.get(field) for cfg in configs]
                if values[0] is None or any(value != values[0] for value in values):
                    raise ValueError(f"三种子运行来源/协议不同，拒绝混合汇总: {field}")
        if name == "RoBERTa evasion-boost":
            configs = [
                json.loads((C.METRICS / f"{tag}_config.json").read_text(encoding="utf-8"))
                for tag in tags
            ]
            for field in ("data_sha256", "batch", "max_len", "lr", "trunc", "train_n", "val_n"):
                if any(cfg[field] != configs[0][field] for cfg in configs):
                    raise ValueError(f"evasion协议不同: {field}")
        row = {
            "模型": name,
            "n_seeds": len(tags),
            "seeds": ",".join(str(s) for s in seeds) if len(tags) > 1 else "42",
        }
        for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            v = np.array([r[field] for r in records])
            row[field] = float(v.mean())
            row[field + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else None
        rows.append(row)
        inventory[name] = tags
    pd.DataFrame(rows).to_csv(C.METRICS / "table3_final.csv", index=False, encoding="utf-8-sig")
    write_json(C.METRICS / "summary_inventory.json", inventory)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
