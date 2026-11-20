"""Inspect fine-label ambiguity before coarse-label deduplication; read only."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiment"


def main():
    sys.path.insert(0, str(EXP / "src"))
    from clean_data import deep_clean

    source = EXP / "data/raw/train-00000-of-00001.parquet"
    data = pd.read_parquet(source)
    processed = pd.read_csv(EXP / "data/processed/qevasion_train.csv")
    groups = {}
    for i, row in data.iterrows():
        a, b = (str(row[field]).strip() for field in ["question", "interview_answer"])
        key = (deep_clean(a), deep_clean(b))
        groups.setdefault(key, []).append(
            dict(
                raw_index=int(i),
                coarse=str(row["clarity_label"]),
                fine=str(row["evasion_label"]),
                raw_pair=(a, b),
            )
        )
    coarse_conflicts, fine_only = [], []
    for rows in groups.values():
        if len({r["coarse"] for r in rows}) > 1:
            coarse_conflicts.append(rows)
        elif len({r["fine"] for r in rows}) > 1:
            fine_only.append(rows)
    details = []
    for rows in fine_only:
        ids = [r["raw_index"] for r in rows]
        kept = processed[processed["_raw_idx"].isin(ids)]
        details.append(
            dict(
                records=[{k: v for k, v in r.items() if k != "raw_pair"} for r in rows],
                already_identical_before_deep_clean=len({r["raw_pair"] for r in rows}) == 1,
                retained=kept[["_raw_idx", "label", "evasion", "role"]].to_dict("records"),
            )
        )
    result = dict(
        date="2026-11-16",
        raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        coarse_conflicting_groups=len(coarse_conflicts),
        fine_only_conflicting_groups=len(fine_only),
        fine_only_raw_rows=sum(len(r) for r in fine_only),
        retained_roles=dict(Counter(r["role"] for d in details for r in d["retained"])),
        identical_before_deep_clean=sum(d["already_identical_before_deep_clean"] for d in details),
        details=details,
        interpretation="Coarse-conflict removal is implemented; fine-only conflicts retain first row. No relabeling or data change performed.",
    )
    out = ROOT / "审核记录"
    out.mkdir(exist_ok=True)
    (out / "原始细标签冲突审核_20261116.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "details"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
