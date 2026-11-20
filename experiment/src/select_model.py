"""Select the report's class-level/error-analysis model using development scores only."""

import argparse
import json

import config as C
from protocol import write_json


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,2024,3407")
    args = ap.parse_args()
    seeds = list(dict.fromkeys(int(s) for s in args.seeds.split(",")))
    selected_tfidf = json.loads((C.METRICS / "tfidf_selection.json").read_text(encoding="utf-8"))[
        "selected_model"
    ]
    tags = [
        selected_tfidf,
        "st_all-MiniLM-L6-v2",
        *[f"roberta_clarity_s{s}" for s in seeds],
        "roberta_evasion_boost_s42",
    ]
    scores = {}
    for tag in tags:
        cfg = json.loads((C.METRICS / f"{tag}_config.json").read_text(encoding="utf-8"))
        scores[tag] = cfg.get("best_val_macro_f1", cfg.get("val_macro_f1"))
        if scores[tag] is None or scores[tag] < 0:
            raise ValueError(f"{tag}: missing validation score")
    best = max(scores, key=scores.get)
    write_json(
        C.METRICS / "best_model.json",
        dict(selected_model=best, criterion="validation_macro_f1", scores=scores),
    )
    print("Selected on validation:", best, scores[best])


if __name__ == "__main__":
    main()
