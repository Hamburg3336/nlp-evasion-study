"""Audit repeated content, fixed length bins, and actual tokenizer evidence retention.

Diagnostic only: no sample deletion, model selection, or training data mutation.
"""

import argparse
import hashlib
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def word_shingles(text, n):
    words = re.findall(r"\w+", normalize(text))
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--hierarchy", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, hierarchy, out = (
        args.experiment.resolve(),
        args.hierarchy.resolve(),
        args.output.resolve(),
    )
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "src"))
    import config as C
    from evaluate import compute_metrics
    from protocol import sha256, write_json

    inputs = {}

    def track(p):
        inputs[str(p)] = sha256(p)
        return p

    def table(name, rows):
        pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")

    tr = pd.read_csv(track(root / "data/processed/qevasion_train.csv"), keep_default_na=False)
    te = pd.read_csv(track(root / "data/processed/qevasion_test.csv"), keep_default_na=False)
    parts = {}
    for split, frame in [
        ("train", tr[tr.role == "train"]),
        ("dev", tr[tr.role == "val"]),
        ("test", te),
    ]:
        frame = frame.copy().reset_index(drop=True)
        frame["sample_index"] = np.arange(len(frame))
        frame["audit_split"] = split
        parts[split] = frame
    config = dict(
        normalization="Unicode NFKC + whitespace collapse + casefold; retain punctuation for exact comparison",
        near_answer="word 5-shingle set Jaccard >= 0.85, both >=20 word tokens",
        near_question="word 2-shingle set Jaccard >= 0.85, both >=5 word tokens",
        min_jaccard=0.85,
        bins=["0-128", "129-512", "513-1024", "1025+"],
        eval_masks="dev relative to train; test relative to both train and dev",
        interpretation="Post-hoc diagnostics, original official evaluation retained; no performance-based threshold search.",
    )
    write_json(out / "audit_plan.json", config)
    # Repeated texts are deduplicated for the exhaustive sparse Jaccard search.
    overlaps, near_pairs, exact_pairs = [], [], []
    flags = {
        s: pd.DataFrame(
            {
                "sample_index": np.arange(len(f)),
                "question_exact_seen": False,
                "answer_exact_seen": False,
                "question_near_seen": False,
                "answer_near_seen": False,
                "pair_exact_seen": False,
            }
        )
        for s, f in parts.items()
    }
    for col, short, shingle_n, min_words in [
        ("text_a", "question", 2, 5),
        ("text_b", "answer", 5, 20),
    ]:
        unique = list(dict.fromkeys(normalize(t) for f in parts.values() for t in f[col]))
        vocabulary_index = {t: i for i, t in enumerate(unique)}
        shingle_sets = [word_shingles(t, shingle_n) for t in unique]
        cv = CountVectorizer(analyzer=lambda x: x, binary=True, dtype=np.int32)
        all_vectors = cv.fit_transform(shingle_sets)
        lengths = np.asarray(all_vectors.sum(axis=1)).ravel()
        word_counts = np.array([len(re.findall(r"\w+", t)) for t in unique])
        for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            la = [normalize(t) for t in parts[left][col]]
            ra = [normalize(t) for t in parts[right][col]]
            lmap, rmap = {}, {}
            for i, t in enumerate(la):
                lmap.setdefault(t, []).append(i)
            for i, t in enumerate(ra):
                rmap.setdefault(t, []).append(i)
            shared = set(lmap) & set(rmap)
            for text in sorted(shared):
                exact_pairs.append(
                    dict(
                        left_split=left,
                        right_split=right,
                        field=short,
                        normalized_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        left_rows="|".join(map(str, lmap[text])),
                        right_rows="|".join(map(str, rmap[text])),
                        left_n=len(lmap[text]),
                        right_n=len(rmap[text]),
                        excerpt=text[:200],
                    )
                )
                flags[right].loc[rmap[text], f"{short}_exact_seen"] = True
            li = np.array([vocabulary_index[t] for t in lmap])
            ri = np.array([vocabulary_index[t] for t in rmap])
            intersection = (all_vectors[li] @ all_vectors[ri].T).tocoo()
            jaccard = intersection.data / (
                lengths[li[intersection.row]] + lengths[ri[intersection.col]] - intersection.data
            )
            near_left, near_right = set(), set()
            for k in np.flatnonzero(jaccard >= 0.85):
                lidx, ridx = int(li[intersection.row[k]]), int(ri[intersection.col[k]])
                if (
                    unique[lidx] == unique[ridx]
                    or min(word_counts[lidx], word_counts[ridx]) < min_words
                ):
                    continue
                # Independent set calculation checks every accepted sparse match.
                a, b = shingle_sets[lidx], shingle_sets[ridx]
                score = len(a & b) / len(a | b)
                assert abs(score - jaccard[k]) < 1e-12
                lrows, rrows = lmap[unique[lidx]], rmap[unique[ridx]]
                near_left.update(lrows)
                near_right.update(rrows)
                flags[right].loc[rrows, f"{short}_near_seen"] = True
                near_pairs.append(
                    dict(
                        left_split=left,
                        right_split=right,
                        field=short,
                        jaccard=float(score),
                        left_rows="|".join(map(str, lrows)),
                        right_rows="|".join(map(str, rrows)),
                        left_n=len(lrows),
                        right_n=len(rrows),
                        left_excerpt=unique[lidx][:200],
                        right_excerpt=unique[ridx][:200],
                    )
                )
            overlaps.append(
                dict(
                    left_split=left,
                    right_split=right,
                    field=short,
                    exact_shared_texts=len(shared),
                    exact_left_rows=sum(len(lmap[t]) for t in shared),
                    exact_right_rows=sum(len(rmap[t]) for t in shared),
                    near_nonexact_left_rows=len(near_left),
                    near_nonexact_right_rows=len(near_right),
                )
            )
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        seen = {
            (normalize(a), normalize(b)) for a, b in zip(parts[left].text_a, parts[left].text_b)
        }
        for i, r in parts[right].iterrows():
            if (normalize(r.text_a), normalize(r.text_b)) in seen:
                flags[right].loc[i, "pair_exact_seen"] = True
    table("overlap_summary.csv", overlaps)
    # Explicit columns ensure an empty result is still a readable CSV.
    pd.DataFrame(
        exact_pairs,
        columns=[
            "left_split",
            "right_split",
            "field",
            "normalized_sha256",
            "left_rows",
            "right_rows",
            "left_n",
            "right_n",
            "excerpt",
        ],
    ).to_csv(out / "exact_shared_texts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        near_pairs,
        columns=[
            "left_split",
            "right_split",
            "field",
            "jaccard",
            "left_rows",
            "right_rows",
            "left_n",
            "right_n",
            "left_excerpt",
            "right_excerpt",
        ],
    ).to_csv(out / "near_shared_texts.csv", index=False, encoding="utf-8-sig")
    tok = AutoTokenizer.from_pretrained(root / "models/roberta-base", local_files_only=True)
    retained_ends = {}
    length_summary = []
    for split, frame in parts.items():
        details = []
        for i, r in frame.iterrows():
            q = tok(r.text_a, add_special_tokens=False, verbose=False)["input_ids"]
            a = tok(r.text_b, add_special_tokens=False, verbose=False)["input_ids"]
            enc = tok(
                r.text_a,
                r.text_b,
                truncation="longest_first",
                max_length=512,
                return_offsets_mapping=True,
                padding=False,
                verbose=False,
            )
            seq = enc.sequence_ids()
            retained_q = sum(s == 0 for s in seq)
            retained_a = sum(s == 1 for s in seq)
            answer_offsets = [o for s, o in zip(seq, enc["offset_mapping"]) if s == 1]
            end = max((o[1] for o in answer_offsets), default=0)
            retained_ends[(split, i)] = end
            full_pair = len(q) + len(a) + tok.num_special_tokens_to_add(pair=True)
            b = (
                "0-128"
                if len(a) <= 128
                else "129-512"
                if len(a) <= 512
                else "513-1024"
                if len(a) <= 1024
                else "1025+"
            )
            details.append(
                dict(
                    sample_index=i,
                    label=r.label,
                    question_tokens=len(q),
                    answer_tokens=len(a),
                    full_pair_tokens=full_pair,
                    retained_pair_tokens=len(enc["input_ids"]),
                    retained_question_tokens=retained_q,
                    retained_answer_tokens=retained_a,
                    answer_retention_ratio=retained_a / len(a) if a else 1,
                    retained_answer_char_end=end,
                    truncated=full_pair > 512,
                    length_bin=b,
                )
            )
        detail = pd.DataFrame(details)
        merged = flags[split].merge(detail, on="sample_index", validate="one_to_one")
        merged["exact_seen"] = (
            merged.question_exact_seen | merged.answer_exact_seen | merged.pair_exact_seen
        )
        merged["near_or_exact_seen"] = (
            merged.exact_seen | merged.question_near_seen | merged.answer_near_seen
        )
        merged.to_csv(out / f"{split}_sample_audit.csv", index=False, encoding="utf-8-sig")
        flags[split] = merged
        length_summary.append(
            dict(
                split=split,
                n=len(frame),
                truncated_n=int(detail.truncated.sum()),
                truncated_rate=float(detail.truncated.mean()),
                median_answer_tokens=float(detail.answer_tokens.median()),
                p95_answer_tokens=float(detail.answer_tokens.quantile(0.95)),
                exact_seen_n=int(merged.exact_seen.sum()),
                near_or_exact_seen_n=int(merged.near_or_exact_seen.sum()),
                normalized_pair_seen_n=int(merged.pair_exact_seen.sum()),
            )
        )
        print("AUDIT", split, length_summary[-1], flush=True)
    table("length_and_overlap_summary.csv", length_summary)
    # Metric files always use fixed three-label macro averages, including absent labels.
    metric_rows = []
    for split in ["dev", "test"]:
        for seed, target in itertools.product(C.SEEDS, ["clarity", "evasion_boost"]):
            tag = f"roberta_{target}_s{seed}"
            path = (
                hierarchy / f"dev_{tag}.csv"
                if split == "dev"
                else root / f"outputs/revised/predictions/{tag}.csv"
            )
            pred = pd.read_csv(track(path), keep_default_na=False)
            gold = pred.label.to_numpy() if split == "dev" else pred.gold.to_numpy()
            predicted = (
                pred.mapped_prediction.to_numpy()
                if split == "dev" and target == "evasion_boost"
                else pred.prediction.to_numpy()
            )
            assert np.array_equal(gold, parts[split].label)
            audit = flags[split]
            masks = {
                "all": np.ones(len(audit), dtype=bool),
                "exclude_exact_seen": ~audit.exact_seen.to_numpy(),
                "exclude_near_or_exact_seen": ~audit.near_or_exact_seen.to_numpy(),
                "only_near_or_exact_seen": audit.near_or_exact_seen.to_numpy(),
                "truncated": audit.truncated.to_numpy(),
                "not_truncated": ~audit.truncated.to_numpy(),
            }
            for b in config["bins"]:
                masks[f"answer_tokens_{b}"] = (audit.length_bin == b).to_numpy()
            for mask_name, mask in masks.items():
                count = int(mask.sum())
                if not count:
                    metric_rows.append(
                        dict(split=split, seed=seed, model=target, subset=mask_name, n=0)
                    )
                    continue
                metrics = compute_metrics(gold[mask], predicted[mask], labels=C.CLARITY_LABELS)
                metric_rows.append(
                    dict(
                        split=split,
                        seed=seed,
                        model=target,
                        subset=mask_name,
                        n=count,
                        accuracy=metrics["accuracy"],
                        macro_f1=metrics["macro_f1"],
                        **{
                            f"support_{c}": metrics["per_class"][c]["support"]
                            for c in C.CLARITY_LABELS
                        },
                        **{f"f1_{c}": metrics["per_class"][c]["f1"] for c in C.CLARITY_LABELS},
                    )
                )
    table("sensitivity_and_length_metrics.csv", metric_rows)
    # Check evidence spans for both the six new cases and five original cases.
    case_path = track(hierarchy / "reviewed_cases.json")
    original_path = track(root / "data/analysis/error_annotations.json")
    cases = [
        (r["sample_index"], r["evidence"], "hierarchy_pair_case")
        for r in json.loads(case_path.read_text(encoding="utf-8"))["cases"]
    ]
    cases += [
        (r["sample_index"], r["answer_excerpt"], "original_error_case")
        for r in json.loads(original_path.read_text(encoding="utf-8"))["samples"]
    ]
    coverage = []
    for index, excerpt, kind in cases:
        answer = te.iloc[index].text_b
        start = answer.find(excerpt)
        assert start >= 0, (index, excerpt)
        end = start + len(excerpt)
        retained_end = retained_ends[("test", index)]
        coverage.append(
            dict(
                sample_index=index,
                case_set=kind,
                excerpt_char_start=start,
                excerpt_char_end=end,
                retained_answer_char_end=retained_end,
                evidence_fully_retained=end <= retained_end,
                evidence_partly_retained=start < retained_end < end,
                full_pair_tokens=int(flags["test"].iloc[index].full_pair_tokens),
                evidence=excerpt,
            )
        )
    table("case_evidence_coverage.csv", coverage)
    for path, h in inputs.items():
        assert sha256(path) == h, path
    write_json(
        out / "audit_manifest.json",
        dict(
            script_sha256=sha256(Path(__file__)),
            inputs=inputs,
            outputs={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
            input_hashes_unchanged=True,
            near_search="Exhaustive sparse set-Jaccard over distinct texts across each split pair",
            note="Similarity is a diagnostic flag, not proof of label leakage; bins are observational and not a truncation ablation.",
        ),
    )
    print("AUDIT COMPLETED", out, flush=True)


if __name__ == "__main__":
    main()
