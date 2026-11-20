"""基线 A：多数类基线 + TF-IDF + Logistic Regression（模板必做）。

要点（对应模板要求）：
  - question 与 answer 通过显式分隔符拼接
  - 比较 unigram 与 unigram+bigram
  - 记录是否使用 class_weight
  - 所有配置共享同一数据划分、标签映射与主指标

用法:
    python src/train_tfidf.py
    python src/train_tfidf.py --seeds 42,2024,3407     # 多随机种子，给均值±标准差
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import numpy as np
import pandas as pd
from evaluate import compute_metrics, evaluate_and_save, majority_baseline
from protocol import sha256, write_json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_text(df: pd.DataFrame) -> pd.Series:
    """问题与回答显式分隔拼接。保证两段文本边界清晰，便于 TF-IDF 学到「回答段」的特征。"""
    return (df["text_a"].fillna("") + C.SEP + df["text_b"].fillna("")).astype(str)


def make_pipeline(ngram_range, class_weight, seed) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=C.TFIDF_MAX_FEATURES,
                    min_df=C.TFIDF_MIN_DF,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def run_once(cfg, tr, te, seed: int, save: bool):
    name = f"{cfg['name']}_s{seed}"
    Xtr, Xte = build_text(tr), build_text(te)
    ytr, yte = tr["label"].values, te["label"].values

    pipe = make_pipeline(cfg["ngram_range"], cfg["class_weight"], seed)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)

    # 保证概率列顺序与 CLARITY_LABELS 一致
    cls = list(pipe.named_steps["clf"].classes_)
    order = [cls.index(c) for c in C.CLARITY_LABELS if c in cls]
    proba = proba[:, order]

    if not save:
        m = compute_metrics(yte, pred)
        return m["macro_f1"], m
    m = evaluate_and_save(
        name, yte, pred, proba, class_names=C.CLARITY_LABELS, texts=te["text_a"].values
    )
    return m["macro_f1"], m


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(C.SEED), help="逗号分隔，如 42,2024,3407")
    ap.add_argument("--no-save-pred", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    all_tr = pd.read_csv(C.DATA_PROC / "qevasion_train.csv")
    tr = all_tr[all_tr["role"] == "train"]
    val = all_tr[all_tr["role"] == "val"]
    te = pd.read_csv(C.DATA_PROC / "qevasion_test.csv")
    ytr, yte = tr["label"].values, te["label"].values

    print("=" * 62)
    print("基线 A：多数类 + TF-IDF / Logistic Regression")
    print("=" * 62)

    results = {}

    # ---- 多数类基线（必报，Macro-F1 论证的对照）
    maj = majority_baseline(ytr, yte)
    maj_proba = np.array([[float(label == p) for label in C.CLARITY_LABELS] for p in maj])
    m_maj = evaluate_and_save(
        "majority",
        yte,
        maj,
        maj_proba,
        class_names=C.CLARITY_LABELS,
        texts=te["text_a"].values,
        save_pred=not args.no_save_pred,
    )
    results["多数类基线"] = m_maj

    # ---- TF-IDF 各配置
    multi = {}
    val_scores = {}
    for cfg in C.TFIDF_CONFIGS:
        dev_pipe = make_pipeline(cfg["ngram_range"], cfg["class_weight"], seeds[0])
        dev_pipe.fit(build_text(tr), tr["label"])
        val_scores[cfg["name"]] = compute_metrics(val["label"], dev_pipe.predict(build_text(val)))[
            "macro_f1"
        ]
        write_json(
            C.METRICS / f"{cfg['name']}_s{seeds[0]}_config.json",
            {
                **cfg,
                "seed": seeds[0],
                "train_n": len(tr),
                "val_n": len(val),
                "test_n": len(te),
                "analyzer": "word",
                "actual_features": len(dev_pipe["tfidf"].vocabulary_),
                "val_macro_f1": val_scores[cfg["name"]],
                "max_features": C.TFIDF_MAX_FEATURES,
                "data_sha256": sha256(C.DATA_PROC / "qevasion_train.csv"),
            },
        )
        scores = []
        last = None
        for seed in seeds:
            f1, m = run_once(cfg, tr, te, seed, save=(seed == seeds[0] and not args.no_save_pred))
            if seed == seeds[0]:
                last = m
            elif m is not None:
                pass
            scores.append(f1)
        # 多 seed 时，重新跑一遍完整评测以落盘（用第一个 seed 的结果）
        if len(seeds) > 1:
            _, last = run_once(cfg, tr, te, seeds[0], save=not args.no_save_pred)
        results[cfg["name"]] = last
        multi[cfg["name"]] = {
            "macro_f1_mean": round(float(np.mean(scores)), 4),
            "macro_f1_std": round(float(np.std(scores)), 4),
            "n_seeds": len(seeds),
        }
        print(f"  -> {cfg['name']}: macro_f1 = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ---- 汇总成表3 用的 csv
    rows = []
    for name, m in results.items():
        if m is None:
            continue
        rows.append(
            {
                "模型": name,
                "Accuracy": m["accuracy"],
                "Macro-P": m["macro_precision"],
                "Macro-R": m["macro_recall"],
                "Macro-F1": m["macro_f1"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS / "table3_tfidf.csv", index=False, encoding="utf-8-sig")
    with open(C.METRICS / "tfidf_multiseed.json", "w", encoding="utf-8") as f:
        json.dump(multi, f, ensure_ascii=False, indent=2)
    best = max(val_scores, key=val_scores.get)
    write_json(
        C.METRICS / "tfidf_selection.json",
        {
            "criterion": "val_macro_f1",
            "validation_scores": val_scores,
            "selected_model": f"{best}_s{seeds[0]}",
            "train_n": len(tr),
            "val_n": len(val),
            "refit_on_val": False,
        },
    )

    print()
    print(df.to_string(index=False))
    print(f"\n已写出: {C.METRICS / 'table3_tfidf.csv'}")


if __name__ == "__main__":
    main()
