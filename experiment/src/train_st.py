"""基线 B-1：Sentence-Transformer 编码 + 逻辑回归。

思路：用预训练句向量分别编码 question 与 answer，再把句对特征
[u, v, |u-v|, u*v] 拼接后送进逻辑回归。相比 BERT 句对微调，
这种「冻结编码器 + 浅层分类器」的方式显存占用极低，适合做快速对照。

用法:
    python src/train_st.py
    python src/train_st.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import numpy as np
import pandas as pd
from adapters import enable_mirror
from evaluate import evaluate_and_save
from protocol import model_name, sha256


def pair_features(enc, texts_a, texts_b, batch=32):
    """编码句对并构造交互特征。长文本交给模型自身的 max_seq_length 截断。"""
    u = enc.encode(
        list(texts_a), batch_size=batch, show_progress_bar=True, normalize_embeddings=True
    )
    v = enc.encode(
        list(texts_b), batch_size=batch, show_progress_bar=True, normalize_embeddings=True
    )
    return np.hstack([u, v, np.abs(u - v), u * v])


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.ST_MODEL)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    enable_mirror()  # 国内网络：走 hf-mirror
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    tr = pd.read_csv(C.DATA_PROC / "qevasion_train.csv")
    te = pd.read_csv(C.DATA_PROC / "qevasion_test.csv")
    fit = tr[tr["role"] == "train"]
    val = tr[tr["role"] == "val"]

    print("=" * 62)
    print(f"基线 B-1：Sentence-Transformer + LR  ({args.model})")
    print("=" * 62)

    name = "st_" + model_name(args.model)
    enc = SentenceTransformer(args.model, device=args.device)
    enc.max_seq_length = C.MAX_LEN
    print(f"max_seq_length = {enc.max_seq_length}")

    Xtr = pair_features(enc, fit["text_a"], fit["text_b"])
    Xva = pair_features(enc, val["text_a"], val["text_b"])
    Xte = pair_features(enc, te["text_a"], te["text_b"])

    candidates = []
    for cw in (None, "balanced"):
        tag = f"{name}_{'bal' if cw else 'raw'}"
        pipe = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=1.0, max_iter=3000, class_weight=cw, random_state=C.SEED),
                ),
            ]
        )
        pipe.fit(Xtr, fit["label"].values)
        val_f1 = __import__("evaluate").compute_metrics(
            val["label"].values, pipe.predict(Xva), labels=C.CLARITY_LABELS
        )["macro_f1"]
        print(f"  [val] {tag}: macro_f1 = {val_f1:.4f}  (class_weight={cw})")
        candidates.append((val_f1, cw, pipe))

    # 开发集仅选配置；所有原任务模型只用同样3138条训练。
    best_val, best_cw, pipe = max(candidates, key=lambda item: item[0])
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)
    cls = list(pipe.named_steps["clf"].classes_)
    proba = proba[:, [cls.index(c) for c in C.CLARITY_LABELS if c in cls]]

    m = evaluate_and_save(
        name,
        te["label"].values,
        pred,
        proba,
        class_names=C.CLARITY_LABELS,
        texts=te["text_a"].values,
    )
    with open(C.METRICS / f"{name}_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "max_seq_length": C.MAX_LEN,
                "features": "[u, v, |u-v|, u*v]",
                "classifier": "LogisticRegression",
                "class_weight": best_cw,
                "val_macro_f1": best_val,
                "validation_scores": [
                    {"class_weight": cw, "macro_f1": f1} for f1, cw, _ in candidates
                ],
                "train_n": len(fit),
                "val_n": len(val),
                "test_n": len(te),
                "refit_on_val": False,
                "seed": C.SEED,
                "data_sha256": sha256(C.DATA_PROC / "qevasion_train.csv"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return m


if __name__ == "__main__":
    main()
