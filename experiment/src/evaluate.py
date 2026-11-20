"""统一评测模块：所有模型共用，保证「同一测试集、同一标签映射、同一主指标」。

产出：
    outputs/metrics/<模型名>.json    总体指标 + 类别级指标
    outputs/predictions/<模型名>.csv 每个样本的 gold / prediction / 各类别概率
    outputs/figures/cm_<模型名>.png  混淆矩阵

用法（作为库调用）:
    from evaluate import evaluate_and_save
    m = evaluate_and_save("tfidf_uni", y_true, y_pred, proba, class_names)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import numpy as np
import pandas as pd


def _setup_mpl():
    """中文字体 + 白底，避免图表出现方块或透明背景。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = C.MPL_FONT
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.dpi"] = 160
    return plt


# ---------------------------------------------------------------- 指标
def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """Macro-F1 为主指标，附带 Accuracy 与每类 P/R/F1。"""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = labels or C.CLARITY_LABELS
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Expected nonempty, equally sized one-dimensional labels")
    if len(set(labels)) != len(labels) or not labels:
        raise ValueError("Label names must be unique and nonempty")
    if not set(y_true).issubset(labels) or not set(y_pred).issubset(labels):
        raise ValueError("Unknown gold or predicted labels")

    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        lab: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(sup[i]),
        }
        for i, lab in enumerate(labels)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(np.mean(p)), 4),
        "macro_recall": round(float(np.mean(r)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def plot_confusion(y_true, y_pred, name: str, labels=None):
    """绘制混淆矩阵（含归一化），供报告「关键图表」使用。"""
    plt = _setup_mpl()
    from sklearn.metrics import confusion_matrix

    labels = labels or C.CLARITY_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.0))
    for ax, mat, title, fmt in (
        (axes[0], cm, "混淆矩阵（计数）", "d"),
        (axes[1], cm_norm, "混淆矩阵（按真实类别归一化）", ".2f"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
        ax.set_title(title)
        thr = mat.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    format(mat[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] > thr else "#222222",
                    fontsize=10,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{name} — 混淆矩阵", fontsize=13)
    fig.tight_layout()
    path = C.FIG / f"cm_{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 落盘
def evaluate_and_save(
    name: str,
    y_true,
    y_pred,
    proba=None,
    class_names=None,
    texts=None,
    save_pred=True,
    decision_rule="argmax",
) -> dict:
    """评测 + 保存 metrics.json / predictions.csv / 混淆矩阵图。"""
    C.ensure_directories()
    class_names = class_names or C.CLARITY_LABELS
    if proba is not None:
        proba = np.asarray(proba)
        if proba.shape != (len(y_true), len(class_names)):
            raise ValueError("概率矩阵维度与样本数/类别数不符")
        if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
            raise ValueError("概率必须为有限的[0,1]数值")
        if not np.allclose(proba.sum(axis=1), 1, atol=1e-5):
            raise ValueError("各类概率和不为1")
        if decision_rule == "argmax" and not np.array_equal(
            np.asarray(class_names)[proba.argmax(axis=1)], np.asarray(y_pred)
        ):
            raise ValueError("prediction与同次推理概率argmax不一致")
    if not set(y_true).issubset(class_names) or not set(y_pred).issubset(class_names):
        raise ValueError("真实或预测标签超出指定标签集合")
    m = compute_metrics(y_true, y_pred, labels=class_names)
    m["decision_rule"] = decision_rule

    with open(C.METRICS / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in m.items() if k != "report"}, f, ensure_ascii=False, indent=2)
    with open(C.METRICS / f"{name}_report.txt", "w", encoding="utf-8") as f:
        f.write(m["report"])

    if save_pred:
        data = {
            "sample_index": np.arange(len(y_true)),
            "gold": np.asarray(y_true),
            "prediction": np.asarray(y_pred),
        }
        if texts is not None:
            data["text_a"] = list(texts)
        if proba is not None:
            proba = np.asarray(proba)
            for i, cname in enumerate(class_names):
                data[f"prob_{cname}"] = proba[:, i]
        pd.DataFrame(data).to_csv(C.PRED / f"{name}.csv", index=False, encoding="utf-8-sig")

    try:
        plot_confusion(y_true, y_pred, name, labels=class_names)
    except Exception as e:  # 绘图失败不影响指标产出
        print(f"[warn] 混淆矩阵绘制失败: {e}")

    print(
        f"[{name}] accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
        f"macro_p={m['macro_precision']:.4f}  macro_r={m['macro_recall']:.4f}"
    )
    for lab, d in m["per_class"].items():
        print(
            f"    {lab:<16} P={d['precision']:.3f} R={d['recall']:.3f} "
            f"F1={d['f1']:.3f}  n={d['support']}"
        )
    return m


def majority_baseline(y_train, y_test):
    """多数类基线：预测训练集中出现最多的类别。必须进表3。"""
    from collections import Counter

    top = Counter(y_train).most_common(1)[0][0]
    return np.array([top] * len(y_test))


if __name__ == "__main__":
    print("evaluate 模块：请通过其他脚本导入使用。")
