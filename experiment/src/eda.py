"""探索性数据分析（模板必做项）。

产出：
    outputs/figures/fig1_label_dist.png   标签分布（含 train/test 对比）
    outputs/figures/fig2_length_dist.png  长度分布 + 按 clarity 分组比较
    outputs/figures/fig3_president.png    president 分布与同源采访风险
    outputs/metrics/eda_summary.json      文字结论所需数字

用法:
    python src/eda.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import pandas as pd
from evaluate import _setup_mpl


def load_processed():
    tr = pd.read_csv(C.DATA_PROC / "qevasion_train.csv")
    te = pd.read_csv(C.DATA_PROC / "qevasion_test.csv")
    return tr, te


def fig_label_dist(tr, te, plt):
    """图1：标签分布。含 train/test 的 clarity 对比与 train 的 evasion 分布。"""
    fig = plt.figure(figsize=(6.3, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.4], hspace=0.6)

    ax1 = fig.add_subplot(gs[0])
    d = tr["label"].value_counts().reindex(C.CLARITY_LABELS).fillna(0)
    bars = ax1.bar(range(len(d)), d.values, color="#5DCAA5", edgecolor="#0F6E56", linewidth=0.8)
    ax1.set_xticks(range(len(d)), [x.replace(" ", "\n") for x in d.index], fontsize=9)
    ax1.set_ylabel("样本数")
    ax1.set_title("train 的 clarity 分布")
    for b, v in zip(bars, d.values):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{int(v)}\n{v / len(tr):.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.set_ylim(0, d.max() * 1.22)

    ax2 = fig.add_subplot(gs[1])
    d = te["label"].value_counts().reindex(C.CLARITY_LABELS).fillna(0)
    bars = ax2.bar(range(len(d)), d.values, color="#FAEEDA", edgecolor="#854F0B", linewidth=0.8)
    ax2.set_xticks(range(len(d)), [x.replace(" ", "\n") for x in d.index], fontsize=9)
    ax2.set_ylabel("样本数")
    ax2.set_title("test 的 clarity 分布")
    for b, v in zip(bars, d.values):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{int(v)}\n{v / len(te):.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylim(0, d.max() * 1.22)

    ax3 = fig.add_subplot(gs[2])
    d = tr["evasion"].value_counts().reindex(C.EVASION_LABELS).fillna(0)
    colors = [
        "#5DCAA5"
        if C.EVASION_TO_CLARITY[x] == "Clear Reply"
        else "#EF9F27"
        if C.EVASION_TO_CLARITY[x] == "Ambivalent"
        else "#F09595"
        for x in d.index
    ]
    bars = ax3.barh(range(len(d)), d.values, color=colors, edgecolor="#5F5E5A", linewidth=0.6)
    ax3.set_yticks(range(len(d)), d.index, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel("样本数")
    ax3.set_title("train 的 evasion 分布（颜色 = 所属 clarity）")
    for b, v in zip(bars, d.values):
        ax3.text(v + 12, b.get_y() + b.get_height() / 2, str(int(v)), va="center", fontsize=9)

    fig.suptitle("图1  标签分布：Ambivalent 占据近六成，Clear Non-Reply 严重稀缺", fontsize=13)
    p = C.FIG / "fig1_label_dist.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_length_dist(tr, plt):
    """图2：长度分布，并按 clarity 分组比较回答长度。"""
    q = tr["text_a"].str.len()
    a = tr["text_b"].str.len()

    fig = plt.figure(figsize=(6.3, 7.8))
    gs = fig.add_gridspec(3, 1, hspace=0.55)

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(q, bins=50, color="#85B7EB", edgecolor="#185FA5", linewidth=0.5)
    ax1.axvline(
        q.mean(), color="#A32D2D", linestyle="--", linewidth=1.4, label=f"均值 {q.mean():.0f}"
    )
    ax1.set_xlabel("问题字符数")
    ax1.set_ylabel("频数")
    ax1.set_title("问题长度分布")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(a, bins=60, color="#EF9F27", edgecolor="#854F0B", linewidth=0.5)
    for v, c, lab in ((C.MAX_LEN * 4, "#A32D2D", f"512 token 截断线 ≈{C.MAX_LEN * 4} 字符"),):
        ax2.axvline(v, color=c, linestyle="--", linewidth=1.4, label=lab)
    ax2.set_xlabel("回答字符数")
    ax2.set_ylabel("频数")
    ax2.set_title(f"回答长度分布（p50={a.median():.0f}, p95={a.quantile(0.95):.0f}）")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    data = [a[tr["label"] == lab].values for lab in C.CLARITY_LABELS]
    bp = ax3.boxplot(
        data,
        vert=True,
        patch_artist=True,
        showfliers=False,
        tick_labels=[x.replace(" ", "\n") for x in C.CLARITY_LABELS],
    )
    for patch, col in zip(bp["boxes"], ["#5DCAA5", "#EF9F27", "#F09595"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    ax3.set_ylabel("回答字符数")
    ax3.set_title("回答长度按 clarity 分组")
    for i, d in enumerate(data, start=1):
        ax3.text(i, d.mean(), f"{d.mean():.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("图2  长度分布：回答长度远大于问题，且长尾严重", fontsize=13)
    p = C.FIG / "fig2_length_dist.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_president(tr, plt):
    """图3：president 分布与同源采访风险（train 的 title 高度聚集）。

    注意：president 字段只存在于原始 parquet，且 test 全缺。这里用 title 做映射，
    把 president 落到**清洗后**的 tr 上（而不是直接统计原始表），
    否则图上会出现 3448 条、与正文的 3355 条不一致。
    """
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 6.0))

    raw = pd.read_parquet(C.DATA_RAW / "train-00000-of-00001.parquet")
    if "_raw_idx" in tr.columns:
        # 用清洗时保留的原始行号精确关联（title 可能重名，不能用 title 映射）
        pres_series = tr["_raw_idx"].map(raw["president"])
    else:
        title2pres = (
            raw.dropna(subset=[C.FIELD_GROUP])
            .drop_duplicates(C.FIELD_GROUP)
            .set_index(C.FIELD_GROUP)["president"]
        )
        pres_series = tr["group"].map(title2pres)
    pres_series = pres_series.dropna()
    pc = pres_series.value_counts()
    n_mapped = len(pres_series)

    ax1 = axes[0]
    bars = ax1.bar(range(len(pc)), pc.values, color="#AFA9EC", edgecolor="#534AB7", linewidth=0.8)
    ax1.set_xticks(range(len(pc)), [x.replace(" ", "\n") for x in pc.index], fontsize=9)
    ax1.set_ylabel("样本数")
    ax1.set_title("train 按 president 的样本量")
    for b, v in zip(bars, pc.values):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{int(v)}\n{v / max(n_mapped, 1):.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.set_ylim(0, pc.max() * 1.22)

    ax2 = axes[1]
    per_title = tr.groupby("group").size().sort_values()
    ax2.bar(
        range(len(per_title)), per_title.values, color="#B5D4F4", edgecolor="#185FA5", linewidth=0.4
    )
    ax2.axhline(
        per_title.mean(),
        color="#A32D2D",
        linestyle="--",
        linewidth=1.4,
        label=f"均值 {per_title.mean():.1f} 条/组",
    )
    ax2.set_xlabel(f"title 组（共 {len(per_title)} 组，按样本量升序）")
    ax2.set_ylabel("该 title 组包含的 QA 数")
    ax2.set_title(f"标题分组聚集度：{len(tr)} 条来自 {len(per_title)} 个 title 组")
    ax2.legend(fontsize=9)

    fig.suptitle("图3  数据来源与 title 分组分布", fontsize=13)
    fig.tight_layout()
    p = C.FIG / "fig3_president.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    C.ensure_directories()
    plt = _setup_mpl()
    tr, te = load_processed()

    print("=" * 62)
    print("探索性数据分析")
    print("=" * 62)

    summary = {
        "train_size": len(tr),
        "test_size": len(te),
        "clarity_train": tr["label"].value_counts().to_dict(),
        "clarity_test": te["label"].value_counts().to_dict(),
        "evasion_train": tr["evasion"].value_counts().to_dict(),
        "test_has_evasion_label": bool((te["evasion"].fillna("").str.len() > 0).any()),
        "len_train": {
            "question_mean": round(tr["text_a"].str.len().mean(), 1),
            "answer_mean": round(tr["text_b"].str.len().mean(), 1),
            "answer_p50": int(tr["text_b"].str.len().median()),
            "answer_p95": int(tr["text_b"].str.len().quantile(0.95)),
            "answer_max": int(tr["text_b"].str.len().max()),
        },
        "n_title_groups_train": int(tr["group"].nunique()),
        "avg_samples_per_title_group": round(len(tr) / tr["group"].nunique(), 1),
    }

    # 多数类基线的理论值 —— Macro-F1 论证的核心数字
    top = tr["label"].value_counts().idxmax()
    acc = (te["label"] == top).mean()
    f1_pos = 2 * acc / (acc + 1)  # 该类的 F1（P=acc, R=1）
    summary["majority_baseline"] = {
        "class": top,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_pos / len(C.CLARITY_LABELS)), 4),
    }

    for name, fn in (
        ("fig1_label_dist", lambda: fig_label_dist(tr, te, plt)),
        ("fig2_length_dist", lambda: fig_length_dist(tr, plt)),
        ("fig3_president", lambda: fig_president(tr, plt)),
    ):
        p = fn()
        print(f"  已生成 {p.name}")

    with open(C.METRICS / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
