"""可解释错误分析：定位最难类别、归纳错误类型、挑出表5 的典型误判样本。

产出：
    outputs/metrics/error_summary.json          混淆对统计 + 长度刻画
    outputs/metrics/table5_typical_errors.csv   表5 可直接誊抄的 5 条样本
    outputs/figures/fig4_error_analysis.png     错误流向 + 长度对比

用法:
    python src/error_analysis.py --model tfidf_uni_bi_bal_s42
    python src/error_analysis.py --model roberta-base_clarity_s42
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import numpy as np
import pandas as pd
from data_prep import truncate_text
from evaluate import _setup_mpl
from protocol import sha256


def classify_error_type(gold: str, pred: str, ans: str, q: str, cls_median: float = None) -> str:
    """按「表面证据」给出误差类型的初判，供人工复核与报告填表。

    这些规则只是辅助归因，最终结论要在报告中结合具体文本说明。
    cls_median 为该真实类别的回答长度中位数，用于判断样本相对长短。
    """
    n = len(ans)
    short = cls_median is not None and n < cls_median * 0.6
    long_ = cls_median is not None and n > cls_median * 1.6

    if gold == "Ambivalent" and pred == "Clear Reply":
        if short:
            return "部分回答：回答很短却语气肯定，模型把'答了一点'读成了'答清楚了'"
        return "长而空泛：篇幅长、语气肯定，但缺少可核验的具体信息"
    if gold == "Ambivalent" and pred == "Clear Non-Reply":
        return "主题转移：绕开问题焦点，模型读成了明确拒答"
    if gold == "Clear Reply" and pred == "Ambivalent":
        if long_:
            return "隐式回答：信息已给出但埋在大段叙述里，表述不直接"
        return "隐式回答：信息已给出但表述间接，未落在模型预期的形式"
    if gold == "Clear Non-Reply" and pred == "Ambivalent":
        return "明确拒答被淡化：拒答措辞含蓄（如只提'现在不便谈'）"
    if gold == "Clear Reply" and pred == "Clear Non-Reply":
        return "标签边界模糊：回答极简短，形式近似回避"
    if short:
        return "短回答：上下文不足，信息量偏低"
    if long_:
        return "超长回答：关键信息可能落在截断窗口之外"
    return "标签边界模糊：clarity 边界本身存在标注分歧"


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tfidf_uni_bi_bal_s42")
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()
    if args.model == "auto":
        args.model = json.loads((C.METRICS / "best_model.json").read_text(encoding="utf-8"))[
            "selected_model"
        ]

    pred_path = C.PRED / f"{args.model}.csv"
    if not pred_path.exists():
        raise SystemExit(
            f"找不到 {pred_path}\n可用的预测文件: {[p.stem for p in C.PRED.glob('*.csv')]}"
        )

    pred = pd.read_csv(pred_path)
    te = pd.read_csv(C.DATA_PROC / "qevasion_test.csv")
    te = te.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    if len(pred) != len(te) or not np.array_equal(pred.gold.values, te.label.values):
        raise ValueError("预测文件的样本数/标签顺序与测试集不一致")
    if not np.array_equal(pred.text_a.astype(str).values, te.text_a.astype(str).values):
        raise ValueError("预测文件文本顺序与测试集不一致")

    y_true, y_pred = pred["gold"].values, pred["prediction"].values
    ans = te["text_b"].astype(str).values
    que = te["text_a"].astype(str).values
    ans_len = np.array([len(x) for x in ans])
    annotations = {}
    annotation_path = C.ROOT / "data/analysis/error_annotations.json"
    if annotation_path.exists():
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if payload["test_sha256"] != sha256(C.DATA_PROC / "qevasion_test.csv"):
            raise ValueError("人工误判注释的数据版本不匹配")
        annotations = {int(item["sample_index"]): item for item in payload["samples"]}

    wrong = y_true != y_pred
    print("=" * 62)
    print(f"错误分析：{args.model}")
    print("=" * 62)
    print(f"总体: {len(y_true)} 条，错误 {wrong.sum()} 条，错误率 {wrong.mean():.3f}")

    # ---- 错误流向
    flows = (
        pd.DataFrame({"gold": y_true[wrong], "pred": y_pred[wrong]})
        .value_counts()
        .reset_index(name="count")
    )
    flows["占全部错误比例"] = (flows["count"] / wrong.sum()).round(4)
    print("\n主要错误流向：")
    print(flows.to_string(index=False))

    # ---- 长度刻画：错误样本是否更长
    len_stat = {
        "正确样本_平均回答长度": round(float(ans_len[~wrong].mean()), 1),
        "错误样本_平均回答长度": round(float(ans_len[wrong].mean()), 1),
        "正确样本_中位数": int(np.median(ans_len[~wrong])) if (~wrong).any() else 0,
        "错误样本_中位数": int(np.median(ans_len[wrong])) if wrong.any() else 0,
    }
    print("\n长度刻画：")
    for k, v in len_stat.items():
        print(f"  {k} = {v}")

    # ---- 每类召回（谁最难）
    per_class = {}
    for lab in C.CLARITY_LABELS:
        m = y_true == lab
        per_class[lab] = {
            "support": int(m.sum()),
            "recall": round(float((y_pred[m] == lab).mean()), 4) if m.any() else 0.0,
        }
    hardest = min(per_class, key=lambda k: per_class[k]["recall"])
    print(f"\n最难类别: {hardest} (recall={per_class[hardest]['recall']:.3f})")

    # 每类真实标签的回答长度中位数，用于相对长短判断
    cls_median = {
        lab: float(np.median(ans_len[y_true == lab]))
        for lab in C.CLARITY_LABELS
        if (y_true == lab).any()
    }

    def _conf(i):
        return " / ".join(
            f"{c}={pred[f'prob_{c}'].iloc[i]:.2f}"
            for c in C.CLARITY_LABELS
            if f"prob_{c}" in pred.columns
        )

    def _rec(i, k):
        record = {
            "序号": k,
            "测试样本行号": int(i),
            "问题摘要": truncate_text(que[i], "head", 1.0)[:120],
            "回答摘要": truncate_text(ans[i], "head", 1.0)[:220],
            "回答字符数": int(ans_len[i]),
            "真实标签": y_true[i],
            "预测标签": y_pred[i],
            "误差类型/原因": classify_error_type(
                y_true[i], y_pred[i], ans[i], que[i], cls_median.get(y_true[i])
            ),
            "各类别置信度": _conf(i),
            "归因来源": "规则辅助，待人工核读",
        }
        annotation = annotations.get(int(i))
        if annotation and annotation["gold"] == y_true[i] and annotation["prediction"] == y_pred[i]:
            excerpt = annotation["answer_excerpt"]
            if excerpt not in ans[i]:
                raise ValueError(f"样本{i}的人工节选与原文不符")
            record["回答摘要"] = excerpt
            record["误差类型/原因"] = annotation["reason"]
            record["归因来源"] = "完整原文人工核读；机制仅为假设"
        return record

    # ---- 轮转采样：按错误流向频率排序后逐个方向轮流取，保证表5 覆盖多种误差类型
    wrong_idx = np.where(wrong)[0]
    pairs = {}
    for i in wrong_idx:
        pairs.setdefault((y_true[i], y_pred[i]), []).append(i)

    # 每个方向内部按「长度偏离该类中位数最多」排序，挑最有信息量的样本
    for key, idxs in pairs.items():
        med = cls_median.get(key[0], np.median(ans_len[idxs]))
        pairs[key] = sorted(idxs, key=lambda i: -abs(ans_len[i] - med))

    order = sorted(pairs.keys(), key=lambda k: -len(pairs[k]))
    recs, cursor, used = [], {k: 0 for k in order}, set()
    while len(recs) < args.top_n:
        progressed = False
        for k in order:
            if len(recs) >= args.top_n:
                break
            lst, c = pairs[k], cursor[k]
            while c < len(lst) and lst[c] in used:
                c += 1
            if c < len(lst):
                i = lst[c]
                used.add(i)
                cursor[k] = c + 1
                recs.append(_rec(i, len(recs) + 1))
                progressed = True
        if not progressed:
            break

    t5 = pd.DataFrame(recs)
    t5.to_csv(C.METRICS / "table5_typical_errors.csv", index=False, encoding="utf-8-sig")
    print(f"\n表5 典型误判样本已写出: {C.METRICS / 'table5_typical_errors.csv'}")
    for _, r in t5.iterrows():
        print(f"  {r['序号']}. {r['真实标签']} -> {r['预测标签']} | {r['误差类型/原因']}")

    # ---- 图4
    try:
        plt = _setup_mpl()
        fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.0))

        ax = axes[0]
        if len(flows):
            labels = [f"{a}\n→ {b}" for a, b in zip(flows["gold"], flows["pred"])]
            bars = ax.barh(
                range(len(flows)),
                flows["count"],
                color="#F09595",
                edgecolor="#A32D2D",
                linewidth=0.7,
            )
            ax.set_yticks(range(len(flows)), labels, fontsize=9)
            ax.invert_yaxis()
            for b, v in zip(bars, flows["count"]):
                ax.text(
                    v + 0.3, b.get_y() + b.get_height() / 2, str(int(v)), va="center", fontsize=9
                )
            ax.set_xlabel("错误条数")
            ax.set_title("错误流向（真实 → 预测）")

        ax = axes[1]
        ax.hist(
            np.clip(ans_len[~wrong], 0, 6000),
            bins=45,
            alpha=0.75,
            label="正确",
            color="#5DCAA5",
            edgecolor="#0F6E56",
            linewidth=0.4,
        )
        ax.hist(
            np.clip(ans_len[wrong], 0, 6000),
            bins=45,
            alpha=0.75,
            label="错误",
            color="#F09595",
            edgecolor="#A32D2D",
            linewidth=0.4,
        )
        ax.axvline(ans_len[~wrong].mean(), color="#0F6E56", linestyle="--", linewidth=1.3)
        ax.axvline(ans_len[wrong].mean(), color="#A32D2D", linestyle="--", linewidth=1.3)
        ax.set_xlabel("回答字符数（截断显示至 6000）")
        ax.set_ylabel("频数")
        ax.set_title("错误样本 vs 正确样本的回答长度")
        ax.legend(fontsize=9)

        fig.suptitle(f"图4  错误分析（{args.model}）", fontsize=13)
        fig.tight_layout()
        fig.savefig(C.FIG / "fig4_error_analysis.png", bbox_inches="tight")
        plt.close(fig)
        print("已生成 fig4_error_analysis.png")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")

    with open(C.METRICS / "error_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "n_total": int(len(y_true)),
                "n_error": int(wrong.sum()),
                "error_rate": round(float(wrong.mean()), 4),
                "error_flows": flows.to_dict(orient="records"),
                "length_stat": len_stat,
                "per_class_recall": per_class,
                "hardest_class": hardest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
