"""数据准备：读取 parquet -> 构造统一样本 -> 落盘 -> 输出统计。

⚠️ 注意：本脚本只做**字段规范化**，不做文本级清洗。
   报告中全部数字使用的是 `src/clean_data.py` 的产物——它在字段规范化之上
   追加了方括号转录标记移除、空白规范化、行首说话人前缀清理与 train 去重
   （train 3448 → 3355）。
   如果误跑本脚本，会把 data/processed/*.csv 覆盖成**未清洗**版本，
   导致复现结果与报告不一致。复现报告请运行：python src/clean_data.py

输出的统一样本列（迁移实验复用同一套列名）：
    text_a   问题 / 前提句
    text_b   回答 / 待判定句
    label    3 类 clarity（主任务标签）
    evasion  9 类细粒度（仅 train 非空，test 恒为空）
    group    标题组 title，同名标题可跨日期（test 缺失，填 __TEST__）
    split    train / test

用法:
    python src/data_prep.py     # 仅用于对照「清洗前」的基线，一般不单独运行
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import pandas as pd


# ---------------------------------------------------------------- 加载
def load_raw():
    """从本地 parquet 读取原始 train / test。"""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if not all(
        (C.DATA_RAW / f"{split}-00000-of-00001.parquet").exists() for split in ("train", "test")
    ):
        from adapters import http_load_frames

        frames = http_load_frames("ailsntua/QEvasion")
        for split in ("train", "test"):
            frames[split].to_parquet(C.DATA_RAW / f"{split}-00000-of-00001.parquet", index=False)
    tr = pd.read_parquet(C.DATA_RAW / "train-00000-of-00001.parquet")
    te = pd.read_parquet(C.DATA_RAW / "test-00000-of-00001.parquet")
    return tr, te


def _norm(s: pd.Series) -> pd.Series:
    """统一清洗：转字符串、去首尾空白、把 NaN 与纯空白规范化为空串。"""
    out = s.astype("string").fillna("").str.strip()
    return out


# ---------------------------------------------------------------- 清洗
def clean(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """字段规范化。

    注意：模板明确要求「不得在了解字段之前直接删除数据」。
    因此这里只做规范化与标记，不做删除；所有被标记的样本在统计中如实报告。
    """
    d = df.copy()

    for col in [C.FIELD_TEXT_A, C.FIELD_TEXT_B, "clarity_label", "evasion_label"]:
        d[col] = _norm(d[col])

    # 三个质量标志位：只标记不删，报告中给出分布
    for col in ["inaudible", "multiple_questions", "affirmative_questions"]:
        d[col] = d[col].fillna(False).astype(bool)

    # 空文本标记（用于表1 的「缺失样本」列）
    d["empty_text_a"] = d[C.FIELD_TEXT_A].str.len() == 0
    d["empty_text_b"] = d[C.FIELD_TEXT_B].str.len() == 0

    # test 的 title / president 全缺失，用哨兵值占位，避免后续 groupby 丢样本
    if C.FIELD_GROUP in d.columns:
        d[C.FIELD_GROUP] = _norm(d[C.FIELD_GROUP]).replace("", f"__{split.upper()}__")
    else:
        d[C.FIELD_GROUP] = f"__{split.upper()}__"

    d["split"] = split
    return d


# ---------------------------------------------------------------- 统一结构
def to_unified(d: pd.DataFrame) -> pd.DataFrame:
    """映射到统一列名。test 的 evasion 为空串（官方未提供）。"""
    out = pd.DataFrame(
        {
            "text_a": d[C.FIELD_TEXT_A],
            "text_b": d[C.FIELD_TEXT_B],
            "label": d["clarity_label"],
            "evasion": d["evasion_label"],
            "group": d[C.FIELD_GROUP],
            "split": d["split"],
        }
    )
    # 只保留标签合法的样本（label 为空的不参与训练；此处如有会如实计数）
    out["valid_label"] = out["label"].isin(C.CLARITY_LABELS)
    return out


# ---------------------------------------------------------------- 分组切分
def make_group_split(df: pd.DataFrame, val_ratio: float = 0.15, seed: int = None):
    """从 train 中切出验证集，按 title 标题组整体切分。

    同一来源问答跨集合可能高估验证指标。标题分组是保守近似：
    同名title可跨日期与总统，不能把175个标题组称为175场采访。
    未来可结合日期与来源构造更准确的采访标识；当前保持预设划分。
    """
    from sklearn.model_selection import GroupShuffleSplit

    seed = C.SEED if seed is None else seed
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["group"]))

    out = df.copy()
    out["role"] = "unused"
    out.iloc[tr_idx, out.columns.get_loc("role")] = "train"
    out.iloc[va_idx, out.columns.get_loc("role")] = "val"

    # 泄漏自检：两个集合的 group 不能有交集
    overlap = set(out.loc[out["role"] == "train", "group"]) & set(
        out.loc[out["role"] == "val", "group"]
    )
    assert not overlap, f"分组切分失败，group 出现交集: {list(overlap)[:5]}"
    return out


# ---------------------------------------------------------------- 统计（表1）
def describe(df: pd.DataFrame) -> dict:
    """生成表1 所需的一行统计。"""
    q = df[C.FIELD_TEXT_A].str.len()
    a = df[C.FIELD_TEXT_B].str.len()
    return {
        "样本数": int(len(df)),
        "缺失样本": int((df["empty_text_a"] | df["empty_text_b"]).sum()),
        "唯一访谈数": int(df[C.FIELD_GROUP].nunique()),
        "平均问题长度": round(float(q.mean()), 1),
        "平均回答长度": round(float(a.mean()), 1),
        "回答长度中位数": int(a.median()),
        "回答长度p95": int(a.quantile(0.95)),
        "回答长度最大值": int(a.max()),
    }


# ---------------------------------------------------------------- 截断工具
def truncate_text(text: str, mode: str = None, head_ratio: float = None) -> str:
    """字符级截断预览工具（供 EDA 与错误分析查看，不改变原始数据）。

    token 级截断由 tokenizer 的 truncation 参数处理；本函数用于报告里
    说明「截断策略」时给出可读示例。
    """
    mode = mode or C.TRUNC_MODE
    head_ratio = head_ratio or C.HEAD_RATIO
    limit = C.MAX_LEN * 4  # 粗略按 4 字符 ≈ 1 token 估算
    if len(text) <= limit:
        return text
    if mode == "head":
        return text[:limit]
    if mode == "tail":
        return text[-limit:]
    head = int(limit * head_ratio)
    tail = limit - head
    return text[:head] + " …… " + text[-tail:]


# ---------------------------------------------------------------- 主流程
def main():
    C.ensure_directories()
    print("=" * 62)
    print("⚠️  本脚本产出的是**未做文本级清洗**的数据（仅字段规范化）")
    print("    报告的正式数据来自 src/clean_data.py（train 3448 → 3355）")
    print("    继续运行会覆盖 data/processed/*.csv，复现结果将与报告不一致")
    print("=" * 62)
    tr_raw, te_raw = load_raw()
    tr = to_unified(clean(tr_raw, "train"))
    te = to_unified(clean(te_raw, "test"))

    # 从 train 内按 title group 切出验证集，供早停与模型选择。
    tr = make_group_split(tr, val_ratio=0.15)
    te["role"] = "test"

    tr.to_csv(C.DATA_PROC / "qevasion_train.csv", index=False, encoding="utf-8-sig")
    te.to_csv(C.DATA_PROC / "qevasion_test.csv", index=False, encoding="utf-8-sig")

    stats = {"train": describe(clean(tr_raw, "train")), "test": describe(clean(te_raw, "test"))}

    # 泄漏检查：train 内部同一 title 是否被拆分（这里检查 test 与 train 的问题文本重叠）
    tr_q = set(_norm(tr_raw[C.FIELD_TEXT_A]).str.lower())
    te_q = _norm(te_raw[C.FIELD_TEXT_A]).str.lower()
    stats["train_test_question_overlap"] = int(te_q.isin(tr_q).sum())

    # 标签体系核对
    stats["clarity_dist_train"] = tr["label"].value_counts().to_dict()
    stats["clarity_dist_test"] = te["label"].value_counts().to_dict()
    stats["evasion_dist_train"] = tr["evasion"].value_counts().to_dict()
    stats["test_evasion_nonempty"] = int((te["evasion"].str.len() > 0).sum())

    with open(C.DATA_PROC / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("=" * 62)
    print("数据准备完成")
    print("=" * 62)
    for k, v in stats.items():
        print(f"{k}: {v}")

    # 映射自检：用 evasion 反推 clarity，必须与 clarity_label 完全一致
    mapped = tr["evasion"].map(C.EVASION_TO_CLARITY)
    agree = (mapped == tr["label"]).sum()
    print(f"\n[自检] evasion->clarity 映射与 clarity_label 一致: {agree}/{len(tr)}")
    if agree != len(tr):
        bad = tr.loc[mapped != tr["label"], ["evasion", "label"]].head()
        print("[警告] 存在不一致样本：")
        print(bad)


if __name__ == "__main__":
    main()
