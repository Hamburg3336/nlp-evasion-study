"""数据质量体检：把清洗前的所有问题量化出来（表1 与报告「数据清洗」一节的依据）。"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

PATS = {
    "HTML实体": re.compile(r"&[a-zA-Z]+;|&#\d+;"),
    "方括号转录标记": re.compile(
        r"\[\s*(?:inaudible|crosstalk|overlapping|laughter|applause|unknown|music)\s*\]", re.I
    ),
    "圆括号inaudible": re.compile(r"\(\s*inaudible\s*\)", re.I),
    "其他方括号标记": re.compile(r"\[[^\]]{1,30}\]"),
    "连续空白": re.compile(r"\s{2,}"),
    "不可见字符": re.compile("[\u200b-\u200f\u2028-\u202f\ufeff]"),
    "连续标点3+": re.compile(r"[!?.,]{3,}"),
    "URL": re.compile(r"https?://|www\.", re.I),
    "残留XML标签": re.compile(r"<[^>]{1,20}>"),
    "说话人标记": re.compile(r"^\s*[A-Z][A-Za-z.\- ]{0,20}:\s", re.M),
}


def load(split):
    df = pd.read_parquet(RAW / f"{split}-00000-of-00001.parquet")
    # pandas 3.0 默认 ArrowStringArray，pyarrow 的 regex 不支持 \u 转义
    for c in df.columns:
        if df[c].dtype == "string" or str(df[c].dtype).startswith("large_string"):
            df[c] = df[c].astype(object)
    return df


def main():
    tr, te = load("train"), load("test")
    print("=" * 70)
    print(f"train {tr.shape} | test {te.shape}")
    print("=" * 70)

    print("\n【1】转录噪声（命中条数）")
    any_hit = {c: np.zeros(len(tr), bool) for c in ("question", "interview_answer")}
    for name, p in PATS.items():
        cells = []
        for col in ("question", "interview_answer"):
            hit_tr = tr[col].astype(str).map(lambda s: bool(p.search(s)))
            hit_te = te[col].astype(str).map(lambda s: bool(p.search(s)))
            any_hit[col] |= hit_tr.values
            if hit_tr.sum() or hit_te.sum():
                cells.append(f"{col}: tr={hit_tr.sum():>4d} te={hit_te.sum():>3d}")
        if cells:
            print(f"  {name:<16s} " + " | ".join(cells))
    print(
        f"  {'任一噪声合计':<16s} question: tr={any_hit['question'].sum():>4d} | "
        f"interview_answer: tr={any_hit['interview_answer'].sum():>4d}"
    )

    print("\n【2】空值与极短")
    for nm, d in (("train", tr), ("test", te)):
        q = d["question"].astype(str).str.strip()
        a = d["interview_answer"].astype(str).str.strip()
        print(
            f"  {nm}: 空问题={(q == '').sum()} 空回答={(a == '').sum()} "
            f"问题<5字符={(q.str.len() < 5).sum()} 回答<20字符={(a.str.len() < 20).sum()}"
        )

    print("\n【3】inaudible 标志（数据集自带的听不清标记）")
    for nm, d in (("train", tr), ("test", te)):
        print(f"  {nm}: {dict(d['inaudible'].value_counts(dropna=False))}")
    sub = tr[tr["inaudible"] == True]  # noqa: E712
    if len(sub):
        print(f"  train 中 inaudible=True 的标签分布: {dict(sub['clarity_label'].value_counts())}")

    print("\n【4】重复样本")
    print(
        f"  train 完全相同(question+answer): {tr.duplicated(subset=['question', 'interview_answer']).sum()}"
    )
    print(f"  train 同问题不同回答: {tr.duplicated(subset=['question'], keep=False).sum()}")
    print(f"  test  同问题不同回答: {te.duplicated(subset=['question'], keep=False).sum()}")

    print("\n【5】长度分布")
    for nm, d in (("train", tr), ("test", te)):
        a = d["interview_answer"].astype(str).str.len()
        q = d["question"].astype(str).str.len()
        print(f"  {nm} question: p50={q.median():.0f} max={q.max()}")
        print(
            f"  {nm} answer  : p50={a.median():.0f} p95={a.quantile(0.95):.0f} "
            f"max={a.max()} 超10000字符={(a > 10000).sum()}"
        )

    print("\n【6】清洗后是否会出现空文本（需在清洗步骤里防护）")
    a = tr["interview_answer"].astype(str)
    after = a.map(lambda s: re.sub(r"\[[^\]]{1,30}\]", " ", s).strip())
    print(f"  移除方括号标记后变空/过短的回答: {(after.str.len() < 20).sum()}")


if __name__ == "__main__":
    main()
