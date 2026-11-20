"""数据清洗：在 data_prep 的字段规范化之上，追加文本级清洗与训练集去重。

清洗原则（写进报告「数据清洗」一节）：
  1. **只去噪声，不按长度删样本**。QEvasion 里大量回答极短但是有效标注
     （如 "No." / "It doesn't." 就是 Clear Reply），按长度过滤会直接删掉正样本。
  2. **test 一律不删**，保持官方 308 条评测集完整，否则指标无法与论文对齐。
  3. 去重只在 train 内部做（且在分组切分之前），跨 split 去重会破坏官方划分。
  4. 所有被改动/删除的样本都计数，产出清洗前后对照表，不做静默处理。

清洗项：
  - HTML 实体反解（&amp; &#39; 等）
  - 方括号转录标记（[inaudible] [cbs news] [david gregory, nbc news] [.] 等）
  - 圆括号听不清标记（(inaudible)）
  - 不可见字符（零宽空格、BOM、行分隔符）
  - 行首说话人前缀（Q: / A: / MR. X: / PRESIDENT OBAMA:）
  - 连续标点归一（??? -> ??，..... -> ...）
  - 空白规范化（制表符、多空格、超量换行）
  - train 内完全重复样本去重；同文本标签冲突时整组剔除并计数

用法:
    python src/clean_data.py              # 清洗并覆盖 processed 数据
    python src/clean_data.py --dry-run    # 只出报告，不写文件
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import numpy as np
import pandas as pd
from data_prep import clean, load_raw, make_group_split, to_unified

# ---------------------------------------------------------------- 正则
RE_BRACKET = re.compile(r"\[[^\]\n]{1,40}\]")
RE_PAREN_INAUD = re.compile(r"\(\s*(?:inaudible|crosstalk|unclear|overlap\w*)\s*\)", re.I)
RE_INVISIBLE = re.compile("[\u200b-\u200f\u2028-\u202f\ufeff\u00ad]")
RE_SPEAKER = re.compile(
    r"(?m)^\s*(?:Q|A|QUESTION|ANSWER|MR\.|MS\.|MRS\.|DR\.|"
    r"PRESIDENT\s+[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*)?)\s*:\s*"
)
RE_REPEAT_PUNCT = re.compile(r"([!?.,])\1{2,}")
RE_ELLIPSIS = re.compile(r"\.{4,}")
RE_WS = re.compile(r"[ \t\r\f\v\u00a0]+")
RE_NL = re.compile(r"\n{3,}")


def deep_clean(text: str) -> str:
    """文本级清洗。每一步都只删除「非语言内容」，不改写任何实词。"""
    if not isinstance(text, str) or not text:
        return ""
    s = html.unescape(text)
    s = RE_BRACKET.sub(" ", s)  # 记者署名 / [inaudible] / [.]
    s = RE_PAREN_INAUD.sub(" ", s)
    s = RE_INVISIBLE.sub("", s)
    s = RE_SPEAKER.sub("", s)  # 说话人前缀
    s = RE_REPEAT_PUNCT.sub(r"\1\1", s)
    s = RE_ELLIPSIS.sub("...", s)
    s = RE_WS.sub(" ", s)
    s = RE_NL.sub("\n\n", s)
    return s.strip()


def count_noise(s: pd.Series) -> dict:
    """统计每类噪声的命中条数（清洗前）。"""
    s = s.astype(str)
    return {
        "HTML实体": int(s.map(lambda x: "&" in x and ";" in x).sum()),
        "方括号标记": int(s.map(lambda x: bool(RE_BRACKET.search(x))).sum()),
        "圆括号inaudible": int(s.map(lambda x: bool(RE_PAREN_INAUD.search(x))).sum()),
        "不可见字符": int(s.map(lambda x: bool(RE_INVISIBLE.search(x))).sum()),
        "说话人前缀": int(s.map(lambda x: bool(RE_SPEAKER.search(x))).sum()),
        "连续标点3+": int(s.map(lambda x: bool(RE_REPEAT_PUNCT.search(x))).sum()),
        "多空白": int(s.map(lambda x: bool(RE_WS.search(x))).sum()),
    }


# ---------------------------------------------------------------- 去重
def dedup_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """train 内按 (text_a, text_b) 去重。

    同文本但标签冲突的组属于标注噪声：整组剔除（保留哪一条都会引入随机性，
    且冲突本身说明该样本的 clarity 不可靠）。
    """
    key = df["text_a"].astype(str) + "\u0001" + df["text_b"].astype(str)
    df = df.assign(_key=key)

    grp = df.groupby("_key")
    n_dup_groups = int((grp.size() > 1).sum())

    # 标签冲突组
    conflict_keys = set()
    for k, g in grp:
        if g["label"].nunique() > 1:
            conflict_keys.add(k)

    before = len(df)
    df_dedup = df[~df["_key"].isin(conflict_keys)].drop_duplicates(subset="_key", keep="first")
    df_dedup = df_dedup.drop(columns=["_key"])

    info = {
        "去重前": before,
        "去重后": len(df_dedup),
        "重复组数": n_dup_groups,
        "删除的纯重复行": before - len(df_dedup) - int(df["_key"].isin(conflict_keys).sum()),
        "标签冲突组数": len(conflict_keys),
        "因标签冲突剔除的行": int(df["_key"].isin(conflict_keys).sum()),
    }
    return df_dedup.reset_index(drop=True), info


# ---------------------------------------------------------------- 主流程
def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只出报告，不覆盖 processed 数据")
    ap.add_argument("--keep-brackets", action="store_true", help="保留方括号标记（做消融对比用）")
    args = ap.parse_args()

    tr_raw, te_raw = load_raw()

    # ---- 清洗前统计
    report = {"before": {}, "after": {}, "noise": {}, "dedup": {}}
    for nm, raw in (("train", tr_raw), ("test", te_raw)):
        report["before"][nm] = {
            "样本数": len(raw),
            "问题平均长度": round(float(raw[C.FIELD_TEXT_A].astype(str).str.len().mean()), 1),
            "回答平均长度": round(float(raw[C.FIELD_TEXT_B].astype(str).str.len().mean()), 1),
            "空回答": int((raw[C.FIELD_TEXT_B].astype(str).str.strip() == "").sum()),
        }
        report["noise"][nm] = {
            "question": count_noise(raw[C.FIELD_TEXT_A]),
            "interview_answer": count_noise(raw[C.FIELD_TEXT_B]),
        }

    # ---- 结构规范化 + 文本清洗
    tr = to_unified(clean(tr_raw, "train"))
    te = to_unified(clean(te_raw, "test"))

    # 保留原始行号：cleaning 会改写文本，之后无法再用文本反查原始字段
    # （president 只存在于原始 parquet，且 title 可能重名，必须靠行号精确关联）
    tr["_raw_idx"] = np.arange(len(tr), dtype=int)
    te["_raw_idx"] = np.arange(len(te), dtype=int)

    if not args.keep_brackets:
        for col in ("text_a", "text_b"):
            tr[f"{col}_orig_len"] = tr[col].str.len()
            tr[col] = tr[col].map(deep_clean)
            te[f"{col}_orig_len"] = te[col].str.len()
            te[col] = te[col].map(deep_clean)

    # ---- train 去重（必须在分组切分之前）
    tr, dedup_info = dedup_train(tr)
    report["dedup"] = dedup_info

    # ---- 清洗后为空的样本：原文整条都是转录噪声（如 "[Inaudible]"），
    #      没有任何可用于判断 clarity 的语言内容，标签不可靠，从 train 剔除。
    #      test 不删（保持官方 308 条），只计数。
    empty_mask = (tr["text_a"].str.len() == 0) | (tr["text_b"].str.len() == 0)
    report["dropped_empty_train"] = int(empty_mask.sum())
    report["dropped_empty_examples"] = tr.loc[empty_mask, "text_b"].head(5).tolist()
    tr = tr[~empty_mask].reset_index(drop=True)

    # ---- 分组切分 + 落盘
    tr = make_group_split(tr, val_ratio=0.15)
    te["role"] = "test"

    # 清洗后仍为空的样本（应极少；如出现需人工确认）
    report["empty_after"] = {
        "train": int((tr["text_a"].str.len() == 0).sum() + (tr["text_b"].str.len() == 0).sum()),
        "test": int((te["text_a"].str.len() == 0).sum() + (te["text_b"].str.len() == 0).sum()),
    }

    for nm, d in (("train", tr), ("test", te)):
        report["after"][nm] = {
            "样本数": int(len(d)),
            "问题平均长度": round(float(d["text_a"].str.len().mean()), 1),
            "回答平均长度": round(float(d["text_b"].str.len().mean()), 1),
        }

    # 标签分布不因清洗而错乱
    report["label_dist_train"] = tr["label"].value_counts().to_dict()
    report["label_dist_test"] = te["label"].value_counts().to_dict()
    mapped = tr.evasion.map(C.EVASION_TO_CLARITY)
    if not (mapped == tr.label).all():
        raise ValueError("细粒度到高层标签映射不一致")
    from protocol import data_record

    report["source_sha256"] = data_record(
        [C.DATA_RAW / "train-00000-of-00001.parquet", C.DATA_RAW / "test-00000-of-00001.parquet"]
    )
    report["mapping_check"] = {"matched": int((mapped == tr.label).sum()), "n": len(tr)}

    if not args.dry_run:
        # 备份清洗前的 processed（若存在）
        bak = C.DATA_PROC / "_before_clean"
        bak.mkdir(exist_ok=True)
        for f in ("qevasion_train.csv", "qevasion_test.csv"):
            src = C.DATA_PROC / f
            if src.exists() and not (bak / f).exists():
                (bak / f).write_bytes(src.read_bytes())

        keep = ["text_a", "text_b", "label", "evasion", "group", "split", "valid_label", "_raw_idx"]
        tr_out, te_out = tr[keep + ["role"]], te[keep + ["role"]]
        tr_out.to_csv(C.DATA_PROC / "qevasion_train.csv", index=False, encoding="utf-8-sig")
        te_out.to_csv(C.DATA_PROC / "qevasion_test.csv", index=False, encoding="utf-8-sig")

    with open(C.METRICS / "clean_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 清洗前后对照表（报告可直接引用）
    rows = []
    for nm in ("train", "test"):
        b, a = report["before"][nm], report["after"][nm]
        rows.append(
            {
                "数据集": nm,
                "清洗前样本数": b["样本数"],
                "清洗后样本数": a["样本数"],
                "清洗前问题均长": b["问题平均长度"],
                "清洗后问题均长": a["问题平均长度"],
                "清洗前回答均长": b["回答平均长度"],
                "清洗后回答均长": a["回答平均长度"],
            }
        )
    t = pd.DataFrame(rows)
    t.to_csv(C.METRICS / "table_clean_before_after.csv", index=False, encoding="utf-8-sig")

    print("=" * 66)
    print("数据清洗完成")
    print("=" * 66)
    print(t.to_string(index=False))
    print("\n去重：", dedup_info)
    print("清洗后空文本：", report["empty_after"])
    print("\n标签分布 train:", report["label_dist_train"])
    print("标签分布 test :", report["label_dist_test"])
    print("\n噪声命中（train / 清洗前）：")
    for col, d in report["noise"]["train"].items():
        nz = {k: v for k, v in d.items() if v}
        if nz:
            print(f"  {col}: {nz}")
    print(f"\n镜像报告: {C.METRICS / 'clean_report.json'}")
    if args.dry_run:
        print("（dry-run，未覆盖 processed 数据）")


if __name__ == "__main__":
    main()
