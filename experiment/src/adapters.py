"""统一字段适配器 —— 跨场景迁移实验的核心复用点。

设计思想：把「数据集本身的字段差异」全部收敛到本文件的一张注册表里。
只要为每个数据集声明「文本对字段 / 标签字段 / 标签映射 / 分组字段」，
后续的分词、建模、训练、评测、报告流程一行都不用改。

统一输出列（与原任务完全一致）：
    text_a, text_b, label, group, split

默认使用官方站点；网络入口可通过HF_ENDPOINT配置。
本模块也提供纯HTTP下载回退，供Hub客户端不可用的环境使用。

用法:
    python src/adapters.py --list                  # 列出所有已注册数据集
    python src/adapters.py --preview qnli          # 打印真实列名（字段不符时先跑这个）
    python src/adapters.py --build qnli            # 拉取并落盘为统一样本
    python src/adapters.py --build qnli --http     # 强制走 HTTP 直取
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import pandas as pd

MIRROR = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# 迁移候选数据集注册表
# 字段名以「实际加载结果」为准。若某数据集列名与下表不符，
# 先运行 --preview 查看真实列名，再修改本表对应项即可。
ADAPTERS = {
    # ---------------- 原任务 ----------------
    "qevasion": {
        "hf_path": "ailsntua/QEvasion",
        "hf_config": None,
        "subdir": "",  # parquet 所在根目录，"" = 全仓库递归
        "splits": {"train": "train", "test": "test"},
        "fields": {
            "text_a": "question",
            "text_b": "interview_answer",
            "label": "clarity_label",
            "group": "title",
        },
        "label_map": None,
        "label_names": C.CLARITY_LABELS,
        "subsample": None,
        "note": "原任务。test 的 title/president 缺失，evasion_label 为空。",
    },
    # ---------------- 迁移候选 1：QNLI（推荐首选，已确认镜像可用）----------------
    "qnli": {
        "hf_path": "nyu-mll/glue",
        "hf_config": "qnli",
        "subdir": "qnli",
        "splits": {
            "train": "train",
            "test": "validation",
        },  # 官方 test 标签被隐藏（-1），用 validation 当评测集
        "fields": {"text_a": "question", "text_b": "sentence", "label": "label", "group": None},
        "label_map": {0: "entailment", 1: "not_entailment"},
        "label_names": ["entailment", "not_entailment"],
        "subsample": {"train": 30000},  # 控制迭代速度；全量 10.5 万条
        "note": "question + sentence -> 是否蕴含答案。2 分类，与原任务结构同构。"
        "注意：GLUE 版 test 标签隐藏为 -1，故用 validation 作评测集。",
    },
    # ---------------- 迁移候选 2：LCQMC（中文，已确认镜像可用）----------------
    "lcqmc": {
        "hf_path": "C-MTEB/LCQMC",
        "hf_config": None,
        "subdir": "data",
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "fields": {"text_a": "sentence1", "text_b": "sentence2", "label": "label", "group": None},
        "label_map": {0: "not_duplicate", 1: "duplicate"},
        "label_names": ["not_duplicate", "duplicate"],
        "subsample": {"train": 30000},  # 全量 23.9 万条
        "note": "中文句对语义匹配。需换用中文主干（hfl/chinese-roberta-wwm-ext）。",
    },
    # ---------------- 迁移候选 3：EZ-STANCE（镜像不可用，保留备用）----------------
    "ezstance": {
        "hf_path": "Harelix/EZ-STANCE",
        "hf_config": None,
        "subdir": "",
        "splits": {"train": "train", "test": "test"},
        "fields": {"text_a": "target", "text_b": "text", "label": "label", "group": None},
        "label_map": {"FAVOR": "favor", "AGAINST": "against", "NONE": "neutral"},
        "label_names": ["favor", "against", "neutral"],
        "subsample": None,
        "note": "⚠️ 该仓库在镜像上返回鉴权错误，不可用。若要用请自行寻找可访问的 EZ-STANCE 镜像仓库。",
    },
    # ---------------- 迁移候选 4：WildGuardMix（语义最贴近拒答）----------------
    "wildguard": {
        "hf_path": "allenai/wildguardmix",
        "hf_config": "wildguardtrain",
        "subdir": "",
        "splits": {"train": "train", "test": None},
        "fields": {
            "text_a": "prompt",
            "text_b": "response",
            "label": "response_refusal_label",
            "group": None,
        },
        "label_map": None,
        "label_names": None,
        "subsample": {"train": 30000},
        "note": "prompt + response -> 是否拒答。列名先用 --preview 确认；需自建验证集。",
    },
}

PREFERRED_TRANSFER = ["qnli", "lcqmc"]


# ---------------------------------------------------------------- 网络
def enable_mirror():
    """必须在 import datasets / huggingface_hub 之前调用。"""
    os.environ.setdefault("HF_ENDPOINT", MIRROR)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return os.environ["HF_ENDPOINT"]


def _tree(repo_id, path="", revision="main"):
    import httpx

    url = f"{MIRROR}/api/datasets/{repo_id}/tree/{revision}/{path}".rstrip("/")
    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _list_parquet(repo_id, path="", revision="main"):
    """递归列出仓库中的 parquet 文件路径。"""
    out = []
    for it in _tree(repo_id, path, revision):
        if it["type"] == "directory":
            out += _list_parquet(repo_id, it["path"], revision)
        elif it["path"].endswith(".parquet"):
            out.append(it["path"])
    return out


def http_load_frames(repo_id, subdir="", revision="main") -> dict:
    """纯 HTTP 直取镜像 parquet，绕过 huggingface_hub。

    本机 huggingface_hub 走镜像时会把 parquet 落成 0 字节文件，
    而 httpx 直取正常，因此提供这条回退路径。
    """
    import httpx

    cache = C.DATA_RAW / "hf" / repo_id.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)

    files = _list_parquet(repo_id, subdir, revision)
    if not files:
        raise RuntimeError(f"{repo_id} 在镜像上找不到 parquet 文件（subdir={subdir!r}）")

    frames = {}
    for f in files:
        m = re.search(r"(train|test|validation|dev)", Path(f).name)
        split = m.group(1) if m else Path(f).stem
        local = cache / Path(f).name
        if not local.exists() or local.stat().st_size == 0:
            url = f"{MIRROR}/datasets/{repo_id}/resolve/{revision}/{f}"
            with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
                r.raise_for_status()
                with open(local, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 16):
                        fh.write(chunk)
        df = pd.read_parquet(local)
        frames[split] = pd.concat([frames[split], df], ignore_index=True) if split in frames else df
        print(f"    [http] {split:<11} {Path(f).name}  -> {len(df)} 行")
    return frames


def _load_frames(spec, force_http=False) -> dict:
    """统一返回 {split: DataFrame}。优先 datasets 库，失败则回退 HTTP 直取。"""
    enable_mirror()
    if not force_http:
        try:
            from datasets import load_dataset

            path, conf = spec["hf_path"], spec["hf_config"]
            ds = load_dataset(path, conf) if conf else load_dataset(path)
            print(f"    [datasets] 已加载 {path} / {conf}")
            return {k: v.to_pandas() for k, v in ds.items()}
        except Exception as e:
            print(
                f"    [warn] datasets 加载失败（{type(e).__name__}: {str(e)[:80]}）"
                f"，回退 HTTP 直取镜像 parquet"
            )
    return http_load_frames(spec["hf_path"], spec.get("subdir", ""))


# ---------------------------------------------------------------- 查看
def preview(key: str, force_http=False):
    """打印数据集的真实列名与前两行，字段对不上时用它定位。"""
    spec = ADAPTERS[key]
    print(f"[{key}] {spec['hf_path']} / config={spec['hf_config']}  (镜像: {enable_mirror()})")
    frames = _load_frames(spec, force_http)
    for split, df in frames.items():
        print(f"\n--- split={split}  rows={len(df)}")
        print("columns:", list(df.columns))
        if len(df):
            for k, v in list(df.iloc[0].items())[:20]:
                sv = str(v)
                print(f"    {k} = {sv[:110]}{'…' if len(sv) > 110 else ''}")
    return frames


# ---------------------------------------------------------------- 适配
def adapt(key: str, splits=None, force_http=False) -> dict:
    """把注册表中的数据集转换为统一样本。返回 {role: DataFrame}。"""
    spec = ADAPTERS[key]
    frames = _load_frames(spec, force_http)
    available = list(frames.keys())
    want = splits or spec["splits"]

    out = {}
    for role, split_name in want.items():
        if split_name is None:
            print(f"[skip] {key}/{role}: 注册表中未指定 split")
            continue
        if split_name not in frames:
            print(f"[warn] {key}/{role}: split '{split_name}' 不存在，可用 {available}")
            continue
        raw = frames[split_name]
        f = spec["fields"]
        missing = [v for v in f.values() if v and v not in raw.columns]
        if missing:
            raise KeyError(
                f"{key} 缺少字段 {missing}；实际列名 = {list(raw.columns)}。"
                f"请先运行: python src/adapters.py --preview {key}"
            )

        d = pd.DataFrame(
            {
                "text_a": raw[f["text_a"]].astype(str),
                "text_b": raw[f["text_b"]].astype(str),
                "label": raw[f["label"]],
            }
        )
        if spec["label_map"]:
            d["label"] = d["label"].map(
                lambda x: spec["label_map"].get(x, spec["label_map"].get(str(x), x))
            )
        d["label"] = d["label"].astype(str)
        d["group"] = raw[f["group"]].astype(str) if f.get("group") else f"__{key}_{role}__"
        d["split"] = role

        # 剔除无效标签（GLUE 类数据集的 test 标签常被隐藏为 -1 / 空）
        bad = d["label"].isin(["-1", "", "nan", "None"])
        if bad.any():
            if bad.mean() > 0.5:
                print(
                    f"[skip] {key}/{role}: {bad.mean():.0%} 的标签无效（如 -1），"
                    f"该划分不可用作评测集，已跳过"
                )
                continue
            print(f"[clean] {key}/{role}: 剔除 {int(bad.sum())} 条无效标签样本")
            d = d[~bad].reset_index(drop=True)

        # 抽样，控制迭代速度（固定随机种子保证可复现）
        sub = (spec.get("subsample") or {}).get(role)
        if sub and len(d) > sub:
            d = d.sample(n=sub, random_state=C.SEED).reset_index(drop=True)
            print(f"    [subsample] {role}: 抽样至 {sub} 条")

        out[role] = d
        print(
            f"[ok] {key}/{role} ({split_name}): {len(d)} 条 | "
            f"标签分布 {d['label'].value_counts().to_dict()}"
        )
    return out


def build_and_save(key: str, force_http=False):
    """构建并落盘到 data/processed/transfer_<key>_<split>.csv。"""
    frames = adapt(key, force_http=force_http)
    if not frames:
        raise SystemExit(f"{key} 未产出任何划分")
    for role, d in frames.items():
        p = C.DATA_PROC / f"transfer_{key}_{role}.csv"
        d.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"    写出 {p.name}")

    # 保留各划分，不能漏掉test，也不能把val合并进训练。
    train_roles = [r for r in ("train", "val", "test") if r in frames]
    if "train" in frames and "test" in frames:
        pd.concat([frames[r] for r in train_roles]).to_csv(
            C.DATA_PROC / f"transfer_{key}_unified.csv", index=False, encoding="utf-8-sig"
        )
        print(f"    写出 transfer_{key}_unified.csv（保留split；缺val时由transfer.py自建）")

    meta = {
        "key": key,
        "hf_path": ADAPTERS[key]["hf_path"],
        "subdir": ADAPTERS[key].get("subdir", ""),
        "note": ADAPTERS[key]["note"],
        "label_names": ADAPTERS[key]["label_names"],
        "sizes": {k: len(v) for k, v in frames.items()},
        "label_dist": {k: v["label"].value_counts().to_dict() for k, v in frames.items()},
    }
    with open(C.METRICS / f"transfer_meta_{key}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return frames


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preview")
    ap.add_argument("--build")
    ap.add_argument("--http", action="store_true", help="强制走 HTTP 直取，跳过 datasets 库")
    args = ap.parse_args()

    if args.list:
        print("已注册的数据集：")
        for k, v in ADAPTERS.items():
            tag = " ★推荐" if k in PREFERRED_TRANSFER else ""
            print(f"  {k:<11}{tag}  {v['hf_path']}")
            print(f"      {v['note']}")
        return
    if args.preview:
        preview(args.preview, args.http)
        return
    if args.build:
        build_and_save(args.build, args.http)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
