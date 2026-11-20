"""为已产出的预测文件补写各类别概率列（不重新训练，保持指标与报告一致）。

背景：`train_encoder.py` 早期版本调用 evaluate_and_save 时未传 proba，
导致 RoBERTa 系列与迁移模型的 predictions.csv 缺少概率列（模板要求必须有）。
本脚本从 outputs/models/<tag>/encoder.pt 加载权重做一次推理，
把概率补进现有 CSV，并用「argmax 是否与已存的 prediction 完全一致」做校验——
只有校验通过才写盘，确保补的概率确实来自同一个模型、同一份数据。

用法:
    python src/backfill_proba.py
    python src/backfill_proba.py --tag roberta_clarity_s42
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import numpy as np
import pandas as pd
from train_encoder import PadCollator, PairDataset

# tag -> ('qevasion' | 'qnli', 'clarity' | 'evasion')
JOBS = [
    ("roberta_clarity_s42", "qevasion", "clarity"),
    ("roberta_clarity_clean_s2024", "qevasion", "clarity"),
    ("roberta_clarity_clean_s3407", "qevasion", "clarity"),
    ("roberta_clarity_raw_s42", "qevasion", "clarity"),
    ("roberta_clarity_raw_s2024", "qevasion", "clarity"),
    ("roberta_clarity_raw_s3407", "qevasion", "clarity"),
    ("roberta_evasion_boost_s42", "qevasion", "evasion"),
    ("transfer_qnli_frozen", "qnli", "clarity"),
    ("transfer_qnli_finetuned", "qnli", "clarity"),
]


def load_test(kind):
    if kind == "qevasion":
        return pd.read_csv(C.DATA_PROC / "qevasion_test.csv")
    d = pd.read_csv(C.DATA_PROC / "transfer_qnli_revised_unified.csv")
    return d[d["split"] == "test"].reset_index(drop=True)


def infer(tag, kind, target, device="cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ck = C.CKPT / tag
    if not ck.exists():
        print(f"  [skip] 权重目录不存在: {ck}")
        return None
    cfg_path = C.METRICS / f"{tag}_config.json"
    max_len = 512
    if cfg_path.exists():
        max_len = json.loads(cfg_path.read_text(encoding="utf-8")).get("max_len", 512)

    tok = AutoTokenizer.from_pretrained(str(ck))
    # 早期版本只存了 state_dict + tokenizer，没有 model config，
    # 因此这里用基础主干的 config 重建结构，再从 state_dict 载入权重。
    sd = torch.load(ck / "encoder.pt", map_location="cpu")
    n_lab = int(sd["classifier.out_proj.weight"].shape[0])
    base = C.ROOT / "models" / "roberta-base"
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(base) if base.exists() else "roberta-base")
    cfg.num_labels = n_lab
    model = AutoModelForSequenceClassification.from_config(cfg)
    missing = model.load_state_dict(sd, strict=False)
    if missing.unexpected_keys:
        print(f"  [warn] 未使用键 {len(missing.unexpected_keys)} 个")
    model = model.to(device).eval()

    if n_lab == 2:
        label_names = ["entailment", "not_entailment"]
    elif target == "evasion" and n_lab == 9:
        label_names = C.EVASION_LABELS
    else:
        label_names = C.CLARITY_LABELS

    if "_raw_" in tag:
        test = pd.read_csv(C.DATA_PROC / "_before_clean" / "qevasion_test.csv")
    else:
        test = load_test(kind)
    trunc = (
        json.loads(cfg_path.read_text(encoding="utf-8")).get("trunc", C.TRUNC_MODE)
        if cfg_path.exists()
        else C.TRUNC_MODE
    )
    ds = PairDataset(test, tok, max_len, None, None, trunc_mode=trunc)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, collate_fn=PadCollator(tok.pad_token_id)
    )

    probs = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    proba = np.vstack(probs)
    pred_idx = proba.argmax(1)
    pred_name = np.array([label_names[i] for i in pred_idx])
    # evasion-boost 的预测文件里存的是「映射后的 3 类」，比较时必须先映射，
    # 否则 9 类名 vs 3 类名会 0% 命中。写盘时概率也要按 taxonomy 聚合到 3 类。
    if n_lab == 9:
        pred_cmp = np.array([C.EVASION_TO_CLARITY[label_names[i]] for i in pred_idx])
        proba_save = np.zeros((len(proba), len(C.CLARITY_LABELS)))
        for i, nm in enumerate(label_names):
            proba_save[:, C.CLARITY_LABELS.index(C.EVASION_TO_CLARITY[nm])] += proba[:, i]
        save_names = C.CLARITY_LABELS
    else:
        pred_cmp = pred_name
        proba_save = proba
        save_names = label_names
    return test, proba_save, save_names, pred_cmp


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    jobs = [j for j in JOBS if args.tag is None or j[0] == args.tag]
    ok, fail = [], []
    for tag, kind, target in jobs:
        print(f"\n=== {tag} ===")
        out = infer(tag, kind, target)
        if out is None:
            fail.append(tag)
            continue
        test, proba, label_names, pred_name = out

        csv_path = C.PRED / f"{tag}.csv"
        if not csv_path.exists():
            print(f"  [skip] 预测文件不存在: {csv_path}")
            fail.append(tag)
            continue
        old = pd.read_csv(csv_path)
        if len(old) != len(pred_name):
            print("  !! 样本数不同，拒绝写入")
            fail.append(tag)
            continue
        if "text_a" not in old or not np.array_equal(
            old.text_a.astype(str).values, test.text_a.astype(str).values
        ):
            print("  !! 文本或顺序不同，拒绝写入")
            fail.append(tag)
            continue
        n = len(old)
        agree = (old["prediction"].values[:n] == pred_name[:n]).mean()
        n_diff = int((old["prediction"].values[:n] != pred_name[:n]).sum())
        print(f"  与已有 prediction 一致率: {agree:.4f}  （{n} 条，不一致 {n_diff} 条）")

        # 补写不能改变原来的决策；精度导致的任何翻转都应重新完整评测。
        if agree != 1.0:
            print("  !! 一致率过低，说明权重/数据对不上，跳过写盘")
            fail.append(tag)
            continue

        # 写概率列
        keep = [c for c in old.columns if not c.startswith("prob_")]
        new = old[keep].copy()
        for i, cname in enumerate(label_names):
            new[f"prob_{cname}"] = proba[:n, i]
        new.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  已补概率列: {[f'prob_{c}' for c in label_names]}")
        ok.append(tag)

    print("\n" + "=" * 60)
    print(f"成功 {len(ok)} 个，失败 {len(fail)} 个")
    if fail:
        print("失败:", fail)


if __name__ == "__main__":
    main()
