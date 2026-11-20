"""基线 B-2：RoBERTa / BERT 句对微调（手写训练循环，可控性最好）。

三个关键设计，都直接对应模板要求：
  1. 类别加权损失：针对 Clear Non-Reply 等少数类提高训练损失权重。
  2. 按访谈分组的验证集早停：绝不使用官方 test 做模型选择。
  3. --target evasion 实现论文的 "evasion-boost" 两阶段方案：
     先预测 9 类细粒度 evasion，再按 taxonomy 映射为 3 类 clarity 评测。

用法:
    python src/train_encoder.py                              # 直接 3 分类
    python src/train_encoder.py --target evasion              # evasion-boost（推荐改进项）
    python src/train_encoder.py --frozen                      # 冻结编码器，仅训练分类头（迁移用）
    python src/train_encoder.py --model bert-base-uncased --lr 2e-5 --batch 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import numpy as np
import pandas as pd
import torch
from adapters import enable_mirror
from evaluate import compute_metrics, evaluate_and_save
from protocol import model_name, sha256

# --target 名称 -> 处理后 CSV 中的实际列名（data_prep 产出的是 label / evasion）
TARGET_COL = {"clarity": "label", "evasion": "evasion"}


# ---------------------------------------------------------------- 复现
def set_seed(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- 数据集
class PairDataset:
    """question / answer 句对数据集。

    关键优化：这里**不做 padding**，只做截断，把变长序列交给 collator
    按批内最长补齐（dynamic padding）。回答长度中位数约 300 token，
    若一律补到 512，会白白多算 40%~60% 的无效 token。

    截断方式由 trunc_mode 控制（对应 config.TRUNC_MODE / --trunc）：
      longest_first  HF 原生策略，优先缩短最长序列，默认丢弃右侧token。
                     **已完成实验用的就是这个，改名会让已有指标失效。**
      head           只保留回答开头
      tail           只保留回答结尾
      head_tail      回答取开头 75% + 结尾 25%，避免长回答的结论句被截掉
                     （政治访谈里"到底答没答"往往由最后一句定调）
    """

    def __init__(
        self, df, tokenizer, max_len, labels=None, label2id=None, trunc_mode="longest_first"
    ):
        self.a = df["text_a"].astype(str).tolist()
        self.b = df["text_b"].astype(str).tolist()
        self.tok = tokenizer
        self.max_len = max_len
        self.trunc_mode = trunc_mode
        self.labels = None if labels is None else [label2id[str(x)] for x in df[labels].astype(str)]

    def __len__(self):
        return len(self.a)

    @staticmethod
    def _slice(ids, n, mode):
        """把 ids 裁到 n 个 token。head 取前 n，tail 取后 n，head_tail 取首尾。"""
        if len(ids) <= n:
            return ids
        if mode == "tail":
            return ids[-n:]
        if mode == "head_tail":
            h = int(n * C.HEAD_RATIO)
            t = n - h
            return (ids[:h] + ids[-t:]) if t > 0 else ids[:h]
        return ids[:n]

    def _build_pair(self, ea, eb):
        """拼成句对 ids。transformers 5.x 已移除 build_inputs_with_special_tokens，
        因此按模型族手动构造；老版本仍优先走官方方法。"""
        tok = self.tok
        fn = getattr(tok, "build_inputs_with_special_tokens", None)
        if callable(fn):
            try:
                return fn(ea, eb)
            except Exception:
                pass
        sep = tok.sep_token_id
        if getattr(tok, "cls_token", None) == "<s>":
            # RoBERTa / BART: <s> A </s></s> B </s>
            return [tok.bos_token_id] + ea + [sep, sep] + eb + [sep]
        # BERT 系: [CLS] A [SEP] B [SEP]
        return [tok.cls_token_id] + ea + [sep] + eb + [sep]

    def _encode(self, a, b):
        """按 trunc_mode 编码句对；longest_first 直接交给 tokenizer。"""
        if self.trunc_mode == "longest_first":
            return self.tok(
                a, b, truncation="longest_first", max_length=self.max_len, padding=False
            )
        ea = self.tok(a, add_special_tokens=False)["input_ids"]
        eb = self.tok(b, add_special_tokens=False)["input_ids"]
        budget = max(8, self.max_len - self.tok.num_special_tokens_to_add(pair=True))
        # 问题平均仅 84 字符，最多分它 1/4，其余留给回答
        max_a = min(len(ea), max(16, budget // 4))
        ea = ea[:max_a]
        eb = self._slice(eb, max(1, budget - len(ea)), self.trunc_mode)
        ids = self._build_pair(ea, eb)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def __getitem__(self, i):
        enc = self._encode(self.a[i], self.b[i])
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item, self.a[i]


class PadCollator:
    """按批内最长序列补齐（dynamic padding）。

    写成类而不是闭包，因为 Windows 的多进程 DataLoader 用 spawn 启动，
    需要可 pickle 的 collate_fn，闭包无法序列化。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch

        items, texts = zip(*batch)
        out = {}
        for k in items[0]:
            if k == "labels":
                out[k] = torch.stack([it[k] for it in items])
            else:
                pad_val = self.pad_token_id if k == "input_ids" else 0
                out[k] = torch.nn.utils.rnn.pad_sequence(
                    [it[k] for it in items], batch_first=True, padding_value=pad_val
                )
        return out, list(texts)


def _avg_len(loader, n=64):
    """估一下批内平均序列长度，用于报告与日志。"""
    tot = cnt = 0
    for i, (batch, _) in enumerate(loader):
        tot += int(batch["input_ids"].size(1))
        cnt += 1
        if i + 1 >= n:
            break
    return tot / max(cnt, 1)


# ---------------------------------------------------------------- 训练
def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train=True,
    max_grad=1.0,
    scaler=None,
    use_amp=True,
    sched=None,
):
    """跑一轮。loader 若不含标签（如官方 test），只做预测不算损失。

    混合精度用 bf16：RTX 5070（Blackwell）原生支持 bf16，动态范围与 fp32 一致，
    无需 GradScaler 的 loss scaling，也就不会有 fp16 下梯度被跳过导致的「训练不动」。

    sched 必须**按 step** 前进：get_linear_schedule_with_warmup 是按优化步计数的，
    若放在 epoch 循环里每轮只走一步，学习率会一直卡在 warmup 的起点附近
    （2e-5 × 1/36 ≈ 5.6e-7），表现为 loss 恒定、模型塌缩到单一类别。
    """
    import torch

    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds, golds, probs = [], [], []
    autocast = torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=use_amp and device == "cuda"
    )
    torch.set_grad_enabled(train)
    for batch, _ in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        has_label = "labels" in batch
        labels = batch.get("labels")
        with autocast:
            out = model(**batch)
            loss = criterion(out.logits.float(), labels) if has_label else None
        if train:
            optimizer.zero_grad()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
            if sched is not None:
                sched.step()
        p = out.logits.argmax(-1)
        preds += p.detach().cpu().tolist()
        # 模板要求「每个样本的 gold、prediction、各类别概率」，这里一并收集
        probs.append(torch.softmax(out.logits.float(), dim=-1).detach().cpu().numpy())
        if has_label:
            golds += labels.cpu().tolist()
            correct += (p == labels).sum().item()
            loss_sum += loss.item() * len(p)
        total += len(p)
    torch.set_grad_enabled(True)
    proba = np.vstack(probs) if probs else np.zeros((0, 0))
    return (
        loss_sum / max(total, 1),
        correct / max(total, 1),
        np.array(golds),
        np.array(preds),
        proba,
    )


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.ENCODER_MODEL)
    ap.add_argument("--lr", type=float, default=C.ENCODER_LR)
    ap.add_argument("--batch", type=int, default=C.ENCODER_BATCH)
    ap.add_argument("--epochs", type=int, default=C.ENCODER_EPOCHS)
    ap.add_argument("--max-len", type=int, default=C.MAX_LEN)
    ap.add_argument(
        "--trunc",
        default=C.TRUNC_MODE,
        choices=["longest_first", "head", "tail", "head_tail"],
        help="截断方式。默认 longest_first 与已完成实验一致；"
        "改成 head/tail/head_tail 会改变行为，已有指标与报告数字将失效",
    )
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument(
        "--target",
        choices=["clarity", "evasion"],
        default="clarity",
        help="evasion = 论文的 evasion-boost 两阶段方案",
    )
    ap.add_argument("--frozen", action="store_true", help="冻结编码器，仅训练分类头")
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--tag", default=None, help="输出文件标识，默认自动生成")
    ap.add_argument(
        "--transfer-csv",
        default=None,
        help="迁移实验：用外部统一样本 csv 训练/评测（由 adapters.py 产出）",
    )
    ap.add_argument(
        "--init-from",
        default=None,
        help="从已训好的编码器权重初始化（迁移实验的「直接迁移」需要用）",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="数据目录，默认 data/processed；做清洗前后消融时指向 _before_clean",
    )
    ap.add_argument(
        "--workers", type=int, default=2, help="DataLoader 进程数；Windows 下若报序列化错误改为 0"
    )
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度（默认开启，约提速 2 倍）")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    enable_mirror()
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 62)
    print(f"基线 B-2：{args.model} 句对微调")
    print(f"  target={args.target}  frozen={args.frozen}  device={device}")
    if device == "cuda":
        print(
            f"  GPU: {torch.cuda.get_device_name(0)}  "
            f"显存 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    print("=" * 62)

    # ---------------- 数据
    if args.transfer_csv:
        # 迁移模式：传入的 csv 需含 text_a/text_b/label/split
        d = pd.read_csv(args.transfer_csv)
        fit = d[d["split"] == "train"]
        test = d[d["split"] == "test"]
        val = d[d["split"] == "val"]
        if fit.empty or val.empty or test.empty:
            raise ValueError("迁移数据必须包含独立的train/val/test")
        if set(fit.text_a) & set(val.text_a):
            raise ValueError("迁移train/val问题组重叠")
        for left, right in ((fit, test), (val, test)):
            if set(zip(left.text_a, left.text_b)) & set(zip(right.text_a, right.text_b)):
                raise ValueError("迁移训练或验证文本对与test重叠")
        label_names = sorted(fit["label"].unique().tolist())
        col = "label"  # 迁移用的统一样本固定用 label 列
        print(f"[迁移] 训练 {len(fit)} 条 / 测试 {len(test)} 条 | 标签 {label_names}")
    else:
        data_dir = Path(args.data_dir) if args.data_dir else C.DATA_PROC
        tr = pd.read_csv(data_dir / "qevasion_train.csv")
        te = pd.read_csv(data_dir / "qevasion_test.csv")
        fit = tr[tr["role"] == "train"]
        val = tr[tr["role"] == "val"]
        test = te
        if args.target == "evasion":
            label_names = C.EVASION_LABELS
        else:
            label_names = C.CLARITY_LABELS
        col = TARGET_COL[args.target]
        fit = fit[fit[col].isin(label_names)]
        val = val[val[col].isin(label_names)]

    label2id = {n: i for i, n in enumerate(label_names)}
    if not set(test[col if args.transfer_csv else "label"]).issubset(
        label_names if args.target == "clarity" else C.CLARITY_LABELS
    ):
        raise ValueError("测试集含未知标签")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        id2label={i: n for n, i in label2id.items()},
        label2id=label2id,
    ).to(device)

    if args.frozen:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("[frozen] 编码器已冻结，仅训练分类头")

    # 迁移实验的「直接迁移」必须真的加载原任务训好的编码器，
    # 而不是只用 roberta-base 的原始权重 —— 否则四行设置的语义就不成立。
    init_record = None
    if args.init_from:
        ck = Path(args.init_from)
        if ck.is_dir():
            ck = ck / "encoder.pt"
        if not ck.exists():
            raise SystemExit(f"找不到权重文件: {ck}")
        sd = torch.load(ck, map_location="cpu")
        # 分类头维度不同（3 类 vs 2 类），只加载主干参数
        cur = model.state_dict()
        keep = {
            k: v
            for k, v in sd.items()
            if k in cur and cur[k].shape == v.shape and not k.startswith("classifier")
        }
        missing = model.load_state_dict(keep, strict=False)
        n_loaded = len(keep)
        n_total = len([k for k in cur if not k.startswith("classifier")])
        if n_loaded != n_total:
            raise ValueError(f"源模型主干加载不完整: {n_loaded}/{n_total}")
        init_record = {
            "path": str(ck.resolve()),
            "sha256": sha256(ck),
            "loaded_tensors": n_loaded,
            "expected_tensors": n_total,
        }
        print(f"[init-from] 已加载 {n_loaded}/{n_total} 个主干张量 <- {ck}")
        if missing.unexpected_keys:
            print(f"  未使用键 {len(missing.unexpected_keys)} 个（分类头按预期跳过）")

    collate = PadCollator(tok.pad_token_id)
    nw = args.workers
    fit_ds = PairDataset(fit, tok, args.max_len, col, label2id, trunc_mode=args.trunc)
    train_loader = DataLoader(
        fit_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=nw,
        persistent_workers=nw > 0,
    )
    val_loader = (
        DataLoader(
            PairDataset(val, tok, args.max_len, col, label2id, trunc_mode=args.trunc),
            batch_size=args.batch * 2,
            collate_fn=collate,
            num_workers=nw,
            persistent_workers=nw > 0,
        )
        if val is not None
        else None
    )
    test_loader = DataLoader(
        PairDataset(test, tok, args.max_len, None, None, trunc_mode=args.trunc),
        batch_size=args.batch * 2,
        collate_fn=collate,
        num_workers=nw,
    )

    # ---------------- 类别加权损失
    if args.no_class_weight:
        weights = np.ones(len(label_names))
    else:
        cnt = np.array([(fit[col] == n).sum() for n in label_names], dtype=float)
        weights = cnt.sum() / (len(label_names) * np.maximum(cnt, 1))
    print(f"类别权重: {dict(zip(label_names, weights.round(3)))}")
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    # ---------------- 优化器 + 混合精度
    # bf16 动态范围等同 fp32，不需要 loss scaling，因此不使用 GradScaler。
    use_amp = (not args.no_amp) and device == "cuda"
    scaler = None
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * C.ENCODER_WARMUP_RATIO))
    sched = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(
        f"混合精度 AMP(bf16) = {use_amp} | DataLoader workers = {args.workers} "
        f"| 每轮步数 = {len(train_loader)} | 总步数 = {total_steps} | warmup = {warmup_steps} 步"
    )

    # ---------------- 训练 + 早停
    best_f1, best_state, bad, log = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            True,
            scaler=scaler,
            use_amp=use_amp,
            sched=sched,
        )
        lr_now = sched.get_last_lr()[0]
        line = {
            "epoch": ep,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "lr": f"{lr_now:.2e}",
        }
        if val_loader is not None:
            _, va_acc, g, p, _ = run_epoch(
                model,
                val_loader,
                optimizer,
                criterion,
                device,
                False,
                scaler=scaler,
                use_amp=use_amp,
            )
            # run_epoch 返回的是类别下标，需转回标签名才能与 label_names 对齐
            g_name = np.array([label_names[int(i)] for i in g])
            p_name = np.array([label_names[int(i)] for i in p])
            if args.target == "evasion" and not args.transfer_csv:
                line["val_evasion_macro_f1"] = compute_metrics(g_name, p_name, labels=label_names)[
                    "macro_f1"
                ]
                g_eval = [C.EVASION_TO_CLARITY[x] for x in g_name]
                p_eval = [C.EVASION_TO_CLARITY[x] for x in p_name]
                va_f1 = compute_metrics(g_eval, p_eval, labels=C.CLARITY_LABELS)["macro_f1"]
            else:
                va_f1 = compute_metrics(g_name, p_name, labels=label_names)["macro_f1"]
            line.update({"val_acc": round(va_acc, 4), "val_macro_f1": round(va_f1, 4)})
            if va_f1 > best_f1:
                best_f1, bad = va_f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        log.append(line)
        print("  " + json.dumps(line, ensure_ascii=False), flush=True)
        if val_loader is not None and bad >= C.ENCODER_EARLY_STOP:
            print(f"  早停：验证集 Macro-F1 连续 {bad} 轮未提升")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  已恢复最佳权重 (val macro_f1 = {best_f1:.4f})")
    elapsed = time.time() - t0

    # ---------------- 评测
    _, te_acc, gold_raw, pred_raw, te_proba = run_epoch(
        model, test_loader, optimizer, criterion, device, False, scaler=scaler, use_amp=use_amp
    )

    if args.target == "evasion" and not args.transfer_csv:
        # evasion-boost：把 9 类预测按 taxonomy 映射回 3 类 clarity 再评测
        inv = {i: n for n, i in label2id.items()}
        pred = np.array([C.EVASION_TO_CLARITY[inv[int(i)]] for i in pred_raw])
        gold = test["label"].values
        eval_labels = C.CLARITY_LABELS
        # 概率同样要聚合到 3 类：把归入同一 clarity 的 9 类概率相加
        proba_save = np.zeros((len(te_proba), len(C.CLARITY_LABELS)))
        for i, nm in enumerate(label_names):
            proba_save[:, C.CLARITY_LABELS.index(C.EVASION_TO_CLARITY[nm])] += te_proba[:, i]
        print("\n[evasion-boost] 9 类预测已映射为 3 类 clarity 进行评测（概率已按映射聚合）")
        print("[注意] 官方 test 的 evasion_label 为空，无法报告 9 类分数（9 类指标在验证集上报告）")
    else:
        pred = np.array([label_names[int(i)] for i in pred_raw])
        gold = test["label"].values
        eval_labels = label_names
        proba_save = te_proba

    name = args.tag or (
        f"{model_name(args.model)}_{args.target}"
        f"{'_frozen' if args.frozen else ''}{'_noweight' if args.no_class_weight else ''}_s{args.seed}"
    )
    decision_rule = (
        "fine_argmax_then_map" if args.target == "evasion" and not args.transfer_csv else "argmax"
    )
    m = evaluate_and_save(
        name,
        gold,
        pred,
        proba=proba_save,
        class_names=eval_labels,
        texts=test["text_a"].values,
        decision_rule=decision_rule,
    )

    # 权重落盘：迁移实验的「直接迁移」要加载原任务训好的编码器，没有这一步就无从谈起
    ck_dir = C.CKPT / name
    ck_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck_dir / "encoder.pt")
    tok.save_pretrained(ck_dir)
    # 必须同时存模型 config：否则别人拿到权重目录无法 from_pretrained（缺 model_type）
    model.save_pretrained(ck_dir, safe_serialization=True)
    print(f"权重已保存: {ck_dir / 'encoder.pt'}")

    with open(C.METRICS / f"{name}_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "target": args.target,
                "frozen": args.frozen,
                "class_weight": not args.no_class_weight,
                "lr": args.lr,
                "batch": args.batch,
                "epochs_run": len(log),
                "max_len": args.max_len,
                "trunc": args.trunc,
                "seed": args.seed,
                "device": device,
                "elapsed_sec": round(elapsed, 1),
                "best_val_macro_f1": best_f1,
                "train_log": log,
                "amp": use_amp,
                "dynamic_padding": True,
                "workers": args.workers,
                "avg_batch_seq_len": round(_avg_len(train_loader), 1),
                "train_n": len(fit),
                "val_n": len(val) if val is not None else 0,
                "test_n": len(test),
                "labels": label_names,
                "class_weights": dict(zip(label_names, weights.tolist())),
                "init_from": init_record,
                "run_id": args.run_id,
                "data_sha256": sha256(args.transfer_csv)
                if args.transfer_csv
                else sha256(data_dir / "qevasion_train.csv"),
                "prediction_precision": "bf16" if use_amp else "fp32",
                "decision_rule": decision_rule,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n耗时 {elapsed:.1f}s  模型输出: {name}")
    return m


if __name__ == "__main__":
    main()
