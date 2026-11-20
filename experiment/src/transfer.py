"""目标任务独立开发集选参；模板四行迁移对照，失败立即终止。"""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

import config as C
import pandas as pd
from adapters import ADAPTERS, build_and_save
from evaluate import compute_metrics, evaluate_and_save
from protocol import data_record, sha256, validate_splits, write_json
from sklearn.model_selection import GroupShuffleSplit


def prepare_target(key):
    C.ensure_directories()
    train_path = C.DATA_PROC / f"transfer_{key}_train.csv"
    test_path = C.DATA_PROC / f"transfer_{key}_test.csv"
    if train_path.exists() and test_path.exists():
        raw_train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    else:
        frames = build_and_save(key)
        raw_train, test = frames["train"], frames["test"]
    # 只使用题目文本分组，不使用测试标签；相同问题不跨训练/开发集合。
    raw_train = raw_train.copy()
    raw_train["group"] = raw_train.text_a.astype(str).str.strip().str.casefold()
    idx_fit, idx_val = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=C.SEED).split(
            raw_train, groups=raw_train.group
        )
    )
    fit, val = raw_train.iloc[idx_fit].copy(), raw_train.iloc[idx_val].copy()
    test = test.copy()
    test["group"] = test.text_a.astype(str).str.strip().str.casefold()
    validate_splits(fit, val, test)
    for frame, split in ((fit, "train"), (val, "val"), (test, "test")):
        frame["split"] = split
        frame.to_csv(
            C.DATA_PROC / f"transfer_{key}_revised_{split}.csv", index=False, encoding="utf-8-sig"
        )
    unified = C.DATA_PROC / f"transfer_{key}_revised_unified.csv"
    pd.concat([fit, val, test], ignore_index=True).to_csv(
        unified, index=False, encoding="utf-8-sig"
    )
    write_json(
        C.METRICS / f"transfer_meta_{key}.json",
        {
            "key": key,
            "hf_path": ADAPTERS[key]["hf_path"],
            "evaluation_split": "GLUE validation" if key == "qnli" else "test",
            "sizes": {"train": len(fit), "val": len(val), "test": len(test)},
            "seed": C.SEED,
            "grouping": "normalized question",
            "question_group_overlap": {
                "train_val": len(set(fit.group) & set(val.group)),
                "train_test": len(set(fit.group) & set(test.group)),
                "val_test": len(set(val.group) & set(test.group)),
            },
            "test_policy": "retain official evaluation split; exact text pairs checked separately",
            "source_sha256": data_record([train_path, test_path]),
            "unified_sha256": sha256(unified),
            "label_dist": {
                s: d.label.value_counts().to_dict()
                for s, d in (("train", fit), ("val", val), ("test", test))
            },
        },
    )
    return fit, val, test, unified


def tfidf_within_domain(fit, val, test, name):
    from train_tfidf import build_text, make_pipeline

    labels = sorted(fit.label.unique())
    candidates, validation_scores = [], {}
    for cfg in C.TFIDF_CONFIGS:
        pipe = make_pipeline(cfg["ngram_range"], cfg["class_weight"], C.SEED)
        pipe.fit(build_text(fit), fit.label)
        score = compute_metrics(val.label, pipe.predict(build_text(val)), labels=labels)["macro_f1"]
        validation_scores[cfg["name"]] = score
        candidates.append((score, cfg, pipe))
        print(f"[dev] {cfg['name']}: {score:.4f}", flush=True)
    _, chosen, pipe = max(candidates, key=lambda c: c[0])
    # 仅选定配置在最终评测集上预测一次，不用测试标签选择模型。
    x_test = build_text(test)
    proba = pipe.predict_proba(x_test)
    order = [list(pipe["clf"].classes_).index(label) for label in labels]
    tag = f"transfer_{name}_tfidf"
    metrics = evaluate_and_save(
        tag,
        test.label,
        pipe.predict(x_test),
        proba[:, order],
        class_names=labels,
        texts=test.text_a,
    )
    write_json(
        C.METRICS / f"{tag}_config.json",
        {
            "selected_config": chosen,
            "validation_scores": validation_scores,
            "criterion": "val_macro_f1",
            "refit_on_val": False,
            "train_n": len(fit),
            "val_n": len(val),
            "test_n": len(test),
            "seed": C.SEED,
        },
    )
    return metrics, chosen["name"]


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--target", default="qnli")
    ap.add_argument("--with-encoder", action="store_true")
    ap.add_argument("--source-model-name", default="roberta_clarity_s42")
    ap.add_argument("--encoder-dir", default=str(C.ROOT / "models/roberta-base"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--controls", action="store_true", help="额外从原始RoBERTa做相同目标任务训练")
    args = ap.parse_args()
    if args.list:
        for key, spec in ADAPTERS.items():
            print(key, spec["hf_path"], spec["note"])
        return
    if args.target not in ADAPTERS or args.target == "qevasion":
        raise ValueError("请选择已注册的目标任务")
    src_metrics = C.METRICS / f"{args.source_model_name}.json"
    src_ckpt = C.CKPT / args.source_model_name / "encoder.pt"
    if not src_metrics.exists():
        raise FileNotFoundError(f"缺少本次原任务结果: {src_metrics}")
    if args.with_encoder and not src_ckpt.is_file():
        raise FileNotFoundError(f"缺少源任务权重，拒绝改用原始权重: {src_ckpt}")
    run_id = args.run_id or uuid.uuid4().hex
    fit, val, test, unified = prepare_target(args.target)
    src = json.loads(src_metrics.read_text(encoding="utf-8"))
    rows = [
        {
            "实验设置": "原方法基线",
            "Accuracy": src["accuracy"],
            "Macro-F1": src["macro_f1"],
            "主要结论": "QEvasion三分类，seed42；不与QNLI绝对分数横比",
        }
    ]
    baseline, config = tfidf_within_domain(fit, val, test, args.target)
    rows.append(
        {
            "实验设置": "迁移数据基线",
            "Accuracy": baseline["accuracy"],
            "Macro-F1": baseline["macro_f1"],
            "主要结论": f"开发集选{config}；目标域TF-IDF参照",
        }
    )
    if args.with_encoder:
        for frozen, label in ((True, "直接迁移"), (False, "适配后训练")):
            tag = f"transfer_{args.target}_{'frozen' if frozen else 'finetuned'}"
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("train_encoder.py")),
                "--model",
                args.encoder_dir,
                "--transfer-csv",
                str(unified),
                "--init-from",
                str(src_ckpt),
                "--tag",
                tag,
                "--run-id",
                run_id,
            ]
            if frozen:
                cmd.append("--frozen")
            subprocess.run(cmd, check=True)
            cfg = json.loads((C.METRICS / f"{tag}_config.json").read_text(encoding="utf-8"))
            if cfg.get("run_id") != run_id or cfg.get("init_from", {}).get("sha256") != sha256(
                src_ckpt
            ):
                raise ValueError("迁移结果不是本次运行或源权重不匹配")
            m = json.loads((C.METRICS / f"{tag}.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "实验设置": label,
                    "Accuracy": m["accuracy"],
                    "Macro-F1": m["macro_f1"],
                    "主要结论": "源任务编码器冻结，目标标签训练新头"
                    if frozen
                    else "源任务初始化，目标域整体微调；开发集早停",
                }
            )
        if args.controls:
            for frozen in (True, False):
                tag = f"control_{args.target}_{'frozen' if frozen else 'finetuned'}"
                cmd = [
                    sys.executable,
                    str(Path(__file__).with_name("train_encoder.py")),
                    "--model",
                    args.encoder_dir,
                    "--transfer-csv",
                    str(unified),
                    "--tag",
                    tag,
                    "--run-id",
                    run_id,
                ]
                if frozen:
                    cmd.append("--frozen")
                subprocess.run(cmd, check=True)
    else:
        for label in ("直接迁移", "适配后训练"):
            rows.append(
                {
                    "实验设置": label,
                    "Accuracy": None,
                    "Macro-F1": None,
                    "主要结论": "本次未运行编码器；未读取历史结果",
                }
            )
    pd.DataFrame(rows).to_csv(
        C.METRICS / f"table7_transfer_{args.target}.csv", index=False, encoding="utf-8-sig"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
