"""从本次逐轮日志生成图5，包含实际选择的最佳epoch。"""

import json

import config as C
from evaluate import _setup_mpl


def main():
    C.ensure_directories()
    cfg = json.loads((C.METRICS / "roberta_clarity_s42_config.json").read_text(encoding="utf-8"))
    log = cfg["train_log"]
    plt = _setup_mpl()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.5))
    epoch = [r["epoch"] for r in log]
    axes[0].plot(epoch, [r["train_loss"] for r in log], "o-", label="训练损失")
    axes[0].set(xlabel="epoch", ylabel="loss", title="训练损失")
    axes[1].plot(epoch, [r["val_macro_f1"] for r in log], "o-", label="验证 Macro-F1")
    axes[1].plot(epoch, [r["val_acc"] for r in log], "s--", label="验证 Accuracy")
    best = max(log, key=lambda r: r["val_macro_f1"])
    axes[1].axvline(best["epoch"], linestyle=":", color="gray", label=f"选中epoch={best['epoch']}")
    axes[1].set(xlabel="epoch", ylabel="score", title="独立验证集选择模型")
    axes[1].legend()
    fig.suptitle("图5 RoBERTa训练过程（seed=42）")
    fig.tight_layout()
    fig.savefig(C.FIG / "fig5_training_curve.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
