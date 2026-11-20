"""Plot observational diagnostics and optionally the completed 2x2 control study."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--control-analysis", type=Path)
    args = ap.parse_args()
    out = args.audit
    bins = ["0-128", "129-512", "513-1024", "1025+"]
    labels = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
    colors = ["#4477AA", "#EEAA33", "#228833"]
    test = pd.read_csv(out / "test_sample_audit.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(4)
    for lab, color in zip(labels, colors):
        counts = np.array([int(((test.length_bin == b) & (test.label == lab)).sum()) for b in bins])
        ax.bar(bins, counts, bottom=bottom, label=lab, color=color)
        for i, n in enumerate(counts):
            if n >= 4:
                ax.text(i, bottom[i] + n / 2, str(n), ha="center", va="center", fontsize=10)
        bottom += counts
    ax.set(
        xlabel="Full answer length in tokens",
        ylabel="Test samples",
        title="Length bins have different label composition (n=308)",
    )
    ax.set_xticks(range(4), [f"{b}\nn={int((test.length_bin == b).sum())}" for b in bins])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "test_length_label_composition.png", dpi=160)
    plt.close(fig)
    metrics = pd.read_csv(out / "sensitivity_and_length_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, color, offset, label in [
        ("clarity", "#4477AA", -0.08, "Direct 3-class"),
        ("evasion_boost", "#CC6677", 0.08, "9-class then map"),
    ]:
        means = []
        for i, b in enumerate(bins):
            rows = metrics[
                (metrics.split == "test")
                & (metrics.model == model)
                & (metrics.subset == f"answer_tokens_{b}")
            ]
            values = rows.macro_f1.to_numpy()
            means.append(values.mean())
            ax.scatter(np.full(len(values), i + offset), values, color=color, alpha=0.6, s=23)
        ax.plot(np.arange(4) + offset, means, color=color, marker="D", label=label)
    ax.set_xticks(range(4), [f"{b}\nn={int((test.length_bin == b).sum())}" for b in bins])
    ax.set(
        xlabel="Full answer tokens",
        ylabel="Macro-F1 over the fixed 3 labels",
        ylim=(0, 1),
        title="Observational length diagnostics: dots are seeds, diamonds are means",
    )
    ax.legend(loc="upper right")
    fig.text(
        0.5,
        0.015,
        "Different label mix and very small rare-class counts prevent a causal truncation claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "test_length_model_comparison.png", dpi=160)
    plt.close(fig)
    if args.control_analysis:
        p = args.control_analysis
        summary = pd.read_csv(p / "four_arm_summary.csv")
        data = summary[summary.subset == "all"]
        contrasts = pd.read_csv(p / "paired_contrasts_cluster_bootstrap.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        names = {
            "clarity_weighted": "3-class / weighted",
            "evasion_boost_weighted": "9-class / weighted",
            "clarity_unweighted": "3-class / unweighted",
            "evasion_boost_unweighted": "9-class / unweighted",
        }
        x = np.arange(len(data))
        axes[0].bar(
            x,
            data.macro_f1_mean,
            yerr=data.macro_f1_sample_std,
            capsize=4,
            color=["#4477AA", "#CC6677", "#66CCEE", "#EE99AA"],
        )
        axes[0].set_xticks(x, [names[s] for s in data.arm], rotation=25, ha="right")
        axes[0].set(ylabel="Test Macro-F1", ylim=(0, 1), title="Mean and sample SD over 3 seeds")
        short = {
            "granularity_with_weights": "9 vs 3, weighted",
            "granularity_without_weights": "9 vs 3, unweighted",
            "weighting_direct": "Weighting, 3-class",
            "weighting_hierarchy": "Weighting, 9-class",
            "granularity_by_weighting_interaction": "Interaction",
        }
        y = np.arange(len(contrasts))
        for i, r in contrasts.iterrows():
            axes[1].plot([r.bootstrap_low, r.bootstrap_high], [i, i], color="#4477AA", linewidth=2)
            axes[1].scatter(r.delta_unrounded_mean, i, color="#CC6677", s=35, zorder=3)
        axes[1].axvline(0, color="gray", linestyle="--")
        axes[1].set_yticks(y, [short[s] for s in contrasts.contrast])
        axes[1].invert_yaxis()
        axes[1].set(
            xlabel="Paired difference in Macro-F1", title="95% answer-cluster bootstrap intervals"
        )
        fig.text(
            0.5,
            0.015,
            "Bootstrap intervals condition on the trained checkpoints; they do not include retraining uncertainty.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(p / "weighting_control_comparison.png", dpi=160)
        plt.close(fig)
    print("Plots saved", flush=True)


if __name__ == "__main__":
    main()
