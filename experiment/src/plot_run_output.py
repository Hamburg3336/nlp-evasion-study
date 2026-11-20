"""Generate an image of verified run-output excerpts for the report."""

import json
from datetime import date

import config as C
import pandas as pd
from evaluate import _setup_mpl, plot_confusion


def main():
    C.ensure_directories()
    raw = pd.read_parquet(C.DATA_RAW / "train-00000-of-00001.parquet")
    agree = int((raw.evasion_label.map(C.EVASION_TO_CLARITY) == raw.clarity_label).sum())
    train = pd.read_csv(C.DATA_PROC / "qevasion_train.csv")
    lines = [
        f"QEvasion revised experiment {date.today().isoformat()}",
        f"Raw train rows / taxonomy matched: {len(raw)} / {agree}",
        f"Fit / dev / test: {(train.role == 'train').sum()} / {(train.role == 'val').sum()} / 308",
        f"Fit/dev title-group overlap: {len(set(train[train.role == 'train'].group) & set(train[train.role == 'val'].group))}",
    ]
    for tag in [
        "majority",
        "roberta_clarity_s42",
        "transfer_qnli_frozen",
        "transfer_qnli_finetuned",
        "control_qnli_frozen",
        "control_qnli_finetuned",
        "length_only_s42",
    ]:
        m = json.loads((C.METRICS / f"{tag}.json").read_text(encoding="utf-8"))
        lines.append(f"{tag}: acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")
    supplement = json.loads((C.METRICS / "supplement_comparison.json").read_text(encoding="utf-8"))
    evasion = supplement["evasion"]
    lines.append(f"Evasion 3 seeds: F1={evasion['mean']:.4f} +/- {evasion['sample_std']:.4f}")
    check = json.loads((C.METRICS / "artifact_verification.json").read_text(encoding="utf-8"))
    lines.append(f"Artifact verification: {check['status']}")
    text = "\n".join(lines)
    print(text)
    (C.METRICS / "run_output.txt").write_text(text, encoding="utf-8")
    plt = _setup_mpl()
    fig = plt.figure(figsize=(11, 4.8), facecolor="#15202b")
    fig.text(
        0.03,
        0.94,
        text,
        ha="left",
        va="top",
        fontfamily="DejaVu Sans Mono",
        fontsize=10.5,
        color="#e5f1fa",
        linespacing=1.6,
    )
    fig.savefig(C.FIG / "run_output.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    chosen = json.loads((C.METRICS / "tfidf_selection.json").read_text(encoding="utf-8"))[
        "selected_model"
    ]
    best = json.loads((C.METRICS / "best_model.json").read_text(encoding="utf-8"))["selected_model"]
    for tag in [best, chosen]:
        data = pd.read_csv(C.PRED / f"{tag}.csv")
        plot_confusion(data.gold, data.prediction, tag)


if __name__ == "__main__":
    main()
