"""Plan or execute the primary study and its supplementary controls."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiment"
SUP = ROOT / "补充实验"


def jobs(stage):
    commands = []
    if stage in {"all", "primary"}:
        commands.append([EXP / "run_all.py", "--stage", "all"])
    if stage in {"all", "supplement"}:
        commands.extend(
            [
                [ROOT / "scripts/record_inputs.py"],
                [
                    ROOT / "scripts/hierarchy_analysis.py",
                    "--experiment",
                    EXP,
                    "--output",
                    SUP / "层级分析_20261116",
                ],
                [
                    ROOT / "scripts/audit_overlap_length.py",
                    "--experiment",
                    EXP,
                    "--hierarchy",
                    SUP / "层级分析_20261116",
                    "--output",
                    SUP / "数据隔离与长度_20261116",
                ],
                [
                    ROOT / "scripts/run_weight_controls.py",
                    "--experiment",
                    EXP,
                    "--output",
                    SUP / "权重消融_20261116",
                ],
                [
                    ROOT / "scripts/export_control_dev.py",
                    "--experiment",
                    EXP,
                    "--controls",
                    SUP / "权重消融_20261116",
                    "--output",
                    SUP / "权重消融开发集_20261116",
                ],
                [
                    ROOT / "scripts/analyze_weight_controls.py",
                    "--experiment",
                    EXP,
                    "--controls",
                    SUP / "权重消融_20261116",
                    "--audit",
                    SUP / "数据隔离与长度_20261116",
                    "--development",
                    SUP / "权重消融开发集_20261116",
                    "--output",
                    SUP / "权重消融分析_20261116",
                ],
                [
                    ROOT / "scripts/run_frozen_probe.py",
                    "--experiment",
                    EXP,
                    "--output",
                    SUP / "冻结表示跨源种子_20261117",
                    "--source-seeds",
                    "42",
                    "2024",
                    "3407",
                ],
                [ROOT / "scripts/audit_raw_fine_labels.py"],
                [
                    ROOT / "scripts/run_ambiguity_controls.py",
                    "--experiment",
                    EXP,
                    "--output",
                    SUP / "细类歧义拟合排除_20261117",
                ],
                [
                    ROOT / "scripts/analyze_ambiguity_controls.py",
                    "--experiment",
                    EXP,
                    "--controls",
                    SUP / "细类歧义拟合排除_20261117",
                    "--output",
                    SUP / "细类歧义排除分析_20261117",
                ],
            ]
        )
    return [[sys.executable, *map(str, command)] for command in commands]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "primary", "supplement"], default="all")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List commands without downloading, writing, or training",
    )
    args = parser.parse_args(argv)
    env = dict(os.environ, NLP_OUTPUT_DIR=str(EXP / "outputs/revised"))
    for command in jobs(args.stage):
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
