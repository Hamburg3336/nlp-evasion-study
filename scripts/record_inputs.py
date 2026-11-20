"""Record acquired local inputs; optionally require the original byte hashes."""

import argparse
import json
from pathlib import Path

from artifact_io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "experiment")
    parser.add_argument("--strict-reference", action="store_true")
    args = parser.parse_args()
    reference = read_json(ROOT / "provenance/reference_input_checksums.json")
    result = {}
    for relative, original in reference.items():
        source = args.experiment / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing input: {source}")
        record = dict(sha256=sha256_file(source), bytes=source.stat().st_size)
        if args.strict_reference and record != original:
            raise ValueError(f"Input differs from original snapshot: {relative}")
        result[relative] = record
    destination = args.experiment / "data/source_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(recorded=len(result), matches_reference=result == reference), indent=2))


if __name__ == "__main__":
    main()
