"""Shared split validation and provenance for the revised experiments."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_name(value):
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1]


def validate_splits(fit, val, test):
    if any(frame.empty for frame in (fit, val, test)):
        raise ValueError("train / val / test must all be nonempty")
    if set(fit["group"]) & set(val["group"]):
        raise ValueError("train and val groups overlap")
    for a, b in ((fit, val), (fit, test), (val, test)):
        left = set(zip(a.text_a, a.text_b))
        right = set(zip(b.text_a, b.text_b))
        if left & right:
            raise ValueError("identical text pairs occur across splits")


def data_record(paths):
    return {str(Path(p).resolve()): sha256(p) for p in paths}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
