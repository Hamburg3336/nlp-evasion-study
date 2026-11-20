"""Small UTF-8 and streaming-hash helpers shared by delivery scripts."""

import hashlib
import json
from pathlib import Path


def read_json(path):
    """Accept UTF-8 JSON with or without the BOM produced by PowerShell."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path):
    """Hash large model files without loading the complete file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
