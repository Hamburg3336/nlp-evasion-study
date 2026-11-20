"""用httpx下载模型文件，默认官方站点，也支持HF_ENDPOINT指定入口。

该HTTP兼容路径用于无法通过常规Hub客户端下载的环境；
不假定所有网络都需要镜像或都存在同一种代理问题。

用法:
    python src/fetch_model.py roberta-base
    python src/fetch_model.py sentence-transformers/all-MiniLM-L6-v2
    python src/fetch_model.py hfl/chinese-roberta-wwm-ext

下载到 models/<repo 名>/ 后，直接把这个目录路径传给各脚本的 --model 参数：
    python src/train_encoder.py --model models/roberta-base
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

MIRROR = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
MODELS_DIR = C.ROOT / "models"

# 这些文件不需要（体积大或无用）
SKIP_PATTERNS = [
    r"^\.gitattributes$",
    r"^README\.md$",
    r"^LICENSE",
    r"^\.gitignore$",
    r"\.(onnx|h5|msgpack|tflite|ot)$",  # 其他框架的权重
    r"^(rust_model|tf_model|flax_model)\.",  # 非 pytorch 权重
]
# 若仓库同时有 safetensors 与 bin，只留 safetensors
BIN_PATTERNS = [r"\.bin$", r"\.bin\.index\.json$"]


def _get_json(url: str, retries: int = 6):
    """带重试的 GET JSON。经代理时偶发 SSL 中断，必须重试。"""
    import time

    import httpx

    last = None
    for attempt in range(1, retries + 1):
        try:
            r = httpx.get(url, timeout=httpx.Timeout(90, connect=30), follow_redirects=True)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"      列举重试 {attempt}/{retries}：{type(e).__name__}: {str(e)[:70]}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"列举失败 {url}: {last}")


# 当镜像的 tree API 不可用时的兜底文件名清单（逐个探测，不存在的自动跳过）
CANDIDATE_FILES = [
    # 模型配置与权重
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    # 分词器
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spm.model",
    # sentence-transformers 专用
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
]


def list_repo_files(repo_id: str, kind: str = "models", revision: str = "main", path: str = ""):
    """递归列出仓库中的文件；API 不可用时回退到候选清单。"""
    url = f"{MIRROR}/api/{kind}/{repo_id}/tree/{revision}/{path}".rstrip("/")
    try:
        out = []
        for it in _get_json(url, retries=3):
            if it["type"] == "directory":
                out += list_repo_files(repo_id, kind, revision, it["path"])
            else:
                out.append(it["path"])
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] tree API 不可用（{type(e).__name__}），改用候选文件清单探测")
    return list(CANDIDATE_FILES)


def select_files(files):
    """筛选需要下载的文件。"""
    keep = [f for f in files if not any(re.search(p, f) for p in SKIP_PATTERNS)]
    has_safetensors = any(f.endswith(".safetensors") for f in keep)
    if has_safetensors:
        keep = [f for f in keep if not any(re.search(p, f) for p in BIN_PATTERNS)]
    return keep


class _NotFound(Exception):
    """镜像上不存在该文件（404）。"""


def _download_one(url: str, dest: Path, retries: int = 5):
    """下载单个文件，带重试。大文件经代理易出现 SSL 中断，因此必须重试。"""
    import time

    import httpx

    last = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=httpx.Timeout(600, connect=30)
            ) as r:
                if r.status_code == 404:
                    raise _NotFound(url)
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 18):
                        fh.write(chunk)
                tmp.replace(dest)
            return dest.stat().st_size
        except _NotFound:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"      重试 {attempt}/{retries}：{type(e).__name__}: {str(e)[:80]}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"下载失败 {url}: {last}")


def fetch(repo_id: str, kind: str = "models", force: bool = False, quiet: bool = False) -> Path:
    """下载模型仓库全部必需文件到 models/<name>/，返回本地目录。"""
    name = repo_id.split("/")[-1]
    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = select_files(list_repo_files(repo_id, kind))
    if not files:
        raise SystemExit(f"{repo_id} 未找到可下载文件")

    total_mb = 0.0
    for f in files:
        dest = out_dir / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        # 已存在且非空则跳过（断点续传友好）
        if dest.exists() and dest.stat().st_size > 0 and not force:
            if not quiet:
                print(f"  [skip] {f}")
            continue
        # 注意 URL 前缀：数据集是 /datasets/<repo>/resolve/...，模型是 /<repo>/resolve/...
        prefix = "" if kind == "models" else f"/{kind}"
        url = f"{MIRROR}{prefix}/{repo_id}/resolve/main/{f}"
        try:
            size = _download_one(url, dest)
        except _NotFound:
            print(f"  [skip] {f}  （镜像上不存在）")
            continue
        # 镜像对不存在的文件有时返回 200 + 字面量 "Entry not found"，需识别并剔除
        with dest.open("rb") as stream:
            head = stream.read(32)
        if b"Entry not found" in head or head.strip() in (b"Not Found",):
            dest.unlink(missing_ok=True)
            print(f"  [skip] {f}  （镜像上不存在）")
            continue
        mb = size / 1024**2
        total_mb += mb
        if not quiet:
            print(f"  [ok]   {f}  ({mb:.1f} MB)")

    # 写一份来源记录，方便报告里写「模型来源与版本」
    (out_dir / "_source.json").write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "kind": kind,
                "mirror": MIRROR,
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n完成 -> {out_dir}   本次下载 {total_mb:.1f} MB，共 {len(files)} 个文件")
    print(f"用法：把 --model {out_dir.as_posix()} 传给训练脚本")
    return out_dir


def main():
    C.ensure_directories()
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="例如 roberta-base 或 sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--kind", default="models", choices=["models", "datasets"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fetch(args.repo_id, args.kind, args.force)


if __name__ == "__main__":
    main()
