"""Repo paths and .env loading.

Call ``load_env()`` before importing diffusers or huggingface_hub so ``HF_HOME``
is already set when the hub client is created.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ensure_root_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_env() -> None:
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file(env_path)
    else:
        load_dotenv(env_path, override=False)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(name: str, fallback: str = "run") -> str:
    slug = re.sub(r"[^\w.-]+", "-", name.strip(), flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return (slug or fallback)[:80]


def rel_posix(path: Path, start: Path | None = None) -> str:
    base = ROOT if start is None else start
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
