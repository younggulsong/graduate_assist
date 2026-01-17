from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    base = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{base}_v{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = _unique_path(Path(path))
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def write_yaml(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = _unique_path(Path(path))
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def write_markdown(path: str | Path, text: str) -> Path:
    path = _unique_path(Path(path))
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    return path


def write_text(path: str | Path, text: str) -> Path:
    path = _unique_path(Path(path))
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    return path


def artifacts_root(base_dir: str | Path, run_id: str) -> Path:
    root = Path(base_dir) / run_id
    ensure_dir(root)
    return root
