"""Small reproducibility, logging, and serialization helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def setup_logging(output_dir: str | Path, verbose: bool = False) -> logging.Logger:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("loopquant")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    return device


def runtime_metadata(started_at: float | None = None) -> dict[str, Any]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "runtime_seconds": None if started_at is None else time.time() - started_at,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_version": torch.version.cuda,
        "gpu_model": gpu,
        "loopquant_version": "1.0.0",
        "git_commit": commit,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
