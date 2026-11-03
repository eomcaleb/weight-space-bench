"""Paths and configuration shared by the tests."""
import json
import os
import tomllib
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent                                   # repository root
DATA = Path(os.environ.get("WSB_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("WSB_RESULTS", ROOT / "results"))

# ---- test 1: write MinAtar policies from scratch (exp 5) -- the leaderboard test
MINATAR_ASSETS = PKG / "minatar" / "assets"
MINATAR_RESULTS = RESULTS / "minatar"

# ---- optional test: edit Qwen3.5-0.8B for math (exp 8) -- needs a GPU, not on the leaderboard yet
MATH_ASSETS = PKG / "math_edit" / "assets"
MATH_RESULTS = RESULTS / "math_edit"
MODEL_DIR = DATA / "model"                          # text-only Qwen3.5-0.8B (mounted read-only at /model)
PRACTICE_DIR = DATA / "practice"                    # practice problems (mounted read-only at /practice)
BENCH_DIR = DATA / "bench"                          # MATH-500 / GSM8K test: never visible to the editing model
QWEN_REPO = "Qwen/Qwen3.5-0.8B"
QWEN_REV = "2fc06364715b967f1860aea9cf38778875588b17"
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")


def _toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(path=None):
    """math-edit settings."""
    return _toml(path or ROOT / "config" / "math_edit.toml")


def load_minatar_config(path=None):
    return _toml(path or ROOT / "config" / "minatar.toml")


def load_models(path=None):
    return _toml(path or ROOT / "config" / "models.toml")["model"]


def model_by_name(name, path=None):
    for m in load_models(path):
        if m["name"] == name or m["slug"] == name:
            return m
    raise SystemExit(f"model {name!r} is not in config/models.toml")


def read_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)
