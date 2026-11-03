"""Edited checkpoints: the base model with every <edits>/<name>.npy replacing the original tensor, cast to bfloat16
(round to nearest even, as torch's .to(torch.bfloat16)); the same procedure as the release's build_ckpt.py.
Used for forward passes on the edited weights and for the final evaluation of a submission."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import st_numpy as S
from .config import FROZEN, MODEL_DIR


def edits_in(d):
    d = Path(d)
    return sorted(d.glob("*.npy")) if d.is_dir() else []


def edits_hash(d):
    h = hashlib.sha256()
    for p in edits_in(d):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def build(edits_dir, out_dir, base_dir=MODEL_DIR):
    """Write a full checkpoint folder; returns {tensor: relative change}. Raises on a frozen or mis-shaped tensor."""
    out_dir, base_dir = Path(out_dir), Path(base_dir)
    edits = {p.stem: p for p in edits_in(edits_dir)}
    bad = [n for n in edits if n in FROZEN]
    assert not bad, f"out-of-scope tensors in the submission: {bad}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copyfile(f, out_dir / f.name)
    tensors = S.load(base_dir / "model.safetensors")
    changed = {}
    for name, p in edits.items():
        assert name in tensors, f"unknown tensor {name}"
        dt, base = tensors[name]
        a = np.load(p).astype(np.float32)
        assert list(a.shape) == list(base.shape), (name, a.shape, base.shape)
        assert np.isfinite(a).all(), f"{name}: non-finite values"
        bits = S.bf16_bits(a)
        o = S.bf16_to_f32(base)
        changed[name] = float(np.linalg.norm(S.bf16_to_f32(bits) - o) / max(np.linalg.norm(o), 1e-30))
        tensors[name] = ("BF16", bits)
    S.save(out_dir / "model.safetensors", tensors, {"format": "pt"})
    json.dump({"edited": sorted(edits), "relative_change": changed}, open(out_dir / "edits.json", "w"), indent=1)
    return changed


def cached(edits_dir, cache_root):
    """Build (or reuse) the checkpoint for the current edits; keeps only the latest build to save disk."""
    if not edits_in(edits_dir):
        return MODEL_DIR
    cache_root = Path(cache_root)
    out = cache_root / edits_hash(edits_dir)
    if (out / "model.safetensors").exists():
        return out
    if cache_root.exists():
        for old in cache_root.iterdir():
            shutil.rmtree(old, ignore_errors=True)
    build(edits_dir, out)
    return out


def apply_reference_edit(edit_dir, out_edits_dir, base_dir=MODEL_DIR):
    """Rebuild a released edit (manifest.json + edit.npz, release format) as float32 .npy submissions."""
    man = json.load(open(Path(edit_dir) / "manifest.json"))
    z = np.load(Path(edit_dir) / "edit.npz")
    names = [t["name"] for t in man["tensors"]]
    base = S.load(Path(base_dir) / "model.safetensors", set(names))
    Path(out_edits_dir).mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(man["tensors"]):
        b = base[t["name"]][1]
        if t["form"] == "lowrank":
            w = S.bf16_to_f32(b).astype(np.float64)
            a, bb = z[f"t{i}_a"], z[f"t{i}_b"]
            for k in range(a.shape[1]):
                w = w + np.outer(a[:, k], bb[k])
            e = S.bf16_bits(w.astype(np.float32)).reshape(-1)
        else:
            e = b.copy().reshape(-1)
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(b.shape)
        assert hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() == t["edited_sha256"], t["name"]
        np.save(Path(out_edits_dir) / f"{t['name']}.npy", S.bf16_to_f32(e))
    return names
