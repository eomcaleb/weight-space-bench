"""One-time setup: the model the editors work on, the practice problems, and the evaluation sets.

  data/model/      text-only Qwen3.5-0.8B (the release's language-model tensors, exact bfloat16 bits, pinned revision),
                   with the experiment's config, generation config, tokenizer config, chat template and tokenizer.json
  data/practice/   the practice problems the paper's editor had (vendored from the release)
  data/bench/      MATH-500 and GSM8K test (from the Hugging Face hub); never mounted for the editing model
"""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from . import st_numpy as S
from .config import ASSETS, BENCH_DIR, MODEL_DIR, PRACTICE_DIR, QWEN_REPO, QWEN_REV

HF = f"https://huggingface.co/{QWEN_REPO}/resolve/{QWEN_REV}"
WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"


def _download(url, dst):
    dst = Path(dst)
    if dst.exists():
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    print(f"downloading {url}")
    with urllib.request.urlopen(req) as r, open(tmp, "ab" if have else "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) + have
        done = have
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dst)
    return dst


def setup_model(cache):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "model.safetensors"
    if not out.exists():
        src = _download(f"{HF}/{WEIGHTS_FILE}", Path(cache) / WEIGHTS_FILE)
        full = S.load(src)
        text = {k: v for k, v in full.items() if k.startswith("model.language_model.")}
        assert len(text) == 320, f"expected 320 language-model tensors, found {len(text)}"
        S.save(out, text, {"format": "pt"})
        print(f"wrote {out} ({len(text)} tensors)")
    for f in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(ASSETS / "qwen35_text" / f, MODEL_DIR / f)
    _download(f"{HF}/tokenizer.json", MODEL_DIR / "tokenizer.json")
    # integrity: the tensors the paper's edits were made on must match bit for bit
    want = json.load(open(ASSETS / "base_sha256.json"))
    got = S.load(out, set(want))
    bad = [n for n, h in want.items() if hashlib.sha256(np.ascontiguousarray(got[n][1]).tobytes()).hexdigest() != h]
    assert not bad, f"base tensors differ from the paper's: {bad[:3]}"
    print(f"model OK: {len(want)} reference tensors match the paper's base hashes")


def setup_practice():
    PRACTICE_DIR.mkdir(parents=True, exist_ok=True)
    for f in (ASSETS / "practice").iterdir():
        shutil.copyfile(f, PRACTICE_DIR / f.name)
    print(f"practice problems -> {PRACTICE_DIR}")


def setup_bench():
    from datasets import load_dataset
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        with open(BENCH_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} problems")

    if not (BENCH_DIR / "math500.jsonl").exists():
        dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                         for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
    if not (BENCH_DIR / "gsm8k.jsonl").exists():
        dump("gsm8k", [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                        "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                       for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))])


def main(cache):
    Path(cache).mkdir(parents=True, exist_ok=True)
    setup_model(cache)
    setup_practice()
    setup_bench()
