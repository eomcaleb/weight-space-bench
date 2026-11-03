"""Evaluation of a submission: the paper's protocol (release: exp8_math/code/evaluate.py, protocol think35).

The submitted tensors are cast to bfloat16 and replace the originals; vLLM samples with the chat template (thinking
on), user message = problem + "\\nPlease reason step by step, and put your final answer within \\boxed{}.",
temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens, seed 0; the last \\boxed{}
answer is graded with math-verify. Score per set = mean accuracy over all samples.

Results: <run>/eval/<profile>/scores.json (+ per_problem.json, raw.jsonl with every generated text). Interrupted
evaluations resume where they stopped.
"""
import json
import random
import shutil
from pathlib import Path

from . import ckpt as CK
from . import mathqa as Q
from ..config import BENCH_DIR, MODEL_DIR, write_json
from .forward import run_worker

SETS = ("math500", "gsm8k")


def load_set(name, limit=None):
    rows = [json.loads(l) for l in open(BENCH_DIR / f"{name}.jsonl", encoding="utf-8")]
    if limit and limit < len(rows):
        keep = set(random.Random(0).sample(range(len(rows)), limit))      # fixed subset, the same for every model
        rows = [r for i, r in enumerate(rows) if i in keep]
    return rows


def evaluate(run_dir, cfg, profile="full", edits_dir=None):
    run_dir = Path(run_dir)
    prof = cfg["eval"][profile]
    out = run_dir / "eval" / profile
    out.mkdir(parents=True, exist_ok=True)
    if (out / "scores.json").exists():
        print(f"{run_dir.name} ({profile}) already evaluated: {out / 'scores.json'}")
        return json.load(open(out / "scores.json", encoding="utf-8"))
    edits_dir = Path(edits_dir) if edits_dir else run_dir / "submission"
    if CK.edits_in(edits_dir):
        ck = out / "ckpt"
        if not (ck / "model.safetensors").exists():
            CK.build(edits_dir, ck)
    else:
        ck = MODEL_DIR
    S = cfg["sampling"]
    rows, items = {}, []
    for name in SETS:
        if name not in prof:
            continue
        rows[name] = load_set(name, prof[name].get("limit"))
        items += [{"key": f"{name}/{r['id']}", "user": Q.build_prompt(r["problem"]), "n": prof[name]["samples"],
                   "seed": 0} for r in rows[name]]
    raw = out / "raw.jsonl"
    job = {"ckpt": str(ck), "out": str(raw), "sampling": S, "chunk": 64,
           "gpu_memory_utilization": cfg["forward"]["gpu_memory_utilization"], "items": items}
    print(f"evaluating {run_dir.name} ({profile}): {sum(i['n'] for i in items)} generations", flush=True)
    run_worker("generate", job, out / "job.json", out / "gpu.log")
    got = {}
    for line in open(raw, encoding="utf-8"):
        d = json.loads(line)
        got[d["key"]] = d["samples"]
    scores, per = {}, {}
    for name, rs in rows.items():
        ok = tot = cut = toks = noans = 0
        per[name] = []
        for r in rs:
            c = []
            for s in got[f"{name}/{r['id']}"]:
                g = Q.grade(s["text"], r["answer"], s["finish"])
                c.append(int(g["correct"]))
                cut += s["finish"] == "length"
                noans += g["outcome"] == "no_answer_finished"
                toks += s["tokens"]
            per[name].append({"id": r["id"], "correct": c})
            ok += sum(c)
            tot += len(c)
        scores[name] = {"acc": ok / tot, "problems": len(rs), "samples": prof[name]["samples"], "ok": ok, "total": tot,
                        "cut_share": cut / tot, "mean_tokens": toks / tot, "no_answer_finished": noans}
    scores["_protocol"] = {"profile": profile, "sets": prof, "sampling": S, "prompt": Q.INSTRUCTION,
                           "edited_tensors": sorted(p.stem for p in CK.edits_in(edits_dir))}
    write_json(out / "per_problem.json", per)
    write_json(out / "scores.json", scores)
    if ck != MODEL_DIR:
        shutil.rmtree(ck, ignore_errors=True)
    print(" ".join(f"{k} {100 * v['acc']:.1f}" for k, v in scores.items() if k in SETS))
    return scores
