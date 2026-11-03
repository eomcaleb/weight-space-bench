"""The forward-pass tool the editing model calls with a ```forward block (see the task message for the request format).

Each pass runs on the local GPU in a fresh worker process. A request that fails raises ForwardError: the harness
reports it and does not count it against the budget.
"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from . import ckpt as CK
from . import mathqa as Q
from .config import MODEL_DIR, PRACTICE_DIR, ROOT


class ForwardError(Exception):
    pass


def seed_of(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def run_worker(cmd, job, job_path, log_path):
    job_path = Path(job_path)
    job["status"] = str(job_path.with_suffix(".status.json"))
    job_path.write_text(json.dumps(job), encoding="utf-8")
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.run([sys.executable, "-m", "weight_space_bench.gpu_worker", cmd, str(job_path)], cwd=str(ROOT),
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        st = json.loads(Path(job["status"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {"error": f"worker exited with code {p.returncode} and no status (see {log_path})"}
    if "error" in st:
        raise ForwardError(st["error"])
    return st


class ForwardTool:
    def __init__(self, cfg, work_dir, run_dir, show):
        self.cfg = cfg                       # full benchmark config
        self.work = Path(work_dir)
        self.run = Path(run_dir)
        self.show = show                     # host path -> the path the model sees (str -> str)
        self.practice = {}
        for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
            for line in open(PRACTICE_DIR / f, encoding="utf-8"):
                r = json.loads(line)
                self.practice[r["id"]] = r
        (self.run / "jobs").mkdir(parents=True, exist_ok=True)

    def _ckpt(self, which):
        if which == "original":
            return str(MODEL_DIR)
        try:
            return str(CK.cached(self.work / "edited", self.run / "ckpt_cache"))
        except AssertionError as e:
            raise ForwardError(f"cannot build the edited weights: {e}") from e

    # ------------------------------------------------------------------------------------------ requests
    def __call__(self, req, k):
        if not isinstance(req, dict):
            raise ForwardError("the forward block must hold one JSON object")
        t = req.get("type")
        if t == "generate":
            return self.generate(req, k)
        if t == "teacher_forced":
            return self.teacher_forced(req, k)
        raise ForwardError('"type" must be "generate" or "teacher_forced"')

    def generate(self, req, k):
        F, S = self.cfg["forward"], self.cfg["sampling"]
        model = req.get("model", "original")
        if model not in ("original", "edited", "both"):
            raise ForwardError('"model" must be "original", "edited" or "both"')
        problems = req.get("problems") or []
        prompts = req.get("prompts") or []
        if not isinstance(problems, list) or not isinstance(prompts, list):
            raise ForwardError('"problems" and "prompts" must be lists')
        unknown = [p for p in problems if p not in self.practice]
        if unknown:
            raise ForwardError(f"unknown practice ids: {unknown[:5]}")
        if not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ForwardError('"prompts" must be non-empty strings')
        n = req.get("n", 1)
        if not isinstance(n, int) or n < 1:
            raise ForwardError('"n" must be a positive integer')
        items = len(problems) + len(prompts)
        if items == 0:
            raise ForwardError("no problems or prompts")
        cap = F["gen_max_items"] // (2 if model == "both" else 1)
        if items * n > cap:
            raise ForwardError(f"items x n = {items * n}; at most {cap}" + (" per model with \"both\"" if model == "both" else ""))
        mt = req.get("max_tokens", S["max_tokens"])
        if not isinstance(mt, int) or not 1 <= mt <= S["max_tokens"]:
            raise ForwardError(f'"max_tokens" must be 1..{S["max_tokens"]}')
        act = req.get("activations")
        if act is not None:
            layers = act.get("layers") if isinstance(act, dict) else None
            if not isinstance(layers, list) or not layers or any(not isinstance(l, int) or not 0 <= l <= 24 for l in layers):
                raise ForwardError('"activations": {"layers": [indices 0..24], "mlp": true or false}')
        rows = [{"key": pid, "id": pid, "source": self.practice[pid]["source"], "user": Q.build_prompt(self.practice[pid]["problem"]),
                 "answer": self.practice[pid]["answer"]} for pid in problems]
        rows += [{"key": f"prompt_{i}", "id": f"prompt_{i}", "source": "prompt", "user": p, "answer": None}
                 for i, p in enumerate(prompts)]
        which = ["original", "edited"] if model == "both" else [model]
        t0 = time.time()
        res = {}
        for m in which:
            ck = self._ckpt(m)
            tag = f"gen_{k}" if model != "both" else f"gen_{k}_{m}"
            raw = self.run / "jobs" / f"{tag}.raw.jsonl"
            raw.unlink(missing_ok=True)
            job = {"ckpt": ck, "out": str(raw), "sampling": S, "gpu_memory_utilization": F["gpu_memory_utilization"],
                   "items": [{"key": r["key"], "user": r["user"], "n": n, "max_tokens": mt,
                              "seed": seed_of(r["key"])} for r in rows]}
            run_worker("generate", job, self.run / "jobs" / f"{tag}.json", self.run / "gpu.log")
            got = {}
            for line in open(raw, encoding="utf-8"):
                d = json.loads(line)
                got[d["key"]] = d
            recs = []
            for r in rows:
                g = got[r["key"]]
                for j, s in enumerate(g["samples"]):
                    rec = {"id": r["id"], "source": r["source"], "sample": j, "text": s["text"], "tokens": s["tokens"],
                           "finish": s["finish"], "answer": r["answer"]}
                    if r["answer"] is not None:
                        rec.update(Q.grade(s["text"], r["answer"], s["finish"]))
                    else:
                        rec.update({"pred": Q.extract(s["text"])[0], "correct": None, "outcome": None})
                    rec["prompt"] = g["prompt"]
                    recs.append(rec)
            path = self.work / f"{tag}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summ = self._summary(recs)
            summ["file"] = self.show(str(path))
            if act is not None:
                adir = self.work / f"{tag}_acts"
                ajob = {"ckpt": ck, "outdir": str(adir), "layers": act["layers"], "mlp": bool(act.get("mlp")),
                        "act_max_tokens": F["act_max_tokens"],
                        "items": [{"prompt": rec["prompt"], "text": rec["text"]} for rec in recs]}
                a = run_worker("acts", ajob, self.run / "jobs" / f"{tag}_acts.json", self.run / "gpu.log")
                idx = {"files": a["files"], "layers": a["layers"],
                       "mlp_layers": "MLP down_proj inputs of decoder layers " + str(a["mlp_decoder_layers"]) +
                                     " (the block that writes residual layer l is decoder layer l-1)",
                       "samples": [{"row": i, "id": rec["id"], "sample": rec["sample"], "correct": rec["correct"],
                                    "outcome": rec["outcome"], "tokens": rec["tokens"]} for i, rec in enumerate(recs)]}
                adir.mkdir(parents=True, exist_ok=True)
                (adir / "index.json").write_text(json.dumps(idx), encoding="utf-8")
                summ["activations"] = {"dir": self.show(str(adir)) + "/", "files": a["files"]}
            res[m] = (summ, recs)
        out = {"ok": True, "type": "generate", "model": model}
        if model == "both":
            out["files"] = {m: res[m][0].pop("file") for m in which}
            acts_ = {m: res[m][0].pop("activations", None) for m in which}
            out["activations"] = acts_ if any(acts_.values()) else None
            for m in which:
                out[m] = res[m][0]
            out["paired"] = self._paired(res["original"][1], res["edited"][1])
        else:
            s = res[model][0]
            out["file"] = s.pop("file")
            out["activations"] = s.pop("activations", None)
            out.update(s)
        out["s"] = round(time.time() - t0, 1)
        return out, out.get("file") or " and ".join(out["files"].values())

    @staticmethod
    def _summary(recs):
        g = [r for r in recs if r["correct"] is not None]
        by = {}
        for r in g:
            by.setdefault(r["source"], []).append(r["correct"])
        fail = {"wrong_answer": 0, "no_answer_finished": 0, "cut_at_max_tokens": 0}
        for r in g:
            if r["outcome"] in fail:
                fail[r["outcome"]] += 1
        return {"generations": len(recs), "graded": len(g),
                "accuracy": sum(r["correct"] for r in g) / len(g) if g else None,
                "accuracy_by_source": {s: round(sum(v) / len(v), 4) for s, v in sorted(by.items())},
                "mean_tokens": round(sum(r["tokens"] for r in recs) / len(recs)),
                "cut_share": round(sum(r["finish"] == "length" for r in recs) / len(recs), 4),
                "failures": fail}

    @staticmethod
    def _paired(orig, edit):
        a, b = {}, {}
        for r in orig:
            if r["correct"] is not None:
                a.setdefault(r["id"], {})[r["sample"]] = int(r["correct"])
        for r in edit:
            if r["correct"] is not None:
                b.setdefault(r["id"], {})[r["sample"]] = int(r["correct"])
        ids = [i for i in a if i in b]
        d = [sum(b[i].values()) / len(b[i]) - sum(a[i].values()) / len(a[i]) for i in ids]
        w2r = sum(1 for i in ids for j in a[i] if j in b[i] and a[i][j] == 0 and b[i][j] == 1)
        r2w = sum(1 for i in ids for j in a[i] if j in b[i] and a[i][j] == 1 and b[i][j] == 0)
        m = sum(d) / len(d) if d else 0.0
        se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else None
        return {"items": len(ids), "accuracy_difference": round(m, 4), "standard_error": None if se is None else round(se, 4),
                "samples_wrong_to_right": w2r, "samples_right_to_wrong": r2w}

    def teacher_forced(self, req, k):
        F = self.cfg["forward"]
        model = req.get("model", "original")
        if model not in ("original", "edited"):
            raise ForwardError('"model" must be "original" or "edited" for a teacher-forced pass')
        outd = self.work / f"fwd_{k}"
        job = {"ckpt": self._ckpt(model), "outdir": str(outd), "request": req,
               "limits": {x: F[x] for x in ("tf_max_texts", "tf_max_tokens", "tf_total_tokens", "tf_max_gb")}}
        t0 = time.time()
        st = run_worker("tf", job, self.run / "jobs" / f"fwd_{k}.json", self.run / "gpu.log")
        out = {"ok": True, "type": "teacher_forced", "model": model, "dir": self.show(str(outd)) + "/",
               "texts": st["texts"], "tokens_total": st["tokens_total"], "max_len": st["max_len"],
               "files": st["files"], "s": round(time.time() - t0, 1)}
        return out, self.show(str(outd)) + "/"
