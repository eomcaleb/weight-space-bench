"""The editing session: one model, one run. Drives the model through OpenRouter, runs its ```python blocks in the
sandboxed kernel and its ```forward blocks on the GPU, and keeps the budgets, as in the paper's harness:

- the task message (assets/task_template.txt = the paper's final prompt) with the forward-pass budget filled in, plus
  the experimenters' notes that reached the paper's runs, folded in from the start (assets/task_notes.txt);
- one ```python or ```forward block per reply; a forward request that fails is not counted;
- the session runs until every forward pass is used; then the model makes its final choice and replies without code;
- every `handoff_every` replies a fresh conversation starts with the task, the notebook and the current edit (the
  kernel keeps running);
- everything is logged and the run can be resumed after a crash (`--resume` is automatic for an unfinished run).
"""
import json
import re
import shutil
import time
from pathlib import Path

from .config import ASSETS, MODEL_DIR, RESULTS, read_json, write_json
from .forward import ForwardError, ForwardTool
from .kernel import Kernel
from .llm import ContextTooLong, NotIsolated, OpenRouter

BLOCK = re.compile(r"```(python|forward)[^\n]*\n(.*?)```", re.S)
SYSTEM = "Answer the user's request."          # the paper's instructions file, verbatim
RESTART_NOTE = "[note: the harness was restarted, so your kernel restarted too and its variables are gone]\n"


def settings_hash(cfg):
    """Fingerprint of everything that defines the editing task (prompt, budgets, sandbox limits, forward-pass tool,
    sampling). Runs are comparable only when it matches; the leaderboard checks it."""
    import hashlib
    keep = {k: cfg[k] for k in ("task", "kernel", "forward", "sampling")}
    keep["kernel"] = {k: v for k, v in keep["kernel"].items() if k not in ("mode", "image", "memory", "cpus")}
    keep["task"] = {k: v for k, v in keep["task"].items() if k != "max_cost_usd"}
    blob = json.dumps(keep, sort_keys=True) + task_text(cfg) + SYSTEM
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def task_text(cfg):
    t = (ASSETS / "task_template.txt").read_text(encoding="utf-8").replace("{BUDGET}", str(cfg["task"]["forward_passes"]))
    t = t.replace("You may run code up to 300 times", f"You may run code up to {cfg['task']['python_runs']} times")
    if not cfg["task"]["explain_every_step"]:
        t = re.sub(r"\nEXPLAIN EVERY STEP\..*?\n\n", "\n\n", t, flags=re.S)
    notes = (ASSETS / "task_notes.txt").read_text(encoding="utf-8")
    return t.rstrip() + "\n\n" + notes.rstrip()


class Session:
    def __init__(self, model_cfg, cfg, run_dir=None):
        self.m, self.cfg = model_cfg, cfg
        self.dir = Path(run_dir or RESULTS / model_cfg["name"])
        self.work = self.dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.llm = OpenRouter(model_cfg)
        self.kernel = Kernel(cfg["kernel"], self.work, self.dir / "kernel.log")
        self.paths = self.kernel.paths
        self.tool = ForwardTool(cfg, self.work, self.dir, self.show_path)
        self.task = self.show(task_text(cfg))
        self.state = read_json(self.dir / "state.json") or {
            "model": model_cfg, "budget": cfg["task"]["forward_passes"], "turn": 0, "runs": 0, "passes": 0,
            "requests": 0, "since_handoff": 0, "handoffs": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0,
            "nudges": 0, "unexplained_streak": 0, "pending": None, "awaiting": None, "done": False,
            "ended": None, "started": time.time(), "settings_hash": settings_hash(cfg),
            "tool_free": True, "tool_free_turns": 0, "discarded_not_tool_free": 0, "invalid": None}
        if self.state.get("effective_effort"):          # a fallback to "high" sticks across resumes
            self.llm.effort = self.state["effective_effort"]
        if self.state["settings_hash"] != settings_hash(cfg):
            raise SystemExit(f"{self.dir} was started with other benchmark settings; use a new results folder")
        self.conv = read_json(self.dir / "conversation.json") or []

    # ------------------------------------------------------------------------------------------ paths & logs
    def show(self, text):
        """the task's paths (/model, /practice, /work) as the model's kernel sees them."""
        for a, b in self.paths.items():
            if a != b:
                text = re.sub(rf"(?<![\w./]){re.escape(a)}(?=[/\s'\",.;:)\]]|$)", lambda _m, b=b: b, text)
        return text

    def show_path(self, host_path):
        p = str(host_path).replace("\\", "/")
        for w in (self.work.resolve().as_posix(), self.work.as_posix()):
            p = p.replace(w, self.paths["/work"])
        return p

    def log(self, name, obj):
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def save(self):
        write_json(self.dir / "state.json", self.state)
        write_json(self.dir / "conversation.json", self.conv)

    # ------------------------------------------------------------------------------------------ setup
    def prepare_work(self):
        p = self.work / "eval_prompt_format.txt"
        if not p.exists():
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            s = self.cfg["sampling"]
            fmt = tok.apply_chat_template(
                [{"role": "user", "content": "<PROBLEM>\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=s["enable_thinking"])
            p.write_text("The exact text the evaluation feeds the model (<PROBLEM> = the problem statement); the "
                         "model's answer follows it:\n" + fmt, encoding="utf-8")
        (self.work / "edited").mkdir(exist_ok=True)
        write_json(self.dir / "run.json", {"model": self.m, "benchmark": self.cfg, "sandbox_paths": self.paths})

    def handoff_message(self, last):
        st, T = self.state, self.cfg["task"]
        notes = ""
        try:
            notes = (self.work / "notes.md").read_text(encoding="utf-8", errors="replace")[-8000:]
        except OSError:
            pass
        files = sorted(p.name for p in self.work.iterdir())
        edited = sorted(p.stem for p in (self.work / "edited").glob("*.npy"))
        return (f"{self.task}\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {st['turn']} replies happened "
                f"before. Forward passes used: {st['passes']} of {st['budget']}; Python runs used: {st['runs']} of "
                f"{T['python_runs']}. Your kernel is still running with its variables. Tensors currently edited: {edited}. "
                f"Files in {self.paths['/work']}: {files}.\nYOUR NOTEBOOK ({self.paths['/work']}/notes.md, last 8,000 "
                f"characters):\n```\n{notes}\n```\nContinue from here.\n\nLast result:\n{last}")

    # ------------------------------------------------------------------------------------------ one reply
    def ask(self, text):
        msgs = [{"role": "system", "content": SYSTEM}] + self.conv + [{"role": "user", "content": text}]
        try:
            r = self.llm.chat(msgs)
        except ContextTooLong:
            if not self.conv:
                raise
            print("  context too long: starting a fresh conversation (handoff)", flush=True)
            self.conv = []
            self.state["since_handoff"] = 0
            self.state["handoffs"] += 1
            text = self.handoff_message(text)
            return self.ask(text)
        self.conv += [{"role": "user", "content": text}, {"role": "assistant", "content": r["text"]}]
        st = self.state
        st["turn"] += 1
        st["since_handoff"] += 1
        st["cost_usd"] += r["cost"]
        st["tokens_in"] += int(r["usage"].get("prompt_tokens") or 0)
        st["tokens_out"] += int(r["usage"].get("completion_tokens") or 0)
        st["tool_free_turns"] += bool(r["audit"]["tool_free"])
        st["effective_effort"] = r.get("effort")
        st["discarded_not_tool_free"] += r["discarded_not_tool_free"]
        self.log("transcript.jsonl", {"turn": st["turn"], "t": time.time(), "sent": text, "reply": r["text"],
                                      "reasoning": r["reasoning"], "usage": r["usage"], "cost": r["cost"],
                                      "audit": r["audit"], "discarded_not_tool_free": r["discarded_not_tool_free"],
                                      "effort": r.get("effort"),
                                      "finish": r["finish"], "provider": r["provider"], "s": r["s"]})
        print(f"turn {st['turn']:3d} | passes {st['passes']}/{st['budget']} | runs {st['runs']} | "
              f"${st['cost_usd']:.2f} | {r['s']:.0f}s", flush=True)
        return r["text"]

    def act(self, reply):
        """Execute the reply's block; returns the next message, or None when the session is over."""
        st, T = self.state, self.cfg["task"]
        blocks = BLOCK.findall(reply)
        if len(blocks) > 1:
            return (f"Your reply contained {len(blocks)} code blocks, so nothing was run. Reply with exactly one "
                    "```python or ```forward block.")
        if not blocks:
            if st["passes"] >= st["budget"]:
                return None
            st["nudges"] += 1
            if st["nudges"] >= 3:
                st["ended"] = "stopped replying with code before using every forward pass"
                return None
            return (f"Replying without a code block does not end the session: {st['budget'] - st['passes']} forward "
                    "passes are left. Continue your work.")
        st["nudges"] = 0
        kind, body = blocks[0]
        prose = reply[:reply.find("```")].strip()
        if T["explain_every_step"] and len(prose) < 150 and st["unexplained_streak"] < 2:
            st["unexplained_streak"] += 1
            return ("(Your block was not run. EXPLAIN EVERY STEP: before your code or forward block, explain in plain "
                    "English what the last result showed, what you conclude, what you will do now and why, and what "
                    "outcome would confirm or refute your hypothesis. Then send the block again.)")
        st["unexplained_streak"] = 0
        if kind == "python":
            if st["runs"] >= T["python_runs"]:
                return f"The Python run limit ({T['python_runs']}) is reached; no more code can be run."
            st["runs"] += 1
            out, restarted, s = self.kernel.run(body)
            self.log("tool_calls.jsonl", {"t": time.time(), "kind": "python", "run": st["runs"], "s": round(s, 1),
                                          "code": body, "out": out})
            return f"OUTPUT of Python run {st['runs']} of at most {T['python_runs']}:\n```\n{out}\n```"
        # forward
        if st["passes"] >= st["budget"]:
            return ("All forward passes are used; no more can be run. Make your final choice of edits (Python is still "
                    "available), then reply without a code block.")
        st["requests"] += 1
        k = st["requests"]
        try:
            req = json.loads(body)
        except ValueError as e:
            self.log("tool_calls.jsonl", {"t": time.time(), "kind": "forward", "k": k, "request": body, "error": str(e)})
            return f"The forward pass failed and was not counted: the block is not valid JSON ({e})"
        print(f"  forward pass {st['passes'] + 1}/{st['budget']}: {req.get('type')} {req.get('model')}", flush=True)
        try:
            summary, where = self.tool(req, k)
        except ForwardError as e:
            shutil.rmtree(self.work / f"fwd_{k}", ignore_errors=True)
            self.log("tool_calls.jsonl", {"t": time.time(), "kind": "forward", "k": k, "request": req, "error": str(e)})
            return f"The forward pass failed and was not counted: {e}"
        st["passes"] += 1
        self.log("tool_calls.jsonl", {"t": time.time(), "kind": "forward", "k": k, "pass": st["passes"],
                                      "request": req, "summary": summary})
        write_json(self.dir / "requests" / f"fwd_{k}.done.json", summary)
        left = st["budget"] - st["passes"]
        msg = (f"Forward pass {st['passes']} of {st['budget']} done ({req['type']}); results in {where}. Summary:\n"
               f"```\n{json.dumps(summary, indent=1)}\n```\nForward passes left: {left}.")
        if left == 0:
            msg += "\nAll forward passes are used. Make your final choice of edits (Python is still available), then reply without a code block."
        return msg

    # ------------------------------------------------------------------------------------------ the loop
    def run(self):
        st, T = self.state, self.cfg["task"]
        if st["done"]:
            print(f"{self.dir} is finished; nothing to do")
            return
        self.prepare_work()
        resumed = st["turn"] > 0
        self.kernel.start()
        if st["awaiting"]:                       # crashed while executing a block: run it again
            nxt = self.act(st["awaiting"])
            st["awaiting"] = None
        else:
            nxt = st["pending"] or self.task
        if resumed and nxt is not None:
            nxt = RESTART_NOTE + nxt
        final = None
        try:
            while nxt is not None:
                limit = None
                if st["turn"] >= T["max_turns"]:
                    limit = f"the maximum of {T['max_turns']} replies"
                elif st["cost_usd"] >= T["max_cost_usd"]:
                    limit = f"the cost limit of ${T['max_cost_usd']:.0f}"
                if limit:
                    st["ended"] = f"limit: {limit}"
                    nxt = (f"The session has reached {limit} and ends now. Make your final choice: the tensors in "
                           f"{self.paths['/work']}/edited are submitted as they are. Reply without a code block, saying "
                           "what you changed, the evidence for it and why you expect it to help.")
                if T["handoff_every"] and st["since_handoff"] >= T["handoff_every"]:
                    self.conv = []
                    st["since_handoff"] = 0
                    st["handoffs"] += 1
                    nxt = self.handoff_message(nxt)
                st["pending"] = nxt
                self.save()
                try:
                    reply = self.ask(nxt)
                except NotIsolated as e:
                    st["tool_free"] = False
                    st["invalid"] = f"not tool-free: {e}"
                    st["ended"] = "invalid: the model kept calling tools"
                    print(f"  RUN INVALID: {e}", flush=True)
                    break
                if limit:
                    final = reply
                    break
                st["awaiting"] = reply
                self.save()
                nxt = self.act(reply)
                st["awaiting"] = None
                if nxt is None:
                    final = reply
            st["pending"] = None
        finally:
            self.save()
            self.kernel.kill()
        self.finish(final)

    def finish(self, final):
        st = self.state
        (self.dir / "final_answer.md").write_text(final or "", encoding="utf-8")
        sub = self.dir / "submission"
        shutil.rmtree(sub, ignore_errors=True)
        shutil.copytree(self.work / "edited", sub)
        st["done"] = True
        st["finished"] = time.time()
        st["submitted_tensors"] = sorted(p.stem for p in sub.glob("*.npy"))
        self.save()
        shutil.rmtree(self.dir / "ckpt_cache", ignore_errors=True)
        print(f"done: {len(st['submitted_tensors'])} tensors submitted -> {sub}; cost ${st['cost_usd']:.2f}")
