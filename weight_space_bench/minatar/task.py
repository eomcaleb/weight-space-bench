"""Test 1: write a MinAtar policy network from scratch, in one reply, with no tools (Experiment 5 of the paper).

For every model x game x attempt: send the paper's exact prompt (a fresh conversation, the paper's one-line system
instructions), read the reply's last ```edits block into an all-zero 600-32-32-6 network, and score it on the game.

results/minatar/<model>/<game>/attempt_<k>/
  prompt.txt       what was sent        reply.md         the model's reply
  response.json    usage, cost, tool audit, effort     edits.json       how the edits were read
  network.npz      the network as scored              score.json       the game score
A finished step is never redone, so an interrupted run resumes where it stopped.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import MINATAR_ASSETS, MINATAR_RESULTS, read_json, write_json
from ..llm import NotIsolated, OpenRouter
from . import edits as E
from . import play as P

SYSTEM = "Answer the user's request."          # the paper's instructions file, verbatim
PARSER_VERSION = 1


def prompt(game):
    return (MINATAR_ASSETS / "prompts" / f"{game}.txt").read_text(encoding="utf-8")


def settings_hash(cfg):
    """Fingerprint of everything that defines the test; runs are ranked together only when it matches."""
    blob = json.dumps({"task": {k: cfg["task"][k] for k in ("games", "attempts")}, "eval": {
        k: cfg["eval"][k] for k in ("episodes", "max_steps", "seed")}, "parser": PARSER_VERSION, "system": SYSTEM},
        sort_keys=True) + "".join(prompt(g) for g in cfg["task"]["games"])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def ask(model_cfg, game, d):
    """One reply (skipped if already saved). Returns True when the reply is usable."""
    if (d / "reply.md").exists():
        return True
    if (d / "invalid.json").exists():
        return False
    p = prompt(game)
    (d / "prompt.txt").write_text(p, encoding="utf-8")
    client = OpenRouter(model_cfg)
    t0 = time.time()
    try:
        r = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}])
    except NotIsolated as e:
        write_json(d / "invalid.json", {"reason": f"not tool-free: {e}"})
        return False
    write_json(d / "response.json", {"model": model_cfg, "usage": r["usage"], "cost": r["cost"], "audit": r["audit"],
                                     "discarded_not_tool_free": r["discarded_not_tool_free"], "effort": r["effort"],
                                     "max_tokens": r.get("max_tokens"), "finish": r["finish"],
                                     "provider": r["provider"], "s": round(time.time() - t0, 1),
                                     "reasoning": r["reasoning"]})
    (d / "reply.md").write_text(r["text"], encoding="utf-8")
    return True


def build(d):
    """Read the reply's edits into the network (skipped if done)."""
    if (d / "network.npz").exists():
        return
    net, rep = E.apply((d / "reply.md").read_text(encoding="utf-8"))
    write_json(d / "edits.json", rep)
    E.save_npz(net, d / "network.npz")


def run_model(model_cfg, cfg, log=print):
    """Ask every game x attempt (in parallel), then score each network."""
    root = MINATAR_RESULTS / model_cfg["name"]
    h = settings_hash(cfg)
    meta = read_json(root / "run.json")
    if meta and meta.get("settings_hash") != h:
        raise SystemExit(f"{root} was made with other benchmark settings; move it away to start over")
    write_json(root / "run.json", {"model": model_cfg, "settings_hash": h, "config": cfg})
    jobs = [(g, k, root / g / f"attempt_{k}") for g in cfg["task"]["games"] for k in range(1, cfg["task"]["attempts"] + 1)]
    for _, _, d in jobs:
        d.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (j[2] / "reply.md").exists() and not (j[2] / "invalid.json").exists()]
    if todo:
        log(f"{model_cfg['name']}: asking {len(todo)} replies")
        with ThreadPoolExecutor(max(1, cfg["task"]["concurrency"])) as ex:
            futs = {ex.submit(ask, model_cfg, g, d): (g, k) for g, k, d in todo}
            for f in as_completed(futs):
                g, k = futs[f]
                try:
                    ok = f.result()
                    log(f"  {model_cfg['name']} {g} #{k}: {'reply saved' if ok else 'INVALID (used tools)'}")
                except Exception as e:  # noqa: BLE001 - one failed call must not stop the others; rerun resumes it
                    log(f"  {model_cfg['name']} {g} #{k}: API error ({e}); rerun to retry")
    for g, k, d in jobs:
        if not (d / "reply.md").exists() or (d / "score.json").exists():
            continue
        build(d)
        net = E.load_npz(d / "network.npz")
        ev = cfg["eval"]
        s = P.score(net, g, episodes=ev["episodes"], seed=ev["seed"], max_steps=ev["max_steps"], workers=ev["workers"])
        write_json(d / "score.json", s)
        rep = read_json(d / "edits.json")
        log(f"  {model_cfg['name']} {g} #{k}: score {s['mean']:.1f} +/- {s['se']:.1f} "
            f"({rep['applied']}/{rep['statements']} statements, {rep['nonzero_total']} nonzero weights)")


def baselines(cfg, log=print):
    """Random floor: the all-zero network (uniform random actions), scored with the same episodes."""
    p = MINATAR_RESULTS / "_random" / "scores.json"
    have = read_json(p) or {}
    if have.get("_settings") == settings_hash(cfg) and all(g in have for g in cfg["task"]["games"]):
        return have
    ev = cfg["eval"]
    out = {"_settings": settings_hash(cfg)}
    for g in cfg["task"]["games"]:
        out[g] = P.score(E.zeros(), g, episodes=ev["episodes"], seed=ev["seed"], max_steps=ev["max_steps"],
                         workers=ev["workers"])
        log(f"random {g}: {out[g]['mean']:.2f}")
    write_json(p, out)
    return out


def reference(cfg, log=print):
    """Score the paper's own networks (GPT-6 Astra) with this harness: checks the scorer reproduces the paper."""
    p = MINATAR_RESULTS / "_astra_paper_networks" / "scores.json"
    out = read_json(p) or {}
    ev = cfg["eval"]
    for g in cfg["task"]["games"]:
        if g in out:
            continue
        out[g] = P.score(E.load_npz(MINATAR_ASSETS / "reference" / f"astra_{g}.npz"), g, episodes=ev["episodes"],
                         seed=ev["seed"], max_steps=ev["max_steps"], workers=ev["workers"])
        log(f"Astra's {g} network: ours {out[g]['mean']:.1f} +/- {out[g]['se']:.1f}, paper {cfg['reference']['astra'][g]:.1f}")
        write_json(p, out)
    return out
