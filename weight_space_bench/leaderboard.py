"""Leaderboard: every evaluated run of one profile, against the unedited model measured with the same harness.

Score = mean of the MATH-500 and GSM8K accuracies of the edited model. Delta = score minus the unedited model's score,
with a paired standard error over problems (the same problems and samples protocol for every row).
"""
import csv
import json
import math

from .config import ASSETS, RESULTS, load_models, read_json

SETS = ("math500", "gsm8k")
BASELINE = "_unedited"


def _per_problem_means(run, profile):
    p = read_json(RESULTS / run / "eval" / profile / "per_problem.json")
    return {s: {r["id"]: sum(r["correct"]) / len(r["correct"]) for r in p[s]} for s in p} if p else None


def _paired(a, b):
    ids = [i for i in a if i in b]
    d = [a[i] - b[i] for i in ids]
    if len(d) < 2:
        return None, None
    m = sum(d) / len(d)
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))


def build(profile="full"):
    from .agent import settings_hash
    from .config import load_config
    current = settings_hash(load_config())
    groups = {m["name"]: m.get("group", "") for m in load_models()}
    base = _per_problem_means(BASELINE, profile)
    rows = []
    for d in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        sc = read_json(d / "eval" / profile / "scores.json")
        if not sc:
            continue
        st = read_json(d / "state.json") or {}
        why = None
        if not d.name.startswith("_"):
            if st.get("invalid") or not st.get("tool_free", False):
                why = st.get("invalid") or "not tool-free"
            elif st.get("settings_hash") != current:
                why = f"other benchmark settings ({st.get('settings_hash')} != {current})"
            elif not st.get("done"):
                why = "run not finished"
        row = {"run": d.name, "group": groups.get(d.name, "baseline" if d.name.startswith("_") else ""), "excluded": why,
               "math500": sc.get("math500", {}).get("acc"), "gsm8k": sc.get("gsm8k", {}).get("acc"),
               "cut_share": sum(sc[s]["cut_share"] for s in SETS if s in sc) / max(1, sum(s in sc for s in SETS)),
               "passes": st.get("passes"), "effort": st.get("effective_effort"), "turns": st.get("turn"), "cost_usd": st.get("cost_usd"),
               "tensors": len(sc["_protocol"].get("edited_tensors", [])), "ended": st.get("ended")}
        vals = [row[s] for s in SETS if row[s] is not None]
        row["score"] = sum(vals) / len(vals) if vals else None
        mine = _per_problem_means(d.name, profile)
        if base and mine and d.name != BASELINE:
            ds = [_paired(mine[s], base[s]) for s in SETS if s in mine and s in base]
            if ds and all(x[0] is not None for x in ds):
                row["delta"] = sum(x[0] for x in ds) / len(ds)
                row["delta_se"] = math.sqrt(sum(x[1] ** 2 for x in ds)) / len(ds)
        rows.append(row)
    rows.sort(key=lambda r: -(r["score"] or -1))
    return rows


def pct(x):
    return "" if x is None else f"{100 * x:.1f}"


def render(rows, profile):
    out = [f"## Leaderboard ({profile} profile)", "",
           "| # | run | group | effort | MATH-500 | GSM8K | score | gain vs unedited (pp) | cut at limit | passes | tensors edited | API cost |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rank = 0
    excluded = [r for r in rows if r.get("excluded")]
    for r in [r for r in rows if not r.get("excluded")]:
        is_model = not r["run"].startswith("_")
        rank += is_model
        delta = f"{100 * r['delta']:+.1f} +/- {100 * r["delta_se"]:.1f}" if r.get("delta") is not None else ""
        cost = f"${r['cost_usd']:.2f}" if r.get("cost_usd") else ""
        out.append(f"| {rank if is_model else ''} | {r['run']} | {r['group']} | {r.get('effort') or ''} | {pct(r['math500'])} | {pct(r['gsm8k'])} | "
                   f"**{pct(r['score'])}** | {delta} | {pct(r['cut_share'])}% | {r['passes'] or ''} | {r['tensors']} | {cost} |")
    if excluded:
        out += ["", "Not ranked:", ""] + [f"- {r['run']}: {r['excluded']} (score {pct(r['score'])})" for r in excluded]
    if profile == "full":
        ref = json.load(open(ASSETS / "reference" / "paper_results.json"))
        out += ["", "Published numbers (the paper, GPT-6 Astra, same protocol, their hardware):", "",
                "| run | MATH-500 | GSM8K | score |", "|---|---|---|---|"]
        for k, label in (("unedited", "unedited"), ("p10c", "Astra, 10 passes"), ("p25c", "Astra, 25 passes"),
                         ("p50c", "Astra, 50 passes")):
            a, b = ref[k]["math500"]["acc"], ref[k]["gsm8k"]["acc"]
            out.append(f"| {label} | {pct(a)} | {pct(b)} | {pct((a + b) / 2)} |")
    return "\n".join(out) + "\n"


def main(profile="full"):
    rows = build(profile)
    md = render(rows, profile)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"leaderboard_{profile}.md").write_text(md, encoding="utf-8")
    with open(RESULTS / f"leaderboard_{profile}.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["run", "group", "effort", "math500", "gsm8k", "score", "delta", "delta_se", "cut_share", "passes", "turns",
                "tensors", "cost_usd", "ended", "excluded"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(md)
