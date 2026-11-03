"""Leaderboard for test 1 (write MinAtar networks from scratch).

Per game: the model's score = mean over its attempts (each attempt = mean over 1,000 episodes), normalized so that
0 = random play (the all-zero network) and 1 = the best PPO network trained for 500M steps (the paper's baseline).
Overall score = the median of the three games' normalized scores, in % (100 = PPO-500M level), as Atari benchmarks
report the median human-normalized score: one outlier game cannot dominate it.

Writes results/minatar/leaderboard.md, leaderboard.csv and leaderboard.png.
"""
import csv
import statistics
import textwrap

from ..config import MINATAR_RESULTS, load_minatar_config, load_models, read_json
from .task import baselines, settings_hash

ANTHROPIC, OPENAI, OTHER, REFERENCE = "#eb6834", "#2a78d6", "#1baf7a", "#dfddd2"   # validated palette slots + neutral
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5df"
GAME_TITLE = {"breakout": "Breakout", "seaquest": "Seaquest", "asterix": "Asterix"}


def norm(s, rand, ppo):
    return (s - rand) / (ppo - rand)


def collect(cfg):
    games = cfg["task"]["games"]
    rnd = baselines(cfg)
    ppo = cfg["reference"]["ppo500"]
    h = settings_hash(cfg)
    rows = []
    for m in load_models():
        root = MINATAR_RESULTS / m["name"]
        if not root.exists():
            continue
        meta = read_json(root / "run.json") or {}
        row = {"name": m["name"], "label": m.get("label", m["name"]), "group": m.get("group", ""), "kind": "model",
               "excluded": None, "cost": 0.0, "out_tokens": 0, "effort": set()}
        for g in games:
            vals = []
            for k in range(1, cfg["task"]["attempts"] + 1):
                d = root / g / f"attempt_{k}"
                if (d / "invalid.json").exists():
                    row["excluded"] = "used tools (not isolated)"
                r = read_json(d / "response.json")
                if r:
                    row["cost"] += r.get("cost") or 0
                    row["out_tokens"] += int((r.get("usage") or {}).get("completion_tokens") or 0)
                    row["effort"].add(r.get("effort"))
                s = read_json(d / "score.json")
                if s:
                    vals.append(s["mean"])
            row[g] = statistics.mean(vals) if len(vals) == cfg["task"]["attempts"] else None
            row[g + "_best"] = max(vals) if vals else None
            if row[g] is None and not row["excluded"]:
                row["excluded"] = f"incomplete ({len(vals)}/{cfg['task']['attempts']} attempts scored on {g})"
        if meta.get("settings_hash") != h and not row["excluded"]:
            row["excluded"] = "made with other benchmark settings"
        row["effort"] = "/".join(sorted(e for e in row["effort"] if e)) or ""
        rows.append(row)
    refs = [("GPT-6 Astra (paper)", cfg["reference"]["astra"]), ("PPO, 100M steps", cfg["reference"]["ppo100"])]
    for label, sc in refs:
        rows.append({"name": label, "label": label, "group": "reference", "kind": "reference", "excluded": None,
                     **{g: sc[g] for g in games}})
    for r in rows:
        if all(r.get(g) is not None for g in games):
            r["norm"] = {g: norm(r[g], rnd[g]["mean"], ppo[g]) for g in games}
            r["score"] = 100 * statistics.median(r["norm"].values())
        else:
            r["norm"], r["score"] = None, None
    return rows, rnd, ppo


def color(r):
    if r["kind"] == "reference":
        return REFERENCE
    return {"Anthropic": ANTHROPIC, "OpenAI": OPENAI}.get(r["group"], OTHER)


def chart(rows, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

    models = sorted([r for r in rows if r["kind"] == "model" and r["score"] is not None and not r["excluded"]],
                    key=lambda r: -r["score"])
    refs = [r for r in rows if r["kind"] == "reference"]
    bars = models + refs
    fig, ax = plt.subplots(figsize=(10, 5.625), dpi=200)
    fig.patch.set_facecolor("white")
    top = max([r["score"] for r in bars] + [100]) * 1.15
    step = next(s for s in (10, 20, 25, 50, 100, 200, 250, 500) if top / s <= 8)
    top = step * (int(top / step) + 1)
    w = 0.72
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.09, right=0.98)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(0, top)
    bb = ax.get_window_extent()
    aspect = (top / bb.height) / ((len(bars) + 0.2) / bb.width)     # y data units per x data unit, on screen
    rx = 10 / (bb.width / (len(bars) + 0.2))                        # a 10 px corner radius, in x data units
    for i, r in enumerate(bars):
        h = max(r["score"], 0)
        ry = rx * aspect
        if h > 2 * ry:   # rounded data end; square corners at the baseline
            ax.add_patch(FancyBboxPatch((i - w / 2, 0), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                        mutation_aspect=aspect, linewidth=0, facecolor=color(r), zorder=3))
            ax.add_patch(Rectangle((i - w / 2, 0), w, h - ry, linewidth=0, facecolor=color(r), zorder=3))
        elif h > 0:
            ax.add_patch(Rectangle((i - w / 2, 0), w, h, linewidth=0, facecolor=color(r), zorder=3))
        ax.text(i, h + top * 0.015, f"{r['score']:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=INK, zorder=4, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.axhline(100, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(len(bars) - 0.5 + 0.05, 100, "PPO, 500M steps", ha="right", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(0, int(top) + 1, step))
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([textwrap.fill(r["label"], 13) for r in bars], fontsize=10, color=INK, linespacing=1.3)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.6)
    ax.set_ylabel("SCORE (% OF PPO, 500M STEPS)", fontsize=11, color=INK, labelpad=10)
    groups = []
    for r in models:
        g = r["group"] if r["group"] in ("Anthropic", "OpenAI") else "Other labs"
        if g not in groups:
            groups.append(g)
    handles = [Patch(facecolor={"Anthropic": ANTHROPIC, "OpenAI": OPENAI}.get(g, OTHER), label=g) for g in groups]
    handles.append(Patch(facecolor=REFERENCE, label="Reference"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=len(handles),
              bbox_to_anchor=(1, 1.06), handlelength=1, handleheight=1, labelcolor=INK_2)
    fig.suptitle("Writing networks from scratch", fontsize=22, fontweight="bold", color=INK, y=0.985)
    games = ", ".join(GAME_TITLE[g] for g in cfg["task"]["games"])
    fig.text(0.5, 0.885, f"weight-space-bench · MinAtar {games} · median of the three games",
             ha="center", fontsize=12, color=INK_2, family="serif")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    cfg = load_minatar_config()
    games = cfg["task"]["games"]
    rows, rnd, ppo = collect(cfg)
    ranked = sorted([r for r in rows if r["kind"] == "model" and not r["excluded"] and r["score"] is not None],
                    key=lambda r: -r["score"])
    f1 = lambda x: "" if x is None else f"{x:.1f}"      # noqa: E731
    head = " | ".join(f"{GAME_TITLE[g]} (raw / % PPO)" for g in games)
    out = ["## Leaderboard: writing MinAtar networks from scratch", "",
           "Score = median over the three games of the normalized score (0 = random play, 100 = the best PPO "
           f"network trained for 500M steps). Raw = mean game score over {cfg['task']['attempts']} attempts x "
           f"{cfg['eval']['episodes']} episodes.", "",
           f"| # | model | lab | score | {head} | effort | API cost |",
           "|---|---|---|---|" + "---|" * len(games) + "---|---|"]
    for i, r in enumerate(ranked, 1):
        cells = " | ".join(f"{f1(r[g])} / {100 * r['norm'][g]:.0f}%" for g in games)
        out.append(f"| {i} | {r['label']} | {r['group']} | **{r['score']:.1f}** | {cells} | {r['effort']} | ${r['cost']:.2f} |")
    for r in [r for r in rows if r["kind"] == "reference"]:
        cells = " | ".join(f"{f1(r[g])} / {100 * r['norm'][g]:.0f}%" for g in games)
        out.append(f"| | *{r['label']}* | reference | *{r['score']:.1f}* | {cells} | | |")
    cells = " | ".join(f"{ppo[g]:.1f} / 100%" for g in games)
    out.append(f"| | *PPO, 500M steps* | reference | *100.0* | {cells} | | |")
    cells = " | ".join(f"{rnd[g]['mean']:.2f} / 0%" for g in games)
    out.append(f"| | *random play* | reference | *0.0* | {cells} | | |")
    excl = [r for r in rows if r["kind"] == "model" and r["excluded"]]
    if excl:
        out += ["", "Not ranked:", ""] + [f"- {r['label']}: {r['excluded']}" for r in excl]
    md = "\n".join(out) + "\n"
    MINATAR_RESULTS.mkdir(parents=True, exist_ok=True)
    (MINATAR_RESULTS / "leaderboard.md").write_text(md, encoding="utf-8")
    with open(MINATAR_RESULTS / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "model", "lab", "score"] + [f"{g}_{x}" for g in games for x in ("raw", "best", "norm")]
                   + ["effort", "cost_usd", "excluded"])
        for i, r in enumerate(ranked, 1):
            w.writerow([i, r["label"], r["group"], round(r["score"], 2)]
                       + [v for g in games for v in (round(r[g], 3), round(r[g + "_best"], 3), round(r["norm"][g], 4))]
                       + [r["effort"], round(r["cost"], 4), ""])
        for r in excl:
            w.writerow(["", r["label"], r["group"], ""] + [""] * (3 * len(games)) + [r["effort"], round(r["cost"], 4), r["excluded"]])
    chart(rows, MINATAR_RESULTS / "leaderboard.png", cfg)
    print(md)
    print(f"chart -> {MINATAR_RESULTS / 'leaderboard.png'}")
