"""weight-space-bench: can an LLM write a working neural network's weights from scratch?

Test 1 (the leaderboard), Experiment 5 of the paper: write MinAtar policy networks from scratch in one reply.

  python -m weight_space_bench run <model|all>     ask + score (results/minatar/<model>; resumes if interrupted)
  python -m weight_space_bench leaderboard         -> results/minatar/leaderboard.md / .csv / .png
  python -m weight_space_bench reference           score the paper's own networks (checks the scorer, ~3 min CPU)
  python -m weight_space_bench all                 run every model, then the leaderboard

Optional test (needs a Linux GPU; not on the leaderboard yet), Experiment 8: edit Qwen3.5-0.8B for math.
  python -m weight_space_bench math-edit --help

Models: the [[model]] entries of config/models.toml. Settings: config/minatar.toml.
"""
import argparse
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "math-edit":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .math_edit.cli import main as math_main
        return math_main()
    from .config import load_minatar_config, load_models, model_by_name
    ap = argparse.ArgumentParser(prog="weight-space-bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "leaderboard", "reference", "all"])
    ap.add_argument("target", nargs="?", default="all", help="a model name from config/models.toml, or all")
    a = ap.parse_args()
    cfg = load_minatar_config()
    from .minatar import leaderboard, task
    if a.cmd in ("run", "all"):
        names = [m["name"] for m in load_models()] if a.target == "all" else [model_by_name(a.target)["name"]]
        for n in names:
            task.run_model(model_by_name(n), cfg)
    if a.cmd == "reference":
        task.reference(cfg)
    if a.cmd in ("leaderboard", "all"):
        leaderboard.main()


if __name__ == "__main__":
    main()
