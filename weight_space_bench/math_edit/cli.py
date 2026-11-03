"""weight-space-bench (exp 8: edit Qwen3.5-0.8B's weights to make it better at math).

  python -m weight_space_bench math-edit setup                         download + export the model, practice and evaluation sets
  python -m weight_space_bench math-edit check                         smoke-test kernel, GPU worker and grading (no API calls)
  python -m weight_space_bench math-edit baseline [--profile P]        evaluate the unedited model           -> results/_unedited
  python -m weight_space_bench math-edit reference [--profile P]       evaluate the paper's Astra 50-pass edit -> results/_reference_astra-p50c
  python -m weight_space_bench math-edit run <model|all>               editing session(s) through OpenRouter -> results/<model>
  python -m weight_space_bench math-edit eval <model|all> [--profile P]
  python -m weight_space_bench math-edit leaderboard [--profile P]
  python -m weight_space_bench math-edit all [--profile P]             baseline + every model's run + eval + leaderboard
Models are the [[model]] entries of config/models.toml; P is "full" (the paper's protocol, default) or "quick".
"""
import argparse
import sys

from ..config import MATH_ASSETS, DATA, MODEL_DIR, MATH_RESULTS, load_config, load_models, model_by_name


def need_setup():
    if not (MODEL_DIR / "model.safetensors").exists():
        sys.exit("run `python -m weight_space_bench math-edit setup` first")


def cmd_run(names, cfg):
    from .agent import Session
    for n in names:
        print(f"=== editing session: {n}")
        Session(model_by_name(n), cfg).run()


def cmd_eval(names, cfg, profile):
    from .evaluate import evaluate
    for n in names:
        d = MATH_RESULTS / n
        if not (d / "state.json").exists() or not (MATH_RESULTS / n / "submission").exists():
            print(f"skip {n}: no finished run in {d}")
            continue
        evaluate(d, cfg, profile)


def cmd_check(cfg):
    """Fast end-to-end smoke test of the local pieces (a few minutes on one GPU)."""
    import json
    import shutil
    import tempfile
    from pathlib import Path

    from . import mathqa as Q
    from .forward import ForwardTool
    from .kernel import Kernel
    assert Q.is_correct("\\frac{1}{2}", "0.5") and not Q.is_correct("3", "4"), "math-verify grading"
    print("grading OK")
    tmp = Path(tempfile.mkdtemp(prefix="wsb_check_"))
    work = tmp / "work"
    work.mkdir()
    k = Kernel(cfg["kernel"], work, tmp / "kernel.log")
    k.start()
    out, _, _ = k.run("w = load('model.language_model.layers.3.mlp.down_proj.weight')\n"
                      "save_edit('model.language_model.layers.3.mlp.down_proj.weight', w * 0.999 + 1e-4)\n"
                      "try:\n    save_edit('model.language_model.norm.weight', load('model.language_model.norm.weight'))\n"
                      "except PermissionError as e:\n    print('refused as expected')\nlen(NAMES), edited()")
    print("kernel:", out.strip())
    assert "refused as expected" in out and "320" in out, out
    k.kill()
    tool = ForwardTool(cfg, work, tmp, lambda p: p)
    s, _ = tool({"type": "generate", "model": "both", "problems": ["gsm8k_train_0", "math_train_algebra_0"], "n": 1,
                 "max_tokens": 256, "activations": {"layers": [0, 12, 24], "mlp": True}}, 1)
    print("generate:", json.dumps(s["paired"]), "acts:", s["activations"]["original"]["files"])
    s, _ = tool({"type": "teacher_forced", "model": "edited", "texts": ["1 + 1 = 2", "The answer is \\boxed{4}."],
                 "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 24], "mlp_act": [3],
                            "attn_layers": [3]}}, 2)
    print("teacher_forced:", s["files"])
    shutil.rmtree(tmp, ignore_errors=True)
    print("check OK")


def main():
    ap = argparse.ArgumentParser(prog="weight-space-bench math-edit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["setup", "check", "baseline", "reference", "run", "eval", "leaderboard", "all"])
    ap.add_argument("target", nargs="?", default="all")
    ap.add_argument("--profile", default="full", choices=["full", "quick"])
    ap.add_argument("--config", help="benchmark config (default config/benchmark.toml)")
    ap.add_argument("--cache", default=str(DATA / "cache"), help="download cache for setup")
    a = ap.parse_args()
    cfg = load_config(a.config)
    all_models = [m["name"] for m in load_models()]
    names = all_models if a.target == "all" else [model_by_name(a.target)["name"]]

    if a.cmd == "setup":
        from .setup_data import main as setup
        setup(a.cache)
        return
    if a.cmd == "leaderboard":
        from .leaderboard import main as lb
        lb(a.profile)
        return
    need_setup()
    if a.cmd == "check":
        cmd_check(cfg)
    elif a.cmd == "baseline":
        from .evaluate import evaluate
        d = MATH_RESULTS / "_unedited"
        d.mkdir(parents=True, exist_ok=True)
        evaluate(d, cfg, a.profile, edits_dir=d / "none")
    elif a.cmd == "reference":
        from .ckpt import apply_reference_edit
        from .evaluate import evaluate
        d = MATH_RESULTS / "_reference_astra-p50c"
        if not (d / "submission").exists():
            apply_reference_edit(MATH_ASSETS / "reference" / "astra_p50c_edit", d / "submission")
        evaluate(d, cfg, a.profile)
    elif a.cmd == "run":
        cmd_run(names, cfg)
    elif a.cmd == "eval":
        cmd_eval(names, cfg, a.profile)
    elif a.cmd == "all":
        from .evaluate import evaluate
        from .leaderboard import main as lb
        d = MATH_RESULTS / "_unedited"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "eval" / a.profile / "scores.json").exists():
            evaluate(d, cfg, a.profile, edits_dir=d / "none")
        cmd_run(names, cfg)
        cmd_eval(names, cfg, a.profile)
        lb(a.profile)


if __name__ == "__main__":
    main()
