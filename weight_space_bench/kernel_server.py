"""The editing model's persistent Python kernel (runs inside the sandbox; standalone: no imports from weight_space_bench).

Protocol: one JSON object per line on stdin ({"code": "..."}), one per line back on the original stdout
({"out": "..."}). Everything the code prints (Python- and C-level, stdout and stderr) is captured. If the last line of
a block is an expression, its repr is printed, like a notebook.

Defined in advance (as the task message says): NAMES, load(name), load_original(name), save_edit(name, tensor),
drop_edit(name), edited(), tok. Gradients are disabled.
"""
import ast
import json
import os
import sys
import tempfile
import traceback

MODEL = os.environ.get("WSB_MODEL", "/model")
WORK = os.environ.get("WSB_WORK", "/work")
EDIT_DIR = os.path.join(WORK, "edited")
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")

PRELUDE = r'''
import os, json
import numpy as np
import torch
from safetensors import safe_open as _safe_open
from tokenizers import Tokenizer as _Tokenizer
torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
_ST = _safe_open(os.path.join(_MODEL, "model.safetensors"), framework="pt")
NAMES = sorted(_ST.keys())
_SHAPES = {n: tuple(_ST.get_slice(n).get_shape()) for n in NAMES}
tok = _Tokenizer.from_file(os.path.join(_MODEL, "tokenizer.json"))

def load_original(name):
    """the original tensor (float32 torch tensor)."""
    if name not in _SHAPES:
        raise KeyError(f"unknown tensor {name!r}")
    return _ST.get_tensor(name).float()

def load(name):
    """your current edit of a tensor if there is one, else the original (float32 torch tensor)."""
    p = os.path.join(_EDIT_DIR, name + ".npy")
    return torch.from_numpy(np.load(p)) if os.path.exists(p) else load_original(name)

def edited():
    """names of the tensors you have edited."""
    return sorted(f[:-4] for f in os.listdir(_EDIT_DIR) if f.endswith(".npy")) if os.path.isdir(_EDIT_DIR) else []

def drop_edit(name):
    """remove your edit of a tensor (back to the original)."""
    p = os.path.join(_EDIT_DIR, name + ".npy")
    if os.path.exists(p):
        os.remove(p)

def save_edit(name, tensor):
    """replace a tensor by your edited version (same shape); it is used by forward passes on 'edited' weights and
    submitted at the end."""
    if name not in _SHAPES:
        raise KeyError(f"unknown tensor {name!r}")
    if name in _FROZEN:
        raise PermissionError(f"refused: {name} is out of scope (final norm, embedding / output layer)")
    t = torch.as_tensor(tensor).detach().to(torch.float32).cpu().contiguous()
    if tuple(t.shape) != _SHAPES[name]:
        raise ValueError(f"refused: shape {tuple(t.shape)} != {_SHAPES[name]}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("refused: non-finite values")
    o = load_original(name)
    if torch.equal(t, o):
        drop_edit(name)
        print(f"save_edit: {name} equals the original; no edit kept")
        return
    oo = float((o.double() * o.double()).sum())
    if oo > 0:
        c = float((t.double() * o.double()).sum()) / oo
        resid = float((t.double() - c * o.double()).norm())
        if resid <= 1e-4 * float(t.double().norm()) + 1e-12:
            raise PermissionError(f"refused: the edit only multiplies the whole tensor by a constant ({c:.4g})")
    os.makedirs(_EDIT_DIR, exist_ok=True)
    np.save(os.path.join(_EDIT_DIR, name + ".npy"), t.numpy())
'''


def run_cell(code, g):
    tree = ast.parse(code, "<cell>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), g)
    if last is not None:
        v = eval(compile(last, "<cell>", "eval"), g)
        if v is not None:
            print(repr(v))


def main():
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    inp = sys.stdin
    sys.stdin = open(os.devnull)
    g = {"__name__": "__main__", "_MODEL": MODEL, "_EDIT_DIR": EDIT_DIR, "_FROZEN": FROZEN}
    try:
        os.chdir(WORK)
    except OSError:
        pass
    exec(PRELUDE, g)
    for k in ("_MODEL",):
        g.pop(k, None)
    proto.write(json.dumps({"ready": True, "tensors": len(g["NAMES"])}) + "\n")
    for line in inp:
        if not line.strip():
            continue
        code = json.loads(line)["code"]
        with tempfile.TemporaryFile() as tf:
            sys.stdout.flush()
            sys.stderr.flush()
            o1, o2 = os.dup(1), os.dup(2)
            os.dup2(tf.fileno(), 1)
            os.dup2(tf.fileno(), 2)
            try:
                run_cell(code, g)
            except BaseException as e:  # noqa: BLE001 - everything is reported back, like a notebook
                tb = e.__traceback__        # hide the harness's own frames: start at the first frame of the cell
                while tb is not None and tb.tb_frame.f_code.co_filename != "<cell>":
                    tb = tb.tb_next
                traceback.print_exception(type(e), e, tb)     # (a syntax error has no cell frame: message only)
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:  # noqa: BLE001
                    pass
                os.dup2(o1, 1)
                os.dup2(o2, 2)
                os.close(o1)
                os.close(o2)
            tf.seek(0)
            out = tf.read().decode("utf-8", "replace")
        proto.write(json.dumps({"out": out}) + "\n")


if __name__ == "__main__":
    main()
