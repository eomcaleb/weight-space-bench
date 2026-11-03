"""Reads a reply's ```edits block into the network's weights, as the prompt defines the language:

  W1[12, 0] = 2.5                      one entry
  W1[0:6, 3] = [1, 0, -1, 0, 2, 0]     explicit values along a range (as many values as entries)
  b1[0:4] = -1                         fill a range (a:b half-open, a:b:s strided, ':' the whole axis)
  W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]   a 2-D list for a 2-D block
  W3[:, 2] *= 2                        scale (also += and -=)

'#' starts a comment; a long list may continue over the next lines; statements apply in order; only the LAST edits
block of the reply is read; a statement that cannot be read is skipped (the others still apply); |v| <= 1e6.
Every weight starts at 0.
"""
import ast
import re

import numpy as np

SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
            "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
MAX_ABS = 1e6
BLOCK = re.compile(r"```edits[^\n]*\n(.*?)(```|\Z)", re.S)
STMT = re.compile(r"^\s*([A-Za-z_]\w*)\s*(\[(?P<idx>[^\]]*)\])?\s*(?P<op>[*+\-]?=)\s*(?P<val>.+?)\s*$", re.S)


def zeros():
    return {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}


def last_block(text):
    """(block text or None, cut_off): the last ```edits block; an unclosed final block (a reply cut at the token
    limit) is read up to the end of the reply."""
    blocks = list(BLOCK.finditer(text))
    if not blocks:
        return None, False
    m = blocks[-1]
    return m.group(1), m.group(2) != "```"


def statements(block):
    """Split into statements: comments removed, lines joined while brackets are open."""
    out, cur, depth = [], "", 0
    for line in block.split("\n"):
        line = line.split("#", 1)[0]
        if not line.strip() and depth == 0:
            continue
        cur = (cur + " " + line) if cur else line
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            if cur.strip():
                out.append(cur.strip())
            cur, depth = "", 0
    if cur.strip():
        out.append(cur.strip())             # unfinished (cut-off) statement: will fail to read and be skipped
    return out


def _index(s, ndim):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != ndim:
        raise ValueError(f"expected {ndim} indices")
    idx = []
    for p in parts:
        if ":" in p:
            bits = p.split(":")
            if len(bits) > 3:
                raise ValueError(f"bad range {p!r}")
            idx.append(slice(*[int(b) if b.strip() else None for b in bits]))
        else:
            idx.append(int(p))
    return tuple(idx)


def _value(s):
    v = ast.literal_eval(s)
    a = np.array(v, dtype=np.float64)
    if a.dtype == object or not np.isfinite(a).all():
        raise ValueError("values must be finite numbers")
    if np.abs(a).max(initial=0) > MAX_ABS:
        raise ValueError(f"|value| > {MAX_ABS:g}")
    return a


def apply(text, net=None):
    """-> (weights dict, report). The report mirrors the paper's apply.json."""
    net = net or zeros()
    block, cut = last_block(text)
    rep = {"block": block is not None, "statements": 0, "applied": 0, "entries_written": 0, "errors": [],
           "cut_off": cut}
    for st in (statements(block) if block is not None else []):
        rep["statements"] += 1
        try:
            m = STMT.match(st)
            if not m:
                raise ValueError("not a statement")
            name, op = m.group(1), m.group("op")
            if name not in SHAPES:
                raise ValueError(f"unknown tensor {name!r}")
            t = net[name]
            idx = _index(m.group("idx"), t.ndim) if m.group("idx") is not None else tuple(slice(None) for _ in t.shape)
            target = t[idx]
            v = _value(m.group("val"))
            tshape = np.shape(target)
            if v.ndim and v.shape != tshape:
                raise ValueError(f"{v.size} values for a block of shape {tshape}")
            cur = np.asarray(target, np.float64)
            new = {"=": lambda: np.broadcast_to(v, tshape), "*=": lambda: cur * v, "+=": lambda: cur + v,
                   "-=": lambda: cur - v}[op]()
            new = np.asarray(new, np.float64)
            if np.abs(new).max(initial=0) > MAX_ABS:
                raise ValueError(f"result |value| > {MAX_ABS:g}")
            if tshape == ():
                t[idx] = np.float32(new)
            else:
                t[idx] = new.astype(np.float32)
            rep["applied"] += 1
            rep["entries_written"] += int(np.size(target))
        except Exception as e:  # noqa: BLE001 - an unreadable statement is skipped, the others still apply
            rep["errors"].append({"statement": st[:200], "error": f"{type(e).__name__}: {e}"})
    rep["nonzero"] = {k: int(np.count_nonzero(v)) for k, v in net.items()}
    rep["nonzero_total"] = sum(rep["nonzero"].values())
    rep["max_abs"] = {k: float(np.abs(v).max()) for k, v in net.items()}
    return net, rep


def save_npz(net, path):
    np.savez(path, **{NPZ_KEYS[k]: v for k, v in net.items()})


def load_npz(path):
    z = np.load(path)
    return {k: z[v].astype(np.float32) for k, v in NPZ_KEYS.items()}
