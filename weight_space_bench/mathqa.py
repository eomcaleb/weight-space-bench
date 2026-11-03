"""Answer extraction and grading, exactly as in the release (exp8_math/code/mathqa.py, MIT; the hendrycks/math
helpers it wraps are MIT too). Trimmed to what the benchmark uses: the prompt, the last-\\boxed{} reading, and the
math-verify equivalence check.
"""
import functools
import re
import threading

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(problem):
    """The user turn: the problem, a newline, Qwen's math instruction."""
    return problem + "\n" + INSTRUCTION


def answer_text(text):
    """The text the extraction reads: a stray reasoning block cut off (up to the last '</think>')."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text):
    """(answer or None, number of boxes in the text)."""
    t = answer_text(text)
    return last_boxed(t), len(_BOX_FORMS.findall(t))


def _timeouts():
    return 5 if threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s, to):
    import math_verify
    return math_verify.parse("\\boxed{" + s + "}", parsing_timeout=to)


def is_correct(pred, gold):
    """math-verify equivalence of a boxed answer and the gold answer (False for no answer / unparseable gold)."""
    if pred is None or gold is None:
        return False
    import math_verify
    to = _timeouts()
    g, p = _parsed(gold, to), _parsed(pred, to)
    if not g or not p:
        return False
    try:
        return bool(math_verify.verify(g, p, timeout_seconds=to))
    except Exception:  # noqa: BLE001
        return False


def grade(text, gold, finish):
    """One generated answer -> record fields used by both the forward-pass tool and the evaluation."""
    pred, _ = extract(text)
    ok = is_correct(pred, gold)
    if ok:
        kind = "correct"
    elif finish == "length":
        kind = "cut_at_max_tokens"
    elif pred is None:
        kind = "no_answer_finished"
    else:
        kind = "wrong_answer"
    return {"pred": pred if pred is None or len(pred) <= 200 else pred[:200] + "...", "correct": bool(ok),
            "outcome": kind}
