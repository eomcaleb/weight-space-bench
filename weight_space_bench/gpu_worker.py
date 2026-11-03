"""GPU work, always in a fresh subprocess (a vLLM engine does not release the GPU reliably in-process):

  python -m weight_space_bench.gpu_worker generate <job.json>   vLLM sampling; appends one JSON line per sample to job["out"]
                                                        and skips keys already there (so evaluations resume)
  python -m weight_space_bench.gpu_worker acts <job.json>       transformers: residual stream / MLP inputs of generated answers
  python -m weight_space_bench.gpu_worker tf <job.json>         transformers: a teacher-forced read (log-probs, top-k, hidden
                                                        states, MLP activations, attention probabilities)
A failure writes {"error": "..."} to job["status"] and exits 2.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def render(tok, messages, enable_thinking, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=enable_thinking)


# ------------------------------------------------------------------------------------------------ generation (vLLM)
def generate(job):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    out = Path(job["out"])
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["key"])
            except (ValueError, KeyError):
                pass
    items = [it for it in job["items"] if it["key"] not in done]
    if not items:
        return {"generated": 0}
    s = job["sampling"]
    tok = AutoTokenizer.from_pretrained(job["ckpt"])
    llm = LLM(model=job["ckpt"], dtype="bfloat16", gpu_memory_utilization=job.get("gpu_memory_utilization", 0.8),
              max_model_len=s["max_tokens"] + 4096, seed=0)
    chunk = job.get("chunk", 10 ** 9)
    n_done = 0
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        prompts = [render(tok, [{"role": "user", "content": it["user"]}], s["enable_thinking"]) for it in part]
        params = [SamplingParams(n=it["n"], temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                                 presence_penalty=s["presence_penalty"], max_tokens=it.get("max_tokens", s["max_tokens"]),
                                 seed=it["seed"]) for it in part]
        outs = llm.generate(prompts, params)
        bad = sum(1 for o in outs for x in o.outputs if x.finish_reason not in ("stop", "length"))
        if bad:
            raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens")
        with open(out, "a", encoding="utf-8") as f:
            for it, p, o in zip(part, prompts, outs):
                f.write(json.dumps({"key": it["key"], "prompt": p, "samples": [
                    {"text": x.text, "tokens": len(x.token_ids), "finish": x.finish_reason} for x in o.outputs]},
                    ensure_ascii=False) + "\n")
        n_done += len(part)
        print(f"generated {n_done}/{len(items)}", flush=True)
    return {"generated": n_done}


# ------------------------------------------------------------------------------------------------ transformers side
def load_hf(ckpt, eager=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    kw = {"attn_implementation": "eager"} if eager else {}
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, **kw).to("cuda").eval()
    torch.set_grad_enabled(False)
    return tok, model


class Taps:
    """Forward hooks: residual stream (0 = embeddings, l = after decoder layer l), MLP down_proj inputs per decoder
    layer, attention probabilities of full-attention layers."""

    def __init__(self, model, resid=(), mlp=(), attn=()):
        self.base = model.model
        self.layers = self.base.layers
        self.store, self.handles = {}, []
        for l in resid:
            mod = self.base.embed_tokens if l == 0 else self.layers[l - 1]
            self.handles.append(mod.register_forward_hook(self._out(("resid", l))))
        for l in mlp:
            self.handles.append(self.layers[l].mlp.down_proj.register_forward_pre_hook(self._inp(("mlp", l))))
        for l in attn:
            self.handles.append(self.layers[l].self_attn.register_forward_hook(self._attn(("attn", l))))

    def _out(self, key):
        def f(_m, _i, o):
            self.store[key] = (o[0] if isinstance(o, tuple) else o)[0].float()
        return f

    def _inp(self, key):
        def f(_m, i):
            self.store[key] = i[0][0].float()
        return f

    def _attn(self, key):
        def f(_m, _i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                self.store[key] = o[1][0].float()
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def acts(job):
    """For every generated answer: the residual stream at the requested layers (and MLP inputs) at the last prompt
    token and averaged over the answer's first act_max_tokens tokens."""
    import torch
    tok, model = load_hf(job["ckpt"])
    layers = job["layers"]
    mlp_layers = [l - 1 for l in layers if l >= 1] if job.get("mlp") else []
    taps = Taps(model, resid=layers, mlp=mlp_layers)
    n = len(job["items"])
    outd = Path(job["outdir"])
    outd.mkdir(parents=True, exist_ok=True)
    H = model.config.hidden_size
    I = model.config.intermediate_size
    hl = np.lib.format.open_memmap(outd / "hidden_last_prompt.npy", "w+", np.float32, (n, len(layers), H))
    hm = np.lib.format.open_memmap(outd / "hidden_mean_answer.npy", "w+", np.float32, (n, len(layers), H))
    if mlp_layers:
        ml = np.lib.format.open_memmap(outd / "mlp_last_prompt.npy", "w+", np.float32, (n, len(mlp_layers), I))
        mm = np.lib.format.open_memmap(outd / "mlp_mean_answer.npy", "w+", np.float32, (n, len(mlp_layers), I))
    for i, it in enumerate(job["items"]):
        p_ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        a_ids = tok(it["text"], add_special_tokens=False)["input_ids"][:job["act_max_tokens"]]
        ids = torch.tensor([p_ids + a_ids], device="cuda")
        taps.store.clear()
        model.model(input_ids=ids, use_cache=False)
        lp = len(p_ids) - 1
        for j, l in enumerate(layers):
            r = taps.store[("resid", l)]
            hl[i, j] = r[lp].cpu().numpy()
            hm[i, j] = (r[lp + 1:].mean(0) if a_ids else r[lp]).cpu().numpy()
        for j, l in enumerate(mlp_layers):
            r = taps.store[("mlp", l)]
            ml[i, j] = r[lp].cpu().numpy()
            mm[i, j] = (r[lp + 1:].mean(0) if a_ids else r[lp]).cpu().numpy()
    taps.remove()
    files = {"hidden_last_prompt.npy": [n, len(layers), H], "hidden_mean_answer.npy": [n, len(layers), H]}
    if mlp_layers:
        files.update({"mlp_last_prompt.npy": [n, len(mlp_layers), I], "mlp_mean_answer.npy": [n, len(mlp_layers), I]})
    return {"files": files, "layers": layers, "mlp_decoder_layers": mlp_layers}


def teacher_forced(job):
    import torch
    r = job["request"]
    ret = r.get("return") or {}
    attn_layers = list(ret.get("attn_layers") or [])
    tok, model = load_hf(job["ckpt"], eager=bool(attn_layers))
    lim = job["limits"]
    full_attn = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
    if "texts" in r:
        texts = r["texts"]
    elif "chats" in r:
        texts = [render(tok, c, bool(r.get("enable_thinking")), add_generation_prompt=False) for c in r["chats"]]
    else:
        raise ValueError('give "texts" or "chats"')
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError('"texts" must be a non-empty list of strings')
    if len(texts) > lim["tf_max_texts"]:
        raise ValueError(f"at most {lim['tf_max_texts']} texts per pass")
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    lens = [len(x) for x in ids]
    if max(lens) > lim["tf_max_tokens"]:
        raise ValueError(f"a text has {max(lens)} tokens; at most {lim['tf_max_tokens']}")
    if sum(lens) > lim["tf_total_tokens"]:
        raise ValueError(f"{sum(lens)} tokens in total; at most {lim['tf_total_tokens']}")
    hs = list(ret.get("hidden_states") or [])
    mlp = list(ret.get("mlp_act") or [])
    k = int(ret.get("topk") or 0)
    if k > 50:
        raise ValueError("topk is at most 50")
    if any(not (0 <= l <= model.config.num_hidden_layers) for l in hs):
        raise ValueError("hidden_states layers are 0..24")
    if any(not (0 <= l < model.config.num_hidden_layers) for l in mlp):
        raise ValueError("mlp_act layers are 0..23")
    if attn_layers:
        if any(l not in full_attn for l in attn_layers):
            raise ValueError(f"attn_layers must be full-attention layers {full_attn}")
        if len(texts) > 2 or max(lens) > 2048:
            raise ValueError("attn_layers: at most 2 texts of at most 2,048 tokens")
    n, T = len(texts), max(lens)
    H, I = model.config.hidden_size, model.config.intermediate_size
    nheads = model.config.num_attention_heads
    gb = n * T * (2 * H * len(hs) + 2 * I * len(mlp) + 4 + 8 * k) / 1e9 + n * nheads * T * T * 4 * len(attn_layers) / 1e9
    if gb > lim["tf_max_gb"]:
        raise ValueError(f"the result would take {gb:.1f} GB; at most {lim['tf_max_gb']} GB per pass")
    outd = Path(job["outdir"])
    outd.mkdir(parents=True, exist_ok=True)
    files = {}

    def mm(name, dtype, shape, fill):
        a = np.lib.format.open_memmap(outd / name, "w+", dtype, shape)
        a[:] = fill
        files[name] = list(shape)
        return a

    tokens = mm("tokens.npy", np.int32, (n, T), -1)
    want_lp = bool(ret.get("token_logprobs", True))
    lp = mm("token_logprobs.npy", np.float32, (n, T), np.nan) if want_lp else None
    if k:
        tki = mm("topk_ids.npy", np.int32, (n, T, k), -1)
        tkl = mm("topk_logprobs.npy", np.float32, (n, T, k), np.nan)
    hsa = {l: mm(f"hidden_{l}.npy", np.float16, (n, T, H), 0) for l in hs}
    mla = {l: mm(f"mlp_act_{l}.npy", np.float16, (n, T, I), 0) for l in mlp}
    ata = {l: mm(f"attn_{l}.npy", np.float32, (n, nheads, T, T), 0) for l in attn_layers}
    taps = Taps(model, resid=hs, mlp=mlp, attn=attn_layers)
    W = model.get_output_embeddings().weight
    for i, x in enumerate(ids):
        L = len(x)
        tokens[i, :L] = x
        t = torch.tensor([x], device="cuda")
        taps.store.clear()
        h = model.model(input_ids=t, use_cache=False, output_attentions=bool(attn_layers)).last_hidden_state[0]
        if want_lp or k:
            for c0 in range(0, L - 1, 2048):                  # logits in chunks: the vocabulary is 248,320 wide
                c1 = min(L - 1, c0 + 2048)
                logp = torch.log_softmax((h[c0:c1] @ W.T).float(), -1)
                if want_lp:
                    nxt = t[0, c0 + 1:c1 + 1]
                    lp[i, c0 + 1:c1 + 1] = logp.gather(1, nxt[:, None])[:, 0].cpu().numpy()
                if k:
                    v, ix = logp.topk(k, -1)
                    tkl[i, c0 + 1:c1 + 1] = v.cpu().numpy()
                    tki[i, c0 + 1:c1 + 1] = ix.cpu().numpy()
        for l in hs:
            hsa[l][i, :L] = taps.store[("resid", l)].cpu().numpy().astype(np.float16)
        for l in mlp:
            mla[l][i, :L] = taps.store[("mlp", l)].cpu().numpy().astype(np.float16)
        for l in attn_layers:
            a = taps.store.get(("attn", l))
            if a is None:
                raise RuntimeError("attention probabilities were not returned by the model")
            ata[l][i, :, :L, :L] = a.cpu().numpy()
    taps.remove()
    meta = {"texts": texts if sum(len(s) for s in texts) < 2_000_000 else None, "lengths": lens, "files": files,
            "notes": {"tokens": "token ids, right-padded with -1",
                      "token_logprobs": "log-prob of token t given tokens < t (position 0 is NaN)",
                      "topk": "the k most likely tokens at position t given tokens < t (aligned with token_logprobs)",
                      "hidden_<l>": "residual stream (float16): 0 = embeddings, l = after layer l (before the final "
                                    "norm)",
                      "mlp_act_<l>": "input of layer l's mlp.down_proj (float16)",
                      "attn_<l>": "attention probabilities [text, head, query, key]"}}
    json.dump(meta, open(outd / "meta.json", "w"))
    files["meta.json"] = None
    return {"texts": n, "tokens_total": sum(lens), "max_len": T, "files": files}


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    job = json.load(open(path, encoding="utf-8"))
    t0 = time.time()
    try:
        res = {"generate": generate, "acts": acts, "tf": teacher_forced}[cmd](job)
        res["s"] = round(time.time() - t0, 1)
        json.dump(res, open(job["status"], "w"))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        json.dump({"error": f"{type(e).__name__}: {e}"}, open(job["status"], "w"))
        sys.exit(2)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
