"""OpenRouter chat client (OpenAI-compatible /chat/completions) with retries, usage and cost accounting."""
import os
import time

import requests

URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") + "/chat/completions"


class ContextTooLong(Exception):
    pass


class LLMError(Exception):
    pass


class NotIsolated(Exception):
    pass


def audit(d):
    """The paper's tool audit, applied to an OpenRouter response: a turn is tool-free when the reply is only text
    (and reasoning). Tool calls, function calls or web-search citations make it not tool-free."""
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    found = []
    if msg.get("tool_calls"):
        found.append("tool_calls")
    if msg.get("function_call"):
        found.append("function_call")
    if any((a or {}).get("type") == "url_citation" for a in msg.get("annotations") or []):
        found.append("web_citations")
    if (d.get("choices") or [{}])[0].get("finish_reason") in ("tool_calls", "function_call"):
        found.append("finish_reason=tool_calls")
    return {"tool_free": not found, "found": found}


class OpenRouter:
    """No tools are ever offered to the editing model: no `tools` / `functions` field, no plugins, no `:online`
    (web search) variants. The model can act only by writing ```python / ```forward blocks, which the harness runs
    in the sandbox."""

    def __init__(self, model_cfg, api_key=None):
        self.m = model_cfg
        if ":online" in model_cfg["slug"] or model_cfg.get("plugins") or model_cfg.get("tools"):
            raise SystemExit(f"{model_cfg['slug']}: web search / plugins / tools are not allowed in this benchmark")
        self.effort = model_cfg.get("reasoning_effort")
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise SystemExit("set OPENROUTER_API_KEY")

    def _messages(self, messages):
        """Anthropic models need explicit cache breakpoints; one on the system message and one on the latest user
        message caches the growing conversation prefix (OpenAI, DeepSeek, Moonshot and Gemini cache automatically)."""
        if not self.m["slug"].startswith("anthropic/"):
            return messages
        out = [dict(x) for x in messages]
        marks = [0, len(out) - 1] if len(out) > 1 else [0]
        for i in marks:
            c = out[i]["content"]
            if isinstance(c, str):
                out[i]["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
        return out

    def chat(self, messages, max_retries=8):
        body = {"model": self.m["slug"], "messages": self._messages(messages),
                "max_tokens": self.m.get("max_output_tokens", 32000), "usage": {"include": True}}
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        if self.m.get("provider"):
            body["provider"] = self.m["provider"]
        if "temperature" in self.m:
            body["temperature"] = self.m["temperature"]
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "weight-space-bench"}
        assert not any(k in body for k in ("tools", "functions", "tool_choice", "plugins")), "no tools allowed"
        delay = 10
        dirty = 0
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                r = requests.post(URL, json=body, headers=headers, timeout=3600)
            except requests.RequestException as e:
                err = f"network: {e}"
            else:
                if r.status_code == 200:
                    d = r.json()
                    if d.get("error"):
                        err = str(d["error"])
                    elif not d.get("choices"):
                        err = f"no choices: {str(d)[:300]}"
                    elif not audit(d)["tool_free"]:
                        dirty += 1                          # discarded, never shown to the harness or the model
                        err = f"reply not tool-free ({audit(d)['found']}); discarded"
                        if dirty >= 3:
                            raise NotIsolated(f"{self.m['slug']}: {dirty} replies in a row used tools: {audit(d)['found']}")
                    else:
                        msg = d["choices"][0]["message"]
                        text = msg.get("content") or ""
                        if isinstance(text, list):
                            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                        u = d.get("usage") or {}
                        if not text.strip() and d["choices"][0].get("finish_reason") not in ("stop", "end_turn"):
                            err = f"empty reply (finish_reason={d['choices'][0].get('finish_reason')})"
                        else:
                            return {"text": text, "reasoning": msg.get("reasoning"), "usage": u, "audit": audit(d),
                                    "effort": self.effort,
                                    "discarded_not_tool_free": dirty,
                                    "cost": float(u.get("cost") or 0.0), "finish": d["choices"][0].get("finish_reason"),
                                    "provider": d.get("provider"), "s": round(time.time() - t0, 1)}
                else:
                    err = f"HTTP {r.status_code}: {r.text[:500]}"
                    low = r.text.lower()
                    if (r.status_code in (400, 422) and self.effort not in (None, "high") and "effort" in low):
                        # the provider does not accept this effort (e.g. "max" / "xhigh"): use "high" for the rest
                        # of the run (sticky, so every turn of one run uses the same effort)
                        print(f"  [llm] effort {self.effort!r} rejected; falling back to 'high'", flush=True)
                        self.effort = "high"
                        body["reasoning"] = {"effort": "high"}
                        continue
                    if r.status_code in (400, 413) and any(s in low for s in ("context", "too long", "maximum", "tokens")):
                        raise ContextTooLong(err)
                    if r.status_code in (401, 402, 403):
                        raise LLMError(err)
            print(f"  [llm] attempt {attempt + 1} failed: {err}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)
        raise LLMError(err)
