"""NRP managed-LLM client (OpenAI-compatible Envoy AI gateway).

Thin wrapper over the NRP Nautilus managed-LLM endpoint. Token is read from
`~/.llmtoken` (as on Atlas). Reasoning models need `enable_thinking=False` and a
larger token budget; fast models answer directly. Returns (text, status) where
status is one of "ok" | "throttle" | "error" so the admission controller can react.
"""

from __future__ import annotations

import os

DEFAULT_BASE = "https://ellm.nrp-nautilus.io/v1"
REASONING = {"gpt-oss", "glm-5", "qwen3", "minimax-m2", "kimi"}


def load_token(path: str = "~/.llmtoken") -> str:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return f.read().strip()


def maxtok_for(model: str, fast: int = 24, reason: int = 320) -> int:
    return reason if model in REASONING else fast


def call(
    model: str,
    prompt: str,
    *,
    token: str,
    base: str = DEFAULT_BASE,
    temperature: float = 0.9,
    system: str | None = None,
    timeout: int = 120,
):
    """Single completion. Returns (text, status). status in {ok, throttle, error}.

    Requires `requests`; imported lazily so the pure-logic modules stay dependency-free.
    """
    import requests

    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    base_body = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": maxtok_for(model),
    }
    hdr = {"Authorization": "Bearer " + token}
    # try with thinking disabled first (reasoning models), then plain
    for body in ({**base_body, "chat_template_kwargs": {"enable_thinking": False}}, base_body):
        try:
            r = requests.post(base + "/chat/completions", json=body, headers=hdr, timeout=timeout)
        except requests.RequestException:
            return "", "error"
        if r.status_code == 200:
            m = r.json()["choices"][0]["message"]
            return (m.get("content") or m.get("reasoning") or ""), "ok"
        if r.status_code in (429, 503):
            return "", "throttle"
        if r.status_code in (500, 502, 504):
            return "", "error"
        if r.status_code == 400:
            continue  # retry without chat_template_kwargs
    return "", "error"
