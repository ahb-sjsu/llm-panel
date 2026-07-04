"""Forced-choice prompting + robust parsing for LLM-panel experiments.

Output-only forced choice (no chain-of-thought leakage): the subject must end
with a single tagged line, e.g. `FINAL CHOICE=A`. Parsing prefers the last
explicit tag, then falls back to the last standalone token -- generalized from
the CHSH ladder's `parse_outcomes`, but for a single categorical choice.
"""

from __future__ import annotations

import re

TAG_RE = re.compile(r"FINAL\s*CHOICE\s*=\s*([A-Za-z]+)", re.IGNORECASE)


def parse_choice(text: str, options: tuple[str, ...] = ("A", "B")) -> str | None:
    """Return the chosen option (upper-cased, from `options`) or None if unparseable.

    Strategy: last `FINAL CHOICE=<opt>` tag wins (reasoning models emit several
    lines); else the last standalone occurrence of any option token.
    """
    if not text:
        return None
    opts = tuple(o.upper() for o in options)
    tags = TAG_RE.findall(text)
    for tok in reversed(tags):
        if tok.upper() in opts:
            return tok.upper()
    # fallback: last standalone option token (word-boundary) anywhere in the text
    best_pos, best = -1, None
    up = text.upper()
    for o in opts:
        for m in re.finditer(rf"(?<![A-Z]){re.escape(o)}(?![A-Z])", up):
            if m.start() > best_pos:
                best_pos, best = m.start(), o
    return best


def build_prompt(body: str, options: tuple[str, ...] = ("A", "B")) -> str:
    """Append the fixed output-only answer instruction to a scenario body."""
    opts = " or ".join(o.upper() for o in options)
    return f"{body}\nAnswer with ONLY this line: FINAL CHOICE=<{opts}>"
