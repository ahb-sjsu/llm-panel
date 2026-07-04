"""Pre-registration freeze/verify -- sha256 over frozen inputs.

Reusable across experiments (CHSH, projection-gap, ...). Freeze any set of
JSON-serializable components (model constants, stimuli, predictions) into a lock
with a combined sha256; verify later that nothing changed before the confirmatory
run. Same discipline as a self-test, applied to human/LLM confirmatory studies.
"""

from __future__ import annotations

import hashlib
import json


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def freeze(components: dict, *, stamp: str, meta: dict | None = None) -> dict:
    """components: {name -> json-able}. Returns a lock dict with per-component and
    combined hashes. `stamp` is an externally-supplied ISO-8601 UTC time (kept out
    of the hash so the lock is reproducible)."""
    comp = {k: sha256(v) for k, v in components.items()}
    combined = sha256(comp)
    return {
        "frozen_at": stamp,
        "combined_sha256": combined,
        "component_sha256": comp,
        "meta": meta or {},
    }


def verify(components: dict, lock: dict) -> bool:
    """True iff re-hashing `components` reproduces the lock's combined hash."""
    comp = {k: sha256(v) for k, v in components.items()}
    return sha256(comp) == lock.get("combined_sha256")
