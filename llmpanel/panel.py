"""Panel runner: AIMD-governed concurrent execution with idempotent checkpointing.

Ties together the admission controller, a call function (NRP by default, or any
injected callable for tests), the forced-choice parser, and the durable result
store. Concurrency tracks the AIMD window in real time, so the panel bursts
politely: it speeds up while the gateway is healthy and backs off on throttle.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .admission import AIMDController
from .checkpoint import ResultStore
from .forcedchoice import parse_choice


@dataclass
class WorkItem:
    id: str
    model: str
    prompt: str
    system: str | None = None
    meta: dict | None = None


def run(
    items,
    call_fn,
    store: ResultStore,
    *,
    controller: AIMDController | None = None,
    options=("A", "B"),
    max_workers: int = 64,
    poll: float = 0.01,
) -> ResultStore:
    """Execute `items` (idempotently) via `call_fn(model, prompt, system)->(text,status)`.

    Concurrency is capped by the live AIMD window. Completed ids (in `store`) are
    skipped. Results record the raw text, parsed choice, and status.
    """
    ctrl = controller or AIMDController()
    pending = [it for it in items if not store.done(it.id)]
    lock = threading.Lock()
    inflight = {"n": 0}

    def work(it: WorkItem):
        text, status = call_fn(it.model, it.prompt, it.system)
        with lock:
            if status == "ok":
                ctrl.on_success()
            elif status == "throttle":
                ctrl.on_throttle()
            else:
                ctrl.on_error()
            inflight["n"] -= 1
        choice = parse_choice(text, options) if status == "ok" else None
        store.put(
            it.id,
            {
                "model": it.model,
                "status": status,
                "choice": choice,
                "raw": text[:500],
                **(it.meta or {}),
            },
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        i = 0
        futures = []
        while i < len(pending):
            with lock:
                room = ctrl.window() - inflight["n"]
            if room <= 0:
                time.sleep(poll)
                continue
            for _ in range(min(room, len(pending) - i)):
                with lock:
                    inflight["n"] += 1
                futures.append(ex.submit(work, pending[i]))
                i += 1
        for f in futures:
            f.result()
    return store
