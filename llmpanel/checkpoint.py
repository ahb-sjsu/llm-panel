"""Idempotent, preemption-tolerant result store (append-only JSONL).

Every work item has a deterministic id (e.g. contrast x subject x sample). Results
are keyed by id; completed ids are skipped on restart. This is the local analogue
of the JetStream durable result stream in the Polite-Bursting plane: NRP pods can
be preempted mid-run and nothing is lost or double-counted.
"""

from __future__ import annotations

import json
import os
import threading


class ResultStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._done: set[str] = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    def done(self, item_id: str) -> bool:
        return item_id in self._done

    def pending(self, item_ids):
        return [i for i in item_ids if i not in self._done]

    def put(self, item_id: str, record: dict) -> None:
        """Append a result. Idempotent: a repeated id is ignored."""
        with self._lock:
            if item_id in self._done:
                return
            row = {"id": item_id, **record}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._done.add(item_id)

    def __len__(self) -> int:
        return len(self._done)
