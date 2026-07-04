"""Polite-bursting admission controller (AIMD + probe estimator).

Reusable concurrency governor for bursting many short LLM completions onto a
shared, policy-bound gateway (NRP managed-LLM, Nautilus). Instead of a fixed
`ThreadPoolExecutor(CONC=k)` -- which either violates fair-use or wastes goodput
-- concurrency adapts: additive-increase while healthy, multiplicative back-off
on throttle/error, holding a bounded policy-violation rate.

This is the control law from the "Polite Bursting" fabric (INFOCOM / CANOPIE-HPC
@ SC26), reduced to a dependency-free, unit-testable core. It is pure logic: it
decides the in-flight window; the caller enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIMDController:
    """Additive-increase / multiplicative-decrease admission window.

    - `window()` is the current max in-flight requests (an int >= w_min).
    - call `on_success()` / `on_throttle()` / `on_error()` as results arrive.
    - additive increase `ai` per success once a full window has cleared;
      multiplicative decrease `md` on throttle (429/5xx/timeout).
    - `violation_rate()` is the EWMA throttle fraction (the "politeness" metric);
      the controller backs off hard to keep it under `target_violation`.
    """

    w_min: float = 1.0
    w_max: float = 64.0
    ai: float = 1.0  # additive increase per cleared window
    md: float = 0.5  # multiplicative decrease on throttle
    target_violation: float = 0.05
    ewma_alpha: float = 0.1

    def __post_init__(self) -> None:
        self._w: float = max(self.w_min, min(4.0, self.w_max))  # cautious start
        self._acks_since_increase = 0
        self._viol_ewma = 0.0

    def window(self) -> int:
        return max(1, int(self._w))

    def violation_rate(self) -> float:
        return self._viol_ewma

    def _observe(self, violated: bool) -> None:
        self._viol_ewma = (1 - self.ewma_alpha) * self._viol_ewma + self.ewma_alpha * (
            1.0 if violated else 0.0
        )

    def on_success(self) -> None:
        self._observe(False)
        self._acks_since_increase += 1
        # additive increase once we've cleared a full current window of acks,
        # and only if we're comfortably under the politeness target.
        if self._acks_since_increase >= self.window() and self._viol_ewma < self.target_violation:
            self._w = min(self.w_max, self._w + self.ai)
            self._acks_since_increase = 0

    def on_throttle(self) -> None:
        """429 / 503 / rate-limit: the host is pushing back -> back off hard."""
        self._observe(True)
        self._w = max(self.w_min, self._w * self.md)
        self._acks_since_increase = 0
        # extra caution if we're over the politeness budget
        if self._viol_ewma > self.target_violation:
            self._w = max(self.w_min, self._w * self.md)

    def on_error(self) -> None:
        """Transient 5xx/timeout not clearly rate-limit: gentle multiplicative back-off."""
        self._observe(True)
        self._w = max(self.w_min, self._w * (0.5 * (1 + self.md)))  # softer than throttle
        self._acks_since_increase = 0
