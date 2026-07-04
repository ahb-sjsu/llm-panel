"""Persona sampling -- induce between-subject heterogeneity within a model.

An LLM at fixed weights is one "type"; conditioning on a persona system-prompt
turns it into a population of synthetic subjects. The grid is deterministic given
a seed (reproducible / pre-registerable): a subject id maps to a fixed cell of the
value/demographic grid, so the same panel can be re-run exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

# Pre-registered persona axes. Extend deliberately (changes the panel identity).
AXES: dict[str, tuple[str, ...]] = {
    "risk": ("risk-averse", "risk-neutral", "risk-seeking"),
    "prosocial": ("self-interested", "fair-minded", "altruistic"),
    "culture": ("individualist", "collectivist"),
    "numeracy": ("low-numeracy", "high-numeracy"),
}


@dataclass(frozen=True)
class Persona:
    id: str
    traits: dict

    def system_prompt(self) -> str:
        desc = ", ".join(f"{v}" for v in self.traits.values())
        return (
            "You are a human participant in an economics study. Decide as a real person "
            f"who is {desc}. Answer honestly and consistently with that disposition."
        )


def grid() -> list[Persona]:
    """The full deterministic persona grid (cartesian product of the axes)."""
    keys = list(AXES)
    out = []
    for combo in product(*(AXES[k] for k in keys)):
        traits = dict(zip(keys, combo))
        pid = "-".join(combo)
        out.append(Persona(id=pid, traits=traits))
    return out


def n_personas() -> int:
    n = 1
    for v in AXES.values():
        n *= len(v)
    return n
