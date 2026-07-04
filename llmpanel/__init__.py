"""llm-panel -- a reusable framework for LLM-panel behavioral experiments.

An LLM at fixed weights, conditioned on sampled personas, is a scalable population
of synthetic subjects. This package provides the shared machinery for running
pre-registered forced-choice experiments across an NRP model ladder, politely and
reproducibly:

    admission   -- AIMD polite-bursting concurrency controller
    nrp         -- NRP managed-LLM client (OpenAI-compatible)
    forcedchoice-- output-only forced-choice prompt + robust parser
    personas    -- deterministic persona grid (between-subject heterogeneity)
    prereg      -- sha256 freeze/verify of frozen predictions (pre-registration)
    checkpoint  -- idempotent, preemption-tolerant result store
    panel       -- the runner tying them together

Used by: sqnd-probe/chsh_ladder (CbD contextuality) and geometric-economics
projection-gap (decision-geometry subsumption).
"""

from .admission import AIMDController
from .checkpoint import ResultStore
from .forcedchoice import build_prompt, parse_choice
from .panel import WorkItem, run
from .personas import Persona, grid, n_personas

__version__ = "0.1.0"
__all__ = [
    "AIMDController",
    "ResultStore",
    "build_prompt",
    "parse_choice",
    "WorkItem",
    "run",
    "Persona",
    "grid",
    "n_personas",
]
