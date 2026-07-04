# llm-panel

A small, reusable framework for running **pre-registered LLM-panel behavioral experiments** on
shared, policy-bound clusters (built for NRP Nautilus). An LLM at fixed weights, conditioned on a
grid of sampled personas, is a scalable *population of synthetic subjects* — this package is the
shared machinery for probing that population **politely, reproducibly, and pre-registered**.

Extracted from two experiments that needed the same infrastructure:
- **CbD contextuality ladder** (`sqnd-probe/chsh_ladder`) — is LLM normative judgment quantum-contextual?
- **Projection-gap** (`geometric-economics`) — does a shared decision geometry predict cross-domain
  behavior that scalar theories provably cannot?

## What it provides

| Module | Role |
|---|---|
| `admission` | **AIMD polite-bursting controller** — adaptive concurrency (additive-increase / multiplicative-back-off, bounded policy-violation rate) instead of a fixed thread pool. The control law from the *Polite Bursting* fabric (INFOCOM / CANOPIE-HPC @ SC26), reduced to a testable core. |
| `nrp` | NRP managed-LLM client (OpenAI-compatible; token from `~/.llmtoken`; reasoning-model handling). |
| `forcedchoice` | Output-only forced-choice prompt + robust last-tag parser (no chain-of-thought leakage). |
| `personas` | Deterministic persona grid → between-subject heterogeneity within one model. |
| `prereg` | `sha256` freeze/verify of frozen predictions — pre-registration you can prove. |
| `checkpoint` | Idempotent, preemption-tolerant result store (NRP pods can be evicted; nothing is lost or double-counted). |
| `panel` | The runner tying them together; concurrency tracks the live AIMD window. |

## Design commitments

- **Polite by construction.** Bursting many short completions onto a multi-tenant gateway is a
  control problem, not a fixed rate. The AIMD controller targets the polite-efficient point — fast
  when the host is healthy, backing off on throttle — rather than a static `CONC=k`.
- **Pre-registered.** Predictions are hashed and frozen before any subject is run; the hash voids if
  the model, stimuli, or predictions change afterwards.
- **Preemption-tolerant.** Every work item has a deterministic id; results are keyed by id, so
  restarts and pod evictions are free (superset of per-model checkpointing).

## Relationship to `fun-hypothesis`

[`fun-hypothesis`](https://github.com/ahb-sjsu/fun-hypothesis) is a sibling project: a specific
*application* (does prompt framing change LLM output quality?) with a rigorous statistical
methodology (paired/Wilcoxon tests, Bonferroni, Cohen's d, power, repeated-measures ANOVA). `llm-panel`
is the general *infrastructure* underneath such studies; the analysis layer here reuses
`fun-hypothesis`'s statistical vocabulary so the two share one methodology.

## Status

Core is implemented and unit-tested (pure-logic modules run with no network). The NRP execution plane
and the analysis layer are wired for the projection-gap study. MIT.
