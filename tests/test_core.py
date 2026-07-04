"""Unit tests for the pure-logic core (no network)."""

from llmpanel import (
    AIMDController,
    ResultStore,
    WorkItem,
    build_prompt,
    grid,
    n_personas,
    parse_choice,
    run,
)
from llmpanel import prereg


# ---- forced choice ----------------------------------------------------------
def test_parse_choice_last_tag_wins():
    txt = "I think A is fine.\nActually reconsidering.\nFINAL CHOICE=B"
    assert parse_choice(txt) == "B"


def test_parse_choice_reasoning_multiple_tags():
    txt = "FINAL CHOICE=A ... wait ... FINAL CHOICE=B"
    assert parse_choice(txt) == "B"  # last tag


def test_parse_choice_fallback_and_none():
    assert parse_choice("clearly option B here") == "B"
    assert parse_choice("no decision") is None
    assert parse_choice("") is None


def test_build_prompt_has_instruction():
    p = build_prompt("Pick one.")
    assert "FINAL CHOICE=" in p and "A or B" in p


# ---- AIMD admission ---------------------------------------------------------
def test_aimd_increases_on_success_and_backs_off_on_throttle():
    c = AIMDController(w_min=1, w_max=32, ai=1.0, md=0.5)
    start = c.window()
    for _ in range(200):
        c.on_success()
    grown = c.window()
    assert grown > start
    c.on_throttle()
    assert c.window() < grown  # multiplicative decrease


def test_aimd_respects_bounds_and_politeness():
    c = AIMDController(w_min=2, w_max=8, target_violation=0.05)
    for _ in range(500):
        c.on_success()
    assert c.window() <= 8
    for _ in range(50):
        c.on_throttle()
    assert c.window() >= 2
    assert c.violation_rate() > 0.05  # throttles registered


# ---- personas ---------------------------------------------------------------
def test_persona_grid_size_and_determinism():
    g1, g2 = grid(), grid()
    assert len(g1) == n_personas() == 3 * 3 * 2 * 2
    assert [p.id for p in g1] == [p.id for p in g2]
    assert "You are a human participant" in g1[0].system_prompt()


# ---- prereg hash ------------------------------------------------------------
def test_prereg_freeze_verify_roundtrip():
    comps = {"model": {"sigma": 1.0}, "pred": [{"id": "x", "delta": 0.1}]}
    lock = prereg.freeze(comps, stamp="2026-07-04T00:00:00Z")
    assert prereg.verify(comps, lock)
    comps2 = {"model": {"sigma": 1.0}, "pred": [{"id": "x", "delta": 0.2}]}  # changed
    assert not prereg.verify(comps2, lock)


def test_prereg_stamp_not_in_hash():
    comps = {"a": 1}
    l1 = prereg.freeze(comps, stamp="2026-07-04T00:00:00Z")
    l2 = prereg.freeze(comps, stamp="2027-01-01T00:00:00Z")
    assert l1["combined_sha256"] == l2["combined_sha256"]  # timestamp excluded


# ---- checkpoint + panel runner (with a fake call_fn) ------------------------
def test_panel_runs_idempotently(tmp_path):
    store = ResultStore(str(tmp_path / "r.jsonl"))
    calls = {"n": 0}

    def fake_call(model, prompt, system):
        calls["n"] += 1
        return "FINAL CHOICE=A", "ok"

    items = [WorkItem(id=f"i{k}", model="m", prompt="p") for k in range(20)]
    run(items, fake_call, store)
    assert len(store) == 20
    assert calls["n"] == 20
    # rerun: everything already done -> no new calls
    store2 = ResultStore(str(tmp_path / "r.jsonl"))
    run(items, fake_call, store2)
    assert calls["n"] == 20  # idempotent, no extra calls
    assert store2.done("i0")


def test_panel_reacts_to_throttle(tmp_path):
    store = ResultStore(str(tmp_path / "r.jsonl"))
    c = AIMDController(w_min=1, w_max=16)

    seq = {"n": 0}

    def flaky(model, prompt, system):
        seq["n"] += 1
        if seq["n"] % 5 == 0:
            return "", "throttle"
        return "FINAL CHOICE=B", "ok"

    items = [WorkItem(id=f"i{k}", model="m", prompt="p") for k in range(40)]
    run(items, flaky, store, controller=c)
    assert len(store) == 40  # all recorded (throttled ones with choice=None)
    oks = 0
    import json

    with open(store.path) as f:
        for line in f:
            if json.loads(line)["status"] == "ok":
                oks += 1
    assert oks > 0
