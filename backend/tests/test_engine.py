import json

import numpy as np
import pytest

from snn.engine import run_simulation
from snn.helpers import generate_puzzle, get_puzzle


@pytest.fixture(scope="module")
def run40():
    puzzle = generate_puzzle(40, np.random.default_rng(1))
    return puzzle, list(run_simulation(puzzle, seed=42, max_sim_ms=20000))


def test_event_order_and_solved(run40):
    _, events = run40
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-1] == "done"
    assert set(types[1:-1]) == {"step"}
    done = events[-1]
    assert done["solved"] is True and done["resume"] is False
    assert done["classical"]["verdict"] == "match"
    assert done["final_grid"] == events[-2]["grid"]
    assert len(done["cells"]["counts"]) == 81 and len(done["cells"]["margin"]) == 81


def test_meta_connections(run40):
    meta = run40[1][0]
    c = meta["connections"]
    assert c["total"] == 521235
    assert c["inhibitory"] == 510300
    assert c["noise"] == c["stimulus"] == c["recorder"] == 3645
    assert c["stimulus_active"] == meta["givens"] * 5


def test_step_payload(run40):
    for step in (e for e in run40[1] if e["type"] == "step"):
        assert len(json.dumps(step)) < 3_000
        assert len(step["grid"]) == 81
        assert step["rate_winners_hz"] > step["rate_losers_hz"]
        assert step["t_end_ms"] - step["t_start_ms"] == 100


def test_continuation_offsets_time():
    puzzle = generate_puzzle(36, np.random.default_rng(3))
    events = list(run_simulation(puzzle, seed=5, t_offset_ms=500, segment=2, max_sim_ms=200))
    assert events[0]["segment"] == 2 and events[0]["t_offset_ms"] == 500
    assert events[1]["t_start_ms"] == 500


def test_wall_budget_hands_over():
    events = list(run_simulation(get_puzzle(6), seed=3, wall_budget_s=1.0))
    done = events[-1]
    assert done["type"] == "done"
    assert done["solved"] is False and done["resume"] is True
    assert done["t_end_ms"] == events[-2]["t_end_ms"]


def test_kicks_and_resets_run():
    # patience=1 forces interventions in almost every window
    events = list(run_simulation(get_puzzle(6), seed=11, max_sim_ms=4000, patience=1))
    kinds = {e["kick"]["kind"] for e in events if e["type"] == "step" and e["kick"]}
    assert kinds == {"kick", "reset"}
    assert events[-1]["type"] == "done"


def test_error_event():
    events = list(run_simulation([[1, 2]], max_sim_ms=100))
    assert events[-1]["type"] == "error"


def test_hints_added_and_carried_over():
    puzzle = get_puzzle(6)
    events = list(run_simulation(puzzle, seed=11, max_sim_ms=4000, patience=1, hint_after_ms=0))
    hinted = [c for e in events if e["type"] == "step" and e["kick"] and e["kick"]["kind"] == "hint" for c in e["kick"]["cells"]]
    assert hinted, "expected at least one hint restart"
    assert all(puzzle.reshape(81)[c] == 0 for c in hinted)
    done = events[-1]
    assert done["hints"] == hinted
    follow_up = list(run_simulation(puzzle, seed=12, max_sim_ms=100, hints=done["hints"]))
    meta = follow_up[0]
    assert meta["hints"] == done["hints"]
    assert meta["connections"]["stimulus_active"] == (meta["givens"] + len(hinted)) * 5
