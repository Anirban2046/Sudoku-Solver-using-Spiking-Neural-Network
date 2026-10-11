import json

import numpy as np
import pytest

from snn.encoding import decode_b64
from snn.engine import run_simulation
from snn.helpers import generate_puzzle, get_puzzle, solve_all, validate_solution


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
    final = np.array(done["final_grid"]).reshape(9, 9)
    assert validate_solution(run40[0], final)[0]
    assert np.array_equal(final, solve_all(run40[0], 1)[0])  # unique puzzle: the one solution
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


def test_network_view(run40):
    _, events = run40
    meta, last_step, done = events[0], events[-2], events[-1]
    net = done["network"]
    neurons = decode_b64(net["spikes"]["neuron_b64"], "<u2")
    times = decode_b64(net["spikes"]["t_b64"], "<u2")
    assert len(neurons) == len(times) == last_step["total_spikes"]
    assert neurons.max() < meta["n_neurons"] and times.max() <= meta["chunk_ms"] * 10
    n = net["vm"]["samples"]
    winner = decode_b64(net["vm"]["winner_b64"], "<i2") / 100
    rival = decode_b64(net["vm"]["rival_b64"], "<i2") / 100
    assert len(winner) == len(rival) == 81 * n
    assert winner.max() <= meta["neuron"]["V_th"] + 0.5  # V_m never sits above threshold
    assert all(r != d for r, d in zip(net["vm"]["rival_digit"], done["final_grid"]))
    rates = net["rates"]
    assert rates["winners_hz"] > 5 * rates["others_hz"]
    assert sum(rates["hist_winners"]) == 81 * meta["pop_size"]
    assert sum(rates["hist_winners"]) + sum(rates["hist_others"]) == meta["n_neurons"]
    assert len(json.dumps(done)) < 400_000
    assert meta["connections"]["total"] == 521235  # the multimeter is not counted


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


def test_kicks_leave_given_cells_alone():
    from snn.control import kick_targets

    sol = solve_all(get_puzzle(4), 1)[0]
    grid = sol.copy()
    grid[0, 0], grid[0, 3] = grid[0, 3], grid[0, 0]  # swap: breaks columns 1 and 4 and two boxes
    givens = np.zeros((9, 9), dtype=int)
    givens[0, 0] = sol[0, 0]  # a given cell now showing a wrong digit
    givens[1, 0] = sol[1, 0]  # a given cell showing its own digit, inside the conflict's column
    targets = dict(kick_targets(givens, grid, wide=True))
    assert targets[0] == grid[0, 0]  # the wrong digit on the given cell is silenced
    assert 9 not in targets  # a correct given cell is never silenced
    assert 3 in targets  # the free cell in the conflict is


def test_closing_the_stream_stops_the_worker():
    from snn import engine

    events = run_simulation(get_puzzle(6), seed=3, max_sim_ms=60_000)
    assert next(events)["type"] == "meta"
    with engine._workers_lock:
        procs = list(engine._workers)
    assert len(procs) == 1 and procs[0].poll() is None
    events.close()  # e.g. the client disconnected
    assert procs[0].poll() is not None
    assert not engine._workers


def test_threads_shared_between_runs():
    from snn import engine

    first = run_simulation(get_puzzle(6), seed=1, max_sim_ms=60_000)
    second = run_simulation(get_puzzle(6), seed=2, max_sim_ms=60_000)
    try:
        t1 = next(first)["threads"]
        t2 = next(second)["threads"]
        cpus = engine.available_cpus()
        assert t1 == cpus and t2 == max(1, cpus // 2)
    finally:
        first.close()
        second.close()

