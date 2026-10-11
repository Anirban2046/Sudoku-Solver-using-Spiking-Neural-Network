import json

import numpy as np
import pytest

from snn.encoding import decode_b64_u16
from snn.engine import run_simulation
from snn.helpers import generate_puzzle


@pytest.fixture(scope="module")
def run40():
    puzzle = generate_puzzle(40, np.random.default_rng(1))
    return puzzle, list(run_simulation(puzzle, max_sim_ms=20000, seed=42))


def test_event_order_and_solved(run40):
    _, events = run40
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert types.count("step") >= 1
    assert set(types[1:-1]) == {"step"}
    done = events[-1]
    assert done["solved"] is True
    assert done["analysis"]["classical"]["verdict"] == "match"


def test_meta_connections(run40):
    meta = run40[1][0]
    c = meta["connections"]
    assert c["total"] == 521235
    assert c["inhibitory"] == 510300
    assert c["noise"] == c["stimulus"] == c["recorder"] == 3645
    assert c["stimulus_active"] == meta["givens"] * 5


def test_step_payload(run40):
    puzzle, events = run40
    meta = events[0]
    for step in (e for e in events if e["type"] == "step"):
        assert len(json.dumps(step)) < 25_000
        assert len(step["grid"]) == 81
        counts = decode_b64_u16(step["pop_counts_b64"])
        assert len(counts) == 729 and counts.sum() == step["total_spikes"]
        pops = decode_b64_u16(step["raster"]["pop_b64"])
        ts = decode_b64_u16(step["raster"]["t_b64"])
        assert len(pops) == len(ts)
        assert pops.max() < meta["n_populations"]
        assert ts.max() <= meta["chunk_ms"] * 10


def test_analysis_numbers(run40):
    a = run40[1][-1]["analysis"]
    d = a["neuron"]["derived"]
    assert d["V_inf_mv"] == pytest.approx(-25.0)
    assert d["tonic_rate_hz"] == pytest.approx(72.7, abs=0.1)
    assert d["t_peak_ms"] == pytest.approx(9.24, abs=0.01)
    assert d["psp"]["inhibitory"]["peak_mv"] == pytest.approx(-2.52, abs=0.01)
    assert d["psp"]["stimulus"]["peak_mv"] == pytest.approx(16.4, abs=0.05)
    assert d["psp"]["noise"]["peak_mv"] == pytest.approx(20.2, abs=0.05)
    assert d["mean_current_noise"] == pytest.approx(2.8)
    assert d["mean_current_stimulus"] == pytest.approx(1.3)
    assert a["network"]["fan_out_inhibitory"] == 140
    assert a["network"]["fan_in"]["total"] == 142
    assert len(json.dumps(a)) < 400_000


def test_dream_terminates():
    events = list(run_simulation(np.zeros((9, 9), dtype=int), max_sim_ms=5000, seed=7))
    assert events[0]["type"] == "meta"
    done = events[-1]
    assert done["type"] == "done"
    assert isinstance(done["solved"], bool)
    assert done["sim_ms_total"] <= 5000


def test_error_event():
    events = list(run_simulation([[1, 2]], max_sim_ms=100))
    assert events[-1]["type"] == "error"
