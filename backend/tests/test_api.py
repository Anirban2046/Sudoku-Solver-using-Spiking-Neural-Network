import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from snn.helpers import generate_puzzle, get_puzzle


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["nest"] == "3.10.0" and body["spikes"] > 0


@pytest.mark.parametrize(
    "puzzle",
    [
        [[0] * 9] * 8,  # wrong shape
        [[10] + [0] * 8] + [[0] * 9] * 8,  # out of range
        get_puzzle(0).tolist(),  # too few givens
    ],
)
def test_invalid_board_422(client, puzzle):
    assert client.post("/api/solve", json={"puzzle": puzzle}).status_code == 422


def test_unsolvable_422(client):
    p = np.zeros((9, 9), dtype=int)
    p[0, 1:5] = [1, 2, 3, 4]
    p[1:3, 0] = [5, 6]
    p[1, 1], p[2, 2], p[5, 0] = 7, 8, 9
    p[8, 8], p[7, 7], p[6, 6], p[8, 4], p[7, 3], p[6, 5], p[4, 8], p[3, 7], p[4, 4] = 1, 2, 3, 5, 6, 7, 8, 9, 1
    r = client.post("/api/solve", json={"puzzle": p.tolist()})
    assert r.status_code == 422
    assert "no solution" in r.text


def test_bad_hints_422(client):
    r = client.post("/api/solve", json={"puzzle": get_puzzle(4).tolist(), "hints": [81]})
    assert r.status_code == 422


def test_duplicate_422(client):
    p = get_puzzle(4).copy()
    p[0, 0] = 1  # 1 already at (0, 1)
    r = client.post("/api/solve", json={"puzzle": p.tolist()})
    assert r.status_code == 422
    assert "appears more than once in row 1" in r.text


def test_unverifiable_422(client, monkeypatch):
    import snn.helpers

    monkeypatch.setattr(snn.helpers, "has_solution", lambda puzzle: None)
    r = client.post("/api/solve", json={"puzzle": get_puzzle(4).tolist()})
    assert r.status_code == 422
    assert "couldn't confirm" in r.text


def test_health_busy_answers_at_once(client):
    import time

    assert main._run_lock.acquire(blocking=False)
    try:
        t = time.time()
        body = client.get("/api/health").json()
        assert body["busy"] is True and body["nest"] == "3.10.0"
        assert time.time() - t < 2
    finally:
        main._run_lock.release()


def test_busy_429(client):
    assert main._run_lock.acquire(blocking=False)
    try:
        r = client.post("/api/solve", json={"puzzle": get_puzzle(4).tolist()})
        assert r.status_code == 429
        assert r.json() == {"detail": "busy"}
    finally:
        main._run_lock.release()


def test_stream(client, monkeypatch):
    monkeypatch.setenv("WALL_BUDGET_S", "10")
    puzzle = generate_puzzle(36, np.random.default_rng(3)).tolist()
    body = {"puzzle": puzzle, "seed": 5, "t_offset_ms": 1000, "segment": 1}
    with client.stream("POST", "/api/solve", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        assert r.headers["x-accel-buffering"] == "no"
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[0]["type"] == "meta" and events[0]["seed"] == 5
    assert events[0]["segment"] == 1 and events[1]["t_start_ms"] == 1000
    assert events[-1]["type"] == "done"
    assert events[-1]["solved"] or events[-1]["resume"]
    assert not main._run_lock.locked()  # released after the stream
