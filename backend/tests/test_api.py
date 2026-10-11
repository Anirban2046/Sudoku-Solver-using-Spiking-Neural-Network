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


def test_duplicate_422(client):
    p = get_puzzle(4).copy()
    p[0, 0] = 1  # 1 already at (0, 1)
    r = client.post("/api/solve", json={"puzzle": p.tolist()})
    assert r.status_code == 422
    assert "appears more than once in row 1" in r.text


def test_busy_429(client):
    assert main._run_lock.acquire(blocking=False)
    try:
        r = client.post("/api/solve", json={"puzzle": get_puzzle(4).tolist()})
        assert r.status_code == 429
        assert r.json() == {"detail": "busy"}
    finally:
        main._run_lock.release()


def test_stream(client, monkeypatch):
    monkeypatch.setenv("MAX_SIM_MS", "300")
    puzzle = generate_puzzle(36, np.random.default_rng(3)).tolist()
    with client.stream("POST", "/api/solve", json={"puzzle": puzzle, "seed": 5}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        assert r.headers["x-accel-buffering"] == "no"
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[0]["type"] == "meta" and events[0]["seed"] == 5
    assert events[-1]["type"] == "done"
    assert 1 <= sum(e["type"] == "step" for e in events) <= 3
    assert not main._run_lock.locked()  # released after the stream
