"""FastAPI app: ``GET /api/health`` and the streaming ``POST /api/solve``."""

import json
import os
import threading
import weakref

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Sudoku Solver using Spiking Neural Network")

_cors = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors:
    app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["GET", "POST"], allow_headers=["*"])

# The NEST kernel is global to the process, so only one simulation runs at a time.
_run_lock = threading.Lock()


def _env_float(name, default, lo, hi):
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        value = default
    return min(max(value, lo), hi)


def settings():
    return {
        "noise_rate": _env_float("NOISE_RATE", 350.0, 0.0, 5000.0),
        # Wall-clock seconds one request may simulate before handing over to a
        # follow-up request; keep it well below the function timeout (300 s).
        "wall_budget_s": _env_float("WALL_BUDGET_S", 240.0, 10.0, 280.0),
        # Simulated time after which restarts start adding hints (negative: never).
        "hint_after_ms": _hint_after(),
    }


def _hint_after():
    value = _env_float("HINT_AFTER_MS", 15000.0, -1.0, 1e9)
    return None if value < 0 else int(value)


def _duplicates(puzzle):
    """Human-readable messages for repeated digits in rows, columns and boxes."""
    msgs = []
    units = []
    for i in range(9):
        units.append((f"row {i + 1}", [(i, c) for c in range(9)]))
        units.append((f"column {i + 1}", [(r, i) for r in range(9)]))
        br, bc = 3 * (i // 3), 3 * (i % 3)
        units.append((f"box {i + 1}", [(br + a, bc + b) for a in range(3) for b in range(3)]))
    for name, cells in units:
        seen = set()
        for r, c in cells:
            d = puzzle[r][c]
            if d and d in seen:
                msgs.append(f"{d} appears more than once in {name}")
            seen.add(d)
    return msgs


class SolveRequest(BaseModel):
    puzzle: list[list[int]]
    seed: int | None = Field(default=None, ge=1, le=2**31 - 1)
    # Continuation of an earlier request that ran out of its wall-clock budget.
    t_offset_ms: int = Field(default=0, ge=0, le=10**9)
    segment: int = Field(default=0, ge=0, le=10**6)
    # Cells (0..80) hinted by earlier requests of the same run.
    hints: list[int] = Field(default_factory=list, max_length=81)

    @field_validator("hints")
    @classmethod
    def check_hints(cls, hints):
        if any(not 0 <= c <= 80 for c in hints):
            raise ValueError("hints must be cell indices from 0 to 80")
        return sorted(set(hints))

    @field_validator("puzzle")
    @classmethod
    def check_puzzle(cls, puzzle):
        if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
            raise ValueError("puzzle must be 9 rows of 9 numbers")
        if any(not 0 <= v <= 9 for row in puzzle for v in row):
            raise ValueError("every cell must be a number from 0 (empty) to 9")
        dups = _duplicates(puzzle)
        if dups:
            raise ValueError("; ".join(dups))
        givens = sum(1 for row in puzzle for v in row if v)
        if givens < 17:
            raise ValueError(f"at least 17 given digits are needed (got {givens})")
        from snn.helpers import has_solution

        if has_solution(puzzle) is False:
            raise ValueError("this puzzle has no solution")
        return puzzle


@app.get("/api/health")
def health():
    import time

    from snn.engine import on_nest_thread

    def tiny_sim():
        import nest

        nest.ResetKernel()
        nest.verbosity = nest.VerbosityLevel.ERROR
        n = nest.Create("iaf_psc_exp", 10, {"I_e": 500.0})
        sr = nest.Create("spike_recorder")
        nest.Connect(n, sr)
        nest.Simulate(100.0)
        return nest.__version__, sr.n_events

    def version():
        import nest

        return nest.__version__

    t = time.time()
    # Don't reset the kernel under a running solve; just report that it's busy.
    if not _run_lock.acquire(blocking=False):
        return {"nest": on_nest_thread(version), "cpus": os.cpu_count(), "busy": True}
    try:
        nest_version, spikes = on_nest_thread(tiny_sim)
    finally:
        _run_lock.release()
    return {"nest": nest_version, "cpus": os.cpu_count(), "spikes": spikes, "seconds": round(time.time() - t, 3)}


def _release_once():
    done = threading.Lock()

    def release():
        if done.acquire(blocking=False):
            _run_lock.release()

    return release


def _ndjson(req, cfg, release):
    try:
        from snn.engine import run_simulation

        options = dict(cfg, seed=req.seed, t_offset_ms=req.t_offset_ms, segment=req.segment, hints=req.hints)
        for event in run_simulation(req.puzzle, **options):
            yield json.dumps(event, separators=(",", ":")) + "\n"
    finally:
        release()


@app.post("/api/solve")
def solve(req: SolveRequest):
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="busy")
    release = _release_once()
    try:
        gen = _ndjson(req, settings(), release)
        # If the client goes away before the stream starts, the generator body
        # never runs; the finalizer still frees the lock.
        weakref.finalize(gen, release)
    except BaseException:
        release()
        raise
    return StreamingResponse(
        gen,
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
