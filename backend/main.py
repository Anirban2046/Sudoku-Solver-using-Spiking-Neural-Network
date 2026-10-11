"""FastAPI app: ``GET /api/health`` and the streaming ``POST /api/solve``."""

import json
import os
import threading
import weakref
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel, Field, field_validator

@asynccontextmanager
async def lifespan(_app):
    yield
    # On shutdown, stop running simulations instead of waiting for their budgets.
    from snn.engine import stop_all

    stop_all()


app = FastAPI(title="Sudoku Solver using Spiking Neural Network", lifespan=lifespan)

_cors = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors:
    app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["GET", "POST"], allow_headers=["*"])



def _env_float(name, default, lo, hi):
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        value = default
    return min(max(value, lo), hi)


# Each simulation runs in its own worker process with its own NEST kernel
# (about 300 MB); this caps how many run at once on one server instance.
MAX_CONCURRENT_RUNS = int(_env_float("MAX_CONCURRENT_RUNS", 3, 1, 16))
_run_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)


def settings():
    return {
        "noise_rate": _env_float("NOISE_RATE", 350.0, 0.0, 5000.0),
        # Wall-clock seconds one request may simulate before handing over to a
        # follow-up request; keep it well below the function timeout (300 s).
        "wall_budget_s": _env_float("WALL_BUDGET_S", 240.0, 10.0, 280.0),
    }


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

        solvable = has_solution(puzzle)
        if solvable is False:
            raise ValueError("this puzzle has no solution")
        if solvable is None:
            # Don't start a run that might never end.
            raise ValueError("couldn't confirm that this puzzle has a solution")
        return puzzle


@app.get("/api/health")
def health():
    import time
    from importlib.metadata import version

    from snn.engine import available_cpus
    from snn.engine import health as run_health_check

    t = time.time()
    base = {"cpus": available_cpus(), "max_concurrent_runs": MAX_CONCURRENT_RUNS}
    # When every slot is taken, report that without starting another worker.
    if not _run_slots.acquire(blocking=False):
        return {"nest": version("nest-simulator"), **base, "busy": True}
    try:
        result = run_health_check()
    finally:
        _run_slots.release()
    if result.get("type") != "health":
        raise HTTPException(status_code=500, detail=result.get("message", "health check failed"))
    return {"nest": result["nest"], **base, "spikes": result["spikes"], "seconds": round(time.time() - t, 3)}


def _release_once():
    done = threading.Lock()

    def release():
        if done.acquire(blocking=False):
            _run_slots.release()

    return release


def _ndjson(req, cfg, handle):
    from snn.engine import run_simulation

    options = dict(cfg, seed=req.seed, t_offset_ms=req.t_offset_ms, segment=req.segment)
    for event in run_simulation(req.puzzle, handle=handle, **options):
        yield json.dumps(event, separators=(",", ":")) + "\n"


async def _stream(req, cfg, release):
    """Relay the run's events. When the client disconnects, the server cancels
    this coroutine; the ``finally`` then stops the run's worker at once (closing
    the underlying generator is not guaranteed to happen promptly)."""
    from snn.engine import cancel

    handle = {}
    try:
        async for line in iterate_in_threadpool(_ndjson(req, cfg, handle)):
            yield line
    finally:
        # Stop the worker on a thread (never block the event loop), even though
        # this coroutine is being cancelled.
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(cancel, handle)
        release()


@app.post("/api/solve")
def solve(req: SolveRequest):
    if not _run_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="busy")
    release = _release_once()
    try:
        stream = _stream(req, settings(), release)
        # If the client goes away before the stream starts, its body never
        # runs (and no worker is started); the finalizer still frees the slot.
        weakref.finalize(stream, release)
    except BaseException:
        release()
        raise
    return StreamingResponse(
        stream,
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
