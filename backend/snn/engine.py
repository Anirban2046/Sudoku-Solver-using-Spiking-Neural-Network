"""Public entry point for running the Sudoku SNN.

``run_simulation`` is a generator of JSON-serialisable event dicts (``meta``,
``step`` ..., then ``done`` or ``error``); see ``simulation.py`` for the loop.

NEST (with OpenMP) segfaults when it is imported on one Python thread and
driven from another, e.g. ResetKernel on a web worker thread after the module
was imported elsewhere. So ``nest`` is never imported at module level: every
NEST import and call happens on one long-lived thread owned by this module.
"""

import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

_nest_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nest")
_END = object()


def on_nest_thread(fn, *args, **kwargs):
    """Run ``fn`` on the dedicated NEST thread and return its result."""
    return _nest_thread.submit(fn, *args, **kwargs).result()


def _safe_run(*args):
    try:
        from .simulation import run

        yield from run(*args)
    except Exception as exc:  # noqa: BLE001 - reported to the client
        traceback.print_exc()
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


def run_simulation(
    puzzle, noise_rate=350.0, max_sim_ms=20000, chunk_ms=100, seed=None, raster_per_pop=1, pop_size=5
):
    """Generator of simulation events. Never raises: failures become one
    ``{"type": "error", "message": ...}`` event.

    The simulation itself runs on the dedicated NEST thread; events are handed
    over through a queue, so the caller may iterate from any thread. Closing the
    generator early stops the simulation after the current window.
    """
    events = queue.Queue()
    stop = threading.Event()

    def worker():
        try:
            for event in _safe_run(puzzle, noise_rate, max_sim_ms, chunk_ms, seed, raster_per_pop, pop_size):
                if stop.is_set():
                    break
                events.put(event)
        finally:
            events.put(_END)

    _nest_thread.submit(worker)
    try:
        while (event := events.get()) is not _END:
            yield event
    finally:
        stop.set()
