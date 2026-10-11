"""Public entry point for running the Sudoku SNN.

``run_simulation`` is a generator of JSON-serialisable event dicts (``meta``,
``step`` ..., then ``done`` or ``error``); see ``simulation.py`` for the loop.

NEST keeps one global kernel per process, so every simulation runs in its own
short-lived worker process (``worker.py``). Several visitors can therefore
solve at the same time on one server instance, each with an independent
network. NEST is never imported in the web server process itself.
"""

import json
import os
import subprocess
import sys
import threading

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_END_EVENTS = ("done", "error", "health")

_workers = set()  # running worker processes
_workers_lock = threading.Lock()


def available_cpus():
    """CPUs this process may actually use. cpu_count() reports the host's CPUs
    in containers; the affinity mask and the cgroup CPU quota (e.g. 1 vCPU on
    a serverless function) are both honoured."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, -(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _jsonable(obj):
    if hasattr(obj, "tolist"):  # numpy arrays and scalars
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _stop(proc):
    """Stop a worker (if still running) and forget it. Workers hold no state
    worth a graceful shutdown, so they are killed outright."""
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    with _workers_lock:
        _workers.discard(proc)


def _events(job, handle=None):
    """Run one job in a worker process and yield its events. Closing the
    generator, or calling ``cancel(handle)``, terminates the worker."""
    env = dict(os.environ)
    # the worker must find the same packages as this process
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYNEST_QUIET"] = "1"
    # NEST's OpenMP threads spin while waiting by default, which starves the
    # other simulations on the instance; passive waiting costs a lone run little.
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    proc = subprocess.Popen(
        [sys.executable, "-m", "snn.worker"],
        cwd=_BACKEND_DIR,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    with _workers_lock:
        _workers.add(proc)
    if handle is not None:
        handle["proc"] = proc
    try:
        proc.stdin.write(json.dumps(job, default=_jsonable))
        proc.stdin.close()
        finished = False
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            finished = event.get("type") in _END_EVENTS
            yield event
            if finished:
                break
        if not finished:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            yield {"type": "error", "message": f"simulation worker exited unexpectedly (code {proc.returncode})"}
    finally:
        _stop(proc)


def run_simulation(puzzle, handle=None, **options):
    """Generator of simulation events (see ``simulation.run`` for the options).
    Never raises: failures become one ``{"type": "error", "message": ...}`` event.

    Pass a dict as ``handle`` to be able to stop the run from another thread
    with ``cancel(handle)``, e.g. when the client disconnects."""
    if "threads" not in options:
        # share the CPUs between the simulations running on this instance
        with _workers_lock:
            active = len(_workers) + 1
        options["threads"] = max(1, available_cpus() // active)
    try:
        yield from _events({"mode": "run", "puzzle": puzzle, "options": options}, handle)
    except Exception as exc:  # noqa: BLE001 - e.g. the worker could not be started
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


def cancel(handle):
    """Stop the worker of a run started with ``handle`` (no-op if not started or
    done). Blocks briefly while the process exits, so call it off the event loop."""
    proc = handle.get("proc") if handle else None
    if proc is not None:
        _stop(proc)


def health():
    """Run a tiny NEST simulation in a worker; returns its ``health`` event."""
    events = list(_events({"mode": "health"}))
    return events[-1] if events else {"type": "error", "message": "no response from worker"}


def stop_all():
    """Terminate every running worker (used on server shutdown)."""
    with _workers_lock:
        procs = list(_workers)
    for proc in procs:
        _stop(proc)
