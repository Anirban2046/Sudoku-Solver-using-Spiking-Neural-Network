"""Worker process: runs one job with its own NEST kernel.

Started by ``engine.py`` as ``python -m snn.worker``. Reads one JSON job from
stdin and writes JSON events to stdout, one per line:

* ``{"mode": "run", "puzzle": [[...]], "options": {...}}`` streams the events
  of ``simulation.run`` (``meta``, ``step`` ..., ``done``);
* ``{"mode": "health"}`` runs a 10-neuron smoke test and writes one
  ``health`` event.

A failure becomes one ``{"type": "error", "message": ...}`` event.
"""

import json
import os
import sys
import traceback


def _health():
    import nest

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.ERROR
    neurons = nest.Create("iaf_psc_exp", 10, {"I_e": 500.0})
    recorder = nest.Create("spike_recorder")
    nest.Connect(neurons, recorder)
    nest.Simulate(100.0)
    return {"type": "health", "nest": nest.__version__, "spikes": int(recorder.n_events)}


def main():
    # Keep a private copy of stdout for the event stream, then send everything
    # else written to stdout (e.g. NEST's start-up banner) to stderr.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def emit(event):
        out.write(json.dumps(event, separators=(",", ":")) + "\n")

    try:
        job = json.load(sys.stdin)
        if job.get("mode") == "health":
            emit(_health())
        else:
            from .simulation import run

            for event in run(job["puzzle"], **job.get("options", {})):
                emit(event)
    except Exception as exc:  # noqa: BLE001 - reported to the client
        traceback.print_exc()
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        out.flush()


if __name__ == "__main__":
    main()
