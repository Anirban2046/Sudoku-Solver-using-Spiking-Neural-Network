"""Builds the NEST Sudoku network and runs it window by window.

Runs inside a worker process (``worker.py``), one per simulation (see ``engine.py``).

``run`` is a generator of JSON-serialisable events (``meta``, ``step`` ...,
``done``). The network and the readout are those of ``sudoku_solver.py`` from
the NEST examples: simulate in ``chunk_ms`` windows and read each cell out as
the digit whose population spiked most (ties broken at random).

The upstream loop simply gives up after a fixed time, and the network can get
stuck: a few cells keep a conflict for seconds, or the network settles on a
valid grid that overrides a given digit (valid grids are stable attractors).
So when progress stalls, a controller perturbs the network:

* **kick** - when the number of conflicts hasn't improved for ``patience``
  windows, the winning populations of the conflicting cells (and wrong digits
  sitting on given cells) are silenced for ``kick_ms`` with a strong negative
  bias current, so a rival digit can take over: the spiking analogue of a
  local-search "flip". If a kick doesn't help, the next one silences every
  non-given cell sharing a row, column or box with a conflict, so that whole
  region re-solves against the rest of the grid.
* **reset** - if several kicks in a row bring no improvement, all membrane
  potentials are re-randomised (``SudokuNet.reset_V_m``), which restarts the
  search from a fresh state while keeping the givens' stimulation.

A request stops at the first valid grid or when its wall-clock budget is used
up. In the latter case ``done.resume`` is true and the client starts a new
request (a rebuilt network with a new seed), so there is no overall time limit
while every request stays within the host's function timeout.
"""

import time
from collections import deque

import nest
import numpy as np

from . import sudoku_net
from .analysis import cell_summary, count_violations, network_view
from .control import kick_targets
from .engine import available_cpus
from .helpers import validate_solution
from .sudoku_net import SudokuNet

KICK_I_E = -5.0  # bias current (nA scale, as neuron_params) that silences a population
RESETS_AFTER_FUTILE_KICKS = 4
VM_INTERVAL_MS = 1.0  # membrane-potential sampling of one neuron per population
RATE_WINDOW_MS = 1000  # span of the final firing-rate statistics


def _connections(net):
    # Filtering by target, or reading all 510k weights, takes 5-50 s in NEST 3.10,
    # so neuron->neuron connections are counted as (all from neurons) - (neuron->recorder),
    # and the inhibitory sign is checked on the first population's outgoing synapses.
    stim_weights = np.asarray(nest.GetConnections(source=net.stim).get("weight"), dtype=float)
    n_noise = len(nest.GetConnections(source=net.noise))
    n_recorder = len(nest.GetConnections(target=net.spikerecorders))
    n_neuron_to_neuron = len(nest.GetConnections(source=net.neurons)) - n_recorder
    sample = nest.GetConnections(source=net.neurons[: net.pop_size])
    sample_weights = np.asarray(sample.get("weight"), dtype=float)
    to_neurons = np.asarray(sample.get("target")) <= net.neurons[-1].global_id
    if not (sample_weights[to_neurons] < 0).all():
        raise RuntimeError("expected all neuron-to-neuron synapses to be inhibitory")
    return {
        "inhibitory": int(n_neuron_to_neuron),
        "noise": int(n_noise),
        "stimulus": int(len(stim_weights)),
        "stimulus_active": int((stim_weights > 0).sum()),
        "recorder": int(n_recorder),
        "total": int(nest.num_connections),
    }


def _vm_matrix(multimeter, first_gid, pop_size, t_start, chunk_ms):
    """Last window's V_m samples as a (729, samples) array, one row per population."""
    n = int(round(chunk_ms / VM_INTERVAL_MS))
    vm = np.full((729, n), np.nan)
    ev = multimeter.get("events")
    pops = (np.asarray(ev["senders"]) - first_gid) // pop_size
    cols = np.rint((np.asarray(ev["times"]) - t_start) / VM_INTERVAL_MS).astype(int) - 1
    ok = (cols >= 0) & (cols < n)
    vm[pops[ok], cols[ok]] = np.asarray(ev["V_m"])[ok]
    # fill any gap (none expected) with the previous sample
    for row in vm:
        for i in range(1, n):
            if np.isnan(row[i]):
                row[i] = row[i - 1]
    return np.nan_to_num(vm, nan=-65.0)


def _readout(counts, rng):
    """Argmax readout per cell with random tie-breaking, as in sudoku_solver.py.
    ``counts`` has shape (9, 9, 9) = (row, col, digit)."""
    grid = np.zeros((9, 9), dtype=np.int64)
    for row in range(9):
        for col in range(9):
            spike_counts = counts[row, col]
            winners = np.flatnonzero(spike_counts == spike_counts.max())
            grid[row, col] = int(rng.choice(winners)) + 1
    return grid


def run(
    puzzle,
    noise_rate=350.0,
    chunk_ms=100,
    seed=None,
    pop_size=5,
    wall_budget_s=240.0,
    t_offset_ms=0,
    segment=0,
    max_sim_ms=None,
    patience=5,
    kick_ms=50,
    threads=None,
):
    t_run = time.time()
    puzzle = np.asarray(puzzle, dtype=np.int64).reshape(9, 9)
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    threads = int(threads or available_cpus())

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.ERROR
    nest.local_num_threads = threads
    nest.rng_seed = int(seed)
    rng = np.random.default_rng(seed)  # tie-breaking in the readout

    t_build = time.time()
    net = SudokuNet(pop_size=pop_size, input=puzzle, noise_rate=noise_rate)
    build_seconds = time.time() - t_build

    connections = _connections(net)

    # Record V_m of the first neuron of every population (created after counting
    # connections, so the counts describe the network itself).
    multimeter = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": VM_INTERVAL_MS})
    nest.Connect(multimeter, net.neurons[::pop_size])
    first_gid = net.neurons[0].global_id

    yield {
        "type": "meta",
        "seed": int(seed),
        "segment": int(segment),
        "t_offset_ms": int(t_offset_ms),
        "nest_version": nest.__version__,
        "threads": threads,
        "chunk_ms": chunk_ms,
        "noise_rate": float(noise_rate),
        "stim_rate": float(net.stim_rate),
        "pop_size": int(net.pop_size),
        "n_populations": int(net.n_populations),
        "n_neurons": int(net.n_total),
        "neuron_model": "iaf_psc_exp",
        "neuron": {k: sudoku_net.neuron_params[k] for k in ("V_th", "V_reset", "E_L", "tau_m", "t_ref")},
        "connections": connections,
        "givens": int((puzzle != 0).sum()),
        "build_seconds": round(build_seconds, 3),
        "wall_budget_s": float(wall_budget_s),
        "kick_ms": kick_ms,
        "patience": patience,
    }

    win_s = chunk_ms / 1000.0
    prev_grid = puzzle.copy()
    grid = puzzle.copy()
    pop_counts = np.zeros(729, dtype=np.int64)
    neuron_idx = np.zeros(0, dtype=np.int64)
    spike_times = np.zeros(0)
    recent_neuron_counts = deque(maxlen=max(1, RATE_WINDOW_MS // chunk_ms))
    valid = False
    k = 0
    pending_kick = []  # (cell, digit) populations to silence at the start of the next window
    best_bad = None  # fewest (violations + overridden givens) since the last kick
    stuck = 0
    best_overall = None
    futile_kicks = 0

    while True:
        t_step = time.time()
        t_start = t_offset_ms + k * chunk_ms
        t_end = t_start + chunk_ms

        net.reset_spike_recorders()
        multimeter.n_events = 0
        if pending_kick:
            ids = sorted(int(net.neuron_indices[c // 9, c % 9, d - 1, n]) for c, d in pending_kick for n in range(pop_size))
            silenced = net.neurons[ids]
            silenced.I_e = KICK_I_E
            nest.Simulate(kick_ms)
            silenced.I_e = sudoku_net.neuron_params["I_e"]
            nest.Simulate(chunk_ms - kick_ms)
            pending_kick = []
        else:
            nest.Simulate(chunk_ms)

        trains = net.get_spike_trains()
        pop_counts = np.array([len(tr["times"]) for tr in trains], dtype=np.int64)
        neuron_idx = np.concatenate([np.asarray(tr["senders"], dtype=np.int64) for tr in trains]) - first_gid
        spike_times = np.concatenate([np.asarray(tr["times"], dtype=float) for tr in trains]) - t_start
        recent_neuron_counts.append(np.bincount(neuron_idx, minlength=net.n_total))

        grid = _readout(pop_counts.reshape(9, 9, 9), rng)
        valid, boxes, rows, cols = validate_solution(puzzle, grid)
        performance = 1.0 if valid else float((boxes.sum() + rows.sum() + cols.sum()) / 27)
        violations = count_violations(grid)
        overridden = int(((puzzle != 0) & (puzzle != grid)).sum())
        winners = np.zeros(729, dtype=bool)
        winners[np.arange(81) * 9 + grid.reshape(81) - 1] = True
        n_win = 81 * pop_size

        # Stall detection and perturbation.
        kick = None
        bad = violations + overridden
        if not valid:
            best_overall = bad if best_overall is None else min(best_overall, bad)
            if best_bad is None or bad < best_bad:
                best_bad, stuck = bad, 0
            else:
                stuck += 1
            if stuck >= patience:
                if futile_kicks >= RESETS_AFTER_FUTILE_KICKS:
                    net.reset_V_m()
                    kick = {"kind": "reset", "cells": []}
                    futile_kicks = 0
                else:
                    # escalate: a kick that didn't help is followed by a wider one
                    pending_kick = kick_targets(puzzle, grid, wide=futile_kicks > 0)
                    kick = {"kind": "kick", "cells": sorted({c for c, _ in pending_kick})}
                    futile_kicks = futile_kicks + 1 if bad >= best_overall else 0
                best_bad, stuck = None, 0

        yield {
            "type": "step",
            "k": k,
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "grid": grid.reshape(81).tolist(),
            "valid_rows": rows.tolist(),
            "valid_cols": cols.tolist(),
            "valid_boxes": boxes.reshape(9).tolist(),
            "givens_overridden": overridden,
            "performance": round(performance, 4),
            "violations": violations,
            "cells_changed": int((grid != prev_grid).sum()),
            "total_spikes": int(pop_counts.sum()),
            "rate_all_hz": round(float(pop_counts.sum()) / (net.n_total * win_s), 2),
            "rate_winners_hz": round(float(pop_counts[winners].sum()) / (n_win * win_s), 2),
            "rate_losers_hz": round(float(pop_counts[~winners].sum()) / ((net.n_total - n_win) * win_s), 2),
            "kick": kick,
            "wall_seconds": round(time.time() - t_step, 3),
        }

        prev_grid = grid
        k += 1
        if valid:
            break
        if max_sim_ms is not None and k * chunk_ms >= max_sim_ms:
            break
        # Stop early enough that the next window still fits in the budget.
        if time.time() - t_run + 2 * (time.time() - t_step) > wall_budget_s:
            break

    yield {
        "type": "done",
        "solved": bool(valid),
        "resume": not valid and max_sim_ms is None,
        "solved_at_ms": t_start if valid else None,
        "t_end_ms": t_end,
        "final_grid": grid.reshape(81).tolist(),
        "wall_seconds": round(time.time() - t_run, 3),
        "cells": cell_summary(pop_counts),
        "network": network_view(
            grid=grid,
            pop_counts=pop_counts,
            pop_size=pop_size,
            chunk_ms=chunk_ms,
            neuron_idx=neuron_idx,
            spike_t=spike_times,
            vm=_vm_matrix(multimeter, first_gid, pop_size, t_start, chunk_ms),
            recent_neuron_counts=list(recent_neuron_counts),
        ),
    }
