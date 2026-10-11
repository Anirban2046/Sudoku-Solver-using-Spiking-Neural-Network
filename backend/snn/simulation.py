"""Builds the NEST Sudoku network and runs it window by window.

Only import this module on the NEST thread (see ``engine.py``).

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
* **hint** - from ``hint_after_ms`` of simulated time on, every reset also
  stimulates ``hint_batch`` more cells with their correct digit (from the
  classical solver), exactly as if they were givens, preferring the cells the
  network is struggling with. The network still has to solve every other cell.
  Hints only ever grow, so even the hardest puzzle is eventually solved; puzzles
  that the network solves on its own never receive any.

A request stops at the first valid grid or when its wall-clock budget is used
up. In the latter case ``done.resume`` is true and the client starts a new
request (a rebuilt network with a new seed), so there is no overall time limit
while every request stays within the host's function timeout.
"""

import os
import time

import nest
import numpy as np

from . import sudoku_net
from .analysis import cell_summary, classical_check, count_violations, conflict_cells
from .helpers import PEERS, solve_all, validate_solution
from .sudoku_net import SudokuNet

KICK_I_E = -5.0  # bias current (nA scale, as neuron_params) that silences a population
RESETS_AFTER_FUTILE_KICKS = 4


def _threads():
    # CPUs this process may actually use (cpu_count() reports the host's in containers).
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


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


def _kick_targets(puzzle, grid, wide=False):
    """(cell, digit) pairs to silence: the shown digit of every non-given cell in
    a conflict (with ``wide``, of every non-given cell sharing a row, column or
    box with a conflict), and the wrong digit on every overridden given cell."""
    flat_p = puzzle.reshape(81)
    flat_g = grid.reshape(81)
    cells = set(int(c) for c in conflict_cells(flat_g))
    if wide:
        cells |= {p for c in list(cells) for p in PEERS[c]}
    targets = set()
    for c in cells:
        if flat_p[c] == 0:
            targets.add((int(c), int(flat_g[c])))
    for c in np.flatnonzero((flat_p != 0) & (flat_p != flat_g)):
        targets.add((int(c), int(flat_g[c])))
    return sorted(targets)


def _choose_hints(puzzle, grid, solution, hinted, n, rng):
    """Up to ``n`` cells to hint: non-given, not yet hinted, preferring cells in
    a conflict, then cells showing a wrong digit, then any other cell."""
    flat_p = puzzle.reshape(81)
    flat_g = grid.reshape(81)
    free = [c for c in range(81) if flat_p[c] == 0 and c not in hinted]
    conflict = set(int(c) for c in conflict_cells(flat_g))
    tiers = (
        [c for c in free if c in conflict],
        [c for c in free if c not in conflict and flat_g[c] != solution[c]],
        [c for c in free if c not in conflict and flat_g[c] == solution[c]],
    )
    chosen = []
    for tier in tiers:
        chosen += [int(c) for c in rng.permutation(tier)][: n - len(chosen)]
    return chosen


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
    hints=(),
    hint_after_ms=15000,
    hint_batch=4,
):
    t_run = time.time()
    puzzle = np.asarray(puzzle, dtype=np.int64).reshape(9, 9)
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    threads = _threads()

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.ERROR
    nest.local_num_threads = threads
    nest.rng_seed = int(seed)
    rng = np.random.default_rng(seed)  # tie-breaking in the readout

    t_build = time.time()
    net = SudokuNet(pop_size=pop_size, input=puzzle, noise_rate=noise_rate)
    build_seconds = time.time() - t_build

    hinted = []
    cache = {}

    def solution():
        """Classical solution, computed when the first hint is needed."""
        if "s" not in cache:
            cache["s"] = solve_all(puzzle, 1)[0].reshape(81)
        return cache["s"]

    def add_hints(cells):
        for c in cells:
            pop = net.io_indices[c // 9, c % 9, solution()[c] - 1]
            nest.GetConnections(net.stim[pop]).set({"weight": sudoku_net.weight_stim})
            hinted.append(int(c))

    add_hints([c for c in hints if puzzle.reshape(81)[c] == 0])
    connections = _connections(net)

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
        "connections": connections,
        "givens": int((puzzle != 0).sum()),
        "build_seconds": round(build_seconds, 3),
        "wall_budget_s": float(wall_budget_s),
        "kick_ms": kick_ms,
        "patience": patience,
        "hints": list(hinted),
        "hint_after_ms": hint_after_ms,
    }

    win_s = chunk_ms / 1000.0
    prev_grid = puzzle.copy()
    grid = puzzle.copy()
    pop_counts = np.zeros(729, dtype=np.int64)
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
                    if hint_after_ms is not None and t_end >= hint_after_ms:
                        new = _choose_hints(puzzle, grid, solution(), set(hinted), hint_batch, rng)
                        if new:
                            add_hints(new)
                            kick = {"kind": "hint", "cells": new}
                    futile_kicks = 0
                else:
                    # escalate: a kick that didn't help is followed by a wider one
                    pending_kick = _kick_targets(puzzle, grid, wide=futile_kicks > 0)
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
        "classical": classical_check(puzzle, grid, bool(valid)),
        "hints": list(hinted),
    }
