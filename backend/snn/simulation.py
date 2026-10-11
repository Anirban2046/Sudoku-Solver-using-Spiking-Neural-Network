"""Builds the NEST Sudoku network and runs it window by window.

Only import this module on the NEST thread (see ``engine.py``).

``run`` is a plain generator of JSON-serialisable event dicts
(``meta``, ``step`` ..., ``done``). It follows the simulation loop
of ``sudoku_solver.py`` from the NEST examples: simulate in ``chunk_ms``
windows, read each cell out as the digit whose population spiked most (ties
broken at random), and stop at the first valid grid or at ``max_sim_ms``.
"""

import os
import time

import nest
import numpy as np

from . import sudoku_net
from .analysis import AnalysisAccumulator, count_violations
from .encoding import b64_u16
from .helpers import givens_kept, validate_solution
from .sudoku_net import SudokuNet

# Initial membrane potentials are drawn from this range by sudoku_net.neuron_params
# (``nest.random.uniform(min=-65, max=-55)``); NEST parameters can't be read back.
V_M_INIT = [-65.0, -55.0]


def _meta(net, puzzle, seed, threads, chunk_ms, max_sim_ms, noise_rate, raster_per_pop, build_seconds):
    # Filtering by target, or reading all 510k weights, takes 5-50 s in NEST 3.10,
    # so neuron->neuron connections are counted as (all from neurons) - (neuron->recorder),
    # and the inhibitory sign is checked on the first population's outgoing synapses.
    stim_conns = nest.GetConnections(source=net.stim)
    stim_weights = np.asarray(stim_conns.get("weight"), dtype=float)
    n_noise = len(nest.GetConnections(source=net.noise))
    n_recorder = len(nest.GetConnections(target=net.spikerecorders))
    n_neuron_to_neuron = len(nest.GetConnections(source=net.neurons)) - n_recorder
    sample = nest.GetConnections(source=net.neurons[: net.pop_size])
    sample_weights = np.asarray(sample.get("weight"), dtype=float)
    sample_targets = np.asarray(sample.get("target"))
    to_neurons = sample_targets <= net.neurons[-1].global_id
    if not (sample_weights[to_neurons] < 0).all():
        raise RuntimeError("expected all neuron-to-neuron synapses to be inhibitory")

    connections = {
        "inhibitory": int(n_neuron_to_neuron),
        "noise": int(n_noise),
        "stimulus": int(len(stim_weights)),
        "stimulus_active": int((stim_weights > 0).sum()),
        "recorder": int(n_recorder),
        "total": int(nest.num_connections),
    }
    params = {k: float(v) for k, v in sudoku_net.neuron_params.items() if k != "V_m"}
    params["V_m_init"] = V_M_INIT
    return {
        "type": "meta",
        "seed": int(seed),
        "nest_version": nest.__version__,
        "threads": int(threads),
        "chunk_ms": chunk_ms,
        "max_sim_ms": max_sim_ms,
        "noise_rate": float(noise_rate),
        "stim_rate": float(net.stim_rate),
        "pop_size": int(net.pop_size),
        "n_populations": int(net.n_populations),
        "n_neurons": int(net.n_total),
        "raster_neurons_per_pop": int(raster_per_pop),
        "neuron_model": "iaf_psc_exp",
        "neuron_params": params,
        "weights": {
            "inhibitory": sudoku_net.inter_neuron_weight,
            "stimulus": sudoku_net.weight_stim,
            "noise": sudoku_net.weight_noise,
            "delay_ms": sudoku_net.delay,
        },
        "connections": connections,
        "givens": int((puzzle != 0).sum()),
        "build_seconds": round(build_seconds, 3),
    }


def _readout(counts, rng):
    """Argmax readout per cell with random tie-breaking, as in sudoku_solver.py.

    ``counts`` has shape (9, 9, 9) = (row, col, digit). Returns the grid (digits
    1..9) and boolean masks for cells whose maximum was shared (``tie``, among
    non-silent cells) and cells with no spikes at all (``silent``).
    """
    grid = np.zeros((9, 9), dtype=np.int64)
    tie = np.zeros((9, 9), dtype=bool)
    silent = np.zeros((9, 9), dtype=bool)
    for row in range(9):
        for col in range(9):
            spike_counts = counts[row, col]
            winners = np.flatnonzero(spike_counts == spike_counts.max())
            grid[row, col] = int(rng.choice(winners)) + 1
            silent[row, col] = spike_counts.max() == 0
            tie[row, col] = len(winners) > 1 and not silent[row, col]
    return grid, tie, silent


def run(puzzle, noise_rate, max_sim_ms, chunk_ms, seed, raster_per_pop, pop_size):
    t_run = time.time()
    puzzle = np.asarray(puzzle, dtype=np.int64).reshape(9, 9)
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    threads = os.cpu_count() or 1

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.ERROR
    nest.local_num_threads = threads
    nest.rng_seed = int(seed)
    rng = np.random.default_rng(seed)  # tie-breaking in the readout

    t_build = time.time()
    net = SudokuNet(pop_size=pop_size, input=puzzle, noise_rate=noise_rate)
    build_seconds = time.time() - t_build

    meta = _meta(net, puzzle, seed, threads, chunk_ms, max_sim_ms, noise_rate, raster_per_pop, build_seconds)
    yield meta

    first_gid = net.neurons[0].global_id
    acc = AnalysisAccumulator(puzzle, meta)
    n_windows = int(max_sim_ms // chunk_ms)
    prev_grid = puzzle.copy()
    valid = False
    solved_at = None
    grid = puzzle.copy()
    k = 0

    for k in range(n_windows):
        t_step = time.time()
        t_start = k * chunk_ms
        t_end = t_start + chunk_ms

        net.reset_spike_recorders()
        nest.Simulate(chunk_ms)
        trains = net.get_spike_trains()

        pop_counts = np.array([len(tr["times"]) for tr in trains], dtype=np.int64)
        senders = np.concatenate([np.asarray(tr["senders"], dtype=np.int64) for tr in trains])
        times = np.concatenate([np.asarray(tr["times"], dtype=float) for tr in trains])
        neuron_idx = senders - first_gid
        neuron_counts = np.bincount(neuron_idx, minlength=net.n_total)

        grid, tie, silent = _readout(pop_counts.reshape(9, 9, 9), rng)
        valid, boxes, rows, cols = validate_solution(puzzle, grid)
        performance = 1.0 if valid else float((boxes.sum() + rows.sum() + cols.sum()) / 27)
        cells_changed = int((grid != prev_grid).sum())
        kept = givens_kept(puzzle, grid)
        violations = count_violations(grid)

        keep = (neuron_idx % pop_size) < raster_per_pop
        r_pop = neuron_idx[keep] // pop_size
        r_t = np.clip(np.rint((times[keep] - t_start) * 10), 0, 0xFFFF).astype(np.int64)
        order = np.lexsort((r_t, r_pop))

        wall = time.time() - t_step
        acc.add_window(
            k=k,
            t_start=t_start,
            t_end=t_end,
            pop_counts=pop_counts,
            neuron_counts=neuron_counts,
            neuron_idx=neuron_idx,
            times=times,
            grid=grid,
            tie=tie,
            silent=silent,
            rows=rows,
            cols=cols,
            boxes=boxes,
            performance=performance,
            cells_changed=cells_changed,
            givens_kept=kept,
            violations=violations,
        )

        yield {
            "type": "step",
            "k": k,
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "grid": grid.flatten().tolist(),
            "tie": tie.flatten().tolist(),
            "silent": silent.flatten().tolist(),
            "valid_rows": rows.tolist(),
            "valid_cols": cols.tolist(),
            "valid_boxes": boxes.flatten().tolist(),
            "givens_kept": kept,
            "performance": round(performance, 4),
            "cells_changed": cells_changed,
            "violations": violations,
            "total_spikes": int(pop_counts.sum()),
            "pop_counts_b64": b64_u16(np.minimum(pop_counts, 0xFFFF)),
            "raster": {"pop_b64": b64_u16(r_pop[order]), "t_b64": b64_u16(r_t[order])},
            "wall_seconds": round(wall, 3),
        }

        prev_grid = grid
        if valid:
            solved_at = t_start
            break

    wall_total = time.time() - t_run
    sim_total = (k + 1) * chunk_ms
    yield {
        "type": "done",
        "solved": bool(valid),
        "solved_at_ms": solved_at,
        "final_grid": grid.flatten().tolist(),
        "sim_ms_total": sim_total,
        "wall_seconds_total": round(wall_total, 3),
        "analysis": acc.finish(solved=bool(valid), solved_at_ms=solved_at, wall_seconds_total=wall_total),
    }
