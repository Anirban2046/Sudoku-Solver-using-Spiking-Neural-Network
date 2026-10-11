"""Small analysis helpers for the readout of the Sudoku SNN."""

import numpy as np

from .encoding import b64_i16, b64_u16
from .helpers import PEER_PAIRS, solve_all


def _equal_pairs(grid):
    g = np.asarray(grid).reshape(81)
    return PEER_PAIRS[g[PEER_PAIRS[:, 0]] == g[PEER_PAIRS[:, 1]]]


def count_violations(grid):
    """Number of unordered cell pairs sharing a row, column or box that show
    the same digit (the "energy" of the readout; 0 for a valid grid)."""
    return int(len(_equal_pairs(grid)))


def conflict_cells(grid):
    """Indices (0..80) of the cells involved in at least one conflicting pair."""
    return np.unique(_equal_pairs(grid))


def cell_summary(pop_counts):
    """Per-cell competition in the last window: the 9 digit spike counts of each
    cell and its decision margin (top - second) / top."""
    counts = np.asarray(pop_counts).reshape(81, 9)
    srt = np.sort(counts, axis=1)
    margin = (srt[:, -1] - srt[:, -2]) / np.maximum(srt[:, -1], 1)
    return {
        "counts": counts.tolist(),
        "margin": [round(float(m), 3) for m in margin],
        "margin_mean": round(float(margin.mean()), 3),
    }


def classical_check(puzzle, grid, solved):
    """Compare the network's answer with a plain backtracking solver."""
    sols = solve_all(puzzle, limit=2)
    match = bool(sols) and np.array_equal(sols[0], np.asarray(grid).reshape(9, 9))
    if not solved:
        verdict = "not_solved"
    elif match:
        verdict = "match"
    else:
        verdict = "valid_but_different"
    return {"n_solutions": min(len(sols), 2), "verdict": verdict}


def network_view(*, grid, pop_counts, pop_size, chunk_ms, neuron_idx, spike_t, vm, recent_neuron_counts):
    """What the spiking network looked like in its final window.

    * every spike of every neuron (neuron index 0..3644, time in 0.1 ms from the
      window start), so the frontend can draw any cell's 45-neuron raster;
    * the membrane potential (centi-mV, 1 ms steps) of the first neuron of each
      cell's winning population and of its strongest rival;
    * firing rates over the last ~1 s: winners versus all other neurons, and a
      histogram of per-neuron rates.

    ``vm`` has shape (729, samples): the recorded neuron of every population.
    ``recent_neuron_counts`` is a list of per-neuron spike counts, one per window.
    """
    g = np.asarray(grid).reshape(81)
    counts = np.asarray(pop_counts).reshape(81, 9).copy()
    winner_pop = np.arange(81) * 9 + g - 1
    counts[np.arange(81), g - 1] = -1
    rival_digit = counts.argmax(axis=1) + 1
    rival_pop = np.arange(81) * 9 + rival_digit - 1

    order = np.lexsort((spike_t, neuron_idx))
    t01 = np.clip(np.rint(np.asarray(spike_t)[order] * 10), 0, chunk_ms * 10)

    window_s = len(recent_neuron_counts) * chunk_ms / 1000.0
    rates = np.sum(recent_neuron_counts, axis=0) / window_s
    is_winner = np.isin(np.arange(len(rates)) // pop_size, winner_pop)
    # round bin widths, at most 20 bins
    width = next(w for w in (5, 10, 20, 25, 50, 100, 200, 500) if rates.max() / w <= 20)
    edges = np.arange(0.0, (np.floor(rates.max() / width) + 2) * width, width)

    return {
        "neurons_per_pop": pop_size,
        "spikes": {"neuron_b64": b64_u16(np.asarray(neuron_idx)[order]), "t_b64": b64_u16(t01)},
        "vm": {
            "dt_ms": chunk_ms / vm.shape[1],
            "samples": int(vm.shape[1]),
            "winner_b64": b64_i16(vm[winner_pop] * 100),
            "rival_b64": b64_i16(vm[rival_pop] * 100),
            "rival_digit": rival_digit.tolist(),
        },
        "rates": {
            "window_ms": int(round(window_s * 1000)),
            "winners_hz": round(float(rates[is_winner].mean()), 1),
            "others_hz": round(float(rates[~is_winner].mean()), 2),
            "hist_edges_hz": [round(float(e), 1) for e in edges],
            "hist_winners": np.histogram(rates[is_winner], bins=edges)[0].tolist(),
            "hist_others": np.histogram(rates[~is_winner], bins=edges)[0].tolist(),
        },
    }
