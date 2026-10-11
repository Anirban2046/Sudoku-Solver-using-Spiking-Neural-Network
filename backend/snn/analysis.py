"""Small analysis helpers for the readout of the Sudoku SNN."""

import numpy as np

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
