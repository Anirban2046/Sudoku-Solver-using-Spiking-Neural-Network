"""Stall controller logic for the Sudoku SNN (no NEST; see simulation.py)."""

import numpy as np

from .analysis import conflict_cells
from .helpers import PEERS


def kick_targets(puzzle, grid, wide=False):
    """(cell, digit) pairs to silence: the shown digit of every free cell in a
    conflict (with ``wide``, of every free cell sharing a row, column or box with
    a conflict), and the wrong digit on every overridden given cell."""
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
