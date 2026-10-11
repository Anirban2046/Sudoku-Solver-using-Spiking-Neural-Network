# -*- coding: utf-8 -*-
#
# helpers.py
#
# Ported from helpers_sudoku.py, which is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

r"""Helper functions for the Sudoku solver
----------------------------------------------------------------

``get_puzzle`` and ``validate_solution`` are ported unchanged from
``helpers_sudoku.py``. The matplotlib drawing functions are not ported; the
browser draws the board instead (with the invalid-box frame placed at
``(row=3*i, col=3*j)``, fixing the upstream ``(3*j, 3*j)`` bug).

The classical backtracking solver, its node-limited solvability check and the
puzzle generator are additions: they reject unsolvable puzzles, check the
network's answer, and generate test puzzles.

:Authors: J Gille, S Furber, A Rowley (ported functions)
"""

import numpy as np


def get_puzzle(puzzle_index):
    """Returns one of 8 Sudoku configuration to be solved.

    Parameters
    ----------
    puzzle_index : int
        index between 0 and 7 indicating the puzzle number

    Returns
    -------
    np.array
        array of shape (9,9) representing the puzzle configuration.
        Array is zero wherever no input is given, and contains the corresponding
        digit otherwise.
    """
    init_config = None

    if puzzle_index == 0:
        # Dream problem: make the network come up with a valid sudoku without
        # any restrictions
        init_config = np.zeros((9, 9), dtype=np.uint8)
    elif puzzle_index == 1:
        # Diabolical problem:
        init_config = [
            [0, 0, 1, 0, 0, 8, 0, 7, 3],
            [0, 0, 5, 6, 0, 0, 0, 0, 1],
            [7, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 9, 0, 8, 1, 0, 0, 0, 0],
            [5, 3, 0, 0, 0, 0, 0, 4, 6],
            [0, 0, 0, 0, 6, 5, 0, 3, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 4],
            [8, 0, 0, 0, 0, 9, 3, 0, 0],
            [9, 4, 0, 5, 0, 0, 7, 0, 0],
        ]

    elif puzzle_index == 2:
        init_config = [
            [2, 0, 0, 0, 0, 6, 0, 3, 0],
            [4, 8, 0, 0, 1, 9, 0, 0, 0],
            [0, 0, 7, 0, 2, 0, 9, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 9, 0],
            [7, 0, 8, 0, 0, 0, 1, 0, 5],
            [0, 4, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 4, 0, 9, 0, 6, 0, 0],
            [0, 0, 0, 6, 4, 0, 0, 1, 9],
            [0, 5, 0, 1, 0, 0, 0, 0, 8],
        ]

    elif puzzle_index == 3:
        init_config = [
            [0, 0, 3, 2, 0, 0, 0, 7, 0],
            [0, 0, 5, 0, 0, 0, 3, 0, 0],
            [0, 0, 8, 9, 7, 0, 0, 5, 0],
            [0, 0, 0, 8, 9, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 6, 1, 0, 0, 0],
            [0, 1, 0, 0, 2, 5, 6, 0, 0],
            [0, 0, 4, 0, 0, 0, 8, 0, 0],
            [0, 9, 0, 0, 0, 7, 5, 0, 0],
        ]

    elif puzzle_index == 4:
        init_config = [
            [0, 1, 0, 0, 0, 0, 0, 0, 2],
            [8, 7, 0, 0, 0, 0, 5, 0, 4],
            [5, 0, 2, 0, 0, 0, 0, 9, 0],
            [0, 5, 0, 4, 0, 9, 0, 0, 1],
            [0, 0, 0, 7, 3, 2, 0, 0, 0],
            [9, 0, 0, 5, 0, 1, 0, 4, 0],
            [0, 2, 0, 0, 0, 0, 4, 0, 8],
            [4, 0, 6, 0, 0, 0, 0, 1, 3],
            [1, 0, 0, 0, 0, 0, 0, 2, 0],
        ]

    elif puzzle_index == 5:
        init_config = [
            [8, 9, 0, 2, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 8, 0, 0, 0, 0],
            [0, 4, 1, 0, 3, 0, 5, 0, 0],
            [2, 5, 8, 0, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [6, 0, 0, 0, 0, 0, 1, 4, 7],
            [0, 0, 7, 0, 1, 0, 4, 3, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 7, 0, 5, 1],
        ]

    elif puzzle_index == 6:
        # "World's hardest sudoku":
        # http://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html
        init_config = [
            [8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 6, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 9, 0, 2, 0, 0],
            [0, 5, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 4, 5, 7, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 3, 0],
            [0, 0, 1, 0, 0, 0, 0, 6, 8],
            [0, 0, 8, 5, 0, 0, 0, 1, 0],
            [0, 9, 0, 0, 0, 0, 4, 0, 0],
        ]

    elif puzzle_index == 7:
        init_config = [
            [1, 0, 0, 4, 0, 0, 0, 0, 0],
            [7, 0, 0, 5, 0, 0, 6, 0, 3],
            [0, 0, 0, 0, 3, 0, 4, 2, 0],
            [0, 0, 9, 0, 0, 0, 0, 3, 5],
            [0, 0, 0, 3, 0, 5, 0, 0, 0],
            [6, 3, 0, 0, 0, 0, 1, 0, 0],
            [0, 2, 6, 0, 5, 0, 0, 0, 0],
            [9, 0, 4, 0, 0, 6, 0, 0, 7],
            [0, 0, 0, 0, 0, 8, 0, 0, 2],
        ]

    else:
        raise ValueError(f"No puzzle for index {puzzle_index} found.")

    return np.array(init_config)


def validate_solution(puzzle, solution):
    """Validates a proposed solution for a sudoku puzzle

    Parameters
    ----------
    puzzle : np.array
        array of shape (9,9) encoding the puzzle.
        see get_puzzle().
    solution : np.array
        array of shape (9,9) encoding the proposed solution.

    Returns
    -------
    bool
        True if the overall solution is valid, False otherwise.
    np.array (3,3)
        True wherever a 3x3 box is valid
    np.array (9,)
        True wherever a row is valid
    np.array (9,)
        True wherever a column is valid
    """

    boxes = np.ones((3, 3), dtype=bool)
    rows = np.ones(9, dtype=bool)
    cols = np.ones(9, dtype=bool)

    expected_numbers = set(range(1, 10))

    # validate boxes
    for i in range(3):
        for j in range(3):
            box = solution[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            if expected_numbers != set(box.flatten()):
                boxes[i, j] = False

    # validate rows and columns
    for i in range(9):
        if expected_numbers != set(solution[i, :]):
            rows[i] = False
        if expected_numbers != set(solution[:, i]):
            cols[i] = False

    # It is possible (in rare cases) that the network finds a valid
    # solution that does not conform to the initial puzzle configuration, that is,
    # one of the cells where input is applied is overridden by the rest of the
    # network. This is taken care of here.
    input_cells = np.where(puzzle != 0)
    puzzle_matched = puzzle[input_cells] == solution[input_cells]

    # overall solution is valid iff all of the components are.
    valid = boxes.all() and rows.all() and cols.all() and puzzle_matched.all()

    return valid, boxes, rows, cols


# ---------------------------------------------------------------------------
# Classical backtracking solver (bitmask candidates, minimum remaining values)
# ---------------------------------------------------------------------------

_ALL = 0x3FE  # bits 1..9


def _peers():
    peers = []
    for i in range(81):
        r, c = divmod(i, 9)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        s = {r * 9 + k for k in range(9)} | {k * 9 + c for k in range(9)}
        s |= {(br + a) * 9 + bc + b for a in range(3) for b in range(3)}
        s.discard(i)
        peers.append(tuple(sorted(s)))
    return tuple(peers)


PEERS = _peers()
PEER_PAIRS = np.array([(i, j) for i in range(81) for j in PEERS[i] if j > i], dtype=np.int16)


class _Budget(Exception):
    pass


def _search(cells, limit, out, budget=None):
    """Depth-first search; appends complete grids to ``out`` until ``limit``.
    ``budget`` is a one-element list of remaining nodes, or None for no limit."""
    if budget is not None:
        budget[0] -= 1
        if budget[0] < 0:
            raise _Budget
    best, best_cands, best_n = -1, 0, 10
    for i in range(81):
        if cells[i]:
            continue
        used = 0
        for p in PEERS[i]:
            if cells[p]:
                used |= 1 << cells[p]
        cands = _ALL & ~used
        n = bin(cands).count("1")
        if n < best_n:
            best, best_cands, best_n = i, cands, n
            if n <= 1:
                break
    if best < 0:
        out.append(list(cells))
        return
    if best_n == 0:
        return
    for d in range(1, 10):
        if best_cands & (1 << d):
            cells[best] = d
            _search(cells, limit, out, budget)
            cells[best] = 0
            if len(out) >= limit:
                return


def solve_all(puzzle, limit=2, max_nodes=None):
    """Returns up to ``limit`` solutions of ``puzzle`` as (9,9) arrays.

    Returns an empty list if the givens already conflict. With ``max_nodes``,
    the search stops after that many nodes and returns what it found so far.
    """
    cells = [int(v) for v in np.asarray(puzzle).flatten()]
    for i, v in enumerate(cells):
        if v and any(cells[p] == v for p in PEERS[i]):
            return []
    out = []
    try:
        _search(cells, limit, out, None if max_nodes is None else [max_nodes])
    except _Budget:
        pass
    return [np.array(s, dtype=np.int64).reshape(9, 9) for s in out]


def has_solution(puzzle, max_nodes=200_000):
    """True or False, or None if the search exceeded ``max_nodes``."""
    cells = [int(v) for v in np.asarray(puzzle).flatten()]
    for i, v in enumerate(cells):
        if v and any(cells[p] == v for p in PEERS[i]):
            return False
    out = []
    try:
        _search(cells, 1, out, [max_nodes])
    except _Budget:
        return None
    return bool(out)


def generate_puzzle(givens=36, rng=None):
    """Random puzzle with a unique solution and about ``givens`` given digits."""
    rng = rng or np.random.default_rng()
    cells = [0] * 81

    def fill(i):
        if i == 81:
            return True
        if cells[i]:
            return fill(i + 1)
        used = {cells[p] for p in PEERS[i]}
        for d in rng.permutation(9) + 1:
            if int(d) not in used:
                cells[i] = int(d)
                if fill(i + 1):
                    return True
        cells[i] = 0
        return False

    fill(0)
    grid = np.array(cells).reshape(9, 9)
    remaining = 81
    for idx in rng.permutation(81):
        if remaining <= givens:
            break
        r, c = divmod(int(idx), 9)
        keep = grid[r, c]
        grid[r, c] = 0
        if len(solve_all(grid, 2)) != 1:
            grid[r, c] = keep
        else:
            remaining -= 1
    return grid
