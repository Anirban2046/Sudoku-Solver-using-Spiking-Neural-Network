import numpy as np

from snn.analysis import count_violations
from snn.encoding import b64_u16, decode_b64_u16
from snn.helpers import PEER_PAIRS, generate_puzzle, get_progress, get_puzzle, givens_kept, solve_all, validate_solution


def solved_grid():
    return solve_all(get_puzzle(4), 1)[0]


def test_validate_solution_valid_grid():
    puzzle = get_puzzle(4)
    sol = solved_grid()
    valid, boxes, rows, cols = validate_solution(puzzle, sol)
    assert valid and boxes.all() and rows.all() and cols.all()
    assert get_progress(puzzle, sol) == 1.0


def test_validate_solution_flags_swap():
    puzzle = np.zeros((9, 9), dtype=int)
    sol = solved_grid().copy()
    sol[0, 0], sol[0, 1] = sol[0, 1], sol[0, 0]  # row 1 still fine, columns 1-2 broken
    valid, boxes, rows, cols = validate_solution(puzzle, sol)
    assert not valid
    assert rows.all()
    assert not cols[0] and not cols[1] and cols[2:].all()
    assert boxes.all()  # swap inside box 1 keeps it valid
    assert get_progress(puzzle, sol) == 25 / 27


def test_overridden_given_is_invalid():
    puzzle = get_puzzle(4)
    sol = solved_grid()
    other = puzzle.copy()
    r, c = np.argwhere(puzzle != 0)[0]
    other[r, c] = sol[r, c] % 9 + 1
    valid, *_ = validate_solution(other, sol)
    assert not valid
    assert not givens_kept(other, sol)


def test_peer_pairs_count():
    assert len(PEER_PAIRS) == 810


def test_count_violations():
    sol = solved_grid()
    assert count_violations(sol) == 0
    bad = sol.copy()
    bad[0, 0] = bad[0, 1]  # duplicates in row 1 and box 1 (same pair) + its column
    assert count_violations(bad) >= 1


def test_solver_counts():
    assert len(solve_all(get_puzzle(4), 2)) == 1
    assert len(solve_all(get_puzzle(0), 2)) == 2
    conflicting = np.zeros((9, 9), dtype=int)
    conflicting[0, 0] = conflicting[0, 5] = 3
    assert solve_all(conflicting, 2) == []


def test_generate_puzzle_unique():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = generate_puzzle(36, rng)
        assert 36 <= (p != 0).sum() <= 40
        assert len(solve_all(p, 2)) == 1


def test_b64_roundtrip():
    arr = np.array([0, 1, 728, 65535])
    assert decode_b64_u16(b64_u16(arr)).tolist() == arr.tolist()
