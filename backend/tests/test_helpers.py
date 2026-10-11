import numpy as np

from snn.analysis import cell_summary, classical_check, conflict_cells, count_violations
from snn.helpers import PEER_PAIRS, generate_puzzle, get_puzzle, has_solution, solve_all, validate_solution


def solved_grid():
    return solve_all(get_puzzle(4), 1)[0]


def test_validate_solution_valid_grid():
    puzzle = get_puzzle(4)
    sol = solved_grid()
    valid, boxes, rows, cols = validate_solution(puzzle, sol)
    assert valid and boxes.all() and rows.all() and cols.all()


def test_validate_solution_flags_swap():
    puzzle = np.zeros((9, 9), dtype=int)
    sol = solved_grid().copy()
    sol[0, 0], sol[0, 1] = sol[0, 1], sol[0, 0]  # row 1 still fine, columns 1-2 broken
    valid, boxes, rows, cols = validate_solution(puzzle, sol)
    assert not valid
    assert rows.all()
    assert not cols[0] and not cols[1] and cols[2:].all()
    assert boxes.all()  # swap inside box 1 keeps it valid


def test_overridden_given_is_invalid():
    puzzle = get_puzzle(4)
    sol = solved_grid()
    other = puzzle.copy()
    r, c = np.argwhere(puzzle != 0)[0]
    other[r, c] = sol[r, c] % 9 + 1
    valid, *_ = validate_solution(other, sol)
    assert not valid


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


def test_has_solution():
    assert has_solution(get_puzzle(6)) is True
    blocked = np.zeros((9, 9), dtype=int)
    blocked[0, 1:5] = [1, 2, 3, 4]
    blocked[1:3, 0] = [5, 6]
    blocked[1, 1], blocked[2, 2], blocked[5, 0] = 7, 8, 9
    assert has_solution(blocked) is False
    assert has_solution(np.zeros((9, 9), dtype=int), max_nodes=3) is None


def test_conflict_cells():
    bad = solved_grid().copy()
    bad[0, 0] = bad[0, 1]
    cells = set(conflict_cells(bad).tolist())
    assert {0, 1} <= cells


def test_cell_summary_margin():
    counts = np.zeros(729, dtype=int)
    counts[0:9] = [10, 5, 0, 0, 0, 0, 0, 0, 0]
    s = cell_summary(counts)
    assert s["margin"][0] == 0.5
    assert s["margin"][1] == 0.0  # silent cell


def test_classical_check():
    puzzle = get_puzzle(4)
    sol = solved_grid()
    assert classical_check(puzzle, sol, True)["verdict"] == "match"
    assert classical_check(puzzle, sol, False)["verdict"] == "not_solved"


def test_b64_roundtrip():
    from snn.encoding import b64_i16, b64_u16, decode_b64

    assert decode_b64(b64_u16([0, 1, 728, 65535]), "<u2").tolist() == [0, 1, 728, 65535]
    assert decode_b64(b64_i16([-31039.6, -5000, 0, 40000]), "<i2").tolist() == [-31040, -5000, 0, 32767]
