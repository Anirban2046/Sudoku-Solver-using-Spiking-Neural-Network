import { describe, expect, it } from "vitest";
import { canStart, countGivens, countSolutions, findConflicts, generatePuzzle, solve, type Grid } from "../lib/sudoku";

// NEST docs puzzle 4 (unique solution)
const PUZZLE_4: Grid = [
  0, 1, 0, 0, 0, 0, 0, 0, 2,
  8, 7, 0, 0, 0, 0, 5, 0, 4,
  5, 0, 2, 0, 0, 0, 0, 9, 0,
  0, 5, 0, 4, 0, 9, 0, 0, 1,
  0, 0, 0, 7, 3, 2, 0, 0, 0,
  9, 0, 0, 5, 0, 1, 0, 4, 0,
  0, 2, 0, 0, 0, 0, 4, 0, 8,
  4, 0, 6, 0, 0, 0, 0, 1, 3,
  1, 0, 0, 0, 0, 0, 0, 2, 0,
];

const at = (r: number, c: number) => r * 9 + c;

function withValues(entries: [number, number, number][]): Grid {
  const g = Array(81).fill(0);
  for (const [r, c, d] of entries) g[at(r, c)] = d;
  return g;
}

describe("findConflicts", () => {
  it("returns nothing for an empty or valid grid", () => {
    expect(findConflicts(Array(81).fill(0)).cells.size).toBe(0);
    expect(findConflicts(PUZZLE_4).cells.size).toBe(0);
    const solved = solve(PUZZLE_4)!;
    expect(findConflicts(solved).cells.size).toBe(0);
  });

  it("flags a row duplicate", () => {
    const { cells, messages } = findConflicts(withValues([[2, 0, 5], [2, 8, 5]]));
    expect([...cells].sort((a, b) => a - b)).toEqual([at(2, 0), at(2, 8)]);
    expect(messages).toEqual(["5 appears twice in row 3"]);
  });

  it("flags a column duplicate", () => {
    const { cells, messages } = findConflicts(withValues([[0, 7, 7], [8, 7, 7]]));
    expect(cells).toEqual(new Set([at(0, 7), at(8, 7)]));
    expect(messages).toEqual(["7 appears twice in column 8"]);
  });

  it("flags a box-only duplicate", () => {
    const { cells, messages } = findConflicts(withValues([[3, 3, 9], [4, 4, 9]]));
    expect(cells).toEqual(new Set([at(3, 3), at(4, 4)]));
    expect(messages).toEqual(["9 appears twice in box 5"]);
  });

  it("does not flag same digit in different units", () => {
    expect(findConflicts(withValues([[0, 0, 4], [1, 3, 4], [3, 1, 4]])).cells.size).toBe(0);
  });

  it("reports row and box when both are violated", () => {
    const { messages } = findConflicts(withValues([[0, 0, 1], [0, 1, 1]]));
    expect(messages).toEqual(["1 appears twice in row 1", "1 appears twice in box 1"]);
  });
});

describe("countSolutions", () => {
  it("counts unique, multiple and impossible", () => {
    expect(countSolutions(PUZZLE_4)).toBe(1);
    expect(countSolutions(Array(81).fill(0))).toBe(2);
    // (0,0) sees 1-4 in its row, 5, 6, 9 in its column and 7, 8 in its box
    const blocked = withValues([[0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4], [1, 0, 5], [2, 0, 6], [1, 1, 7], [2, 2, 8], [5, 0, 9]]);
    expect(countSolutions(blocked)).toBe(0);
  });

  it("returns 0 for conflicting grids", () => {
    expect(countSolutions(withValues([[0, 0, 1], [0, 1, 1]]))).toBe(0);
  });
});

describe("generatePuzzle", () => {
  it("makes unique, conflict-free 36-given puzzles quickly", () => {
    for (let k = 0; k < 50; k++) {
      const t = performance.now();
      const p = generatePuzzle(36);
      const ms = performance.now() - t;
      const n = countGivens(p);
      expect(n).toBeGreaterThanOrEqual(36);
      expect(n).toBeLessThanOrEqual(40);
      expect(findConflicts(p).cells.size).toBe(0);
      expect(countSolutions(p, 2, 5000)).toBe(1);
      expect(ms).toBeLessThan(500); // < 200 ms in a browser; generous for CI
    }
  });
});

describe("canStart", () => {
  it("applies the Start Solving rules", () => {
    expect(canStart(Array(81).fill(0))).toEqual({ ok: false, reason: "Enter at least 17 numbers" });
    const dup = PUZZLE_4.slice();
    dup[at(0, 0)] = 1;
    expect(canStart(dup)).toEqual({ ok: false, reason: "Fix the duplicates shown in red" });
    expect(canStart(PUZZLE_4)).toEqual({ ok: true });
  });
});
