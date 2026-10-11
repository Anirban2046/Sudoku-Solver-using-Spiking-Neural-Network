// Pure Sudoku helpers: conflict detection, a classical solver and a puzzle
// generator. A grid is a flat array of 81 numbers, row-major, 0 = empty.

export type Grid = number[];

export const EMPTY_GRID: Grid = Array(81).fill(0);

export const rowOf = (i: number) => Math.floor(i / 9);
export const colOf = (i: number) => i % 9;
export const boxOf = (i: number) => Math.floor(rowOf(i) / 3) * 3 + Math.floor(colOf(i) / 3);

type Unit = { kind: "row" | "column" | "box"; index: number; cells: number[] };

const UNITS: Unit[] = (() => {
  const units: Unit[] = [];
  for (let k = 0; k < 9; k++) {
    units.push({ kind: "row", index: k, cells: Array.from({ length: 9 }, (_, c) => k * 9 + c) });
    units.push({ kind: "column", index: k, cells: Array.from({ length: 9 }, (_, r) => r * 9 + k) });
    const br = Math.floor(k / 3) * 3;
    const bc = (k % 3) * 3;
    const box: number[] = [];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) box.push((br + a) * 9 + bc + b);
    units.push({ kind: "box", index: k, cells: box });
  }
  return units;
})();

const PEERS: number[][] = Array.from({ length: 81 }, (_, i) => {
  const s = new Set<number>();
  for (const u of UNITS) if (u.cells.includes(i)) u.cells.forEach((c) => c !== i && s.add(c));
  return [...s].sort((a, b) => a - b);
});

export const countGivens = (grid: Grid) => grid.reduce((n, v) => n + (v ? 1 : 0), 0);

const TIMES = ["", "once", "twice", "three times"];

export interface Conflicts {
  cells: Set<number>;
  messages: string[];
}

/** Cells involved in a duplicate within a row, column or 3x3 box, plus messages
 *  such as "5 appears twice in row 3". */
export function findConflicts(grid: Grid): Conflicts {
  const cells = new Set<number>();
  const messages: string[] = [];
  for (const unit of UNITS) {
    const byDigit = new Map<number, number[]>();
    for (const c of unit.cells) {
      const d = grid[c];
      if (!d) continue;
      byDigit.set(d, [...(byDigit.get(d) ?? []), c]);
    }
    for (const [d, where] of [...byDigit.entries()].sort((a, b) => a[0] - b[0])) {
      if (where.length < 2) continue;
      where.forEach((c) => cells.add(c));
      const times = TIMES[where.length] ?? `${where.length} times`;
      messages.push(`${d} appears ${times} in ${unit.kind} ${unit.index + 1}`);
    }
  }
  return { cells, messages };
}

const ALL = 0x3fe; // bits 1..9

function popcount(x: number) {
  let n = 0;
  while (x) {
    x &= x - 1;
    n++;
  }
  return n;
}

/** Backtracking solver with candidate bitmasks and minimum-remaining-values
 *  ordering. Returns the number of solutions found, capped at `limit`
 *  (so 2 means "at least 2"), or "timeout". */
export function countSolutions(grid: Grid, limit = 2, timeoutMs = 1000): number | "timeout" {
  if (findConflicts(grid).cells.size) return 0;
  const result = search(grid.slice(), limit, performance.now() + timeoutMs, null);
  return result.timedOut ? "timeout" : result.count;
}

/** One solution of the grid, or null. */
export function solve(grid: Grid, timeoutMs = 2000): Grid | null {
  if (findConflicts(grid).cells.size) return null;
  const out: Grid[] = [];
  search(grid.slice(), 1, performance.now() + timeoutMs, out);
  return out[0] ?? null;
}

function search(cells: Grid, limit: number, deadline: number, out: Grid[] | null) {
  let count = 0;
  let timedOut = false;
  let nodes = 0;
  const rec = (): void => {
    if (count >= limit || timedOut) return;
    if ((++nodes & 1023) === 0 && performance.now() > deadline) {
      timedOut = true;
      return;
    }
    let best = -1;
    let bestCands = 0;
    let bestN = 10;
    for (let i = 0; i < 81; i++) {
      if (cells[i]) continue;
      let used = 0;
      for (const p of PEERS[i]) if (cells[p]) used |= 1 << cells[p];
      const cands = ALL & ~used;
      const n = popcount(cands);
      if (n < bestN) {
        best = i;
        bestCands = cands;
        bestN = n;
        if (n <= 1) break;
      }
    }
    if (best < 0) {
      count++;
      out?.push(cells.slice());
      return;
    }
    if (bestN === 0) return;
    for (let d = 1; d <= 9; d++) {
      if (!(bestCands & (1 << d))) continue;
      cells[best] = d;
      rec();
      cells[best] = 0;
      if (count >= limit || timedOut) return;
    }
  };
  rec();
  return { count, timedOut };
}

function shuffle<T>(arr: T[], rand: () => number): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Random puzzle with exactly one solution and `givens` given digits (or a few
 *  more, if no further cell can be removed without losing uniqueness). */
export function generatePuzzle(givens = 36, rand: () => number = Math.random): Grid {
  const cells: Grid = Array(81).fill(0);
  const fill = (i: number): boolean => {
    if (i === 81) return true;
    const used = new Set(PEERS[i].map((p) => cells[p]));
    for (const d of shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9], rand)) {
      if (used.has(d)) continue;
      cells[i] = d;
      if (fill(i + 1)) return true;
    }
    cells[i] = 0;
    return false;
  };
  fill(0);

  let remaining = 81;
  for (const i of shuffle(Array.from({ length: 81 }, (_, k) => k), rand)) {
    if (remaining <= givens) break;
    const keep = cells[i];
    cells[i] = 0;
    if (countSolutions(cells, 2, 1000) !== 1) cells[i] = keep;
    else remaining--;
  }
  return cells;
}

export type StartCheck =
  | { ok: true; note?: string }
  | { ok: false; reason: string };

/** Whether "Start Solving" may be pressed for this grid, and why not. */
export function canStart(grid: Grid, conflicts: Conflicts = findConflicts(grid)): StartCheck {
  if (countGivens(grid) < 17) return { ok: false, reason: "Enter at least 17 numbers" };
  if (conflicts.cells.size) return { ok: false, reason: "Fix the duplicates shown in red" };
  const n = countSolutions(grid, 1, 1000);
  // Every solvable puzzle is found in milliseconds; a search this long means the
  // puzzle is (almost certainly) impossible, and a run on it could never end.
  if (n === "timeout") return { ok: false, reason: "This puzzle doesn't seem to have a solution" };
  if (n === 0) return { ok: false, reason: "This puzzle has no solution" };
  return { ok: true };
}
