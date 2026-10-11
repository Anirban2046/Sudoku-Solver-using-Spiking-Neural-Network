import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { solve } from "../lib/api";
import { niceMax } from "../components/ChartCanvas";
import type { DoneEvent, MetaEvent, SolveEvent, StepEvent } from "../lib/types";

vi.mock("../lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api")>()),
  solve: vi.fn(),
}));

// jsdom has no canvas; render charts as placeholders.
vi.mock("../components/ChartCanvas", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../components/ChartCanvas")>()),
  default: ({ label }: { label: string }) => <div role="img" aria-label={label} />,
}));

afterEach(cleanup);

const meta = (segment: number, t_offset_ms: number): MetaEvent => ({
  type: "meta", seed: 7 + segment, segment, t_offset_ms, nest_version: "3.10.0", threads: 1, chunk_ms: 100,
  noise_rate: 350, stim_rate: 200, pop_size: 5, n_populations: 729, n_neurons: 3645,
  connections: { inhibitory: 510300, noise: 3645, stimulus: 3645, stimulus_active: 180, recorder: 3645, total: 521235 },
  givens: 36, build_seconds: 1, wall_budget_s: 240, kick_ms: 50, patience: 5, hints: [], hint_after_ms: 15000,
});

const step = (t: number, grid: number[], solved: boolean): StepEvent => ({
  type: "step", k: 0, t_start_ms: t, t_end_ms: t + 100, grid,
  valid_rows: Array(9).fill(solved), valid_cols: Array(9).fill(solved), valid_boxes: Array(9).fill(solved),
  givens_overridden: 0, performance: solved ? 1 : 0.5, violations: solved ? 0 : 3, cells_changed: 4,
  total_spikes: 11500, rate_all_hz: 31.5, rate_winners_hz: 240, rate_losers_hz: 5, kick: solved ? null : { kind: "hint", cells: [5, 6] },
  wall_seconds: 0.5,
});

const done = (solved: boolean, t_end_ms: number, grid: number[]): DoneEvent => ({
  type: "done", solved, resume: !solved, solved_at_ms: solved ? t_end_ms - 100 : null, t_end_ms, final_grid: grid,
  wall_seconds: 10, cells: { counts: Array.from({ length: 81 }, () => [9, 1, 0, 0, 0, 0, 0, 0, 0]), margin: Array(81).fill(0.9), margin_mean: 0.9 },
  classical: { n_solutions: 1, verdict: solved ? "match" : "not_solved" }, hints: [5, 6],
});

describe("continuation across requests", () => {
  it("continues with a new request, carrying hints, until the puzzle is solved", async () => {
    const calls: unknown[] = [];
    vi.mocked(solve).mockImplementation(async (_puzzle, onEvent: (ev: SolveEvent) => void, _signal, cont) => {
      calls.push(cont);
      const g = Array(81).fill(1);
      if (calls.length === 1) {
        onEvent(meta(0, 0));
        onEvent(step(0, g, false));
        onEvent(done(false, 100, g));
      } else {
        onEvent(meta(1, 100));
        onEvent(step(100, g, true));
        onEvent(done(true, 200, g));
      }
    });

    render(<App />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Randomly Initialize" })));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Start Solving" })));

    expect(calls).toEqual([{ t_offset_ms: 0, segment: 0, hints: [] }, { t_offset_ms: 100, segment: 1, hints: [5, 6] }]);
    expect(await screen.findByText("solved at 100 ms")).toBeTruthy();
    expect(await screen.findByText("How the network solved it", {}, { timeout: 10000 })).toBeTruthy();
    // no kicks; two restarts (the hint restart and the second request), in the live stats and the results tile
    expect(screen.getAllByText("0 / 2")).toHaveLength(2);
    expect(screen.getByText(/2 of the 45 empty cells were hinted/)).toBeTruthy();
  }, 20000);
});

describe("niceMax", () => {
  it("rounds axis maxima up to 1, 2 or 5 x 10^n", () => {
    expect(niceMax(0)).toBe(1000);
    expect(niceMax(1300)).toBe(2000);
    expect(niceMax(2100)).toBe(5000);
    expect(niceMax(5000)).toBe(5000);
    expect(niceMax(73, 10)).toBe(100);
  });
});
