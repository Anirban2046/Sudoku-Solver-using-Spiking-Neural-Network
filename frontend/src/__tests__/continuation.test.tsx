import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { solve, TransientError } from "../lib/api";
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

// jsdom has neither ResizeObserver nor a canvas; the raster just draws nothing.
vi.stubGlobal(
  "ResizeObserver",
  class {
    observe() {}
    disconnect() {}
  },
);
HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;

// 81 cells x 2 samples of int16 zeros, base64
const ZEROS = btoa(String.fromCharCode(...Array(81 * 2 * 2).fill(0)));

const meta = (segment: number, t_offset_ms: number): MetaEvent => ({
  type: "meta", seed: 7 + segment, segment, t_offset_ms, nest_version: "3.10.0", threads: 1, chunk_ms: 100,
  noise_rate: 350, stim_rate: 200, pop_size: 5, n_populations: 729, n_neurons: 3645,
  neuron_model: "iaf_psc_exp", neuron: { V_th: -50, V_reset: -70, E_L: -65, tau_m: 20, t_ref: 2 },
  connections: { inhibitory: 510300, noise: 3645, stimulus: 3645, stimulus_active: 180, recorder: 3645, total: 521235 },
  givens: 36, build_seconds: 1, wall_budget_s: 240, kick_ms: 50, patience: 5,
});

const step = (t: number, grid: number[], solved: boolean): StepEvent => ({
  type: "step", k: 0, t_start_ms: t, t_end_ms: t + 100, grid,
  valid_rows: Array(9).fill(solved), valid_cols: Array(9).fill(solved), valid_boxes: Array(9).fill(solved),
  givens_overridden: 0, performance: solved ? 1 : 0.5, violations: solved ? 0 : 3, cells_changed: 4,
  total_spikes: 11500, rate_all_hz: 31.5, rate_winners_hz: 240, rate_losers_hz: 5, kick: solved ? null : { kind: "reset", cells: [] },
  wall_seconds: 0.5,
});

const done = (solved: boolean, t_end_ms: number, grid: number[]): DoneEvent => ({
  type: "done", solved, resume: !solved, solved_at_ms: solved ? t_end_ms - 100 : null, t_end_ms, final_grid: grid,
  wall_seconds: 10, cells: { counts: Array.from({ length: 81 }, () => [9, 1, 0, 0, 0, 0, 0, 0, 0]), margin: Array(81).fill(0.9), margin_mean: 0.9 },
  network: {
    neurons_per_pop: 5,
    spikes: { neuron_b64: btoa(String.fromCharCode(0, 0, 50, 0)), t_b64: btoa(String.fromCharCode(10, 0, 20, 0)) },
    vm: { dt_ms: 50, samples: 2, winner_b64: ZEROS, rival_b64: ZEROS, rival_digit: Array(81).fill(2) },
    rates: { window_ms: 100, winners_hz: 220, others_hz: 6.5, hist_edges_hz: [0, 100, 200, 300], hist_winners: [0, 10, 395], hist_others: [3000, 240, 0] },
  },
});

describe("continuation across requests", () => {
  it("continues with a new request until the puzzle is solved", async () => {
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

    expect(calls).toEqual([{ t_offset_ms: 0, segment: 0 }, { t_offset_ms: 100, segment: 1 }]);
    expect(await screen.findByText("solved at 100 ms")).toBeTruthy();
    expect(await screen.findByText("Inside the spiking network", {}, { timeout: 10000 })).toBeTruthy();
    expect(screen.getByText("220 Hz")).toBeTruthy(); // winning neurons tile

    // editing after the run keeps the solved board and changes only that cell
    const inputs = screen.getAllByRole("textbox") as HTMLInputElement[];
    act(() => fireEvent.keyDown(inputs[0], { key: "5" }));
    const after = screen.getAllByRole("textbox") as HTMLInputElement[];
    expect(after.filter((el) => el.value).length).toBe(81);
    expect(after[0].value).toBe("5");
  }, 20000);
});

describe("resilience", () => {
  const start = async () => {
    render(<App />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Randomly Initialize" })));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Start Solving" })));
  };
  const g = Array(81).fill(1);

  it("retries after a server error", async () => {
    const calls: unknown[] = [];
    vi.mocked(solve).mockImplementation(async (_p, onEvent: (ev: SolveEvent) => void, _s, cont) => {
      calls.push(cont);
      if (calls.length === 1) throw new TransientError("Server error 504");
      onEvent(meta(1, 0));
      onEvent(step(0, g, true));
      onEvent(done(true, 100, g));
    });
    await start();
    expect(await screen.findByText("solved at 0 ms", {}, { timeout: 8000 })).toBeTruthy();
    expect(calls).toHaveLength(2);
  }, 20000);

  it("continues from the last window after an error event mid-run", async () => {
    const calls: { t_offset_ms?: number }[] = [];
    vi.mocked(solve).mockImplementation(async (_p, onEvent: (ev: SolveEvent) => void, _s, cont) => {
      calls.push(cont ?? {});
      if (calls.length === 1) {
        onEvent(meta(0, 0));
        onEvent(step(0, g, false));
        onEvent({ type: "error", message: "MemoryError" });
      }
      onEvent(meta(1, 100));
      onEvent(step(100, g, true));
      onEvent(done(true, 200, g));
    });
    await start();
    expect(await screen.findByText("solved at 100 ms", {}, { timeout: 8000 })).toBeTruthy();
    expect(calls.map((c) => c.t_offset_ms)).toEqual([0, 100]);
    expect(screen.queryByRole("alert")).toBeNull();
  }, 20000);

  it("shows that it is waiting while the simulator is busy", async () => {
    let release = () => {};
    vi.mocked(solve).mockImplementation(async (_p, onEvent: (ev: SolveEvent) => void, _s, _c, onBusy) => {
      onBusy?.();
      await new Promise<void>((r) => (release = r));
      onEvent(meta(0, 0));
      onEvent(step(0, g, true));
      onEvent(done(true, 100, g));
    });
    await start();
    expect(await screen.findByText("waiting for a free simulator…")).toBeTruthy();
    await act(async () => release());
    expect(await screen.findByText("solved at 0 ms")).toBeTruthy();
  });
});

describe("niceMax", () => {
  it("rounds axis maxima up to a close round number", () => {
    expect(niceMax(0)).toBe(1000);
    expect(niceMax(1300)).toBe(1500);
    expect(niceMax(2100)).toBe(2500);
    expect(niceMax(24100)).toBe(25000);
    expect(niceMax(5000)).toBe(5000);
    expect(niceMax(73, 10)).toBe(80);
  });
});
