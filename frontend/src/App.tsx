import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Controls from "./components/Controls";
import LivePanel, { type RunStatus } from "./components/LivePanel";
import SudokuGrid, { type LiveBoard } from "./components/SudokuGrid";
import { solve, SolveError, TransientError } from "./lib/api";
import { canStart, EMPTY_GRID, findConflicts, generatePuzzle, type Grid } from "./lib/sudoku";
import type { DoneEvent, Segment, SolveEvent, StepEvent } from "./lib/types";

// The report is only needed once a run has finished.
const AnalysisReport = lazy(() => import("./components/AnalysisReport"));

interface Run {
  status: RunStatus;
  /** waiting for a free simulator instance */
  busy: boolean;
  puzzle: Grid;
  segments: Segment[];
  steps: StepEvent[];
  done: DoneEvent | null;
  wallSeconds: number;
}

/** Consecutive dropped connections tolerated before giving up. */
const MAX_DROPS = 3;

const toRows = (g: Grid) => Array.from({ length: 9 }, (_, r) => g.slice(r * 9, r * 9 + 9));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function App() {
  const [puzzle, setPuzzle] = useState<Grid>(EMPTY_GRID);
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewIndex, setViewIndex] = useState(0);
  const abort = useRef<AbortController | null>(null);

  const conflicts = useMemo(() => findConflicts(puzzle), [puzzle]);
  const check = useMemo(() => canStart(puzzle, conflicts), [puzzle, conflicts]);
  const running = run?.status === "starting" || run?.status === "running";

  useEffect(() => () => abort.current?.abort(), []);

  // The board follows the newest window while the stream is live, and lands on
  // the final one when the run ends (the last step and "done" arrive together).
  const nSteps = run?.steps.length ?? 0;
  const status = run?.status;
  useEffect(() => {
    if (nSteps) setViewIndex(nSteps - 1);
  }, [status, nSteps]);

  const clearRun = () => {
    abort.current?.abort();
    setRun(null);
    setError(null);
  };

  // The board on screen after a run (the network's digits), if any. Editing a
  // cell then continues from exactly what is shown instead of dropping back to
  // the original givens.
  const shownAfterRun = useRef<Grid | null>(null);

  const onCellChange = useCallback((i: number, v: number) => {
    const shown = shownAfterRun.current;
    clearRun();
    setPuzzle((g) => {
      const base = shown ?? g;
      return !shown && base[i] === v ? g : base.map((x, k) => (k === i ? v : x));
    });
  }, []);

  const onRandom = () => {
    clearRun();
    setPuzzle(generatePuzzle(36));
  };

  const onClear = () => {
    clearRun();
    setPuzzle(EMPTY_GRID);
  };

  const onEvent = (ev: SolveEvent) => {
    // An error mid-run (e.g. the instance ran out of memory) is retried like a dropped connection.
    if (ev.type === "error") throw new TransientError(ev.message);
    setRun((r) => {
      if (!r) return r;
      switch (ev.type) {
        case "meta":
          return { ...r, status: "running", busy: false, segments: [...r.segments, { meta: ev, done: null }] };
        case "step":
          return { ...r, steps: [...r.steps, ev] };
        case "done": {
          const segments = r.segments.map((s, i) => (i === r.segments.length - 1 ? { ...s, done: ev } : s));
          const wallSeconds = r.wallSeconds + ev.wall_seconds;
          return ev.solved ? { ...r, segments, wallSeconds, status: "done", done: ev } : { ...r, segments, wallSeconds };
        }
      }
    });
  };

  /** Runs the network until it solves the puzzle. Each request stops before the
   *  host's function timeout; the next one continues the timeline with a freshly
   *  built network, so there is no overall time limit. */
  const onStart = async () => {
    if (running || !check.ok) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setError(null);
    setViewIndex(0);
    setRun({ status: "starting", busy: false, puzzle, segments: [], steps: [], done: null, wallSeconds: 0 });

    let offset = 0;
    let segment = 0;
    let hints: number[] = [];
    let drops = 0;
    try {
      for (;;) {
        const last: { done: DoneEvent | null; tEnd: number; hints: number[] } = { done: null, tEnd: offset, hints };
        try {
          await solve(
            toRows(puzzle),
            (ev) => {
              if (ev.type === "step") {
                last.tEnd = ev.t_end_ms;
                if (ev.kick?.kind === "hint") last.hints = [...last.hints, ...ev.kick.cells];
              }
              if (ev.type === "done") last.done = ev;
              onEvent(ev);
            },
            ctrl.signal,
            { t_offset_ms: offset, segment, hints },
            () => setRun((r) => (r ? { ...r, busy: true } : r)),
          );
        } catch (e) {
          if (ctrl.signal.aborted || e instanceof SolveError) throw e;
          // Server error or network hiccup: continue from the last window received.
        }
        if (last.done?.solved) return;
        if (last.done?.resume || last.tEnd > offset) drops = 0; // progress was made
        if (!last.done?.resume) {
          if (++drops > MAX_DROPS) throw new SolveError("The connection to the solver keeps failing. Please try again.");
          await sleep(2000 * drops);
        }
        offset = last.tEnd;
        hints = last.done?.hints ?? last.hints;
        segment += 1;
      }
    } catch (e) {
      if (ctrl.signal.aborted) return;
      const msg = e instanceof SolveError ? e.message : `Could not reach the solver: ${(e as Error).message}`;
      setError(msg);
      setRun((r) => (r ? { ...r, status: "error" } : r));
    }
  };

  const step = run?.steps[viewIndex] ?? null;
  shownAfterRun.current = run && !running && step ? step.grid : null;
  const prev = viewIndex > 0 ? run?.steps[viewIndex - 1] : null;
  const live: LiveBoard | null =
    run && step
      ? {
          k: viewIndex,
          grid: step.grid,
          validRows: step.valid_rows,
          validCols: step.valid_cols,
          validBoxes: step.valid_boxes,
          changed: new Set(step.grid.flatMap((d, i) => ((prev ? prev.grid[i] : run.puzzle[i]) !== d ? [i] : []))),
        }
      : null;
  const lastMeta = run?.segments[run.segments.length - 1]?.meta ?? null;

  return (
    <main>
      <h1>Sudoku Solver using Spiking Neural Network</h1>
      <SudokuGrid
        puzzle={run ? run.puzzle : puzzle}
        editable={!running}
        conflicts={run ? undefined : conflicts.cells}
        live={live}
        onChange={onCellChange}
      />
      <Controls
        running={running}
        check={check}
        conflictMessages={run ? [] : conflicts.messages}
        error={error}
        canClear={!!run || puzzle.some((v) => v)}
        onRandom={onRandom}
        onClear={onClear}
        onStart={onStart}
      />
      {run && (
        <LivePanel
          meta={lastMeta}
          steps={run.steps}
          status={run.status}
          busy={run.busy}
          viewIndex={viewIndex}
          onViewIndex={setViewIndex}
        />
      )}
      {run?.done && lastMeta && (
        <Suspense fallback={<p className="note">Loading the results…</p>}>
          <AnalysisReport meta={lastMeta} steps={run.steps} done={run.done} wallSeconds={run.wallSeconds} />
        </Suspense>
      )}
    </main>
  );
}
