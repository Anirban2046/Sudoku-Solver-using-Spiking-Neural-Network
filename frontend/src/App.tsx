import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AnalysisReport from "./components/AnalysisReport";
import Controls from "./components/Controls";
import LivePanel, { type RunStatus } from "./components/LivePanel";
import SudokuGrid, { type LiveBoard } from "./components/SudokuGrid";
import { solve, SolveError } from "./lib/api";
import { b64ToU16 } from "./lib/decode";
import { canStart, EMPTY_GRID, findConflicts, generatePuzzle, type Grid } from "./lib/sudoku";
import type { DoneEvent, MetaEvent, RasterWindow, SolveEvent, StepEvent } from "./lib/types";

interface Run {
  status: RunStatus;
  puzzle: Grid;
  meta: MetaEvent | null;
  steps: StepEvent[];
  raster: RasterWindow[];
  done: DoneEvent | null;
}

const toRows = (g: Grid) => Array.from({ length: 9 }, (_, r) => g.slice(r * 9, r * 9 + 9));

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

  // While the stream is live, the board follows the newest window.
  const nSteps = run?.steps.length ?? 0;
  useEffect(() => {
    if (running && nSteps) setViewIndex(nSteps - 1);
  }, [running, nSteps]);

  const clearRun = () => {
    abort.current?.abort();
    setRun(null);
    setError(null);
  };

  const onCellChange = useCallback((i: number, v: number) => {
    clearRun();
    setPuzzle((g) => (g[i] === v ? g : g.map((x, k) => (k === i ? v : x))));
  }, []);

  const onRandom = () => {
    clearRun();
    setPuzzle(generatePuzzle(36));
  };

  const onEvent = (ev: SolveEvent) => {
    if (ev.type === "error") throw new SolveError(ev.message);
    setRun((r) => {
      if (!r) return r;
      switch (ev.type) {
        case "meta":
          return { ...r, status: "running", meta: ev };
        case "step": {
          const win: RasterWindow = {
            k: ev.k,
            tStart: ev.t_start_ms,
            pop: b64ToU16(ev.raster.pop_b64),
            t: b64ToU16(ev.raster.t_b64),
            grid: ev.grid,
          };
          const light = { ...ev, raster: { pop_b64: "", t_b64: "" }, pop_counts_b64: "" };
          return { ...r, steps: [...r.steps, light], raster: [...r.raster, win] };
        }
        case "done":
          return { ...r, status: "done", done: ev };
      }
    });
  };

  const onStart = async () => {
    if (running || !check.ok) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setError(null);
    setViewIndex(0);
    setRun({ status: "starting", puzzle, meta: null, steps: [], raster: [], done: null });
    let finished = false;
    try {
      await solve(
        toRows(puzzle),
        (ev) => {
          if (ev.type === "done") finished = true;
          onEvent(ev);
        },
        ctrl.signal,
      );
      if (!finished) throw new SolveError("The connection closed before the run finished.");
    } catch (e) {
      if (ctrl.signal.aborted) return;
      const msg = e instanceof SolveError ? e.message : `Could not reach the solver: ${(e as Error).message}`;
      setError(msg);
      setRun((r) => (r ? { ...r, status: "error" } : r));
    }
  };

  const step = run?.steps[viewIndex] ?? null;
  const prev = viewIndex > 0 ? run?.steps[viewIndex - 1] : null;
  const live: LiveBoard | null =
    run && step
      ? {
          k: step.k,
          grid: step.grid,
          validRows: step.valid_rows,
          validCols: step.valid_cols,
          validBoxes: step.valid_boxes,
          tie: step.tie,
          silent: step.silent,
          changed: new Set(
            step.grid.flatMap((d, i) => ((prev ? prev.grid[i] : run.puzzle[i]) !== d ? [i] : [])),
          ),
        }
      : null;

  return (
    <main>
      <h1>Sudoku Solver using SNN</h1>
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
        onRandom={onRandom}
        onStart={onStart}
      />
      {run && !running && (
        <p className="edit-hint">Type in any cell to edit the puzzle (this clears the results), or press Randomly Initialize.</p>
      )}
      {run && (
        <LivePanel
          meta={run.meta}
          steps={run.steps}
          raster={run.raster}
          status={run.status}
          solved={run.done?.solved ?? null}
          viewIndex={viewIndex}
          onViewIndex={setViewIndex}
        />
      )}
      {run?.done && run.meta && run.done.analysis.outcome && (
        <AnalysisReport analysis={run.done.analysis} meta={run.meta} raster={run.raster} />
      )}
    </main>
  );
}
