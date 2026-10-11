import { useRef, type KeyboardEvent } from "react";
import { boxOf, colOf, rowOf, type Grid } from "../lib/sudoku";

export interface LiveBoard {
  k: number;
  grid: number[];
  validRows: boolean[];
  validCols: boolean[];
  validBoxes: boolean[];
  changed: Set<number>;
  /** cells stimulated with their correct digit as a last-resort hint */
  hinted: Set<number>;
}

interface Props {
  puzzle: Grid;
  editable: boolean;
  conflicts?: Set<number>;
  live?: LiveBoard | null;
  onChange?: (index: number, value: number) => void;
}

const MOVES: Record<string, [number, number]> = {
  ArrowUp: [-1, 0],
  ArrowDown: [1, 0],
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
};

/** The 9x9 board. Editable inputs before a run; afterwards a read-only view of
 *  the network's readout drawn with the rules of helpers_sudoku.plot_field. */
export default function SudokuGrid({ puzzle, editable, conflicts, live, onChange }: Props) {
  const refs = useRef<(HTMLInputElement | null)[]>([]);

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>, i: number) => {
    if (!editable || !onChange) return;
    const move = MOVES[e.key];
    if (move) {
      e.preventDefault();
      const r = Math.min(8, Math.max(0, rowOf(i) + move[0]));
      const c = Math.min(8, Math.max(0, colOf(i) + move[1]));
      refs.current[r * 9 + c]?.focus();
      return;
    }
    if (/^[1-9]$/.test(e.key)) {
      e.preventDefault();
      onChange(i, Number(e.key));
    } else if (e.key === "0" || e.key === "Backspace" || e.key === "Delete") {
      e.preventDefault();
      onChange(i, 0);
    } else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
    }
  };

  // Fallback for virtual keyboards that don't send key events.
  const onInput = (value: string, i: number) => {
    if (!editable || !onChange) return;
    const ch = value.slice(-1);
    onChange(i, /^[1-9]$/.test(ch) ? Number(ch) : 0);
  };

  return (
    <div className="board-wrap">
      <div className={`board${live ? " board-live" : ""}`} role="group" aria-label="Sudoku grid">
        {Array.from({ length: 81 }, (_, i) => {
          const r = rowOf(i);
          const c = colOf(i);
          const b = boxOf(i);
          const given = puzzle[i] !== 0;
          const value = live ? live.grid[i] : puzzle[i];
          const classes = ["cell"];
          if (((Math.floor(b / 3) + (b % 3)) & 1) === 0) classes.push("shade");
          if (c % 3 === 2 && c < 8) classes.push("edge-r");
          if (r % 3 === 2 && r < 8) classes.push("edge-b");
          if (conflicts?.has(i)) classes.push("conflict");
          if (given) classes.push(live && live.grid[i] !== puzzle[i] ? "overridden" : "given");
          else if (live?.hinted.has(i)) classes.push("hinted");
          else if (live) classes.push("network");
          return (
            <div key={i} className={classes.join(" ")}>
              <input
                ref={(el) => {
                  refs.current[i] = el;
                }}
                inputMode="numeric"
                maxLength={1}
                autoComplete="off"
                aria-label={`row ${r + 1} column ${c + 1}`}
                aria-invalid={conflicts?.has(i) || undefined}
                readOnly={!editable}
                tabIndex={editable ? 0 : -1}
                value={value ? String(value) : ""}
                onKeyDown={(e) => onKeyDown(e, i)}
                onChange={(e) => onInput(e.target.value, i)}
              />
              {live?.changed.has(i) && <span key={live.k} className="flash" aria-hidden="true" />}
            </div>
          );
        })}
        {live &&
          live.validRows.map((ok, r) =>
            ok ? null : (
              <span key={`row${r}`} aria-hidden="true">
                <span className="bar bar-h bar-left" style={{ top: `${((r + 0.5) * 100) / 9}%` }} />
                <span className="bar bar-h bar-right" style={{ top: `${((r + 0.5) * 100) / 9}%` }} />
              </span>
            ),
          )}
        {live &&
          live.validCols.map((ok, c) =>
            ok ? null : (
              <span key={`col${c}`} aria-hidden="true">
                <span className="bar bar-v bar-top" style={{ left: `${((c + 0.5) * 100) / 9}%` }} />
                <span className="bar bar-v bar-bottom" style={{ left: `${((c + 0.5) * 100) / 9}%` }} />
              </span>
            ),
          )}
        {live &&
          live.validBoxes.map((ok, b) =>
            ok ? null : (
              // Upstream drew this at (3*j, 3*j); the box is at row 3*i, column 3*j.
              <span
                key={`box${b}`}
                className="box-frame"
                aria-hidden="true"
                style={{ top: `${(Math.floor(b / 3) * 100) / 3}%`, left: `${((b % 3) * 100) / 3}%` }}
              />
            ),
          )}
      </div>
    </div>
  );
}
