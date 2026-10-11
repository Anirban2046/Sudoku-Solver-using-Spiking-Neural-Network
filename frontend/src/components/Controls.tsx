import type { StartCheck } from "../lib/sudoku";

interface Props {
  running: boolean;
  check: StartCheck;
  conflictMessages: string[];
  error?: string | null;
  canClear: boolean;
  onRandom: () => void;
  onClear: () => void;
  onStart: () => void;
}

export default function Controls({ running, check, conflictMessages, error, canClear, onRandom, onClear, onStart }: Props) {
  const startDisabled = running || !check.ok;
  const why = !check.ok ? check.reason : check.note;
  return (
    <div className="controls">
      <div className="buttons">
        <button type="button" onClick={onRandom} disabled={running}>
          Randomly Initialize
        </button>
        <button type="button" onClick={onClear} disabled={running || !canClear}>
          Clear Grid
        </button>
        <button
          type="button"
          className="primary"
          onClick={onStart}
          disabled={startDisabled}
          title={!running && why ? why : undefined}
          aria-describedby={why ? "start-why" : undefined}
        >
          {running ? "Solving…" : "Start Solving"}
        </button>
      </div>
      {why && !running && (
        <p id="start-why" className="start-why">
          {why}
        </p>
      )}
      {conflictMessages.length > 0 && (
        <ul className="conflicts" aria-live="polite">
          {conflictMessages.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
