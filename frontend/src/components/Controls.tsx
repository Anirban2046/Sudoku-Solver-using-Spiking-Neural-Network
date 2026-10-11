import type { StartCheck } from "../lib/sudoku";

interface Props {
  running: boolean;
  check: StartCheck;
  conflictMessages: string[];
  error?: string | null;
  onRandom: () => void;
  onStart: () => void;
}

export default function Controls({ running, check, conflictMessages, error, onRandom, onStart }: Props) {
  const startDisabled = running || !check.ok;
  const hint = !check.ok ? check.reason : check.note;
  return (
    <div className="controls">
      <div className="buttons">
        <button type="button" onClick={onRandom} disabled={running}>
          Randomly Initialize
        </button>
        <button
          type="button"
          className="primary"
          onClick={onStart}
          disabled={startDisabled}
          title={!running && hint ? hint : undefined}
          aria-describedby={hint ? "start-hint" : undefined}
        >
          {running ? "Solving…" : "Start Solving"}
        </button>
      </div>
      {hint && !running && (
        <p id="start-hint" className="hint">
          {hint}
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
