import type { MetaEvent, StepEvent } from "../lib/types";
import { ConvergenceChart, RateChart } from "./RunCharts";

export type RunStatus = "starting" | "running" | "done" | "error";

interface Props {
  meta: MetaEvent | null;
  steps: StepEvent[];
  status: RunStatus;
  segment: number;
  viewIndex: number;
  onViewIndex: (i: number) => void;
}

const fmt = (n: number) => n.toLocaleString("en-US");

export default function LivePanel({ meta, steps, status, segment, viewIndex, onViewIndex }: Props) {
  const step = steps[viewIndex] ?? null;
  const latest = steps[steps.length - 1] ?? null;
  const replaying = status === "done" && step !== latest;
  const kicks = steps.filter((s) => s.kick?.kind === "kick").length;
  const resets = steps.filter((s) => s.kick?.kind === "reset" || s.kick?.kind === "hint").length + segment;
  const hints = steps.reduce((n, s) => n + (s.kick?.kind === "hint" ? s.kick.cells.length : 0), 0);

  let statusText = "starting… (loading NEST and building the network)";
  if (status === "running" && meta) statusText = segment ? `solving… (continued in request ${segment + 1})` : "solving…";
  if (status === "done") statusText = `solved at ${fmt(latest?.t_start_ms ?? 0)} ms`;
  if (status === "error") statusText = "stopped with an error";

  return (
    <section className="live" aria-label="Live simulation">
      <div className="live-top">
        <div className="stats">
          <div className="stat-label">Time progressed:</div>
          <div className="stat-value">{fmt(step ? step.t_end_ms : 0)} ms</div>
          <div className="stat-label">Noise rate:</div>
          <div className="stat-value">{meta ? `${fmt(meta.noise_rate)} Hz` : "–"}</div>
          <div className="stat-label">Status:</div>
          <div className={`stat-value status-${status}`} aria-live="polite">
            {statusText}
          </div>
          {step && (
            <dl className="mini-stats">
              <dt>Performance</dt>
              <dd>{step.performance.toFixed(3)}</dd>
              <dt>Conflicting pairs</dt>
              <dd>{step.violations}</dd>
              {step.givens_overridden > 0 && (
                <>
                  <dt>Givens overridden</dt>
                  <dd>{step.givens_overridden}</dd>
                </>
              )}
              <dt>Cells changed</dt>
              <dd>{step.cells_changed}</dd>
              <dt>Spikes in window</dt>
              <dd>{fmt(step.total_spikes)}</dd>
              <dt>Kicks / restarts</dt>
              <dd>
                {kicks} / {resets}
              </dd>
              {hints > 0 && (
                <>
                  <dt>Hinted cells</dt>
                  <dd>{hints}</dd>
                </>
              )}
            </dl>
          )}
        </div>
        <div className="perf">
          <ConvergenceChart steps={steps} cursorMs={replaying ? step?.t_start_ms : null} />
        </div>
      </div>

      {status === "done" && steps.length > 1 && (
        <div className="replay">
          <label htmlFor="replay">
            Replay window {viewIndex + 1} of {steps.length} ({fmt(step?.t_start_ms ?? 0)}–{fmt(step?.t_end_ms ?? 0)} ms)
          </label>
          <input
            id="replay"
            type="range"
            min={0}
            max={steps.length - 1}
            value={viewIndex}
            onChange={(e) => onViewIndex(Number(e.target.value))}
          />
        </div>
      )}

      <RateChart steps={steps} cursorMs={replaying ? step?.t_start_ms : null} />
      <ul className="note legend">
        <li>
          <span className="key key-kick" />
          <span>
            <b>kick</b>: after {meta ? fmt(meta.patience * meta.chunk_ms) : "–"} ms without progress, the winning digits of the
            conflicting cells (then of their whole rows, columns and boxes) are silenced for {meta?.kick_ms ?? "–"} ms
          </span>
        </li>
        <li>
          <span className="key key-reset" />
          <span>
            <b>restart</b>: all membrane potentials are re-randomised
          </span>
        </li>
        {meta?.hint_after_ms != null && (
          <li>
            <span className="key key-hint" />
            <span>
              <b>hint</b>: from {fmt(meta.hint_after_ms)} ms on, a restart also stimulates a few cells with their correct
              digit (teal on the board)
            </span>
          </li>
        )}
        <li>
          <span className="key key-dot" />
          <span>every row, column and box valid, but a given digit overridden</span>
        </li>
      </ul>
    </section>
  );
}
