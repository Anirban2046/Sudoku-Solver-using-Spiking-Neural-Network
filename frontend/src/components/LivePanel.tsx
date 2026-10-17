import type { MetaEvent, StepEvent } from "../lib/types";
import { PerformanceChart } from "./RunCharts";

export type RunStatus = "starting" | "running" | "done" | "error";

interface Props {
  meta: MetaEvent | null;
  steps: StepEvent[];
  status: RunStatus;
  busy: boolean;
  viewIndex: number;
  onViewIndex: (i: number) => void;
}

const fmt = (n: number) => n.toLocaleString("en-US");

export default function LivePanel({ meta, steps, status, busy, viewIndex, onViewIndex }: Props) {
  const step = steps[viewIndex] ?? null;
  const latest = steps[steps.length - 1] ?? null;
  const replaying = status === "done" && step !== latest;

  let statusText = "building the network in NEST Simulator…";
  if (status === "running" && meta) statusText = "solving…";
  if (busy && status !== "done" && status !== "error") statusText = "waiting for a free simulator…";
  if (status === "done") statusText = `solved at ${fmt(latest?.t_start_ms ?? 0)} ms`;
  if (status === "error") statusText = "stopped with an error";

  return (
    <section className="live panel" aria-label="Live simulation">
      <div className="live-top">
        <div className="stats">
          <div className="stat-label">Time progressed</div>
          <div className="stat-value">{fmt(step ? step.t_end_ms : 0)} ms</div>
          <div className="stat-label">Rule conflicts</div>
          <div className="stat-value">{step ? fmt(step.violations) : "–"}</div>
          <div className="stat-label">Status</div>
          <div className={`stat-value status-${status}`} aria-live="polite">
            {statusText}
          </div>
          {step && (
            <dl className="mini-stats">
              <dt>Performance</dt>
              <dd>{step.performance.toFixed(3)}</dd>
              <dt>Spikes in this window</dt>
              <dd>{fmt(step.total_spikes)}</dd>
              <dt>Network firing rate</dt>
              <dd>{step.rate_all_hz.toFixed(1)} Hz</dd>
              <dt>Winning neurons</dt>
              <dd>{step.rate_winners_hz.toFixed(0)} Hz</dd>
            </dl>
          )}
        </div>
        <div className="perf">
          <PerformanceChart steps={steps} cursorMs={replaying ? step?.t_start_ms : null} />
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
    </section>
  );
}
