import type { MetaEvent, RasterWindow, StepEvent } from "../lib/types";
import PerformanceChart from "./PerformanceChart";
import SpikeRaster from "./SpikeRaster";

export type RunStatus = "starting" | "running" | "done" | "error";

interface Props {
  meta: MetaEvent | null;
  steps: StepEvent[];
  raster: RasterWindow[];
  status: RunStatus;
  solved: boolean | null;
  viewIndex: number;
  onViewIndex: (i: number) => void;
}

const fmt = (n: number) => n.toLocaleString("en-US");

export default function LivePanel({ meta, steps, raster, status, solved, viewIndex, onViewIndex }: Props) {
  const step = steps[viewIndex] ?? null;
  const latest = steps[steps.length - 1] ?? null;
  const replaying = status === "done" && step !== latest;
  const timeMs = step ? step.t_end_ms : 0;

  let statusText = "starting… (loading NEST and building the network)";
  if (status === "running") statusText = meta ? "solving…" : statusText;
  if (status === "done") statusText = solved ? `solved at ${fmt(latest?.t_start_ms ?? 0)} ms` : "no valid solution within the time limit";
  if (status === "error") statusText = "stopped with an error";

  const points = steps.map((s) => ({ x: s.t_start_ms, y: s.performance }));
  const rates = meta ? steps.map((s) => ({ tEnd: s.t_end_ms, hz: s.total_spikes / (meta.n_neurons * (meta.chunk_ms / 1000)) })) : [];

  return (
    <section className="live" aria-label="Live simulation">
      <div className="live-top">
        <div className="stats">
          <div className="stat-label">Time progressed:</div>
          <div className="stat-value">{fmt(timeMs)} ms</div>
          <div className="stat-label">Noise rate:</div>
          <div className="stat-value">{meta ? `${fmt(meta.noise_rate)} Hz` : "–"}</div>
          <div className="stat-label">Status:</div>
          <div className={`stat-value status-${status}${solved ? " solved" : ""}`} aria-live="polite">
            {statusText}
          </div>
          {step && (
            <dl className="mini-stats">
              <dt>Performance</dt>
              <dd>{step.performance.toFixed(3)}</dd>
              <dt>Rule violations</dt>
              <dd>{step.violations}</dd>
              <dt>Cells changed</dt>
              <dd>{step.cells_changed}</dd>
              <dt>Spikes in window</dt>
              <dd>{fmt(step.total_spikes)}</dd>
              <dt>Wall time / window</dt>
              <dd>{step.wall_seconds.toFixed(2)} s</dd>
            </dl>
          )}
        </div>
        <div className="perf">
          <PerformanceChart
            points={points}
            maxSimMs={meta?.max_sim_ms ?? 20000}
            noiseRate={meta?.noise_rate ?? 0}
            cursorMs={replaying ? step?.t_start_ms : null}
          />
        </div>
      </div>

      {status === "done" && steps.length > 1 && (
        <div className="replay">
          <label htmlFor="replay">
            Replay window {viewIndex + 1} of {steps.length} ({fmt(step?.t_start_ms ?? 0)}–{fmt(timeMs)} ms)
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

      {meta && (
        <>
          <h2 className="panel-title">
            Spike raster: last {fmt(2000)} ms, {meta.raster_neurons_per_pop} neuron(s) per population, {fmt(meta.n_populations)} populations
          </h2>
          <SpikeRaster
            windows={raster}
            endMs={Math.max(timeMs, 2000)}
            spanMs={2000}
            chunkMs={meta.chunk_ms}
            rates={rates}
            label="Live spike raster of the last two seconds"
          />
        </>
      )}
    </section>
  );
}
