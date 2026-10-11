import { useMemo, useState, type ReactNode } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import type { ChartConfiguration } from "chart.js/auto";
import type { Analysis, MetaEvent, RasterWindow } from "../lib/types";
import ChartCanvas, { COLORS } from "./ChartCanvas";
import SpikeRaster from "./SpikeRaster";

interface Props {
  analysis: Analysis;
  meta: MetaEvent;
  raster: RasterWindow[];
}

const fmt = (n: number | null | undefined, digits?: number) =>
  n == null
    ? "–"
    : n.toLocaleString("en-US", digits == null ? undefined : { minimumFractionDigits: digits, maximumFractionDigits: digits });

function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = useMemo(() => katex.renderToString(tex, { throwOnError: false, displayMode: display }), [tex, display]);
  return <span className={display ? "tex-display" : "tex"} dangerouslySetInnerHTML={{ __html: html }} />;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="section" open>
      <summary>
        <h3>{title}</h3>
      </summary>
      <div className="section-body">{children}</div>
    </details>
  );
}

const cellName = (i: number) => `r${Math.floor(i / 9) + 1} c${(i % 9) + 1}`;

export default function AnalysisReport({ analysis, meta, raster }: Props) {
  const { outcome, network, neuron, input, readout, firing, convergence, classical } = analysis;
  return (
    <section className="report" aria-label="Analysis">
      <h2>Analysis of this run</h2>
      <Section title="1. Outcome">
        <OutcomeCard outcome={outcome} classical={classical} />
      </Section>
      <NetworkSection network={network} meta={meta} />
      <NeuronSection neuron={neuron} />
      <InputSection input={input} />
      <ReadoutSection readout={readout} firing={firing} />
      <ConvergenceSection readout={readout} convergence={convergence} />
      <FiringSection firing={firing} />
      <Section title="8. Full raster">
        <p>
          Every spike of the {meta.raster_neurons_per_pop} representative neuron(s) per population over the whole run (
          {fmt(outcome.sim_ms_total)} ms). Each band of 81 rows is one row of the board; within it, every 9 rows are the 9
          digits of one cell. The line below is the rate of all {fmt(meta.n_neurons)} neurons.
        </p>
        <SpikeRaster
          windows={raster}
          endMs={outcome.sim_ms_total}
          spanMs={outcome.sim_ms_total}
          chunkMs={meta.chunk_ms}
          rates={firing.t_end_ms.map((t, i) => ({ tEnd: t, hz: firing.rate_all_hz[i] }))}
          label="Spike raster of the whole run"
        />
      </Section>
      <Glossary />
    </section>
  );
}

function OutcomeCard({ outcome: o, classical: c }: Pick<Analysis, "outcome" | "classical">) {
  const verdict = {
    match: "The network's answer is identical to the backtracking solver's solution.",
    valid_but_different: `The network's answer is valid but different from the backtracking solution (the puzzle has ${c.n_solutions_label} solutions).`,
    not_solved: `The network did not reach a valid grid. A backtracking solver finds ${c.n_solutions_label} solution(s) for this puzzle.`,
    no_classical_solution: "The backtracking solver finds no solution for this puzzle.",
  }[c.verdict];
  return (
    <div className={`outcome ${o.solved ? "ok" : "fail"}`}>
      <div className="outcome-head">{o.solved ? "Solved" : "Not solved"}</div>
      <dl className="kv">
        {o.solved ? (
          <>
            <dt>Simulated time to solution</dt>
            <dd>{fmt(o.solved_at_ms)} ms</dd>
          </>
        ) : (
          <>
            <dt>Best performance reached</dt>
            <dd>
              {fmt(o.best_performance, 3)} at {fmt(o.best_at_ms)} ms
            </dd>
            <dt>Still invalid at the end</dt>
            <dd>
              rows [{o.final_invalid_rows.join(", ") || "none"}], columns [{o.final_invalid_cols.join(", ") || "none"}],
              boxes [{o.final_invalid_boxes.join(", ") || "none"}]; {o.final_violations} conflicting cell pairs
            </dd>
          </>
        )}
        <dt>Simulated in total</dt>
        <dd>
          {fmt(o.sim_ms_total)} ms of {fmt(o.max_sim_ms)} ms allowed
        </dd>
        <dt>Wall time</dt>
        <dd>
          {fmt(o.wall_seconds_total, 1)} s (network build {fmt(o.build_seconds, 1)} s, {o.threads} thread(s), NEST {o.nest_version})
        </dd>
        <dt>Speed</dt>
        <dd>{fmt(o.speed_ratio_sim_ms_per_wall_s, 1)} simulated ms per wall-clock second</dd>
        <dt>Given digits kept</dt>
        <dd>{o.givens_kept ? "yes" : "no — the network overrode at least one given"}</dd>
        <dt>Random seed</dt>
        <dd>
          <code>{o.seed}</code>
        </dd>
        <dt>Classical check</dt>
        <dd>{verdict}</dd>
      </dl>
    </div>
  );
}

function TargetsDiagram({ t }: { t: Analysis["network"]["inhibitory_target_populations"] }) {
  // Source population: the digit-5 population of cell r5 c5 (centre of the board).
  const src = { r: 4, c: 4 };
  const s = 22;
  const kind = (r: number, c: number) => {
    if (r === src.r && c === src.c) return "src";
    if (r === src.r) return "row";
    if (c === src.c) return "col";
    if (Math.floor(r / 3) === Math.floor(src.r / 3) && Math.floor(c / 3) === Math.floor(src.c / 3)) return "box";
    return "";
  };
  const fill: Record<string, string> = { src: COLORS.ink, row: COLORS.blue, col: COLORS.green, box: COLORS.orange, "": "#fff" };
  return (
    <figure className="diagram">
      <svg viewBox={`0 0 ${9 * s + 130} ${9 * s + 4}`} width="100%" style={{ maxWidth: 420 }} role="img"
        aria-label="Inhibitory targets of one population">
        <g transform="translate(2,2)">
          {Array.from({ length: 81 }, (_, i) => {
            const r = Math.floor(i / 9);
            const c = i % 9;
            const k = kind(r, c);
            return <rect key={i} x={c * s} y={r * s} width={s} height={s} fill={fill[k]} fillOpacity={k && k !== "src" ? 0.75 : 1} stroke="#bbb" strokeWidth={0.5} />;
          })}
          {[0, 3, 6, 9].map((v) => (
            <g key={v}>
              <line x1={v * s} y1={0} x2={v * s} y2={9 * s} stroke="#222" strokeWidth={1.5} />
              <line x1={0} y1={v * s} x2={9 * s} y2={v * s} stroke="#222" strokeWidth={1.5} />
            </g>
          ))}
          <text x={src.c * s + s / 2} y={src.r * s + s / 2 + 4} textAnchor="middle" fill="#fff" fontSize={12}>5</text>
          {/* the 9 digit populations of the source cell */}
          <g transform={`translate(${9 * s + 18}, ${3 * s})`}>
            {Array.from({ length: 9 }, (_, d) => (
              <g key={d}>
                <rect x={(d % 3) * s} y={Math.floor(d / 3) * s} width={s} height={s}
                  fill={d === 4 ? COLORS.ink : COLORS.purple} fillOpacity={d === 4 ? 1 : 0.7} stroke="#fff" />
                <text x={(d % 3) * s + s / 2} y={Math.floor(d / 3) * s + s / 2 + 4} textAnchor="middle" fill="#fff" fontSize={11}>{d + 1}</text>
              </g>
            ))}
            <text x={1.5 * s} y={-6} textAnchor="middle" fontSize={10} fill="#333">cell r5 c5</text>
          </g>
        </g>
      </svg>
      <figcaption>
        <span className="swatch" style={{ background: COLORS.ink }} /> source: digit 5 in r5 c5{" "}
        <span className="swatch" style={{ background: COLORS.blue }} /> same row ({t.row}){" "}
        <span className="swatch" style={{ background: COLORS.green }} /> same column ({t.column}){" "}
        <span className="swatch" style={{ background: COLORS.orange }} /> rest of box ({t.box_only}){" "}
        <span className="swatch" style={{ background: COLORS.purple }} /> other digits of the cell ({t.same_cell}) = {t.total} target populations
      </figcaption>
    </figure>
  );
}

function NetworkSection({ network: n, meta }: { network: Analysis["network"]; meta: MetaEvent }) {
  const c = n.connections;
  return (
    <Section title="2. How the network is built">
      <p>
        Each of the 81 cells has one population of {n.pop_size} neurons for each digit 1–9, giving {fmt(n.n_populations)}{" "}
        populations and {fmt(n.n_neurons)} neurons. A population being active means “this cell holds this digit”. The rules
        of Sudoku are wired in as inhibition: every population inhibits the populations of the same digit in its row,
        column and box, and the other 8 digits of its own cell (weight {n.weights.inhibitory}, delay {n.weights.delay_ms} ms).
        Background noise makes the network explore, and the inhibition makes states that break a rule unstable.
      </p>
      <TargetsDiagram t={n.inhibitory_target_populations} />
      <table className="table">
        <thead>
          <tr>
            <th>Connection type</th>
            <th className="num">Count</th>
            <th className="num">Weight</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Inhibitory (neuron → neuron)</td><td className="num">{fmt(c.inhibitory)}</td><td className="num">{n.weights.inhibitory}</td></tr>
          <tr><td>Noise (Poisson generator → neuron)</td><td className="num">{fmt(c.noise)}</td><td className="num">{n.weights.noise}</td></tr>
          <tr><td>Stimulus (Poisson generator → neuron)</td><td className="num">{fmt(c.stimulus)}</td><td className="num">0 or {n.weights.stimulus}</td></tr>
          <tr><td className="indent">… of which switched on (given digits)</td><td className="num">{fmt(c.stimulus_active)}</td><td className="num">{n.weights.stimulus}</td></tr>
          <tr><td>Recorder (neuron → spike recorder)</td><td className="num">{fmt(c.recorder)}</td><td className="num">–</td></tr>
          <tr className="total"><td>Total</td><td className="num">{fmt(c.total)}</td><td /></tr>
        </tbody>
      </table>
      <p>
        Fan-out: each neuron inhibits {fmt(n.fan_out_inhibitory)} neurons ({n.inhibitory_target_populations.total} populations ×{" "}
        {n.pop_size}). Fan-in: each neuron receives {fmt(n.fan_in.inhibitory)} inhibitory + {n.fan_in.noise} noise +{" "}
        {n.fan_in.stimulus} stimulus = {fmt(n.fan_in.total)} connections. Built in {fmt(meta.build_seconds, 2)} s.
      </p>
    </Section>
  );
}

const PARAM_INFO: Record<string, [string, string]> = {
  C_m: ["C_m", "membrane capacitance"],
  I_e: ["I_e", "constant bias current"],
  tau_m: ["\\tau_m", "membrane time constant (ms)"],
  t_ref: ["t_{ref}", "refractory period (ms)"],
  tau_syn_ex: ["\\tau_{syn,ex}", "excitatory synaptic time constant (ms)"],
  tau_syn_in: ["\\tau_{syn,in}", "inhibitory synaptic time constant (ms)"],
  V_reset: ["V_{reset}", "reset potential (mV)"],
  E_L: ["E_L", "resting potential (mV)"],
  V_th: ["V_{th}", "spike threshold (mV)"],
  V_m_init: ["V_m(0)", "initial potential, uniform in range (mV)"],
};

function NeuronSection({ neuron }: { neuron: Analysis["neuron"] }) {
  const d = neuron.derived;
  const f = d.formulas;
  const kernel = useMemo<ChartConfiguration>(
    () => ({
      type: "line",
      data: {
        labels: d.kernel_t_ms,
        datasets: (["noise", "stimulus", "inhibitory"] as const).map((k, i) => ({
          label: `${k} (w = ${d.psp[k].weight})`,
          data: d.psp[k].kernel_mv,
          borderColor: [COLORS.grey, COLORS.green, COLORS.red][i],
          borderWidth: 2,
          pointRadius: 0,
        })),
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", title: { display: true, text: "time after spike (ms)" } },
          y: { title: { display: true, text: "ΔV (mV)" } },
        },
      },
    }),
    [d],
  );
  return (
    <Section title="3. Neuron model">
      <p>
        Every neuron is a leaky integrate-and-fire neuron with exponentially decaying synaptic currents (NEST model{" "}
        <code>{neuron.model}</code>). Below threshold:
      </p>
      <Tex display tex={String.raw`\frac{dV}{dt} = -\frac{V - E_L}{\tau_m} + \frac{I_{syn}(t) + I_e}{C_m},\qquad \frac{dI_{syn}}{dt} = -\frac{I_{syn}}{\tau_{syn}}`} />
      <p>
        When <Tex tex="V" /> reaches <Tex tex="V_{th}" /> the neuron spikes, is reset to <Tex tex="V_{reset}" /> and held there
        for <Tex tex="t_{ref}" />. Each incoming spike of weight <Tex tex="w" /> adds <Tex tex="w" /> to{" "}
        <Tex tex="I_{syn}" />. (The parameters were copied from PyNN in nF/nA; since capacitance and currents share the same
        scale factor, the voltage dynamics are unchanged.)
      </p>
      <table className="table">
        <thead>
          <tr><th>Parameter</th><th>Meaning</th><th className="num">Value</th></tr>
        </thead>
        <tbody>
          {Object.entries(neuron.params).map(([k, v]) => (
            <tr key={k}>
              <td><Tex tex={PARAM_INFO[k]?.[0] ?? k} /></td>
              <td>{PARAM_INFO[k]?.[1] ?? ""}</td>
              <td className="num">{Array.isArray(v) ? `[${v.join(", ")}]` : v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>Derived quantities</h4>
      <table className="table derived">
        <tbody>
          <tr>
            <td><Tex tex={f.V_inf} /></td>
            <td className="num">{fmt(d.V_inf_mv, 2)} mV</td>
            <td>{d.V_inf_above_threshold ? "above threshold: the bias alone makes every neuron fire" : "below threshold"}</td>
          </tr>
          <tr>
            <td><Tex tex={f.tonic_isi} /></td>
            <td className="num">{fmt(d.tonic_isi_ms, 2)} ms</td>
            <td>interval between spikes from the bias alone</td>
          </tr>
          <tr>
            <td><Tex tex={f.tonic_rate} /></td>
            <td className="num">{fmt(d.tonic_rate_hz, 1)} Hz</td>
            <td>tonic rate without any input</td>
          </tr>
          <tr>
            <td><Tex tex={f.t_peak} /></td>
            <td className="num">{fmt(d.t_peak_ms, 2)} ms</td>
            <td>time to the peak of a postsynaptic potential</td>
          </tr>
          {(["inhibitory", "stimulus", "noise"] as const).map((k) => (
            <tr key={k}>
              <td>{k === "inhibitory" ? <Tex tex={f.psp} /> : null}</td>
              <td className="num">{d.psp[k].peak_mv > 0 ? "+" : ""}{fmt(d.psp[k].peak_mv, 2)} mV</td>
              <td>peak PSP of one {k} spike (w = {d.psp[k].weight})</td>
            </tr>
          ))}
          <tr>
            <td><Tex tex={f.mean_current} /></td>
            <td className="num">{fmt(d.mean_current_noise, 2)}</td>
            <td>mean input current from noise (every neuron)</td>
          </tr>
          <tr>
            <td />
            <td className="num">{fmt(d.mean_current_stimulus, 2)}</td>
            <td>mean input current from the stimulus (given digits only)</td>
          </tr>
        </tbody>
      </table>
      <h4>Postsynaptic potential kernels</h4>
      <ChartCanvas config={kernel} height={220} label="Postsynaptic potential kernels" />
    </Section>
  );
}

function InputSection({ input }: { input: Analysis["input"] }) {
  const byCell = new Map(input.givens.map((g) => [(g.row - 1) * 9 + g.col - 1, g]));
  return (
    <Section title="4. How the puzzle is fed in">
      <p>
        Every population has its own Poisson generator at {fmt(input.stim_rate_hz)} Hz. Its weight is{" "}
        {input.stim_weight_other} unless the cell is given, in which case the weight into the given digit's population is set
        to {input.stim_weight_given} ({fmt(input.active_stimulus_connections)} connections switched on for {input.n_givens} givens). That
        extra drive makes the given digit win its cell. All neurons also receive independent noise at{" "}
        {fmt(input.noise_rate_hz)} Hz.
      </p>
      <p>
        Population index = row × 81 + column × 9 + (digit − 1), counting rows and columns from 0. Each given cell below shows
        its digit and the index of the stimulated population.
      </p>
      <div className="mini-board input-board">
        {Array.from({ length: 81 }, (_, i) => {
          const g = byCell.get(i);
          return (
            <div key={i} className={`mb-cell${g ? " has" : ""}`} title={g ? `${cellName(i)}: digit ${g.digit}, population ${g.pop}, ${g.rate_hz} Hz` : cellName(i)}>
              {g && (
                <>
                  <span className="mb-digit">{g.digit}</span>
                  <span className="mb-sub">{g.pop}</span>
                </>
              )}
            </div>
          );
        })}
      </div>
    </Section>
  );
}

function ReadoutSection({ readout, firing }: { readout: Analysis["readout"]; firing: Analysis["firing"] }) {
  const [cell, setCell] = useState(() => {
    // start with the least decided cell
    const m = firing.decision_margin;
    return m.indexOf(Math.min(...m));
  });
  const config = useMemo<ChartConfiguration>(
    () => ({
      type: "bar",
      data: {
        labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        datasets: [
          { label: `last window (${readout.chunk_ms} ms)`, data: firing.cell_counts_last[cell], backgroundColor: COLORS.blue, yAxisID: "y" },
          { label: "whole run", data: firing.cell_counts_total[cell], backgroundColor: COLORS.grey, yAxisID: "y2" },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "digit" } },
          y: { position: "left", title: { display: true, text: "spikes, last window" }, beginAtZero: true },
          y2: { position: "right", title: { display: true, text: "spikes, whole run" }, beginAtZero: true, grid: { drawOnChartArea: false } },
        },
      },
    }),
    [cell, firing, readout.chunk_ms],
  );
  const last = readout.trace[readout.trace.length - 1];
  return (
    <Section title="5. How the answer is read out">
      <p>
        Every {readout.chunk_ms} ms the spikes of each population are counted. Rule: {readout.rule}.
      </p>
      <Tex display tex={String.raw`\hat d_{r,c} = \arg\max_{d}\ \#\text{spikes}_{r,c,d}(\text{window}),\qquad \text{performance} = \frac{\#\text{valid boxes}+\#\text{valid rows}+\#\text{valid cols}}{27}`} />
      <p>
        In the last window, {last.ties} cell(s) had a tie and {last.silent} cell(s) were silent. The decision margin of a cell
        is (top − second) / top of its spike counts; its mean over all cells in the last window is{" "}
        <strong>{fmt(firing.decision_margin_mean, 3)}</strong>. Click a cell to see its digit spike counts.
      </p>
      <div className="readout-row">
        <div className="mini-board pick-board" role="group" aria-label="Pick a cell">
          {last.grid.map((d, i) => (
            <button
              key={i}
              type="button"
              className={`mb-cell${i === cell ? " sel" : ""}`}
              onClick={() => setCell(i)}
              aria-label={`${cellName(i)}, digit ${d}`}
              aria-pressed={i === cell}
              style={i === cell ? undefined : { background: `rgba(217, 130, 43, ${(0.55 * (1 - firing.decision_margin[i])).toFixed(3)})` }}
            >
              {d}
            </button>
          ))}
        </div>
        <div className="readout-chart">
          <div className="chart-caption">
            {cellName(cell)}: decision margin {fmt(firing.decision_margin[cell], 3)}
          </div>
          <ChartCanvas config={config} height={240} label={`Digit spike counts for ${cellName(cell)}`} />
        </div>
      </div>
      <p className="note">Orange shading in the picker marks cells with a small decision margin (more orange = less decided).</p>
    </Section>
  );
}

function ConvergenceSection({ readout, convergence }: { readout: Analysis["readout"]; convergence: Analysis["convergence"] }) {
  const tr = readout.trace;
  const energy = useMemo<ChartConfiguration>(
    () => ({
      type: "line",
      data: {
        labels: tr.map((w) => w.t_start_ms),
        datasets: [
          { label: "performance", data: tr.map((w) => w.performance), borderColor: COLORS.blue, pointRadius: 0, borderWidth: 2, yAxisID: "y" },
          { label: "rule violations (conflicting pairs)", data: tr.map((w) => w.violations), borderColor: COLORS.red, pointRadius: 0, borderWidth: 2, yAxisID: "y2" },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "simulation time (ms)" } },
          y: { min: 0, max: 1, title: { display: true, text: "performance" } },
          y2: { position: "right", beginAtZero: true, title: { display: true, text: "violations" }, grid: { drawOnChartArea: false } },
        },
      },
    }),
    [tr],
  );
  const changes = useMemo<ChartConfiguration>(
    () => ({
      type: "bar",
      data: {
        labels: tr.map((w) => w.t_start_ms),
        datasets: [{ label: "cells changed in window", data: tr.map((w) => w.cells_changed), backgroundColor: COLORS.orange }],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "window start (ms)" } }, y: { beginAtZero: true, title: { display: true, text: "cells changed" } } },
      },
    }),
    [tr],
  );
  const maxLock = Math.max(...convergence.lock_in_ms);
  // Kept at or below 0.5 so the ink-coloured text stays at >= 4.5:1 contrast.
  const alpha = (ms: number) => (maxLock ? 0.04 + (0.46 * ms) / maxLock : 0.04);
  const first = convergence.lock_in_order.slice(0, 5).map(cellName).join(", ");
  const lastCells = convergence.lock_in_order.slice(-5).map(cellName).join(", ");
  return (
    <Section title="6. Convergence">
      <p>
        The number of conflicting cell pairs works like the energy of a Hopfield or Boltzmann network: the network relaxes
        towards zero conflicts, and a valid grid has none.
      </p>
      <Tex display tex={String.raw`E = \sum_{\{i,j\}\ \text{share a row, column or box}} [\hat d_i = \hat d_j]`} />
      <ChartCanvas config={energy} height={240} label="Performance and rule violations over time" />
      <ChartCanvas config={changes} height={180} label="Cells changed per window" />
      <h4>When each cell locked in</h4>
      <p>
        Time at which each cell last changed its digit (darker = later). First to settle: {first}. Last: {lastCells}.
      </p>
      <div className="mini-board heat-board">
        {convergence.lock_in_ms.map((ms, i) => (
          <div
            key={i}
            className={`mb-cell${convergence.is_given[i] ? " given" : ""}`}
            style={{ background: `rgba(31, 79, 138, ${alpha(ms)})` }}
            title={`${cellName(i)}: locked in at ${ms} ms, changed ${convergence.changes_per_cell[i]} times`}
          >
            <span className="mb-sub">{fmt(ms)}</span>
          </div>
        ))}
      </div>
      <p className="note">Numbers are milliseconds; underlined cells are givens.</p>
    </Section>
  );
}

function FiringSection({ firing: f }: { firing: Analysis["firing"] }) {
  const rates = useMemo<ChartConfiguration>(
    () => ({
      type: "line",
      data: {
        labels: f.t_end_ms,
        datasets: [
          { label: "all neurons", data: f.rate_all_hz, borderColor: COLORS.ink, pointRadius: 0, borderWidth: 2 },
          { label: `winners (${fmt(f.n_winner_neurons)} neurons)`, data: f.rate_winners_hz, borderColor: COLORS.green, pointRadius: 0, borderWidth: 2 },
          { label: `losers (${fmt(f.n_loser_neurons)} neurons)`, data: f.rate_losers_hz, borderColor: COLORS.grey, pointRadius: 0, borderWidth: 2 },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: "simulation time (ms)" } }, y: { beginAtZero: true, title: { display: true, text: "rate (Hz)" } } },
      },
    }),
    [f],
  );
  const hist = useMemo<ChartConfiguration>(() => {
    const e = f.neuron_rate_hist.edges_hz;
    return {
      type: "bar",
      data: {
        labels: f.neuron_rate_hist.counts.map((_, i) => `${fmt(e[i], 0)}–${fmt(e[i + 1], 0)}`),
        datasets: [{ label: "neurons", data: f.neuron_rate_hist.counts, backgroundColor: COLORS.blue }],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "mean rate over the run (Hz)" } }, y: { beginAtZero: true, title: { display: true, text: "neurons" } } },
      },
    };
  }, [f]);
  return (
    <Section title="7. Firing activity">
      <p>
        Winners are the populations of the digit finally read out in each cell; losers are the other 8 per cell. Rate = spikes
        / (neurons × window length).
      </p>
      <ChartCanvas config={rates} height={240} label="Population rates over time" />
      <ChartCanvas config={hist} height={200} label="Histogram of per-neuron firing rates" />
      <dl className="kv">
        <dt>Total spikes</dt>
        <dd>{fmt(f.total_spikes)}</dd>
        <dt>Mean rate per neuron</dt>
        <dd>{fmt(f.mean_rate_hz, 1)} Hz (highest {fmt(f.max_neuron_rate_hz, 1)} Hz)</dd>
        <dt>ISI coefficient of variation</dt>
        <dd>
          {f.isi_cv_mean == null
            ? "not enough spikes"
            : `mean ${fmt(f.isi_cv_mean, 3)}, median ${fmt(f.isi_cv_median, 3)} over ${f.isi_cv_neurons} winning neurons in the last ${fmt(f.isi_window_ms)} ms`}
        </dd>
      </dl>
      <p className="note">A CV near 0 means clock-like firing; a Poisson process has CV = 1.</p>
    </Section>
  );
}

function Glossary() {
  const items: [string, string][] = [
    ["Cell", "One of the 81 squares that make up the Sudoku field."],
    ["Box", "One of the 9 collections of 3×3 cells; each must contain every digit once."],
    ["Digit", "A number between 1 and 9."],
    ["Population", "The group of neurons coding for a single digit in a cell."],
    ["Window", "A stretch of simulated time (here one readout interval) over which spikes are counted."],
    ["Givens", "The digits of the puzzle; their populations receive extra stimulation."],
    ["Performance", "The fraction of the 27 rows, columns and boxes that contain every digit exactly once (1 = solved)."],
  ];
  return (
    <Section title="9. Glossary">
      <dl className="glossary">
        {items.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}
