import { useMemo, useState, type ReactNode } from "react";
import type { ChartConfiguration } from "chart.js";
import type { Grid } from "../lib/sudoku";
import type { DoneEvent, MetaEvent, StepEvent } from "../lib/types";
import ChartCanvas, { COLORS } from "./ChartCanvas";

interface Props {
  puzzle: Grid;
  meta: MetaEvent;
  steps: StepEvent[];
  done: DoneEvent;
  /** wall-clock seconds of every request of the run */
  wallSeconds: number;
  segments: number;
}

const fmt = (n: number, digits = 0) =>
  n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const cellName = (i: number) => `r${Math.floor(i / 9) + 1} c${(i % 9) + 1}`;

function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export default function AnalysisReport({ puzzle, meta, steps, done, wallSeconds, segments }: Props) {
  const simMs = steps.length ? steps[steps.length - 1].t_end_ms : 0;
  const totalSpikes = steps.reduce((n, s) => n + s.total_spikes, 0);
  const kicks = steps.filter((s) => s.kick?.kind === "kick").length;
  const resets = steps.filter((s) => s.kick?.kind === "reset" || s.kick?.kind === "hint").length + segments - 1;
  const hints = done.hints.length;
  const firstHint = steps.find((s) => s.kick?.kind === "hint");
  const verdict = {
    match: "identical to the backtracking solver's solution",
    valid_but_different: "valid, but different from the backtracking solver's (the puzzle has several solutions)",
    not_solved: "not a valid grid",
  }[done.classical.verdict];

  const tiles: [string, string, string][] = [
    ["Solved at", `${fmt(done.solved_at_ms ?? 0)} ms`, "simulated time"],
    ["Wall time", `${fmt(wallSeconds, 1)} s`, `${fmt(simMs / Math.max(wallSeconds, 1e-9))} simulated ms per second`],
    ["Kicks / restarts", `${kicks} / ${resets}`, hints ? `plus ${hints} hinted cells` : "interventions when progress stalled"],
    ["Neurons", fmt(meta.n_neurons), `${fmt(meta.n_populations)} populations of ${meta.pop_size}`],
    ["Synapses", fmt(meta.connections.total), `${fmt(meta.connections.inhibitory)} inhibitory`],
    ["Spikes", fmt(totalSpikes), `${fmt(totalSpikes / (meta.n_neurons * (simMs / 1000)), 1)} Hz mean rate`],
  ];

  return (
    <section className="report" aria-label="Results">
      <h2>How the network solved it</h2>
      <div className="tiles">
        {tiles.map(([label, value, sub]) => (
          <div className="tile" key={label}>
            <div className="tile-label">{label}</div>
            <div className="tile-value">{value}</div>
            <div className="tile-sub">{sub}</div>
          </div>
        ))}
      </div>
      {hints > 0 && firstHint && (
        <p className="hint-note">
          The network didn't settle on its own within {fmt(firstHint.t_end_ms)} ms, so {hints} of the{" "}
          {81 - meta.givens} empty cells were hinted: their correct digit was stimulated like a given. The network solved
          the other {81 - meta.givens - hints}.
        </p>
      )}
      <p className="note">
        Answer check: {verdict}. Seed {meta.seed}
        {segments > 1 ? ` (last of ${segments} requests)` : ""}, NEST {meta.nest_version}.
      </p>
      <div className="cards">
        <CellCompetition grid={done.final_grid} cells={done.cells} chunkMs={meta.chunk_ms} />
        <LockIn puzzle={puzzle} steps={steps} />
        <Wiring meta={meta} />
      </div>
    </section>
  );
}

function CellCompetition({ grid, cells, chunkMs }: { grid: number[]; cells: DoneEvent["cells"]; chunkMs: number }) {
  const [cell, setCell] = useState(() => cells.margin.indexOf(Math.min(...cells.margin)));
  const config = useMemo<ChartConfiguration>(
    () => ({
      type: "bar",
      data: {
        labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        datasets: [
          {
            label: "spikes",
            data: cells.counts[cell],
            backgroundColor: cells.counts[cell].map((_, d) => (d + 1 === grid[cell] ? COLORS.green : COLORS.grey)),
          },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "digit population" } },
          y: { beginAtZero: true, title: { display: true, text: `spikes in the last ${chunkMs} ms` } },
        },
      },
    }),
    [cell, cells, grid, chunkMs],
  );
  return (
    <Card title="Winner-take-all inside each cell">
      <p>
        Each cell has 9 competing populations, one per digit, and the most active one is the answer. Pick a cell to see the
        competition in the final window. Orange cells were the closest contests; the mean decision margin was{" "}
        {fmt(cells.margin_mean * 100)}%.
      </p>
      <div className="split">
        <div className="mini-board pick-board" role="group" aria-label="Pick a cell">
          {grid.map((d, i) => (
            <button
              key={i}
              type="button"
              className={`mb-cell${i === cell ? " sel" : ""}`}
              onClick={() => setCell(i)}
              aria-label={`${cellName(i)}, digit ${d}`}
              aria-pressed={i === cell}
              style={i === cell ? undefined : { background: `rgba(217, 130, 43, ${(0.55 * (1 - cells.margin[i])).toFixed(3)})` }}
            >
              {d}
            </button>
          ))}
        </div>
        <div className="split-chart">
          <div className="chart-caption">
            {cellName(cell)}: digit {grid[cell]} won, margin {fmt(cells.margin[cell] * 100)}%
          </div>
          <ChartCanvas config={config} height={220} label={`Digit spike counts for ${cellName(cell)}`} />
        </div>
      </div>
    </Card>
  );
}

function LockIn({ puzzle, steps }: { puzzle: Grid; steps: StepEvent[] }) {
  const lockMs = useMemo(() => {
    const out = Array<number>(81).fill(steps[0]?.t_end_ms ?? 0);
    steps.forEach((s, k) => {
      const prev = k ? steps[k - 1].grid : puzzle;
      s.grid.forEach((d, i) => {
        if (d !== prev[i]) out[i] = s.t_end_ms;
      });
    });
    return out;
  }, [puzzle, steps]);
  const max = Math.max(...lockMs);
  return (
    <Card title="Order in which the cells settled">
      <p>
        When each cell took its final digit (darker = later). Given cells (underlined) lock in almost at once; the empty
        cells follow as inhibition pushes the rival digits out.
      </p>
      <div className="mini-board heat-board">
        {lockMs.map((ms, i) => (
          <div
            key={i}
            className={`mb-cell${puzzle[i] ? " given" : ""}`}
            style={{ background: `rgba(31, 79, 138, ${max ? 0.04 + (0.46 * ms) / max : 0.04})` }}
            title={`${cellName(i)}: final digit since ${fmt(ms)} ms`}
          >
            <span className="mb-sub">{fmt(ms)}</span>
          </div>
        ))}
      </div>
      <p className="note">Numbers are milliseconds of simulated time.</p>
    </Card>
  );
}

function Wiring({ meta }: { meta: MetaEvent }) {
  // Source population: digit 5 of the centre cell.
  const s = 20;
  const kind = (r: number, c: number) => {
    if (r === 4 && c === 4) return "src";
    if (r === 4) return "row";
    if (c === 4) return "col";
    if (Math.floor(r / 3) === 1 && Math.floor(c / 3) === 1) return "box";
    return "";
  };
  const fill: Record<string, string> = { src: COLORS.ink, row: COLORS.blue, col: COLORS.green, box: COLORS.orange, "": "#fff" };
  return (
    <Card title="How the Sudoku rules are wired">
      <p>
        The rules are synapses, not code. The population for “5 in the centre cell” inhibits every population that would
        break a rule with it: the 5s in its row, column and box, and the other 8 digits of its own cell, so each neuron
        inhibits {fmt(meta.connections.inhibitory / meta.n_neurons)} others. Given digits get an extra {fmt(meta.stim_rate)} Hz
        input, and {fmt(meta.noise_rate)} Hz of random noise keeps the network exploring until no rule is broken.
      </p>
      <figure className="diagram">
        <svg viewBox={`0 0 ${9 * s + 110} ${9 * s + 4}`} width="100%" style={{ maxWidth: 360 }} role="img"
          aria-label="Populations inhibited by the digit-5 population of the centre cell">
          <g transform="translate(2,2)">
            {Array.from({ length: 81 }, (_, i) => {
              const k = kind(Math.floor(i / 9), i % 9);
              return <rect key={i} x={(i % 9) * s} y={Math.floor(i / 9) * s} width={s} height={s} fill={fill[k]}
                fillOpacity={k && k !== "src" ? 0.75 : 1} stroke="#bbb" strokeWidth={0.5} />;
            })}
            {[0, 3, 6, 9].map((v) => (
              <g key={v}>
                <line x1={v * s} y1={0} x2={v * s} y2={9 * s} stroke="#222" strokeWidth={1.5} />
                <line x1={0} y1={v * s} x2={9 * s} y2={v * s} stroke="#222" strokeWidth={1.5} />
              </g>
            ))}
            <text x={4.5 * s} y={4.5 * s + 4} textAnchor="middle" fill="#fff" fontSize={11}>5</text>
            <g transform={`translate(${9 * s + 18}, ${3 * s})`}>
              {Array.from({ length: 9 }, (_, d) => (
                <g key={d}>
                  <rect x={(d % 3) * s} y={Math.floor(d / 3) * s} width={s} height={s}
                    fill={d === 4 ? COLORS.ink : COLORS.purple} fillOpacity={d === 4 ? 1 : 0.7} stroke="#fff" />
                  <text x={(d % 3) * s + s / 2} y={Math.floor(d / 3) * s + s / 2 + 4} textAnchor="middle" fill="#fff" fontSize={10}>{d + 1}</text>
                </g>
              ))}
              <text x={1.5 * s} y={-6} textAnchor="middle" fontSize={10} fill="#333">centre cell</text>
            </g>
          </g>
        </svg>
        <figcaption>
          <span className="swatch" style={{ background: COLORS.blue }} /> same row
          <span className="swatch" style={{ background: COLORS.green }} /> same column
          <span className="swatch" style={{ background: COLORS.orange }} /> same box
          <span className="swatch" style={{ background: COLORS.purple }} /> other digits of the cell
        </figcaption>
      </figure>
    </Card>
  );
}
