import { useMemo, useState, type ReactNode } from "react";
import type { ChartConfiguration } from "chart.js";
import { b64ToI16, b64ToU16 } from "../lib/decode";
import type { DoneEvent, MetaEvent, StepEvent } from "../lib/types";
import CellRaster from "./CellRaster";
import ChartCanvas, { COLORS } from "./ChartCanvas";
import { RateChart } from "./RunCharts";

interface Props {
  meta: MetaEvent;
  steps: StepEvent[];
  done: DoneEvent;
  /** wall-clock seconds of every request of the run */
  wallSeconds: number;
}

const fmt = (n: number, digits = 0) =>
  n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const cellName = (i: number) => `row ${Math.floor(i / 9) + 1}, column ${(i % 9) + 1}`;

function Card({ title, lead, children }: { title: string; lead: ReactNode; children: ReactNode }) {
  return (
    <section className="card">
      <h3>{title}</h3>
      <p className="lead">{lead}</p>
      {children}
    </section>
  );
}

export default function AnalysisReport({ meta, steps, done, wallSeconds }: Props) {
  const simMs = steps.length ? steps[steps.length - 1].t_end_ms : 0;
  const totalSpikes = steps.reduce((n, s) => n + s.total_spikes, 0);
  const r = done.network.rates;

  const tiles: [string, string, string][] = [
    ["Solved in", `${fmt(done.solved_at_ms ?? 0)} ms`, "of simulated network time"],
    ["Neurons", fmt(meta.n_neurons), `${meta.neuron_model} (leaky integrate-and-fire)`],
    ["Synapses", fmt(meta.connections.total), `${fmt(meta.connections.inhibitory)} of them inhibitory`],
    ["Spikes fired", fmt(totalSpikes), `${fmt(totalSpikes / (meta.n_neurons * (simMs / 1000)), 1)} Hz per neuron on average`],
    ["Winning neurons", `${fmt(r.winners_hz)} Hz`, `vs ${fmt(r.others_hz, 1)} Hz for all the others`],
    ["Wall time", `${fmt(wallSeconds, 1)} s`, `${fmt(simMs / Math.max(wallSeconds, 1e-9))} ms of network time per second`],
  ];

  return (
    <section className="report" aria-label="Inside the spiking network">
      <h2>Inside the spiking network</h2>
      <div className="tiles">
        {tiles.map(([label, value, sub]) => (
          <div className="tile" key={label}>
            <div className="tile-label">{label}</div>
            <div className="tile-value">{value}</div>
            <div className="tile-sub">{sub}</div>
          </div>
        ))}
      </div>
      <CellInspector meta={meta} done={done} />
      <Card
        title="Winner-take-all, across the board"
        lead="In every cell one digit population takes over and silences its rivals. The neurons of the winning digits speed up while everyone else is pushed towards silence."
      >
        <RateChart steps={steps} />
      </Card>
      <RateHistogram rates={r} />
    </section>
  );
}

function CellInspector({ meta, done }: { meta: MetaEvent; done: DoneEvent }) {
  const net = done.network;
  const grid = done.final_grid;
  const [cell, setCell] = useState(() => done.cells.margin.indexOf(Math.min(...done.cells.margin)));

  const spikes = useMemo(
    () => ({ neuron: b64ToU16(net.spikes.neuron_b64), t: b64ToU16(net.spikes.t_b64) }),
    [net.spikes.neuron_b64, net.spikes.t_b64],
  );
  const vm = useMemo(
    () => ({ winner: b64ToI16(net.vm.winner_b64), rival: b64ToI16(net.vm.rival_b64) }),
    [net.vm.winner_b64, net.vm.rival_b64],
  );

  const perPop = net.neurons_per_pop;
  const windowMs = net.vm.samples * net.vm.dt_ms;
  const winner = grid[cell];
  const rival = net.vm.rival_digit[cell];

  const rows = useMemo(() => {
    const out: number[][] = Array.from({ length: 9 * perPop }, () => []);
    const first = cell * 9 * perPop;
    for (let i = 0; i < spikes.neuron.length; i++) {
      const k = spikes.neuron[i] - first;
      if (k >= 0 && k < 9 * perPop) out[k].push(spikes.t[i] / 10);
    }
    return out;
  }, [spikes, cell, perPop]);

  const vmConfig = useMemo<ChartConfiguration>(() => {
    const n = net.vm.samples;
    const trace = (arr: Int16Array) =>
      Array.from(arr.subarray(cell * n, cell * n + n), (v, i) => ({ x: (i + 1) * net.vm.dt_ms, y: v / 100 }));
    return {
      type: "line",
      data: {
        datasets: [
          {
            label: `a neuron of digit ${winner} (winner)`,
            data: trace(vm.winner),
            borderColor: COLORS.sky,
            backgroundColor: COLORS.sky,
            borderWidth: 1.5,
            pointRadius: 0,
          },
          {
            label: `a neuron of digit ${rival} (runner-up)`,
            data: trace(vm.rival),
            borderColor: COLORS.violet,
            backgroundColor: COLORS.violet,
            borderWidth: 1.5,
            pointRadius: 0,
          },
          {
            label: `firing threshold (${meta.neuron.V_th} mV)`,
            data: [
              { x: 0, y: meta.neuron.V_th },
              { x: windowMs, y: meta.neuron.V_th },
            ],
            borderColor: COLORS.amber,
            backgroundColor: COLORS.amber,
            borderDash: [5, 4],
            borderWidth: 1,
            pointRadius: 0,
          },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        parsing: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: { type: "linear", min: 0, max: windowMs, title: { display: true, text: "time in the final window (ms)" } },
          y: { grace: "5%", title: { display: true, text: "membrane potential (mV)" } },
        },
      },
    };
  }, [vm, cell, winner, rival, net.vm.samples, net.vm.dt_ms, windowMs, meta.neuron.V_th]);

  return (
    <Card
      title="One cell, up close"
      lead={
        <>
          Every cell holds 9 populations of {perPop} neurons, one per digit. Pick a cell to see all {9 * perPop} of its
          neurons spike in the final {fmt(windowMs)} ms, and the membrane potential of a winning neuron against one from
          the runner-up digit. Amber cells were the closest contests.
        </>
      }
    >
      <div className="inspector">
        <div className="mini-board pick-board" role="group" aria-label="Pick a cell">
          {grid.map((d, i) => (
            <button
              key={i}
              type="button"
              className={`mb-cell${i === cell ? " sel" : ""}`}
              onClick={() => setCell(i)}
              aria-label={`${cellName(i)}, digit ${d}`}
              aria-pressed={i === cell}
              style={i === cell ? undefined : { background: `rgba(210, 153, 34, ${(0.4 * (1 - done.cells.margin[i])).toFixed(3)})` }}
            >
              {d}
            </button>
          ))}
        </div>
        <div className="inspector-plots">
          <div className="chart-caption">
            {cellName(cell)}: digit <span className="win">{winner}</span> won, digit <span className="rival">{rival}</span>{" "}
            came second
          </div>
          <CellRaster
            rows={rows}
            perPop={perPop}
            windowMs={windowMs}
            winner={winner}
            rival={rival}
            label={`Spike raster of the ${9 * perPop} neurons of ${cellName(cell)}`}
          />
          <ChartCanvas config={vmConfig} height={230} label={`Membrane potentials in ${cellName(cell)}`} />
        </div>
      </div>
    </Card>
  );
}

function RateHistogram({ rates }: { rates: DoneEvent["network"]["rates"] }) {
  const config = useMemo<ChartConfiguration>(() => {
    const e = rates.hist_edges_hz;
    return {
      type: "bar",
      data: {
        labels: rates.hist_winners.map((_, i) => `${fmt(e[i])}–${fmt(e[i + 1])}`),
        datasets: [
          { label: "winning digits' neurons", data: rates.hist_winners, backgroundColor: COLORS.sky, borderRadius: 3 },
          { label: "all other neurons", data: rates.hist_others, backgroundColor: COLORS.grey, borderRadius: 3 },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { boxHeight: 10, boxWidth: 14 } } },
        scales: {
          x: { title: { display: true, text: "firing rate (Hz)" }, ticks: { maxRotation: 0, autoSkip: true } },
          y: {
            type: "logarithmic",
            min: 0.8,
            title: { display: true, text: "neurons (log scale)" },
            // label only 1, 10, 100, 1,000
            ticks: { callback: (v) => (Number.isInteger(Math.log10(Number(v))) ? fmt(Number(v)) : "") },
          },
        },
      },
    };
  }, [rates]);
  return (
    <Card
      title="How fast each neuron fires"
      lead={`Firing rates of all neurons over the last ${fmt(rates.window_ms)} ms. The network splits into two groups: a few hundred winners firing fast, and thousands of neurons held almost silent by inhibition.`}
    >
      <ChartCanvas config={config} height={240} label="Histogram of per-neuron firing rates" />
    </Card>
  );
}
