import { useMemo } from "react";
import type { ChartConfiguration } from "chart.js";
import type { StepEvent } from "../lib/types";
import ChartCanvas, { COLORS, niceMax, type Marker } from "./ChartCanvas";

interface Props {
  steps: StepEvent[];
  /** start time (ms) of the window shown on the board while replaying */
  cursorMs?: number | null;
}

function kickMarkers(steps: StepEvent[], cursorMs?: number | null): Marker[] {
  const lines: Marker[] = steps
    .filter((s) => s.kick)
    .map((s) => ({ x: s.t_end_ms, color: { kick: COLORS.purple, reset: COLORS.red, hint: COLORS.green }[s.kick!.kind], dash: [3, 3] }));
  if (cursorMs != null) lines.push({ x: cursorMs, color: COLORS.orange });
  return lines;
}

const xAxis = (max: number) => ({
  type: "linear" as const,
  min: 0,
  max,
  title: { display: true, text: "simulated time (ms)" },
  ticks: { maxTicksLimit: 8 },
});

/** Performance and rule violations over time: the network relaxing towards a
 *  valid grid, like the energy of a Hopfield network. */
export function ConvergenceChart({ steps, cursorMs }: Props) {
  const config = useMemo<ChartConfiguration>(() => {
    const xMax = niceMax(steps.length ? steps[steps.length - 1].t_end_ms : 0);
    const maxViol = Math.max(4, ...steps.map((s) => s.violations));
    return {
      type: "line",
      data: {
        datasets: [
          {
            label: "performance",
            data: steps.map((s) => ({ x: s.t_start_ms, y: s.performance })),
            borderColor: COLORS.blue,
            backgroundColor: COLORS.blue,
            borderWidth: 2,
            pointRadius: steps.map((s) => (s.givens_overridden ? 3 : 0)),
            pointBackgroundColor: COLORS.orange,
            yAxisID: "y",
          },
          {
            label: "rule violations",
            data: steps.map((s) => ({ x: s.t_start_ms, y: s.violations })),
            borderColor: COLORS.red,
            borderWidth: 1.5,
            pointRadius: 0,
            yAxisID: "y2",
          },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        parsing: false,
        normalized: true,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: xAxis(xMax),
          y: { min: 0, max: 1, title: { display: true, text: "performance" } },
          y2: {
            position: "right",
            min: 0,
            suggestedMax: maxViol,
            ticks: { precision: 0 },
            title: { display: true, text: "conflicting cell pairs" },
            grid: { drawOnChartArea: false },
          },
        },
        plugins: { markers: { lines: kickMarkers(steps, cursorMs) }, legend: { labels: { boxHeight: 2 } } },
      },
    };
  }, [steps, cursorMs]);
  return <ChartCanvas config={config} height={240} label="Performance and rule violations over time" />;
}

/** Firing rate of the populations that currently win their cell versus the
 *  rest: the winner-take-all competition inside every cell. */
export function RateChart({ steps, cursorMs }: Props) {
  const config = useMemo<ChartConfiguration>(() => {
    const xMax = niceMax(steps.length ? steps[steps.length - 1].t_end_ms : 0);
    const yMax = niceMax(Math.max(...steps.map((s) => s.rate_winners_hz), 10) * 1.1, 10);
    const line = (label: string, key: keyof StepEvent, color: string) => ({
      label,
      data: steps.map((s) => ({ x: s.t_start_ms, y: s[key] as number })),
      borderColor: color,
      borderWidth: 2,
      pointRadius: 0,
    });
    return {
      type: "line",
      data: {
        datasets: [
          line("winning digit of each cell", "rate_winners_hz", COLORS.green),
          line("all neurons", "rate_all_hz", COLORS.ink),
          line("other 8 digits", "rate_losers_hz", COLORS.grey),
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        parsing: false,
        normalized: true,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: xAxis(xMax),
          y: { min: 0, max: yMax, title: { display: true, text: "firing rate (Hz)" } },
        },
        plugins: { markers: { lines: kickMarkers(steps, cursorMs) }, legend: { labels: { boxHeight: 2 } } },
      },
    };
  }, [steps, cursorMs]);
  return <ChartCanvas config={config} height={220} label="Firing rate of winning and losing populations over time" />;
}
