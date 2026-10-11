import { useMemo } from "react";
import type { ChartConfiguration } from "chart.js";
import type { StepEvent } from "../lib/types";
import ChartCanvas, { COLORS, niceMax, type Marker } from "./ChartCanvas";

interface Props {
  steps: StepEvent[];
  /** start time (ms) of the window shown on the board while replaying */
  cursorMs?: number | null;
}

const xAxis = (steps: StepEvent[]) => ({
  type: "linear" as const,
  min: 0,
  // grows with the run: 1, 2 or 5 x 10^n ms
  max: niceMax(steps.length ? steps[steps.length - 1].t_end_ms : 0),
  title: { display: true, text: "simulated time (ms)" },
  ticks: { maxTicksLimit: 8 },
});

const cursor = (cursorMs?: number | null): Marker[] => (cursorMs != null ? [{ x: cursorMs, color: COLORS.amber }] : []);

/** Performance (fraction of valid rows, columns and boxes) over simulated time,
 *  as in the NEST example's progress plot. */
export function PerformanceChart({ steps, cursorMs }: Props) {
  const config = useMemo<ChartConfiguration>(() => {
    const top = Math.max(1, ...steps.map((s) => s.performance));
    return {
      type: "line",
      data: {
        datasets: [
          {
            label: "performance",
            data: steps.map((s) => ({ x: s.t_start_ms, y: s.performance })),
            borderColor: COLORS.sky,
            backgroundColor: "rgba(56, 189, 248, 0.12)",
            fill: "origin",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
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
          x: xAxis(steps),
          // starts at 0..1 and stretches if a value ever goes beyond it
          y: { min: 0, suggestedMax: top, title: { display: true, text: "performance" } },
        },
        plugins: { legend: { display: false }, markers: { lines: cursor(cursorMs) } },
      },
    };
  }, [steps, cursorMs]);
  return <ChartCanvas config={config} height={260} label="Performance over simulated time" />;
}

/** Firing rate of the population that currently wins each cell, against all
 *  other neurons: the winner-take-all competition emerging over time. */
export function RateChart({ steps }: Props) {
  const config = useMemo<ChartConfiguration>(() => {
    const line = (label: string, key: "rate_winners_hz" | "rate_losers_hz" | "rate_all_hz", color: string, width = 2) => ({
      label,
      data: steps.map((s) => ({ x: s.t_start_ms, y: s[key] })),
      borderColor: color,
      backgroundColor: color,
      borderWidth: width,
      pointRadius: 0,
      tension: 0.15,
    });
    return {
      type: "line",
      data: {
        datasets: [
          line("winning digit's neurons", "rate_winners_hz", COLORS.sky),
          line("all neurons", "rate_all_hz", COLORS.violet, 1.5),
          line("the other 8 digits' neurons", "rate_losers_hz", COLORS.grey, 1.5),
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        parsing: false,
        normalized: true,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: xAxis(steps),
          y: { beginAtZero: true, grace: "8%", title: { display: true, text: "firing rate (Hz)" } },
        },
      },
    };
  }, [steps]);
  return <ChartCanvas config={config} height={240} label="Firing rate of winning and other neurons over time" />;
}
