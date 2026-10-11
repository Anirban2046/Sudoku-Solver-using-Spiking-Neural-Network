import { useMemo } from "react";
import type { ChartConfiguration } from "chart.js/auto";
import ChartCanvas, { COLORS } from "./ChartCanvas";

interface Props {
  points: { x: number; y: number }[];
  maxSimMs: number;
  noiseRate: number;
  cursorMs?: number | null;
}

/** Performance (fraction of valid rows, columns and boxes) against simulated
 *  time, as in the bottom-left panel of the NEST progress GIF. */
export default function PerformanceChart({ points, maxSimMs, noiseRate, cursorMs }: Props) {
  const config = useMemo<ChartConfiguration>(
    () => ({
      type: "line",
      data: {
        datasets: [
          {
            label: `${noiseRate} Hz`,
            data: points,
            borderColor: COLORS.blue,
            backgroundColor: COLORS.blue,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
          },
          ...(cursorMs != null
            ? [
                {
                  label: "replay",
                  data: [
                    { x: cursorMs, y: 0 },
                    { x: cursorMs, y: 1 },
                  ],
                  borderColor: COLORS.orange,
                  borderWidth: 1,
                  borderDash: [4, 3],
                  pointRadius: 0,
                },
              ]
            : []),
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        parsing: false,
        scales: {
          x: { type: "linear", min: 0, max: maxSimMs, title: { display: true, text: "simulation time (ms)" } },
          y: { min: 0, max: 1, title: { display: true, text: "performance" } },
        },
        plugins: { legend: { display: true, labels: { filter: (item) => item.text !== "replay" } } },
      },
    }),
    [points, maxSimMs, noiseRate, cursorMs],
  );
  return <ChartCanvas config={config} height={230} label="Performance versus simulation time" />;
}
