import { useEffect, useRef } from "react";
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Filler,
  Legend,
  LinearScale,
  LogarithmicScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
  type ChartConfiguration,
  type Plugin,
} from "chart.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  BarController,
  BarElement,
  LinearScale,
  LogarithmicScale,
  CategoryScale,
  Filler,
  Legend,
  Tooltip,
);

// Dark theme defaults for every chart.
Chart.defaults.color = "#9aa8c3";
Chart.defaults.borderColor = "rgba(148, 163, 184, 0.14)";
Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
Chart.defaults.plugins.legend.labels.boxHeight = 2;
Chart.defaults.plugins.tooltip.backgroundColor = "#1b2540";
Chart.defaults.plugins.tooltip.borderColor = "#2c3a5c";
Chart.defaults.plugins.tooltip.borderWidth = 1;

export interface Marker {
  x: number;
  color: string;
  dash?: number[];
}

declare module "chart.js" {
  interface PluginOptionsByType<TType> {
    markers?: { lines: Marker[] };
  }
}

/** Draws vertical lines at given x values (e.g. the replay cursor). */
const markersPlugin: Plugin = {
  id: "markers",
  afterDatasetsDraw(chart) {
    const lines = (chart.options.plugins as { markers?: { lines: Marker[] } }).markers?.lines ?? [];
    const x = chart.scales.x;
    const { top, bottom, left, right } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    for (const m of lines) {
      const px = x.getPixelForValue(m.x);
      if (px < left || px > right) continue;
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 1;
      ctx.setLineDash(m.dash ?? []);
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, bottom);
      ctx.stroke();
    }
    ctx.restore();
  },
};

interface Props {
  config: ChartConfiguration;
  height?: number;
  label: string;
}

/** Thin Chart.js wrapper: creates the chart once, updates it in place (no
 *  animation) when `config` changes, and refits it to the page when printing. */
export default function ChartCanvas({ config, height = 220, label }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const chart = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvas.current) return;
    const c = new Chart(canvas.current, { ...config, plugins: [markersPlugin] });
    chart.current = c;
    const fit = () => c.resize();
    const fitAfterPrint = () => requestAnimationFrame(() => c.resize());
    window.addEventListener("beforeprint", fit);
    window.addEventListener("afterprint", fitAfterPrint);
    return () => {
      window.removeEventListener("beforeprint", fit);
      window.removeEventListener("afterprint", fitAfterPrint);
      c.destroy();
      chart.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    c.data = config.data;
    if (config.options) c.options = config.options;
    c.update("none");
  }, [config]);

  return (
    <div className="chart" style={{ height }}>
      <canvas ref={canvas} role="img" aria-label={label} />
    </div>
  );
}

export const COLORS = {
  text: "#e6ebf5",
  sky: "#38bdf8",
  violet: "#a78bfa",
  green: "#34d399",
  amber: "#fbbf24",
  red: "#f87171",
  grey: "#64748b",
};

/** A round axis maximum at or just above `v` (at least `min`), so the data
 *  fills most of the plot: 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 x 10^n. */
export function niceMax(v: number, min = 1000) {
  const x = Math.max(v, min);
  const p = 10 ** Math.floor(Math.log10(x));
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((m) => Math.round(m * p)).find((m) => m >= x)!;
}
