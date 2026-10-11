import { useEffect, useRef } from "react";
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
  type ChartConfiguration,
  type Plugin,
} from "chart.js";

Chart.register(LineController, LineElement, PointElement, BarController, BarElement, LinearScale, CategoryScale, Legend, Tooltip);

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

/** Draws vertical lines at given x values (kicks, restarts, replay cursor). */
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
  ink: "#1d2b3a",
  blue: "#2f6db5",
  orange: "#d9822b",
  green: "#2e8b57",
  red: "#c0392b",
  purple: "#7d4fb0",
  grey: "#8a96a3",
};

/** A "nice" axis maximum (1, 2 or 5 x 10^n) at or above `v`, at least `min`. */
export function niceMax(v: number, min = 1000) {
  const x = Math.max(v, min);
  const p = 10 ** Math.floor(Math.log10(x));
  return [1, 2, 5, 10].map((m) => m * p).find((m) => m >= x)!;
}
