import { useEffect, useRef } from "react";
import Chart, { type ChartConfiguration } from "chart.js/auto";

interface Props {
  config: ChartConfiguration;
  height?: number;
  label: string;
}

/** Thin Chart.js wrapper: creates the chart once and updates it in place
 *  (no animation) whenever `config` changes. */
export default function ChartCanvas({ config, height = 220, label }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const chart = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvas.current) return;
    chart.current = new Chart(canvas.current, config);
    return () => {
      chart.current?.destroy();
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
  green: "#3a8f5c",
  red: "#c0392b",
  purple: "#7d4fb0",
  grey: "#8a96a3",
};
