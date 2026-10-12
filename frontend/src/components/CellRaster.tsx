import { useEffect, useRef, useState } from "react";
import { COLORS } from "./ChartCanvas";

interface Props {
  /** spike times (ms from the window start) for each of the cell's neurons, row = digit * perPop + neuron */
  rows: number[][];
  perPop: number;
  windowMs: number;
  winner: number; // digit 1..9
  rival: number; // digit 1..9
  label: string;
}

const ROW_H = 6;
const LEFT = 30;
const RIGHT = 74;
const BOTTOM = 22;

/** Spike raster of the 9 digit populations of one cell (9 x perPop neurons). */
export default function CellRaster({ rows, perPop, windowMs, winner, rival, label }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(480);
  const height = 9 * perPop * ROW_H + 8 * 4 + BOTTOM;

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.max(240, Math.floor(el.clientWidth))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvas.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.floor(width * dpr);
    cv.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const plotW = width - LEFT - RIGHT;
    const x = (t: number) => LEFT + (t / windowMs) * plotW;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";

    for (let d = 0; d < 9; d++) {
      const top = d * (perPop * ROW_H + 4);
      const isWin = d + 1 === winner;
      const isRival = d + 1 === rival;
      const color = isWin ? COLORS.sky : isRival ? COLORS.violet : COLORS.grey;
      if (isWin) {
        ctx.fillStyle = "rgba(56, 139, 253, 0.1)";
        ctx.fillRect(LEFT, top - 1, plotW, perPop * ROW_H + 2);
      }
      let n = 0;
      ctx.fillStyle = color;
      for (let k = 0; k < perPop; k++) {
        const y = top + k * ROW_H + 1;
        for (const t of rows[d * perPop + k]) {
          ctx.fillRect(x(t) - 0.75, y, 1.5, ROW_H - 2);
          n++;
        }
      }
      ctx.fillStyle = isWin || isRival ? color : "#9198a1";
      ctx.textAlign = "right";
      ctx.fillText(String(d + 1), LEFT - 10, top + (perPop * ROW_H) / 2);
      ctx.textAlign = "left";
      ctx.fillText(`${n} spikes`, LEFT + plotW + 8, top + (perPop * ROW_H) / 2);
    }

    // time axis
    const axisY = height - BOTTOM + 4;
    ctx.fillStyle = "#9198a1";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const step = windowMs <= 100 ? 20 : windowMs <= 300 ? 50 : 100;
    for (let t = 0; t <= windowMs; t += step) {
      ctx.fillRect(x(t), axisY - 4, 1, 4);
      ctx.fillText(`${t} ms`, x(t), axisY + 1);
    }
  }, [rows, perPop, windowMs, winner, rival, width, height]);

  return (
    <div className="cell-raster" ref={wrap}>
      <canvas ref={canvas} style={{ width: "100%", height }} role="img" aria-label={label} />
    </div>
  );
}
