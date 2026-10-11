import { useEffect, useRef, useState } from "react";
import type { RasterWindow } from "../lib/types";

interface Props {
  windows: RasterWindow[];
  /** right edge of the visible time range (ms) */
  endMs: number;
  /** width of the visible time range (ms) */
  spanMs: number;
  chunkMs: number;
  /** population rate per window (Hz), drawn as a line under the raster */
  rates: { tEnd: number; hz: number }[];
  height?: number;
  label: string;
}

const N_POP = 729;
const WIN = "#1d2b3a";
const LOSE = "#a9bdd4";
const SEP_CELL = "#eef1f4";
const SEP_ROW = "#c9d1da";

function describe(pop: number) {
  const cell = Math.floor(pop / 9);
  return `r${Math.floor(cell / 9) + 1} c${(cell % 9) + 1} digit ${(pop % 9) + 1}`;
}

/** Spike raster: one row per population (ordered cell*9 + digit), one dot per
 *  spike of the representative neuron(s). Dots of the digit currently read out
 *  for that cell are dark, the others light. */
export default function SpikeRaster({ windows, endMs, spanMs, chunkMs, rates, height = 420, label }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const rateCanvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<string | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.max(200, Math.floor(el.clientWidth))));
    ro.observe(el);
    setWidth(Math.max(200, Math.floor(el.clientWidth)));
    return () => ro.disconnect();
  }, []);

  const startMs = Math.max(0, endMs - spanMs);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(width * dpr);
    const H = Math.floor(height * dpr);
    cv.width = W;
    cv.height = H;

    // Draw on an offscreen canvas, then blit in one go.
    const off = document.createElement("canvas");
    off.width = W;
    off.height = H;
    const ctx = off.getContext("2d");
    const out = cv.getContext("2d");
    if (!ctx || !out) return;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);

    const rowH = H / N_POP;
    if (rowH >= 0.5) {
      ctx.fillStyle = SEP_CELL;
      for (let cell = 1; cell < 81; cell++) ctx.fillRect(0, Math.floor(cell * 9 * rowH), W, 1);
    }
    ctx.fillStyle = SEP_ROW;
    for (let row = 1; row < 9; row++) ctx.fillRect(0, Math.floor(row * 81 * rowH), W, 1);

    const pxPerMs = W / spanMs;
    const dot = Math.max(1, Math.round(dpr));
    const dotH = Math.max(1, Math.round(rowH));
    for (const win of windows) {
      if (win.tStart + chunkMs < startMs || win.tStart > endMs) continue;
      for (const pass of [0, 1]) {
        ctx.fillStyle = pass === 0 ? LOSE : WIN;
        for (let i = 0; i < win.pop.length; i++) {
          const p = win.pop[i];
          const isWinner = win.grid[Math.floor(p / 9)] === (p % 9) + 1;
          if (isWinner !== (pass === 1)) continue;
          const t = win.tStart + win.t[i] / 10;
          if (t < startMs || t > endMs) continue;
          ctx.fillRect(Math.floor((t - startMs) * pxPerMs), Math.floor(p * rowH), dot, dotH);
        }
      }
    }
    out.drawImage(off, 0, 0);
  }, [windows, endMs, startMs, spanMs, chunkMs, width, height]);

  useEffect(() => {
    const cv = rateCanvas.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(width * dpr);
    const H = Math.floor(70 * dpr);
    cv.width = W;
    cv.height = H;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    const pad = 16 * dpr;
    const plotH = H - pad;
    const visible = rates.filter((r) => r.tEnd > startMs && r.tEnd - chunkMs <= endMs);
    const maxHz = Math.max(1, ...visible.map((r) => r.hz)) * 1.15;

    // time ticks
    ctx.fillStyle = "#5b6773";
    ctx.font = `${10 * dpr}px Helvetica, Arial, sans-serif`;
    ctx.textBaseline = "bottom";
    const tickStep = spanMs <= 2500 ? 500 : spanMs <= 6000 ? 1000 : spanMs <= 12000 ? 2000 : 5000;
    let lastLabelEnd = -Infinity;
    for (let t = Math.ceil(startMs / tickStep) * tickStep; t <= endMs; t += tickStep) {
      const x = ((t - startMs) / spanMs) * W;
      ctx.fillRect(x, plotH, 1, 4 * dpr);
      const text = `${t.toLocaleString("en-US")} ms`;
      const w = ctx.measureText(text).width;
      const lx = Math.min(x + 2 * dpr, W - w - 2 * dpr);
      if (lx > lastLabelEnd + 6 * dpr) {
        ctx.fillText(text, lx, H);
        lastLabelEnd = lx + w;
      }
    }
    ctx.fillText(`${Math.round(maxHz / 1.15)} Hz`, 2 * dpr, 12 * dpr);

    if (!visible.length) return;
    ctx.strokeStyle = "#2f6db5";
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    visible.forEach((r, i) => {
      const x = ((r.tEnd - chunkMs / 2 - startMs) / spanMs) * W;
      const y = plotH - (r.hz / maxHz) * (plotH - 14 * dpr);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [rates, startMs, endMs, spanMs, chunkMs, width]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pop = Math.floor(((e.clientY - rect.top) / rect.height) * N_POP);
    setHover(pop >= 0 && pop < N_POP ? describe(pop) : null);
  };

  return (
    <div className="raster" ref={wrap}>
      <div className="raster-hover" aria-live="off">
        {hover ?? "Hover the raster to identify a population"}
      </div>
      <canvas
        ref={canvas}
        style={{ width: "100%", height }}
        role="img"
        aria-label={label}
        title={hover ?? undefined}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      />
      <canvas ref={rateCanvas} style={{ width: "100%", height: 70 }} role="img" aria-label="Population rate (Hz)" />
      <div className="raster-legend">
        <span className="swatch" style={{ background: WIN }} /> digit read out for the cell
        <span className="swatch" style={{ background: LOSE }} /> other digits
        <span className="legend-line" /> network rate (Hz, all neurons)
      </div>
    </div>
  );
}
