import type { SolveEvent } from "./types";

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "";

/** A problem retrying won't fix (e.g. the server rejected the puzzle). */
export class SolveError extends Error {}

/** A failure worth retrying: server errors, timeouts, an error event mid-run. */
export class TransientError extends Error {}

/** How long to keep retrying while every simulator instance is busy. */
const BUSY_WAIT_MS = 180_000;

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length)
      return detail.map((d) => String(d.msg ?? d).replace(/^Value error, /, "")).join("; ");
  } catch {
    /* fall through */
  }
  return `Server error ${res.status}`;
}

/** POST the puzzle and call `onEvent` for every NDJSON line as it arrives.
 *
 *  While the simulator is busy (429) it keeps retrying with a growing backoff
 *  for up to three minutes, calling `onBusy` each time. A rejected puzzle
 *  (4xx) throws SolveError; server errors and network failures throw
 *  TransientError (or the fetch error), which the caller may retry. */
export async function solve(
  puzzle: number[][],
  onEvent: (ev: SolveEvent) => void,
  signal?: AbortSignal,
  continuation: { t_offset_ms?: number; segment?: number; seed?: number; hints?: number[] } = {},
  onBusy?: () => void,
): Promise<void> {
  const body = JSON.stringify({ puzzle, ...continuation });
  const started = Date.now();
  let res: Response;
  for (let attempt = 1; ; attempt++) {
    res = await fetch(`${API_BASE}/api/solve`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      signal,
    });
    if (res.status !== 429) break;
    if (Date.now() - started > BUSY_WAIT_MS) throw new SolveError("The simulator is busy. Please try again in a few minutes.");
    onBusy?.();
    await sleep(Math.min(2000 * attempt, 10_000), signal);
  }
  if (res.status >= 500) throw new TransientError(await errorMessage(res));
  if (!res.ok || !res.body) throw new SolveError(await errorMessage(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (value) buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (line) onEvent(JSON.parse(line) as SolveEvent);
      }
      if (done) break;
    }
    const rest = (buffer + decoder.decode()).trim();
    if (rest) onEvent(JSON.parse(rest) as SolveEvent);
  } catch (e) {
    // close the connection if an event handler (or parsing) failed mid-stream
    reader.cancel().catch(() => {});
    throw e;
  }
}
