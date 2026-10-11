import type { SolveEvent } from "./types";

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "";

export class SolveError extends Error {}

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
 *  Retries "busy" (429) responses 3 times with a 2 s backoff. */
export async function solve(
  puzzle: number[][],
  onEvent: (ev: SolveEvent) => void,
  signal?: AbortSignal,
  seed?: number,
): Promise<void> {
  const body = JSON.stringify(seed ? { puzzle, seed } : { puzzle });
  let res: Response | undefined;
  for (let attempt = 0; attempt <= 3; attempt++) {
    res = await fetch(`${API_BASE}/api/solve`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      signal,
    });
    if (res.status !== 429) break;
    if (attempt < 3) await sleep(2000, signal);
  }
  if (!res) throw new SolveError("No response from server");
  if (res.status === 429) throw new SolveError("Server busy, try again in a moment");
  if (res.status === 422) throw new SolveError(await errorMessage(res));
  if (!res.ok || !res.body) throw new SolveError(await errorMessage(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
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
}
