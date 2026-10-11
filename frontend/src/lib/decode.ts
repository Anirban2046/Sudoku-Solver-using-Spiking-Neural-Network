/** Decode base64 little-endian uint16 data (as packed by the backend). */
export function b64ToU16(b64: string): Uint16Array {
  const bin = atob(b64);
  const n = bin.length >> 1;
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
  return out;
}
