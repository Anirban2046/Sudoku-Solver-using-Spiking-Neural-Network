/** Decode base64 little-endian 16-bit arrays packed by the backend. */
function bytes(b64: string): DataView {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new DataView(buf.buffer);
}

export function b64ToU16(b64: string): Uint16Array {
  const v = bytes(b64);
  const out = new Uint16Array(v.byteLength >> 1);
  for (let i = 0; i < out.length; i++) out[i] = v.getUint16(2 * i, true);
  return out;
}

export function b64ToI16(b64: string): Int16Array {
  const v = bytes(b64);
  const out = new Int16Array(v.byteLength >> 1);
  for (let i = 0; i < out.length; i++) out[i] = v.getInt16(2 * i, true);
  return out;
}
