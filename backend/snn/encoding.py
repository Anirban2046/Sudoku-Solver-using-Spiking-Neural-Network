"""Compact binary encoding for the NDJSON stream."""

import base64

import numpy as np


def b64_u16(values) -> str:
    """Pack integers as little-endian uint16 and base64-encode them."""
    arr = np.asarray(values)
    if arr.size and (arr.min() < 0 or arr.max() > 0xFFFF):
        raise ValueError("value out of uint16 range")
    return base64.b64encode(arr.astype("<u2").tobytes()).decode("ascii")


def decode_b64_u16(text: str) -> np.ndarray:
    """Inverse of :func:`b64_u16` (used in tests)."""
    return np.frombuffer(base64.b64decode(text), dtype="<u2")
