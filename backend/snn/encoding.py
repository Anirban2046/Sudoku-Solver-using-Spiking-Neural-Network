"""Compact binary encoding for arrays sent in the NDJSON stream."""

import base64

import numpy as np


def b64_u16(values) -> str:
    """Pack integers as little-endian uint16 and base64-encode them."""
    arr = np.asarray(values)
    if arr.size and (arr.min() < 0 or arr.max() > 0xFFFF):
        raise ValueError("value out of uint16 range")
    return base64.b64encode(arr.astype("<u2").tobytes()).decode("ascii")


def b64_i16(values) -> str:
    """Pack integers as little-endian int16 (clipped) and base64-encode them."""
    arr = np.clip(np.rint(np.asarray(values, dtype=float)), -32768, 32767)
    return base64.b64encode(arr.astype("<i2").tobytes()).decode("ascii")


def decode_b64(text: str, dtype: str) -> np.ndarray:
    """Inverse of the encoders above (used in tests); dtype "<u2" or "<i2"."""
    return np.frombuffer(base64.b64decode(text), dtype=dtype)
