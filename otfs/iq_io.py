"""IQ file export/import for SDR playback and analysis handoff.

Formats:
  .cf32 : interleaved float32 I/Q (GNU Radio / gr-file-source compatible,
          also the interchange format most vector signal analysers accept)
  .sc16 : interleaved int16 I/Q with a scale factor in the sidecar
          (common SDR DAC format; Amarisoft trx_file-style playback and
          most SDR front-ends take this or cf32)

Every file gets a JSON sidecar (<name>.json) carrying sample rate, grid
parameters and provenance so the capture side can reconstruct the frame.
"""

import json
import time
from pathlib import Path

import numpy as np


def _sidecar(path: Path, fs: float, fmt: str, meta: dict | None,
             scale: float | None = None) -> None:
    info = {
        "format": fmt,
        "sample_rate_hz": fs,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "otfs SW model v0.1.0",
    }
    if scale is not None:
        info["int16_scale"] = scale
    if meta:
        info.update(meta)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")


def write_cf32(path, s: np.ndarray, fs: float, meta: dict | None = None) -> None:
    path = Path(path)
    iq = np.empty(2 * len(s), dtype=np.float32)
    iq[0::2] = np.real(s).astype(np.float32)
    iq[1::2] = np.imag(s).astype(np.float32)
    iq.tofile(path)
    _sidecar(path, fs, "cf32 (interleaved float32 I/Q)", meta)


def read_cf32(path) -> np.ndarray:
    iq = np.fromfile(path, dtype=np.float32)
    return iq[0::2].astype(np.float64) + 1j * iq[1::2].astype(np.float64)


def write_sc16(path, s: np.ndarray, fs: float, meta: dict | None = None,
               backoff_db: float = 3.0) -> float:
    """Write int16 IQ with peak scaled to full-scale minus backoff.
    Returns the scale (float amplitude per LSB) recorded in the sidecar."""
    path = Path(path)
    peak = np.abs(np.concatenate([np.real(s), np.imag(s)])).max()
    full = 32767 * 10 ** (-backoff_db / 20)
    scale = peak / full if peak > 0 else 1.0
    iq = np.empty(2 * len(s), dtype=np.int16)
    iq[0::2] = np.round(np.real(s) / scale).astype(np.int16)
    iq[1::2] = np.round(np.imag(s) / scale).astype(np.int16)
    iq.tofile(path)
    _sidecar(path, fs, "sc16 (interleaved int16 I/Q)", meta, scale=scale)
    return scale


def read_sc16(path) -> np.ndarray:
    path = Path(path)
    side = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    scale = side.get("int16_scale", 1.0)
    iq = np.fromfile(path, dtype=np.int16).astype(np.float64) * scale
    return iq[0::2] + 1j * iq[1::2]
