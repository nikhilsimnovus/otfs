"""Gray-coded square QAM mapping / hard demapping.

Supported orders: 4 (QPSK), 16, 64. Constellations are normalised to unit
average symbol energy so Es/N0 bookkeeping stays simple across orders.
"""

import numpy as np

_SUPPORTED = (4, 16, 64)


def _pam_table(bits_per_axis: int) -> np.ndarray:
    """PAM amplitude for each bit-group value, Gray-coded.

    Returns `pam` such that `pam[b]` is the (unnormalised, odd-integer)
    amplitude carried by the bit-group value `b`.
    """
    m = 1 << bits_per_axis
    idx = np.arange(m)
    gray = idx ^ (idx >> 1)          # gray[position] = bit value at that level
    inv = np.argsort(gray)           # inv[bit value] = level position
    return 2 * inv - (m - 1)


def _norm(order: int) -> float:
    m = int(np.sqrt(order))
    es = 2.0 * (m * m - 1) / 3.0     # mean energy of odd-integer square QAM
    return 1.0 / np.sqrt(es)


def qam_constellation(order: int) -> np.ndarray:
    """All constellation points, indexed by symbol bit value (I bits MSB)."""
    if order not in _SUPPORTED:
        raise ValueError(f"order must be one of {_SUPPORTED}")
    bpa = int(np.log2(order)) // 2
    pam = _pam_table(bpa)
    bi, bq = np.meshgrid(np.arange(1 << bpa), np.arange(1 << bpa), indexing="ij")
    return (pam[bi] + 1j * pam[bq]).ravel() * _norm(order)


def qam_map(bits: np.ndarray, order: int) -> np.ndarray:
    """Map a bit array (multiple of log2(order) long) to QAM symbols."""
    if order not in _SUPPORTED:
        raise ValueError(f"order must be one of {_SUPPORTED}")
    bps = int(np.log2(order))
    bpa = bps // 2
    bits = np.asarray(bits, dtype=np.int64).reshape(-1, bps)
    weights = 1 << np.arange(bpa - 1, -1, -1)
    vi = bits[:, :bpa] @ weights
    vq = bits[:, bpa:] @ weights
    pam = _pam_table(bpa)
    return (pam[vi] + 1j * pam[vq]) * _norm(order)


def qam_demap_hard(syms: np.ndarray, order: int) -> np.ndarray:
    """Hard-decision demap: nearest constellation point -> bits."""
    if order not in _SUPPORTED:
        raise ValueError(f"order must be one of {_SUPPORTED}")
    bps = int(np.log2(order))
    bpa = bps // 2
    m = 1 << bpa
    syms = np.asarray(syms).ravel() / _norm(order)

    idx = np.arange(m)
    gray = idx ^ (idx >> 1)

    def axis_bits(vals):
        # quantise to nearest odd-integer level position 0..m-1
        pos = np.clip(np.round((vals + (m - 1)) / 2).astype(np.int64), 0, m - 1)
        return gray[pos]             # bit-group value at that level

    vi = axis_bits(syms.real)
    vq = axis_bits(syms.imag)
    out = np.empty((syms.size, bps), dtype=np.int64)
    for a in range(bpa):
        shift = bpa - 1 - a
        out[:, a] = (vi >> shift) & 1
        out[:, bpa + a] = (vq >> shift) & 1
    return out.ravel()
