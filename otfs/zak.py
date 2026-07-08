"""Discrete Zak transform (DZT) and symplectic finite Fourier transforms.

Conventions
-----------
The delay-Doppler (DD) grid is an (M, N) complex array `x[l, k]`:
  l = 0..M-1  delay bins   (delay resolution 1/B, B = M * subcarrier spacing)
  k = 0..N-1  Doppler bins (Doppler resolution 1/Tf, Tf = frame duration)

A frame is MN time samples. The DZT pair used here is unitary:

  Z[l, k] = (1/sqrt(N)) * sum_q  s[l + q M] e^{-j 2 pi k q / N}
  s[l + q M] = (1/sqrt(N)) * sum_k  Z[l, k] e^{+j 2 pi k q / N}

Vectorisation of DD grids is column-major (order='F'), i.e. index
i = l + k*M, so delay runs fastest.

The ISFFT/SFFT pair maps the DD grid to the time-frequency (TF) grid of a
multicarrier (OFDM-based) OTFS implementation. With rectangular pulses the
ISFFT + per-symbol OFDM IFFT chain is mathematically identical to the
inverse DZT — this is the "OTFS is an overlay on the existing OFDM engine"
property, verified in tests/test_transforms.py.
"""

import numpy as np
from scipy.linalg import dft


def dzt(s: np.ndarray, M: int, N: int) -> np.ndarray:
    """Time signal (MN,) -> DD grid (M, N)."""
    S = np.asarray(s).reshape(N, M).T          # S[l, q] = s[l + q M]
    return np.fft.fft(S, axis=1) / np.sqrt(N)


def idzt(x_dd: np.ndarray) -> np.ndarray:
    """DD grid (M, N) -> time signal (MN,)."""
    M, N = x_dd.shape
    S = np.fft.ifft(x_dd, axis=1) * np.sqrt(N)  # S[l, q]
    return S.T.reshape(-1)                      # s[l + q M]


def isfft(x_dd: np.ndarray) -> np.ndarray:
    """DD grid (M, N) -> TF grid (M subcarriers, N symbols).

    X[m, n] = (1/sqrt(MN)) sum_{l,k} x[l, k] e^{-j2pi m l / M} e^{+j2pi n k / N}
    """
    M, N = x_dd.shape
    return np.fft.fft(np.fft.ifft(x_dd, axis=1) * np.sqrt(N), axis=0) / np.sqrt(M)


def sfft(x_tf: np.ndarray) -> np.ndarray:
    """TF grid (M, N) -> DD grid (M, N). Inverse of isfft."""
    M, N = x_tf.shape
    return np.fft.fft(np.fft.ifft(x_tf, axis=0) * np.sqrt(M), axis=1) / np.sqrt(N)


def mc_otfs_modulate(x_dd: np.ndarray) -> np.ndarray:
    """Multicarrier OTFS TX: ISFFT then per-symbol OFDM modulator (no CP).

    Provided to make the overlay architecture explicit; identical output to
    idzt() for rectangular pulses.
    """
    M, N = x_dd.shape
    x_tf = isfft(x_dd)
    s_mat = np.fft.ifft(x_tf, axis=0) * np.sqrt(M)   # OFDM modulator per symbol
    return s_mat.T.reshape(-1)


def mc_otfs_demodulate(r: np.ndarray, M: int, N: int) -> np.ndarray:
    """Multicarrier OTFS RX: per-symbol OFDM demodulator then SFFT."""
    r_mat = np.asarray(r).reshape(N, M).T
    y_tf = np.fft.fft(r_mat, axis=0) / np.sqrt(M)    # OFDM demodulator
    return sfft(y_tf)


def zak_matrix(M: int, N: int) -> np.ndarray:
    """Unitary DZT as an (MN, MN) matrix acting on time-domain vectors.

    vec(dzt(s)) = Z @ s with order='F' vectorisation of the DD grid.
    """
    W = dft(N) / np.sqrt(N)
    return np.kron(W, np.eye(M))
