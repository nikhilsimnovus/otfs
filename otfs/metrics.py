"""Waveform QA metrics: BER, EVM, PAPR (+CCDF), PSD."""

import numpy as np
from scipy import signal as sig


def ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    bits_tx = np.asarray(bits_tx).ravel()
    bits_rx = np.asarray(bits_rx).ravel()
    return float(np.mean(bits_tx != bits_rx))


def evm_rms(ref_syms: np.ndarray, rx_syms: np.ndarray) -> float:
    """RMS EVM in percent, normalised to reference RMS power."""
    ref = np.asarray(ref_syms).ravel()
    rx = np.asarray(rx_syms).ravel()
    return float(100.0 * np.sqrt(np.mean(np.abs(rx - ref) ** 2)
                                 / np.mean(np.abs(ref) ** 2)))


def papr_db(s: np.ndarray) -> float:
    p = np.abs(np.asarray(s)) ** 2
    return float(10 * np.log10(p.max() / p.mean()))


def papr_ccdf(s: np.ndarray, thresholds_db=None):
    """CCDF of instantaneous power over mean. Returns (thresholds_db, prob)."""
    p = np.abs(np.asarray(s).ravel()) ** 2
    p = p / p.mean()
    if thresholds_db is None:
        thresholds_db = np.linspace(0, 13, 66)
    th = 10 ** (np.asarray(thresholds_db) / 10)
    prob = np.array([(p > t).mean() for t in th])
    return np.asarray(thresholds_db), prob


def psd(s: np.ndarray, fs: float, nfft: int = 1024):
    """Welch PSD, fftshifted, in dB relative to peak. Returns (f_hz, psd_db)."""
    f, pxx = sig.welch(np.asarray(s), fs=fs, nperseg=min(nfft, len(s)),
                       return_onesided=False, detrend=False)
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    pdb = 10 * np.log10(np.maximum(pxx, 1e-20))
    return f, pdb - pdb.max()


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine filter taps (span symbols, sps samples/symbol)."""
    n = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    taps = np.zeros_like(n, dtype=float)
    for i, t in enumerate(n):
        if abs(t) < 1e-10:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(t) - 1 / (4 * beta)) < 1e-10:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * t * (1 - beta))
                   + 4 * beta * t * np.cos(np.pi * t * (1 + beta)))
            den = np.pi * t * (1 - (4 * beta * t) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps ** 2))


def oversample_rrc(s: np.ndarray, sps: int, beta: float = 0.25,
                   span: int = 16) -> np.ndarray:
    """Pulse-shape a sample stream with an RRC interpolator (for PAPR/PSD
    analysis and SDR playback at a higher DAC rate)."""
    taps = rrc_taps(beta, sps, span)
    return sig.upfirdn(taps, np.asarray(s), up=sps) * np.sqrt(sps)
