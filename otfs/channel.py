"""Doubly-dispersive (delay-Doppler) channel models.

Each channel is a set of discrete paths (integer sample delay, real-valued
Doppler, complex gain). Doppler is stored in cycles/sample so it is grid
agnostic; helper factories build 3GPP-style profiles (ETU, EVA) and an
NTN-LEO profile from physical parameters (speed, carrier, sample rate).

Two application modes:
  apply_linear : physically causal FIR-with-phase on an arbitrary-length
                 signal (what an SDR front-end would see).
  apply_cyclic : frame-level quasi-circulant model on an MN-sample frame,
                 equivalent to apply_linear on a CP-extended frame after CP
                 removal (verified in tests). Used to build exact effective
                 channel matrices for the equaliser.
"""

from dataclasses import dataclass, field

import numpy as np

C_LIGHT = 299_792_458.0


@dataclass
class DDPath:
    delay_samp: int          # integer sample delay
    doppler_cps: float       # Doppler in cycles per sample (can be tiny)
    gain: complex


@dataclass
class DDChannel:
    paths: list = field(default_factory=list)
    fs: float = 1.0          # sample rate, for reporting only

    @property
    def max_delay(self) -> int:
        return max((p.delay_samp for p in self.paths), default=0)

    def apply_linear(self, s: np.ndarray) -> np.ndarray:
        """r[n] = sum_p g_p s[n - l_p] e^{j 2 pi nu_p (n - l_p)}, causal."""
        s = np.asarray(s)
        n = np.arange(s.size)
        r = np.zeros(s.size, dtype=complex)
        for p in self.paths:
            l = p.delay_samp
            ph = np.exp(2j * np.pi * p.doppler_cps * (n[l:] - l))
            r[l:] += p.gain * s[: s.size - l] * ph
        return r

    def apply_cyclic(self, s: np.ndarray, n0: int = 0) -> np.ndarray:
        """Quasi-circulant frame model (CP already absorbed).

        r[n] = sum_p g_p s[(n - l_p) mod L] e^{j 2 pi nu_p (n + n0 - l_p)}
        The Doppler phase is NOT wrapped — it tracks true time. n0 is the
        absolute sample index of the first frame sample (e.g. the CP length
        when the frame clock starts at the CP), so the phase matches
        apply_linear on a CP-extended signal after CP removal.
        """
        s = np.asarray(s)
        n = np.arange(s.size)
        r = np.zeros(s.size, dtype=complex)
        for p in self.paths:
            ph = np.exp(2j * np.pi * p.doppler_cps * (n + n0 - p.delay_samp))
            r += p.gain * np.roll(s, p.delay_samp) * ph
        return r

    def time_matrix(self, length: int, n0: int = 0) -> np.ndarray:
        """Dense (length, length) matrix of apply_cyclic."""
        n = np.arange(length)
        H = np.zeros((length, length), dtype=complex)
        for p in self.paths:
            ph = p.gain * np.exp(2j * np.pi * p.doppler_cps * (n + n0 - p.delay_samp))
            cols = (n - p.delay_samp) % length
            H[n, cols] += ph
        return H


def _doppler_hz_to_cps(f_hz: float, fs: float) -> float:
    return f_hz / fs


def _build_multipath(delays_ns, powers_db, fs, fc_hz, speed_kmh, rng,
                     common_doppler_hz=0.0) -> DDChannel:
    f_max = fc_hz * (speed_kmh / 3.6) / C_LIGHT
    delays = np.round(np.asarray(delays_ns) * 1e-9 * fs).astype(int)
    p_lin = 10.0 ** (np.asarray(powers_db) / 10.0)
    p_lin = p_lin / p_lin.sum()
    paths = []
    for d, p in zip(delays, p_lin):
        g = np.sqrt(p / 2) * (rng.standard_normal() + 1j * rng.standard_normal())
        f_d = f_max * np.cos(rng.uniform(0, 2 * np.pi)) + common_doppler_hz
        paths.append(DDPath(int(d), _doppler_hz_to_cps(f_d, fs), g))
    return DDChannel(paths=paths, fs=fs)


def etu_channel(fs, fc_hz, speed_kmh, rng, common_doppler_hz=0.0) -> DDChannel:
    """3GPP Extended Typical Urban profile (long delay spread, 5 us)."""
    delays_ns = [0, 50, 120, 200, 230, 500, 1600, 2300, 5000]
    powers_db = [-1, -1, -1, 0, 0, 0, -3, -5, -7]
    return _build_multipath(delays_ns, powers_db, fs, fc_hz, speed_kmh, rng,
                            common_doppler_hz)


def eva_channel(fs, fc_hz, speed_kmh, rng, common_doppler_hz=0.0) -> DDChannel:
    """3GPP Extended Vehicular A profile."""
    delays_ns = [0, 30, 150, 310, 370, 710, 1090, 1730, 2510]
    powers_db = [0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]
    return _build_multipath(delays_ns, powers_db, fs, fc_hz, speed_kmh, rng,
                            common_doppler_hz)


def ntn_leo_channel(fs, residual_doppler_hz, rng, k_factor_db=10.0,
                    delay_spread_ns=1000.0, n_scatter=2) -> DDChannel:
    """NTN LEO service-link style channel.

    Strong LOS with a residual common Doppler (what remains after NR-NTN
    pre-compensation) plus a few weak scattered paths with small extra
    Doppler spread. residual_doppler_hz is the headline stressor.
    """
    k_lin = 10.0 ** (k_factor_db / 10.0)
    p_los = k_lin / (k_lin + 1.0)
    p_nlos = 1.0 / (k_lin + 1.0)
    paths = [DDPath(0, _doppler_hz_to_cps(residual_doppler_hz, fs),
                    np.sqrt(p_los) * np.exp(2j * np.pi * rng.uniform()))]
    for _ in range(n_scatter):
        d = int(np.round(rng.uniform(0.2, 1.0) * delay_spread_ns * 1e-9 * fs))
        g = np.sqrt(p_nlos / n_scatter / 2) * (rng.standard_normal()
                                               + 1j * rng.standard_normal())
        f_d = residual_doppler_hz * rng.uniform(0.9, 1.1)
        paths.append(DDPath(d, _doppler_hz_to_cps(f_d, fs), g))
    return DDChannel(paths=paths, fs=fs)


def awgn(r: np.ndarray, noise_var: float, rng) -> np.ndarray:
    n = np.sqrt(noise_var / 2) * (rng.standard_normal(r.shape)
                                  + 1j * rng.standard_normal(r.shape))
    return r + n
