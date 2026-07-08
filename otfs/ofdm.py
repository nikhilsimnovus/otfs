"""CP-OFDM baseline modem (the NR waveform OTFS is compared against).

M subcarriers x N symbols per frame, per-symbol CP, one-tap MMSE
equalisation with the exact ICI-free diagonal channel (time-averaged path
phase over each symbol window). This is a *favourable* baseline — real NR
channel estimation is worse — so any error floor it shows versus Doppler is
a lower bound on CP-OFDM degradation, which is the honest comparison for
the "where OTFS wins" narrative.
"""

from dataclasses import dataclass

import numpy as np

from .qam import qam_map, qam_demap_hard


@dataclass
class OFDMConfig:
    M: int = 32              # subcarriers
    N: int = 16              # OFDM symbols per frame
    scs_hz: float = 15e3
    qam_order: int = 16
    cp_len: int = 8          # per-symbol CP

    @property
    def fs(self) -> float:
        return self.M * self.scs_hz

    @property
    def sym_len(self) -> int:
        return self.M + self.cp_len

    @property
    def frame_len(self) -> int:
        return self.sym_len * self.N


class OFDMModem:
    def __init__(self, cfg: OFDMConfig):
        self.cfg = cfg

    @property
    def bits_per_frame(self) -> int:
        return self.cfg.M * self.cfg.N * int(np.log2(self.cfg.qam_order))

    def modulate(self, bits: np.ndarray):
        cfg = self.cfg
        syms = qam_map(bits, cfg.qam_order).reshape(cfg.M, cfg.N, order="F")
        s_mat = np.fft.ifft(syms, axis=0) * np.sqrt(cfg.M)
        out = np.empty(cfg.frame_len, dtype=complex)
        for n in range(cfg.N):
            blk = np.concatenate([s_mat[-cfg.cp_len:, n], s_mat[:, n]])
            out[n * cfg.sym_len:(n + 1) * cfg.sym_len] = blk
        return syms, out

    def _diag_channel(self, chan) -> np.ndarray:
        """Exact ICI-free per-symbol frequency response H[m, n].

        For integer path delays, the diagonal of the per-symbol channel is
        sum_p g_p <e^{j 2 pi nu_p (t - l_p)}>_window e^{-j 2 pi m l_p / M}.
        """
        cfg = self.cfg
        H = np.zeros((cfg.M, cfg.N), dtype=complex)
        m = np.arange(cfg.M)
        for n in range(cfg.N):
            w0 = n * cfg.sym_len + cfg.cp_len          # FFT window start
            win = np.arange(w0, w0 + cfg.M)
            for p in chan.paths:
                avg_ph = np.mean(np.exp(2j * np.pi * p.doppler_cps
                                        * (win - p.delay_samp)))
                H[:, n] += p.gain * avg_ph * np.exp(-2j * np.pi * m
                                                    * p.delay_samp / cfg.M)
        return H

    def demodulate(self, r: np.ndarray, chan, noise_var: float):
        """One-tap MMSE with exact diagonal CSI. Returns (bits, sym ests)."""
        cfg = self.cfg
        Y = np.empty((cfg.M, cfg.N), dtype=complex)
        for n in range(cfg.N):
            w0 = n * cfg.sym_len + cfg.cp_len
            Y[:, n] = np.fft.fft(r[w0:w0 + cfg.M]) / np.sqrt(cfg.M)
        H = self._diag_channel(chan)
        X_hat = np.conj(H) * Y / (np.abs(H) ** 2 + noise_var)
        bits = qam_demap_hard(X_hat.ravel(order="F"), cfg.qam_order)
        return bits, X_hat.ravel(order="F")

    def run_frame(self, bits, chan, noise_var, rng):
        from .channel import awgn
        _, s = self.modulate(bits)
        r = chan.apply_linear(s)
        r = awgn(r, noise_var, rng)
        return self.demodulate(r, chan, noise_var)
