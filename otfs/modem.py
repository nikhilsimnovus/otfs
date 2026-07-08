"""Zak-OTFS modem: bits <-> DD grid <-> time-domain IQ frame.

TX chain : bits -> QAM -> DD grid (data + embedded pilot) -> inverse DZT
           -> frame CP -> IQ samples
RX chain : IQ -> CP removal -> DZT -> pilot channel estimation ->
           LMMSE / MP detection -> QAM demap -> bits

The effective DD channel matrix can be built two ways:
  build_h_true : exact, from the channel object (perfect-CSI reference)
  taps read off the received pilot + h_matrix_from_taps (realistic)
"""

from dataclasses import dataclass, field

import numpy as np

from .qam import qam_map, qam_demap_hard, qam_constellation
from .zak import dzt, idzt, zak_matrix
from .grid import (GridLayout, make_layout, estimate_taps,
                   h_matrix_from_taps, estimate_paths)
from .detectors import lmmse_detect, mp_detect


@dataclass
class ZakOTFSConfig:
    M: int = 32                  # delay bins (also "subcarriers" of overlay OFDM)
    N: int = 16                  # Doppler bins (also OFDM symbols per frame)
    scs_hz: float = 15e3         # subcarrier spacing -> fs = M * scs
    qam_order: int = 16
    cp_len: int = 8              # frame cyclic prefix (>= max delay spread)
    l_max: int = 4               # guard sizing: max expected delay bins
    k_max: int = 3               # guard sizing: max expected Doppler bins
    k_extra: int = 2
    pilot_boost_db: float = 30.0

    @property
    def fs(self) -> float:
        return self.M * self.scs_hz

    @property
    def frame_len(self) -> int:
        return self.M * self.N

    @property
    def frame_duration(self) -> float:
        return self.frame_len / self.fs

    @property
    def delay_res_s(self) -> float:
        return 1.0 / self.fs

    @property
    def doppler_res_hz(self) -> float:
        return 1.0 / self.frame_duration


class ZakOTFSModem:
    def __init__(self, cfg: ZakOTFSConfig):
        self.cfg = cfg
        self.layout: GridLayout = make_layout(
            cfg.M, cfg.N, cfg.l_max, cfg.k_max, cfg.k_extra, cfg.pilot_boost_db)
        self._zmat = None

    # ---------------------------------------------------------------- TX
    @property
    def bits_per_frame(self) -> int:
        return self.layout.n_data * int(np.log2(self.cfg.qam_order))

    def modulate(self, bits: np.ndarray):
        """bits -> (x_dd grid, time-domain frame with CP)."""
        syms = qam_map(bits, self.cfg.qam_order)
        x_dd = self.layout.place_data(syms)
        s = idzt(x_dd)
        s_cp = np.concatenate([s[-self.cfg.cp_len:], s])
        return x_dd, s_cp

    def strip_cp(self, r_cp: np.ndarray) -> np.ndarray:
        return r_cp[self.cfg.cp_len:self.cfg.cp_len + self.cfg.frame_len]

    def to_dd(self, r_frame: np.ndarray) -> np.ndarray:
        return dzt(r_frame, self.cfg.M, self.cfg.N)

    # ------------------------------------------------- channel matrices
    @property
    def zmat(self) -> np.ndarray:
        if self._zmat is None:
            self._zmat = zak_matrix(self.cfg.M, self.cfg.N)
        return self._zmat

    def build_h_true(self, chan) -> np.ndarray:
        """Exact effective DD channel matrix Z H_t Z^H (perfect CSI).

        Uses the quasi-circulant frame model; the CP makes the physical
        linear channel equal to this model (verified in tests). n0=cp_len
        accounts for the Doppler phase accrued while the CP is on air.
        """
        H_t = chan.time_matrix(self.cfg.frame_len, n0=self.cfg.cp_len)
        return self.zmat @ H_t @ self.zmat.conj().T

    def estimate_h(self, y_dd: np.ndarray, noise_var: float, thresh: float = 3.0):
        """Model-free pilot tap readout -> (H_est, taps).

        Exact for on-grid (integer-bin) Doppler; degrades with fractional
        Doppler because rectangular-pulse leakage is truncated at the read
        window. Prefer estimate_h_parametric for realistic channels.
        """
        taps = estimate_taps(y_dd, self.layout, noise_var, thresh)
        return h_matrix_from_taps(taps, self.layout), taps

    def estimate_h_parametric(self, y_dd: np.ndarray, noise_var: float,
                              thresh: float = 3.0, max_comp: int = 10):
        """Parametric (fractional-Doppler) estimate -> (H_est, est_channel).

        Fits Dirichlet components per delay row of the pilot response, then
        rebuilds the exact effective channel from the estimated paths.
        """
        from .channel import DDChannel, DDPath
        trips = estimate_paths(y_dd, self.layout, noise_var,
                               n0=self.cfg.cp_len, thresh=thresh,
                               max_comp=max_comp)
        est = DDChannel(paths=[DDPath(dl, nu, g) for dl, nu, g in trips],
                        fs=self.cfg.fs)
        return self.build_h_true(est), est

    # ---------------------------------------------------------------- RX
    def demodulate(self, y_dd: np.ndarray, H: np.ndarray, noise_var: float,
                   detector: str = "lmmse", **det_kwargs):
        """Equalise + demap. Returns (bits, data symbol estimates)."""
        y = y_dd.ravel(order="F")
        # subtract the known pilot's contribution through the channel
        y_clean = y - H @ self.layout.pilot_grid().ravel(order="F")
        H_d = H[:, self.layout.data_idx_flat]
        if detector == "lmmse":
            x_hat = lmmse_detect(H_d, y_clean, noise_var)
        elif detector == "mp":
            x_hat = mp_detect(H_d, y_clean, noise_var,
                              qam_constellation(self.cfg.qam_order), **det_kwargs)
        else:
            raise ValueError(f"unknown detector {detector}")
        bits = qam_demap_hard(x_hat, self.cfg.qam_order)
        return bits, x_hat

    # --------------------------------------------------------- one-shot
    def run_frame(self, bits, chan, noise_var, rng, csi: str = "est",
                  detector: str = "lmmse", **det_kwargs):
        """Full TX -> channel -> RX pass.

        csi: 'perfect' (exact H from channel object), 'est' (parametric
        pilot estimation), 'est-taps' (model-free tap readout).
        """
        from .channel import awgn
        x_dd, s_cp = self.modulate(bits)
        r_cp = chan.apply_linear(s_cp)
        r_cp = awgn(r_cp, noise_var, rng)
        y_dd = self.to_dd(self.strip_cp(r_cp))
        if csi == "perfect":
            H = self.build_h_true(chan)
        elif csi == "est":
            H, _ = self.estimate_h_parametric(y_dd, noise_var)
        elif csi == "est-taps":
            H, _ = self.estimate_h(y_dd, noise_var)
        else:
            raise ValueError(f"unknown csi mode {csi}")
        return self.demodulate(y_dd, H, noise_var, detector, **det_kwargs)
