"""DD-grid resource layout: embedded pilot, guard region, data mask,
pilot-based channel estimation and effective-channel reconstruction.

The embedded-pilot scheme follows the standard Zak-OTFS approach: a single
high-power point pilot at (l_p, k_p) surrounded by a guard region sized to
the expected channel spreads. At the receiver the effective DD channel taps
h_w[dl, dk] are read off directly from the received pilot neighbourhood
(twisted-convolution response), then an effective channel matrix is
reconstructed for equalisation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class GridLayout:
    M: int
    N: int
    l_p: int                 # pilot delay index
    k_p: int                 # pilot Doppler index
    l_max: int               # max channel delay spread (bins)
    k_max: int               # max channel Doppler spread (bins, one-sided)
    k_extra: int             # extra Doppler guard for fractional-Doppler leakage
    pilot_amp: float         # linear pilot amplitude
    data_mask: np.ndarray    # (M, N) bool, True where data symbols go

    @property
    def n_data(self) -> int:
        return int(self.data_mask.sum())

    @property
    def data_idx_flat(self) -> np.ndarray:
        """Flat (order='F') indices of data cells."""
        return np.flatnonzero(self.data_mask.ravel(order="F"))

    def pilot_grid(self) -> np.ndarray:
        g = np.zeros((self.M, self.N), dtype=complex)
        g[self.l_p, self.k_p] = self.pilot_amp
        return g

    def place_data(self, syms: np.ndarray) -> np.ndarray:
        """Data symbols + pilot -> full (M, N) DD grid."""
        if syms.size != self.n_data:
            raise ValueError(f"expected {self.n_data} symbols, got {syms.size}")
        gf = self.pilot_grid().ravel(order="F")
        gf[self.data_idx_flat] = syms          # same ordering the detector uses
        return gf.reshape(self.M, self.N, order="F")


def make_layout(M, N, l_max, k_max, k_extra=2, pilot_boost_db=30.0) -> GridLayout:
    """Embedded pilot at grid centre with guard region.

    Guard (no data) region:
      delay   l in [l_p - l_max, l_p + l_max]
      Doppler k in [k_p - 2 k_max - k_extra, k_p + 2 k_max + k_extra]
    which keeps data-to-pilot and pilot-to-data interference out of the
    estimation read window under the crystallization condition.
    """
    l_p, k_p = M // 2, N // 2
    mask = np.ones((M, N), dtype=bool)
    dl = np.arange(M).reshape(-1, 1) - l_p
    dk = (np.arange(N).reshape(1, -1) - k_p + N // 2) % N - N // 2
    guard = (np.abs(dl) <= l_max) & (np.abs(dk) <= 2 * k_max + k_extra)
    mask[guard] = False
    pilot_amp = 10.0 ** (pilot_boost_db / 20.0)
    return GridLayout(M, N, l_p, k_p, l_max, k_max, k_extra, pilot_amp, mask)


def estimate_taps(y_dd: np.ndarray, layout: GridLayout, noise_var: float,
                  thresh: float = 3.0):
    """Read effective DD channel taps off the received pilot response.

    Returns list of (dl, dk, h) with dl in [0, l_max], dk in
    [-(k_max + k_extra), +(k_max + k_extra)], keeping taps whose magnitude
    exceeds thresh * sqrt(noise_var).

    The raw readout h = y / pilot_amp is used directly: the taps are
    referenced to the pilot position (l_p, k_p), and h_matrix_from_taps
    references its reconstruction phases to l_p accordingly, so the pair is
    self-consistent (exact for on-grid Doppler; see tests).
    """
    M, N = layout.M, layout.N
    taps = []
    k_read = min(layout.k_max + layout.k_extra, (N - 1) // 2)
    floor = thresh * np.sqrt(noise_var)
    for dl in range(0, layout.l_max + 1):
        for dk in range(-k_read, k_read + 1):
            l = layout.l_p + dl
            k = (layout.k_p + dk) % N
            v = y_dd[l, k]
            if np.abs(v) > floor:
                taps.append((dl, dk, v / layout.pilot_amp))
    return taps


def h_matrix_from_taps(taps, layout: GridLayout) -> np.ndarray:
    """Effective DD channel matrix (MN, MN) from estimated taps.

    Implements the discrete twisted convolution of the Zak domain: a source
    symbol at (l0, k0) contributes to output cell
    (l, k) = ((l0 + dl) mod M, (k0 + dk) mod N) with phase

      e^{j 2 pi dk (l0 - eps M - l_p) / (M N)} * e^{-j 2 pi k0 eps / N}

    where eps = 1 when l0 + dl wraps past M (delay wrap crosses a frame
    boundary, picking up the DD quasi-periodicity phase). The Doppler of
    each tap is approximated by its integer bin offset dk, exact for
    on-grid Doppler and a small mismatch for fractional Doppler (standard
    model-free reconstruction).
    """
    M, N = layout.M, layout.N
    MN = M * N
    H = np.zeros((MN, MN), dtype=complex)
    l0 = np.arange(M).reshape(-1, 1)      # source delay
    k0 = np.arange(N).reshape(1, -1)      # source Doppler
    for dl, dk, h in taps:
        l_t = l0 + dl
        eps = (l_t >= M).astype(int)
        l_t = l_t % M
        k_t = (k0 + dk) % N
        phase = np.exp(2j * np.pi * dk * (l0 - eps * M - layout.l_p) / MN) \
            * np.exp(-2j * np.pi * k0 * eps / N)
        src = (l0 + k0 * M).ravel()                    # order='F' flat index
        dst = (l_t + k_t * M).ravel()
        H[dst, src] += h * np.broadcast_to(phase, (M, N)).ravel()
    return H


def _dirichlet(delta, N: int) -> np.ndarray:
    """Periodic Dirichlet kernel D_N(delta) = (1/N) sum_q e^{j2pi delta q/N}.

    This is the Doppler-domain leakage pattern of a rectangular-pulse frame:
    a path with fractional Doppler nu contributes h * D_N(nu - dk) at
    Doppler offset dk. D_N(integer != 0 mod N) = 0, D_N(0) = 1.
    """
    delta = np.asarray(delta, dtype=float)
    num = np.sin(np.pi * delta)
    den = N * np.sin(np.pi * delta / N)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(np.abs(den) < 1e-12, 1.0, num / np.where(
            np.abs(den) < 1e-12, 1.0, den))
    return np.exp(1j * np.pi * delta * (N - 1) / N) * mag


def estimate_paths(y_dd: np.ndarray, layout: GridLayout, noise_var: float,
                   n0: int = 0, thresh: float = 3.0, max_comp: int = 10):
    """Parametric DD channel estimation from the embedded pilot.

    For each delay row of the pilot read region, fits up to max_comp
    Dirichlet components with fractional Doppler (CLEAN-style: fit the
    strongest component, subtract, repeat). This handles fractional Doppler
    exactly, which the model-free tap readout cannot (rectangular-pulse
    Doppler leakage spans the whole axis). max_comp matters when several
    physical paths collapse into one delay bin with sub-bin Doppler spacing
    (e.g. EVA at low sample rates): extra pseudo-components approximate the
    unresolvable aggregate, cutting estimated-CSI BER several-fold.

    Returns a list of (delay_samples, doppler_cps, complex gain) tuples,
    directly usable to build a DDChannel / exact effective-H. n0 is the
    frame clock offset (CP length) so gains are dereferenced consistently
    with the physical channel.
    """
    M, N = layout.M, layout.N
    MN = M * N
    k_read = min(layout.k_max + layout.k_extra, (N - 1) // 2)
    dks = np.arange(-k_read, k_read + 1)
    floor = thresh * np.sqrt(noise_var) / layout.pilot_amp
    paths = []
    for dl in range(0, layout.l_max + 1):
        row = y_dd[layout.l_p + dl, (layout.k_p + dks) % N] / layout.pilot_amp
        p = row.copy()
        for _ in range(max_comp):
            pk = int(np.argmax(np.abs(p)))
            if np.abs(p[pk]) < floor or abs(dks[pk]) > layout.k_max:
                break
            k0 = dks[pk]

            def neg_fit_power(nu):
                d = _dirichlet(nu - dks, N)
                return -np.abs(np.vdot(d, p)) ** 2 / np.real(np.vdot(d, d))

            res = minimize_scalar(neg_fit_power, bounds=(k0 - 0.6, k0 + 0.6),
                                  method="bounded",
                                  options={"xatol": 1e-6})
            nu = float(res.x)
            d = _dirichlet(nu - dks, N)
            a = np.vdot(d, p) / np.real(np.vdot(d, d))   # LS gain
            p = p - a * d
            # observed amplitude includes the pilot-position/CP phase:
            # a = g * e^{j 2 pi nu (l_p + n0) / MN}
            g = a * np.exp(-2j * np.pi * nu * (layout.l_p + n0) / MN)
            paths.append((dl, nu / MN, g))
    return paths
