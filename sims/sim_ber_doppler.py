"""BER comparison: Zak-OTFS vs CP-OFDM under mobility and NTN Doppler.

Study 1: BER vs UE speed (EVA channel, fc=3.5 GHz FR1, 16QAM, fixed SNR).
Study 2: BER vs SNR at 500 km/h (terrestrial worst case).
Study 3: BER vs residual NTN Doppler (LEO, post pre-compensation).

CP-OFDM uses exact ICI-free one-tap CSI (favourable baseline); Zak-OTFS is
shown with perfect CSI and with realistic embedded-pilot estimation.

Output: results/ber_vs_speed.png, ber_vs_snr.png, ber_vs_ntn_doppler.png
        + printed tables.
"""

import common  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from otfs import (ZakOTFSConfig, ZakOTFSModem, OFDMConfig, OFDMModem,
                  eva_channel, ntn_leo_channel)
from common import RESULTS

rng = np.random.default_rng(2026)

FC = 3.5e9
CFG = dict(M=32, N=16, scs_hz=15e3, qam_order=16, cp_len=8)
# k_max=4 covers 1000 km/h at 3.5 GHz (3.2 kHz = 3.5 Doppler bins); with
# N=16 the pilot guard spans the full Doppler axis either way, so capacity
# is unchanged and the OFDM/OTFS comparison stays rate-fair.
md_otfs = ZakOTFSModem(ZakOTFSConfig(**CFG, k_max=4))
md_ofdm = OFDMModem(OFDMConfig(**CFG))
FS = md_otfs.cfg.fs

# NTN study needs a wider Doppler budget: 5 kHz residual = 5.33 bins
md_otfs_ntn = ZakOTFSModem(ZakOTFSConfig(**CFG, k_max=6))


def run_point(make_chan, snr_db, n_frames, modes=("ofdm", "otfs-perfect",
                                                  "otfs-est"), md=None):
    md_otfs_l = md or md_otfs
    noise_var = 10 ** (-snr_db / 10)
    errs = {m: 0 for m in modes}
    tot = {m: 0 for m in modes}
    for _ in range(n_frames):
        chan = make_chan()
        if "ofdm" in modes:
            b = rng.integers(0, 2, md_ofdm.bits_per_frame)
            rx, _ = md_ofdm.run_frame(b, chan, noise_var, rng)
            errs["ofdm"] += int(np.sum(rx != b))
            tot["ofdm"] += b.size
        for mode, csi in (("otfs-perfect", "perfect"), ("otfs-est", "est")):
            if mode in modes:
                b = rng.integers(0, 2, md_otfs_l.bits_per_frame)
                rx, _ = md_otfs_l.run_frame(b, chan, noise_var, rng, csi=csi)
                errs[mode] += int(np.sum(rx != b))
                tot[mode] += b.size
    return {m: errs[m] / max(tot[m], 1) for m in modes}


LABEL = {"ofdm": "CP-OFDM (ideal one-tap CSI)",
         "otfs-perfect": "Zak-OTFS (perfect CSI)",
         "otfs-est": "Zak-OTFS (pilot-estimated CSI)"}
STYLE = {"ofdm": "o-C3", "otfs-perfect": "s-C0", "otfs-est": "^--C0"}


def plot(x, curves, xlabel, title, fname, xlog=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, y in curves.items():
        y = np.maximum(np.asarray(y), 1e-6)
        ax.semilogy(x, y, STYLE[m][:-2] + STYLE[m][-2:], label=LABEL[m])
    if xlog:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel="uncoded BER", title=title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=130)
    print(f"saved {fname}")


# ---------------------------------------------------- 1: BER vs speed
speeds = [3, 30, 120, 250, 500, 1000]
snr_db = 22.0
n_frames = 24
curves = {m: [] for m in LABEL}
print(f"\n=== BER vs speed (EVA, fc={FC/1e9:.1f} GHz, SNR={snr_db:.0f} dB, "
      f"16QAM, {n_frames} frames/point) ===")
print(f"{'km/h':>6} {'CP-OFDM':>12} {'OTFS perfect':>14} {'OTFS est':>12}")
for v in speeds:
    res = run_point(lambda: eva_channel(FS, FC, v, rng), snr_db, n_frames)
    for m in curves:
        curves[m].append(res[m])
    print(f"{v:>6} {res['ofdm']:>12.2e} {res['otfs-perfect']:>14.2e} "
          f"{res['otfs-est']:>12.2e}")
plot(speeds, curves, "UE speed (km/h)",
     f"EVA channel, {FC/1e9:.1f} GHz, 16QAM, SNR {snr_db:.0f} dB",
     "ber_vs_speed.png")

# ------------------------------------------------------ 2: BER vs SNR
speed = 500
snrs = [8, 12, 16, 20, 24, 28]
n_frames = 24
curves = {m: [] for m in LABEL}
print(f"\n=== BER vs SNR (EVA, {speed} km/h) ===")
print(f"{'SNR dB':>6} {'CP-OFDM':>12} {'OTFS perfect':>14} {'OTFS est':>12}")
for s in snrs:
    res = run_point(lambda: eva_channel(FS, FC, speed, rng), s, n_frames)
    for m in curves:
        curves[m].append(res[m])
    print(f"{s:>6} {res['ofdm']:>12.2e} {res['otfs-perfect']:>14.2e} "
          f"{res['otfs-est']:>12.2e}")
plot(snrs, curves, "SNR (dB)",
     f"EVA channel, {speed} km/h, {FC/1e9:.1f} GHz, 16QAM",
     "ber_vs_snr.png")

# ------------------------------------------- 3: BER vs NTN residual Doppler
resid = [200, 500, 1000, 2000, 3500, 5000]
snr_db = 20.0
n_frames = 24
curves = {m: [] for m in LABEL}
print(f"\n=== BER vs NTN residual Doppler (LEO, SNR={snr_db:.0f} dB) ===")
print(f"{'Hz':>7} {'CP-OFDM':>12} {'OTFS perfect':>14} {'OTFS est':>12}")
for f in resid:
    res = run_point(lambda: ntn_leo_channel(FS, f, rng), snr_db, n_frames,
                    md=md_otfs_ntn)
    for m in curves:
        curves[m].append(res[m])
    print(f"{f:>7} {res['ofdm']:>12.2e} {res['otfs-perfect']:>14.2e} "
          f"{res['otfs-est']:>12.2e}")
plot(resid, curves, "residual Doppler after NTN pre-compensation (Hz)",
     f"NTN-LEO channel, 16QAM, SNR {snr_db:.0f} dB",
     "ber_vs_ntn_doppler.png", xlog=True)
