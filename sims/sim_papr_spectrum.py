"""PAPR CCDF and PSD: Zak-OTFS vs CP-OFDM.

Expected result (and the honest caveat from the assessment): OTFS gives
essentially the SAME PAPR as CP-OFDM — no PA-efficiency win. Spectrum is
also comparable. Output: results/papr_ccdf.png, results/psd.png.
"""

import common  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from otfs import ZakOTFSConfig, ZakOTFSModem, OFDMConfig, OFDMModem, metrics
from common import RESULTS

rng = np.random.default_rng(7)

CFG = dict(M=32, N=16, scs_hz=15e3, qam_order=16, cp_len=8)
md_otfs = ZakOTFSModem(ZakOTFSConfig(**CFG))
# same waveform without the point-pilot power boost, to separate the
# intrinsic waveform PAPR from the embedded-pilot peak
md_flat = ZakOTFSModem(ZakOTFSConfig(**CFG, pilot_boost_db=0.0))
md_ofdm = OFDMModem(OFDMConfig(**CFG))
SPS = 4

frames_otfs, frames_flat, frames_ofdm = [], [], []
for _ in range(60):
    b = rng.integers(0, 2, md_otfs.bits_per_frame)
    _, s = md_otfs.modulate(b)
    frames_otfs.append(metrics.oversample_rrc(s, SPS))
    _, s = md_flat.modulate(b)
    frames_flat.append(metrics.oversample_rrc(s, SPS))
    b = rng.integers(0, 2, md_ofdm.bits_per_frame)
    _, s = md_ofdm.modulate(b)
    frames_ofdm.append(metrics.oversample_rrc(s, SPS))

s_otfs = np.concatenate(frames_otfs)
s_flat = np.concatenate(frames_flat)
s_ofdm = np.concatenate(frames_ofdm)

th, p_otfs = metrics.papr_ccdf(s_otfs)
_, p_flat = metrics.papr_ccdf(s_flat)
_, p_ofdm = metrics.papr_ccdf(s_ofdm)

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(th, np.maximum(p_otfs, 1e-7), "C0-",
            label="Zak-OTFS 16QAM (+30 dB point pilot)")
ax.semilogy(th, np.maximum(p_flat, 1e-7), "C2-.",
            label="Zak-OTFS 16QAM (no pilot boost)")
ax.semilogy(th, np.maximum(p_ofdm, 1e-7), "C3--", label="CP-OFDM 16QAM")
ax.set(xlabel="PAPR threshold (dB)", ylabel="P(PAPR > threshold)",
       title="PAPR CCDF (4x RRC-oversampled) — OTFS gives no PAPR advantage",
       xlim=(4, 13), ylim=(1e-6, 1))
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "papr_ccdf.png", dpi=130)

print(f"peak PAPR: OTFS(+pilot) {metrics.papr_db(s_otfs):.2f} dB, "
      f"OTFS(flat) {metrics.papr_db(s_flat):.2f} dB, "
      f"OFDM {metrics.papr_db(s_ofdm):.2f} dB")

fs_ovs = SPS * md_otfs.cfg.fs
f1, psd1 = metrics.psd(s_otfs, fs_ovs, nfft=2048)
f2, psd2 = metrics.psd(s_ofdm, fs_ovs, nfft=2048)
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(f1 / 1e3, psd1, "C0-", lw=0.9, label="Zak-OTFS")
ax2.plot(f2 / 1e3, psd2, "C3--", lw=0.9, label="CP-OFDM")
ax2.set(xlabel="frequency (kHz)", ylabel="PSD (dB rel. peak)",
        title="Power spectral density (RRC beta=0.25, 4x oversampled)",
        ylim=(-90, 3))
ax2.grid(alpha=0.3)
ax2.legend()
fig2.tight_layout()
fig2.savefig(RESULTS / "psd.png", dpi=130)
print("saved papr_ccdf.png, psd.png")
