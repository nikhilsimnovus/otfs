"""End-to-end Zak-OTFS demo: one frame through an NTN-LEO channel.

Produces:
  results/demo_dd_grids.png      TX/RX DD grids + received pilot response
  results/demo_constellation.png equalised data symbols
  results/demo_frame.cf32/.sc16  SDR-ready IQ (+ JSON sidecars)
and prints frame parameters, estimated channel and link metrics.
"""

import common  # noqa: F401  (path setup)

import numpy as np
import matplotlib.pyplot as plt

from otfs import (ZakOTFSConfig, ZakOTFSModem, ntn_leo_channel, metrics,
                  iq_io)
from otfs.channel import awgn
from common import RESULTS

rng = np.random.default_rng(42)

# k_max sized for the NTN residual Doppler below (5 kHz = 5.33 bins);
# the crystallization condition requires the guard to cover the channel
# Doppler spread, exactly like a real deployment parameterisation.
cfg = ZakOTFSConfig(M=32, N=16, scs_hz=15e3, qam_order=16, cp_len=8, k_max=6)
md = ZakOTFSModem(cfg)

print("=== Zak-OTFS frame parameters ===")
print(f"grid M x N            : {cfg.M} x {cfg.N} (delay x Doppler)")
print(f"sample rate           : {cfg.fs/1e3:.0f} kHz")
print(f"frame duration        : {cfg.frame_duration*1e3:.3f} ms (+CP)")
print(f"delay resolution      : {cfg.delay_res_s*1e6:.2f} us")
print(f"Doppler resolution    : {cfg.doppler_res_hz:.1f} Hz")
print(f"data cells / frame    : {md.layout.n_data} of {cfg.frame_len}")
print(f"bits / frame (16QAM)  : {md.bits_per_frame}")

# NTN LEO: 5 kHz residual Doppler after pre-compensation
resid_dop = 5e3
chan = ntn_leo_channel(cfg.fs, resid_dop, rng)
print(f"\nchannel: NTN-LEO, residual Doppler {resid_dop/1e3:.1f} kHz "
      f"(= {resid_dop/cfg.doppler_res_hz:.2f} Doppler bins)")

snr_db = 20.0
noise_var = 10 ** (-snr_db / 10)

bits = rng.integers(0, 2, md.bits_per_frame)
x_dd, s_cp = md.modulate(bits)
r_cp = awgn(chan.apply_linear(s_cp), noise_var, rng)
y_dd = md.to_dd(md.strip_cp(r_cp))

H_est, est_chan = md.estimate_h_parametric(y_dd, noise_var)
print("\nestimated DD paths (delay bin, Doppler Hz, |gain|):")
for p in est_chan.paths:
    print(f"  l={p.delay_samp}  f_D={p.doppler_cps*cfg.fs:+8.1f} Hz  "
          f"|g|={abs(p.gain):.3f}")
print("true paths:")
for p in chan.paths:
    print(f"  l={p.delay_samp}  f_D={p.doppler_cps*cfg.fs:+8.1f} Hz  "
          f"|g|={abs(p.gain):.3f}")

rx_bits, x_hat = md.demodulate(y_dd, H_est, noise_var)
from otfs.qam import qam_map
ref = qam_map(bits, cfg.qam_order)

print(f"\nSNR {snr_db:.0f} dB -> BER {metrics.ber(bits, rx_bits):.2e}, "
      f"EVM {metrics.evm_rms(ref, x_hat):.1f}%, "
      f"PAPR {metrics.papr_db(s_cp):.2f} dB")

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].imshow(np.abs(x_dd), aspect="auto", origin="lower", cmap="viridis")
ax[0].set(title="TX DD grid |x[l,k]| (pilot + data)",
          xlabel="Doppler bin k", ylabel="delay bin l")
ax[1].imshow(np.abs(y_dd), aspect="auto", origin="lower", cmap="viridis")
ax[1].set(title="RX DD grid |y[l,k]|", xlabel="Doppler bin k")
pr = np.abs(y_dd[md.layout.l_p:md.layout.l_p + md.layout.l_max + 1, :])
ax[2].imshow(pr, aspect="auto", origin="lower", cmap="magma")
ax[2].set(title="received pilot response (channel snapshot)",
          xlabel="Doppler bin k", ylabel="delay offset dl")
fig.tight_layout()
fig.savefig(RESULTS / "demo_dd_grids.png", dpi=130)

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.plot(x_hat.real, x_hat.imag, ".", ms=3, alpha=0.6, label="equalised")
from otfs.qam import qam_constellation
c = qam_constellation(cfg.qam_order)
ax2.plot(c.real, c.imag, "r+", ms=10, label="ideal 16QAM")
ax2.set(title=f"Zak-OTFS equalised constellation @ {snr_db:.0f} dB, "
        f"NTN {resid_dop/1e3:.0f} kHz residual Doppler",
        xlabel="I", ylabel="Q")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(RESULTS / "demo_constellation.png", dpi=130)

# ------------------------------------------------------------ IQ export
meta = {"waveform": "zak-otfs", "M": cfg.M, "N": cfg.N,
        "scs_hz": cfg.scs_hz, "cp_len": cfg.cp_len,
        "qam_order": cfg.qam_order, "pilot": [md.layout.l_p, md.layout.k_p],
        "channel": "clean TX frame (no channel)"}
iq_io.write_cf32(RESULTS / "demo_frame.cf32", s_cp, cfg.fs, meta)
iq_io.write_sc16(RESULTS / "demo_frame.sc16", s_cp, cfg.fs, meta)
# oversampled version for DAC playback / spectrum analysis
s_ovs = metrics.oversample_rrc(s_cp, sps=4)
iq_io.write_cf32(RESULTS / "demo_frame_x4.cf32", s_ovs, 4 * cfg.fs,
                 {**meta, "oversampling": 4, "pulse": "RRC beta=0.25"})
print(f"\nIQ written to {RESULTS}\\demo_frame.[cf32|sc16] "
      f"and demo_frame_x4.cf32 (4x RRC oversampled)")
