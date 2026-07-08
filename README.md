# Zak-OTFS Delay-Doppler Waveform — Software Implementation

Phase-1 (software) deliverable of the **Cohere Zak-OTFS feasibility assessment**
(Confluence: *Cohere Zak-OTFS (Delay-Doppler) Waveform Support — Feasibility & QA
Assessment for Simnovator / UESIM (Amarisoft)*, page 1367048194).

A complete, tested waveform **generator and receiver** for Zak-OTFS
(delay-Doppler modulation) with a CP-OFDM baseline, doubly-dispersive channel
models, DD-domain channel estimation/equalisation, waveform QA metrics and
SDR-ready IQ export. Phase 2 (not in this repo) ports the chain to an SDR
card / Amarisoft TRX IQ injection.

---

## 1. What the customer (Cohere) needs — and what this covers

Cohere Technologies originated OTFS; their current formulation is **Zak-OTFS**
(productised as **Pulsone**, Oct 2025). They position it for **NTN/LEO, high
mobility and ISAC**, claiming it runs as a **2-D pre/post-transform overlay on
existing OFDM hardware**. The feasibility page maps their ask to three scopes:

| Scope | Meaning | This repo |
|---|---|---|
| A. Closed-loop OTFS link | gNB ⇄ UE OTFS at PHY | Gated on Amarisoft — this repo is the **reference model** to validate against when a build lands |
| B. Channel/impairment emulation | quantify where CP-OFDM degrades | **Done in SW**: BER-vs-Doppler / NTN sweeps with a fair CP-OFDM baseline (`sims/sim_ber_doppler.py`) |
| C. IQ playback/capture | open-loop OTFS PHY signal test | **Done in SW**: cf32/sc16 IQ export with JSON sidecars, RRC-oversampled DAC-rate variant (`otfs/iq_io.py`) |

### How others have implemented OTFS (survey)

* **Cohere / academic Zak-OTFS** (Mohammed, Hadani et al.): work directly through
  the discrete Zak transform; point "pulsone" pilot; channel taps read off the
  pilot response when the **crystallization condition** holds (delay/Doppler
  periods exceed the channel spreads). Papers: [pulse shaping & predictability
  tradeoff](https://arxiv.org/abs/2405.02718), [joint sensing +
  communication](https://arxiv.org/abs/2406.06024), [interleaved
  pilots](https://arxiv.org/pdf/2408.09379), [MIMO Zak-OTFS
  ](https://arxiv.org/pdf/2606.26420).
* **MATLAB reference** ([MathWorks OTFS example](https://github.com/mathworks/Wireless-Systems-with-MATLAB-and-Simulink)):
  multicarrier OTFS = ISFFT + OFDM modulator, compares against OFDM — same
  overlay architecture used here.
* **Open-source toolboxes** ([Phy_Mod_OTFS](https://github.com/textremo/Phy_Mod_OTFS),
  MATLAB+Python): embedded pilot + message-passing detection (Raviteja-style) —
  both implemented here.
* **SDR proofs of concept**: [GPU-enabled real-time OTFS SDR](https://arxiv.org/pdf/2309.12861),
  [LabVIEW/USRP OTFS_SDR](https://github.com/NoDuckyAnyMore/OTFS_SDR),
  [FPGA implementation](https://arxiv.org/pdf/2310.09671) — evidence the Phase-2
  SDR port is well-trodden; all use the same DD→time transform + frame CP.

---

## 2. Signal-processing chain implemented

```
TX:  bits → Gray-QAM → DD grid (M×N, embedded pilot+guard) → inverse DZT → frame CP → IQ
                                        │  (identically: ISFFT → OFDM modulator — overlay
                                        ▼   equivalence proven in tests/test_transforms.py)
CH:  P-path doubly-dispersive channel: integer delays, FRACTIONAL Doppler, AWGN
     (3GPP ETU/EVA profiles, NTN-LEO with residual Doppler)

RX:  IQ → CP removal → DZT → DD pilot readout → channel estimation →
     LMMSE (reference) or message-passing (low-complexity) detection → demap → bits
```

Key implementation decisions, all verified by the test suite (26 tests):

* **Zak-native**: TX/RX go through the discrete Zak transform; the multicarrier
  (ISFFT-on-OFDM) path is provided and proven **numerically identical** for
  rectangular pulses — this is exactly the "reuse the OFDM engine" argument in
  the feasibility page (§6) and the basis of the Amarisoft feature request.
* **Exact effective channel**: the frame CP makes the physical linear channel
  equal to a quasi-circulant model (test-pinned, including the Doppler phase
  accrued during the CP), so the perfect-CSI equaliser is exact.
* **Two channel estimators** from the same embedded pilot:
  * *model-free tap readout* — exact for on-grid Doppler (machine precision in
    tests), degrades under fractional Doppler (rectangular-pulse leakage);
  * *parametric (default)* — CLEAN-style per-delay-row Dirichlet-kernel fits
    with fractional Doppler, then the effective H is rebuilt from estimated
    (delay, Doppler, gain) triples. Handles NTN-level Doppler (5 kHz ≈ 5.3
    bins) with zero BER at 20 dB SNR in the demo.
* **Detectors**: exact LMMSE (reference, O((MN)³)) and Gaussian-approximation
  **message passing** over the sparse effective channel (the low-complexity
  algorithm a real-time receiver would run).
* **Honest baseline**: CP-OFDM gets *exact ICI-free one-tap CSI* (better than
  any real NR estimator), so measured OFDM error floors are lower bounds.

### Parameterisation (defaults)

| Parameter | Value | Note |
|---|---|---|
| M × N | 32 × 16 | delay × Doppler bins (M also = overlay subcarriers) |
| SCS / fs | 15 kHz / 480 kHz | delay res 2.08 µs, Doppler res 937.5 Hz |
| Frame | 1.067 ms + CP 8 samples | CP ≥ max delay spread |
| Pilot | point pilot @ (M/2, N/2), +30 dB | guard sized by `l_max`, `k_max` (crystallization budget) |
| QAM | 4 / 16 / 64, Gray | uncoded (FEC is orthogonal to the waveform question) |

`k_max` must be sized to the channel's Doppler spread (in bins) — the SW model
enforces the same crystallization budgeting a real deployment needs.

---

## 3. Results (plots + IQ in `results/`, tables printed by the sims)

* **BER vs UE speed** (EVA, 3.5 GHz, 16QAM, 22 dB SNR): CP-OFDM degrades from
  ~9e-3 (30 km/h) to 9.7e-2 (1000 km/h) — the ICI floor; Zak-OTFS with
  realistic pilot-estimated CSI holds 2.0e-2 at 1000 km/h and ~8e-3 at
  500 km/h (~5x better than OFDM there).
* **BER vs SNR at 500 km/h**: CP-OFDM floors at ~3.4e-2 regardless of SNR;
  Zak-OTFS keeps falling (1.1e-3 est-CSI / 1.4e-4 perfect-CSI at 28 dB) —
  the error-floor separation is the scope-B headline evidence.
* **BER vs NTN residual Doppler** (LEO, 20 dB SNR): at 5 kHz residual Doppler
  (a third of the SCS) CP-OFDM is unusable (BER 0.22) while Zak-OTFS
  estimates and equalises the shift from its own pilot (BER 3.4e-4).
* **PAPR CCDF**: intrinsic Zak-OTFS PAPR ≈ CP-OFDM (11.5 vs 10.9 dB peak here) —
  confirming the assessment's caveat that OTFS brings **no PA-efficiency win**;
  the +30 dB point pilot adds ~3 dB peak (a known cost of point pilots;
  spread/interleaved pilots are the literature's mitigation).
* **End-to-end demo**: NTN-LEO 5 kHz residual Doppler, 20 dB SNR → BER 0,
  EVM ~10%, estimated DD paths within ~1% of truth.

## 4. Repo layout & usage

```
otfs/            package: qam, zak (DZT/ISFFT), channel, grid (pilot+estimation),
                 modem (Zak-OTFS), detectors (LMMSE/MP), ofdm (baseline),
                 metrics (BER/EVM/PAPR/PSD/RRC), iq_io (cf32/sc16 + sidecars)
webui/           Zak-OTFS Waveform Studio — interactive web dashboard (Flask)
sims/            demo_end_to_end.py, sim_ber_doppler.py, sim_papr_spectrum.py
tests/           26 pytest tests pinning every math convention
results/         plots, tables, exported IQ (generated)
```

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q            # 26 passed
python webui\app.py                  # web UI -> http://127.0.0.1:8050
python sims\demo_end_to_end.py       # frame walkthrough + IQ export
python sims\sim_ber_doppler.py       # scope-B evidence curves (~ minutes)
python sims\sim_papr_spectrum.py     # PAPR/PSD comparison
```

### Web UI — Zak-OTFS Waveform Studio

`python webui\app.py` serves an interactive dashboard (dark, product-styled)
on port 8050. It drives the same validated `otfs` package — no duplicated DSP.

* **Live frame** tab: configure grid (M/N/SCS/QAM/pilot boost), channel
  (NTN-LEO / EVA / ETU / custom two-path), receiver (CSI mode, LMMSE or MP
  detector, SNR), then transmit a frame (~0.5 s). Shows metric cards (OTFS vs
  OFDM BER/EVM/PAPR), TX/RX delay-Doppler heatmaps, both constellations,
  channel truth-vs-pilot-estimate table, and PSD. Guard sizing (crystallization
  budget) is computed automatically from the requested channel.
* **BER sweep** tab: speed / NTN-Doppler / SNR studies run in a background
  thread with a progress bar; results plot as interactive log-BER curves.
* **IQ export** buttons download the last TX frame as cf32 / sc16 / 4× RRC
  oversampled cf32 for SDR playback (scope C).

### Server install & self-update (oneclick-style)

```bash
# one-shot install on Ubuntu/Debian or RHEL-family (creates the `otfs`
# systemd service on port 8050; OTFS_PORT env overrides):
curl -fsSL https://github.com/nikhilsimnovus/otfs/archive/refs/heads/main.tar.gz \
  | tar xz && sudo bash otfs-main/scripts/install.sh
```

The **Update** button in the topbar (POST `/api/update`) pulls the latest
main tarball from this repo and re-runs the same idempotent installer via a
narrow sudoers entry (`/usr/local/sbin/otfs-update` only) — bump `VERSION`
in `webui/app.py` on every push so the applied build is visible in the UI.
Reference deployment: `http://192.168.1.36:8050`.

## 5. Phase 2 — SDR / Amarisoft path (next)

1. **Open-loop first (scope C)**: play `results/*_x4.cf32` (RRC-oversampled)
   through an SDR TX / the Amarisoft TRX IQ path; capture and run
   `otfs.metrics` (EVM/PAPR/PSD) + the RX chain on the capture. The IQ sidecar
   JSON carries every parameter the capture side needs.
2. **Receiver on live captures**: timing/frequency sync are the missing blocks
   (SW model assumes frame sync); the pilot pulsone itself is the natural sync
   probe — correlate in the DD domain.
3. **Closed loop (scope A)** is gated on the Amarisoft waveform mode (feature
   request per §8 of the assessment); this model then becomes the golden
   reference for conformance of their implementation.

**Known SW-model simplifications** (documented, deliberate): integer-sample
path delays (delay-fractional support = straightforward sinc interpolation),
rectangular DD pulse (RRC applied at the DAC-oversampling stage, not as a full
DD-domain pulse-shaping filter), no FEC, no MIMO, frame-level sync assumed.

## 6. Sources

* [Zak-OTFS pulse shaping / time-bandwidth vs predictability](https://arxiv.org/abs/2405.02718)
* [Zak-OTFS turbo signal processing for ISAC](https://arxiv.org/abs/2406.06024)
* [Zak-OTFS interleaved pilots](https://arxiv.org/pdf/2408.09379)
* [MIMO Zak-OTFS: estimation, detection, throughput](https://arxiv.org/pdf/2606.26420)
* [MathWorks OTFS example](https://github.com/mathworks/Wireless-Systems-with-MATLAB-and-Simulink)
* [Phy_Mod_OTFS toolbox](https://github.com/textremo/Phy_Mod_OTFS)
* [GPU real-time OTFS SDR PoC](https://arxiv.org/pdf/2309.12861) · [OTFS_SDR (USRP)](https://github.com/NoDuckyAnyMore/OTFS_SDR) · [FPGA OTFS](https://arxiv.org/pdf/2310.09671)
* [OTFS foundational paper (Hadani et al.)](https://arxiv.org/pdf/1608.02993)
