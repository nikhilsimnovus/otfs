import numpy as np

from otfs.ofdm import OFDMConfig, OFDMModem
from otfs.channel import DDPath, DDChannel

RNG = np.random.default_rng(5)


def test_back_to_back_static_multipath():
    md = OFDMModem(OFDMConfig())
    chan = DDChannel(paths=[
        DDPath(0, 0.0, 0.8),
        DDPath(3, 0.0, 0.4 * np.exp(1.2j)),
    ], fs=md.cfg.fs)
    bits = RNG.integers(0, 2, md.bits_per_frame)
    rx, _ = md.run_frame(bits, chan, 1e-8, RNG)
    assert np.array_equal(rx, bits)


def test_doppler_causes_error_floor():
    """Sanity check of the 'why OTFS' effect: at high SNR, adding strong
    Doppler produces residual errors (ICI floor) in CP-OFDM."""
    md = OFDMModem(OFDMConfig())
    fs = md.cfg.fs
    strong_doppler = 0.3 * md.cfg.scs_hz / fs   # 30% of SCS, in cycles/sample
    chan = DDChannel(paths=[
        DDPath(0, strong_doppler, 0.8),
        DDPath(2, -strong_doppler, 0.4 * np.exp(0.5j)),
    ], fs=fs)
    n_err = 0
    for _ in range(5):
        bits = RNG.integers(0, 2, md.bits_per_frame)
        rx, _ = md.run_frame(bits, chan, 1e-6, RNG)
        n_err += int(np.sum(rx != bits))
    assert n_err > 0
