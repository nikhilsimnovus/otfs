import numpy as np
import pytest

from otfs.modem import ZakOTFSConfig, ZakOTFSModem
from otfs.channel import DDPath, DDChannel, awgn
from otfs.grid import estimate_taps, h_matrix_from_taps

RNG = np.random.default_rng(4)


def make_modem(**kw):
    return ZakOTFSModem(ZakOTFSConfig(**kw))


def doppler_bins_to_cps(bins, M, N):
    """nu in cycles/sample for a Doppler offset given in DD grid bins."""
    return bins / (M * N)


def test_back_to_back_no_channel():
    md = make_modem()
    bits = RNG.integers(0, 2, md.bits_per_frame)
    x_dd, s_cp = md.modulate(bits)
    y_dd = md.to_dd(md.strip_cp(s_cp))
    H = np.eye(md.cfg.frame_len)
    rx_bits, _ = md.demodulate(y_dd, H, 1e-8)
    assert np.array_equal(rx_bits, bits)


def _test_channel(M, N, integer_doppler=True):
    k1 = 2 if integer_doppler else 1.6
    k2 = -1 if integer_doppler else -0.7
    return DDChannel(paths=[
        DDPath(0, doppler_bins_to_cps(k1, M, N), 0.7 * np.exp(0.3j)),
        DDPath(2, doppler_bins_to_cps(k2, M, N), 0.5 * np.exp(-1.1j)),
        DDPath(4, doppler_bins_to_cps(0.0, M, N), 0.3 * np.exp(0.9j)),
    ], fs=480e3)


def test_h_true_matches_physical():
    """H built from the channel object reproduces the physical TX->RX map."""
    md = make_modem()
    M, N = md.cfg.M, md.cfg.N
    chan = _test_channel(M, N, integer_doppler=False)
    bits = RNG.integers(0, 2, md.bits_per_frame)
    x_dd, s_cp = md.modulate(bits)
    y_dd = md.to_dd(md.strip_cp(chan.apply_linear(s_cp)))
    H = md.build_h_true(chan)
    assert np.allclose(H @ x_dd.ravel(order="F"), y_dd.ravel(order="F"))


def test_estimated_h_matches_true_integer_doppler():
    """For on-grid Doppler the pilot-readout + twisted-convolution
    reconstruction reproduces the true effective channel."""
    md = make_modem()
    M, N = md.cfg.M, md.cfg.N
    chan = _test_channel(M, N, integer_doppler=True)
    bits = RNG.integers(0, 2, md.bits_per_frame)
    x_dd, s_cp = md.modulate(bits)
    y_dd = md.to_dd(md.strip_cp(chan.apply_linear(s_cp)))

    H_true = md.build_h_true(chan)
    taps = estimate_taps(y_dd, md.layout, noise_var=1e-10, thresh=3.0)
    H_est = h_matrix_from_taps(taps, md.layout)
    # compare action on the data grid (pilot region excluded by design)
    d = md.layout.data_idx_flat
    err = np.linalg.norm(H_est[:, d] - H_true[:, d]) / np.linalg.norm(H_true[:, d])
    assert err < 1e-6


@pytest.mark.parametrize("csi", ["perfect", "est"])
def test_e2e_high_snr_zero_ber(csi):
    md = make_modem()
    chan = _test_channel(md.cfg.M, md.cfg.N, integer_doppler=False)
    noise_var = 10 ** (-30 / 10)  # 30 dB SNR
    n_err = 0
    n_bits = 0
    for _ in range(4):
        bits = RNG.integers(0, 2, md.bits_per_frame)
        rx, _ = md.run_frame(bits, chan, noise_var, RNG, csi=csi)
        n_err += int(np.sum(rx != bits))
        n_bits += bits.size
    assert n_err / n_bits < 1e-3


def test_mp_detector_agrees_with_lmmse():
    md = make_modem(qam_order=4)
    chan = _test_channel(md.cfg.M, md.cfg.N, integer_doppler=True)
    noise_var = 10 ** (-18 / 10)
    bits = RNG.integers(0, 2, md.bits_per_frame)
    rx_l, _ = md.run_frame(bits, chan, noise_var, np.random.default_rng(7),
                           csi="perfect", detector="lmmse")
    rx_m, _ = md.run_frame(bits, chan, noise_var, np.random.default_rng(7),
                           csi="perfect", detector="mp", n_iter=25)
    ber_l = np.mean(rx_l != bits)
    ber_m = np.mean(rx_m != bits)
    assert ber_m <= max(ber_l * 3, 5e-3)
