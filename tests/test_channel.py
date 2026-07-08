import numpy as np

from otfs.channel import DDPath, DDChannel, etu_channel

RNG = np.random.default_rng(3)


def _rand_chan(fs=480e3):
    return DDChannel(paths=[
        DDPath(0, 1.7e-3, 0.6 + 0.2j),
        DDPath(3, -2.3e-3, -0.3 + 0.5j),
        DDPath(7, 0.9e-3, 0.2 - 0.4j),
    ], fs=fs)


def test_cp_makes_linear_equal_cyclic():
    """apply_linear on a CP-extended frame, after CP drop, equals the
    quasi-circulant model with the frame clock offset n0 = cp_len."""
    L, cp = 512, 8
    chan = _rand_chan()
    s = RNG.standard_normal(L) + 1j * RNG.standard_normal(L)
    s_cp = np.concatenate([s[-cp:], s])
    r_lin = chan.apply_linear(s_cp)[cp:cp + L]
    r_cyc = chan.apply_cyclic(s, n0=cp)
    assert np.allclose(r_lin, r_cyc)


def test_time_matrix_matches_apply():
    L = 128
    chan = _rand_chan()
    s = RNG.standard_normal(L) + 1j * RNG.standard_normal(L)
    H = chan.time_matrix(L, n0=8)
    assert np.allclose(H @ s, chan.apply_cyclic(s, n0=8))


def test_profile_power_normalised():
    chan = etu_channel(480e3, 3.5e9, 120, RNG)
    # gains are random; mean power over many draws should be ~1
    p = np.mean([np.sum(np.abs([q.gain for q in
                 etu_channel(480e3, 3.5e9, 120, RNG).paths]) ** 2)
                 for _ in range(400)])
    assert 0.8 < p < 1.2
    assert chan.max_delay <= 4  # 5 us at 480 kHz = 2.4 samples
