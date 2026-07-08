import numpy as np

from otfs import iq_io

RNG = np.random.default_rng(6)


def test_cf32_roundtrip(tmp_path):
    s = RNG.standard_normal(256) + 1j * RNG.standard_normal(256)
    f = tmp_path / "frame.cf32"
    iq_io.write_cf32(f, s, fs=480e3, meta={"M": 32, "N": 16})
    r = iq_io.read_cf32(f)
    assert np.allclose(r, s, atol=1e-6)
    assert (tmp_path / "frame.cf32.json").exists()


def test_sc16_roundtrip(tmp_path):
    s = RNG.standard_normal(256) + 1j * RNG.standard_normal(256)
    f = tmp_path / "frame.sc16"
    iq_io.write_sc16(f, s, fs=480e3)
    r = iq_io.read_sc16(f)
    # int16 quantisation: relative error bounded by scale/2 per sample
    assert np.max(np.abs(r - s)) < np.max(np.abs(s)) * 1e-3
