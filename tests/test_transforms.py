import numpy as np
import pytest

from otfs.zak import (dzt, idzt, isfft, sfft, zak_matrix,
                      mc_otfs_modulate, mc_otfs_demodulate)

RNG = np.random.default_rng(1)


@pytest.mark.parametrize("M,N", [(8, 4), (32, 16), (16, 64)])
def test_dzt_roundtrip(M, N):
    x = RNG.standard_normal((M, N)) + 1j * RNG.standard_normal((M, N))
    assert np.allclose(dzt(idzt(x), M, N), x)
    s = RNG.standard_normal(M * N) + 1j * RNG.standard_normal(M * N)
    assert np.allclose(idzt(dzt(s, M, N)), s)


def test_dzt_unitary():
    M, N = 16, 8
    x = RNG.standard_normal((M, N)) + 1j * RNG.standard_normal((M, N))
    s = idzt(x)
    assert np.isclose(np.linalg.norm(s), np.linalg.norm(x))


def test_isfft_roundtrip():
    x = RNG.standard_normal((32, 16)) + 1j * RNG.standard_normal((32, 16))
    assert np.allclose(sfft(isfft(x)), x)


def test_mc_otfs_equals_zak():
    """ISFFT + OFDM modulator == inverse DZT (overlay equivalence)."""
    M, N = 32, 16
    x = RNG.standard_normal((M, N)) + 1j * RNG.standard_normal((M, N))
    assert np.allclose(mc_otfs_modulate(x), idzt(x))
    s = RNG.standard_normal(M * N) + 1j * RNG.standard_normal(M * N)
    assert np.allclose(mc_otfs_demodulate(s, M, N), dzt(s, M, N))


def test_zak_matrix_matches_dzt():
    M, N = 8, 8
    Z = zak_matrix(M, N)
    assert np.allclose(Z @ Z.conj().T, np.eye(M * N))  # unitary
    s = RNG.standard_normal(M * N) + 1j * RNG.standard_normal(M * N)
    assert np.allclose(Z @ s, dzt(s, M, N).ravel(order="F"))
