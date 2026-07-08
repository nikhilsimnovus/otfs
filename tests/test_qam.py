import numpy as np
import pytest

from otfs.qam import qam_map, qam_demap_hard, qam_constellation

RNG = np.random.default_rng(2)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_roundtrip(order):
    bps = int(np.log2(order))
    bits = RNG.integers(0, 2, size=120 * bps)
    syms = qam_map(bits, order)
    assert np.allclose(np.mean(np.abs(qam_constellation(order)) ** 2), 1.0)
    assert np.array_equal(qam_demap_hard(syms, order), bits)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_gray_neighbours(order):
    """Adjacent constellation points along each axis differ by one bit."""
    bps = int(np.log2(order))
    m = int(np.sqrt(order))
    const = qam_constellation(order)
    levels = np.sort(np.unique(np.round(const.real, 9)))
    for lv in range(m - 1):
        for im in levels:
            a = np.argmin(np.abs(const - (levels[lv] + 1j * im)))
            b = np.argmin(np.abs(const - (levels[lv + 1] + 1j * im)))
            bits_a = qam_demap_hard(const[a], order)
            bits_b = qam_demap_hard(const[b], order)
            assert np.sum(bits_a != bits_b) == 1
