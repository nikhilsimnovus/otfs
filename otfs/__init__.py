"""Zak-OTFS delay-Doppler waveform SW implementation.

Reference software model of a Zak-OTFS (delay-Doppler) transmitter and
receiver, with a CP-OFDM baseline, doubly-dispersive channel models and
SDR-ready IQ export. Built as the Phase-1 (software) deliverable of the
Cohere Zak-OTFS feasibility assessment for Simnovator/UESIM (Amarisoft);
Phase 2 ports the TX/RX to an SDR card / Amarisoft TRX IQ injection.
"""

from .qam import qam_map, qam_demap_hard, qam_constellation
from .zak import dzt, idzt, isfft, sfft, zak_matrix
from .channel import DDPath, DDChannel, etu_channel, eva_channel, ntn_leo_channel
from .grid import (GridLayout, make_layout, estimate_taps,
                   h_matrix_from_taps, estimate_paths)
from .modem import ZakOTFSConfig, ZakOTFSModem
from .detectors import lmmse_detect, mp_detect
from .ofdm import OFDMConfig, OFDMModem
from . import metrics
from . import iq_io

__version__ = "0.1.0"
