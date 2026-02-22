"""
Step 7: Heinzelman First-Order Radio Energy Model.

Implements distance-dependent energy consumption for WSN:

  E_Tx(k, d) = k * E_elec + k * eps * d^n
    where eps = E_fs if d < d0 else E_mp
    and   n   = 2    if d < d0 else 4

  E_Rx(k) = k * E_elec

  E_DA = data aggregation energy at CH

This model is used by the simulation to deduct energy from nodes
after every sensing/transmitting/receiving event.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    E_ELEC, E_FS, E_MP, E_DA, D0,
    PACKET_SIZE_BITS, V_SUPPLY,
    I_SENSE, I_PROC, T_SENSE, T_PROC, T_INFER
)


def energy_transmit(k_bits, distance):
    """
    Energy to transmit k_bits over distance d (meters).
    Uses free-space model if d < d0, multi-path otherwise.
    """
    if distance < D0:
        return k_bits * E_ELEC + k_bits * E_FS * (distance ** 2)
    else:
        return k_bits * E_ELEC + k_bits * E_MP * (distance ** 4)


def energy_receive(k_bits):
    """Energy to receive k_bits."""
    return k_bits * E_ELEC


def energy_aggregation(k_bits, n_signals):
    """Energy for CH to aggregate n_signals of size k_bits each."""
    return E_DA * k_bits * n_signals


def energy_sense():
    """Energy consumed during one sensing operation."""
    return I_SENSE * V_SUPPLY * T_SENSE


def energy_process():
    """Energy consumed during processing (non-inference)."""
    return I_PROC * V_SUPPLY * T_PROC


def energy_infer():
    """Energy consumed during TinyML inference."""
    return I_PROC * V_SUPPLY * T_INFER


def energy_node_epoch(distance_to_ch, transmit=False):
    """
    Total energy consumed by a MEMBER NODE in one epoch (time slot).

    Every epoch: sense + process + infer
    If mismatch (transmit=True): + transmit to CH

    Args:
        distance_to_ch: distance from this node to its CH (meters)
        transmit: whether node transmits this epoch (mismatch detected)

    Returns:
        Energy consumed in Joules
    """
    e = energy_sense() + energy_process() + energy_infer()
    if transmit:
        e += energy_transmit(PACKET_SIZE_BITS, distance_to_ch)
    return e


def energy_ch_epoch(n_transmitting, distances_members, distance_to_bs,
                    k_bits=PACKET_SIZE_BITS):
    """
    Total energy consumed by a CLUSTER HEAD in one epoch.

    CH does:
      - Sensing + processing + inference (same as member)
      - Receive data from n_transmitting members
      - Aggregate received data
      - Transmit aggregated data to BS (if any member transmitted)

    Args:
        n_transmitting: number of member nodes that transmitted
        distances_members: not used for Rx (Rx energy is distance-independent)
        distance_to_bs: distance from CH to base station
        k_bits: packet size in bits

    Returns:
        Energy consumed in Joules
    """
    # CH also senses and infers
    e = energy_sense() + energy_process() + energy_infer()

    if n_transmitting > 0:
        # Receive from members
        e += energy_receive(k_bits) * n_transmitting
        # Aggregate
        e += energy_aggregation(k_bits, n_transmitting)
        # Forward to BS
        e += energy_transmit(k_bits, distance_to_bs)

    return e


def energy_baseline_epoch(distance_to_ch, distance_ch_to_bs, is_ch=False,
                          n_members=0):
    """
    Baseline energy: node ALWAYS transmits every epoch (no prediction).
    This is the comparison scenario without TinyML suppression.
    """
    if is_ch:
        e = energy_sense() + energy_process()
        e += energy_receive(PACKET_SIZE_BITS) * n_members
        e += energy_aggregation(PACKET_SIZE_BITS, n_members)
        e += energy_transmit(PACKET_SIZE_BITS, distance_ch_to_bs)
    else:
        e = energy_sense() + energy_process()
        e += energy_transmit(PACKET_SIZE_BITS, distance_to_ch)
    return e
