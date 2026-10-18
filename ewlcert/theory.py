"""Closed-form results (derived symbolically; see tests/test_theory.py for the exact check against
the density-matrix model).

Local noise on the entangled state ("pre") or after the moves ("post"): per qubit i a
phase-covariant relaxation (g_i, c_i) followed by depolarizing r_i.  Against an opponent playing Q
the deviation payoffs form a DIAGONAL quadratic form (symmetric under A <-> B).  The two gains
that decide the equilibrium are:
  gain_D = M_D - M_Q = -(1-r_A)(1-r_B) [7 c_A c_B - B_stage] / 2
  gain_C = M_C - M_Q = -2 (1-r_A)(1-r_B) c_A c_B            (never positive)
with B_pre  = (1-g_A)(1-g_B) + g_A g_B  and  B_post = (1-g_A)(1-g_B).
Hence (Q,Q) is a Nash equilibrium of the two-parameter game iff 7 c_A c_B >= B_stage, and the
strict margin is  mu = (1-r_A)(1-r_B) * min(2 c_A c_B, (7 c_A c_B - B_stage)/2).
Depolarizing noise rescales the margin but never moves the threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def b_term(gA, gB, stage="pre"):
    base = (1 - gA) * (1 - gB)
    return base + gA * gB if stage == "pre" else base


def gain_D(gA, cA, gB, cB, rA=0.0, rB=0.0, stage="pre"):
    return -(1 - rA) * (1 - rB) * (7 * cA * cB - b_term(gA, gB, stage)) / 2


def gain_C(gA, cA, gB, cB, rA=0.0, rB=0.0, stage="pre"):
    return -2 * (1 - rA) * (1 - rB) * cA * cB


def margin(gA, cA, gB, cB, rA=0.0, rB=0.0, stage="pre"):
    """Strict-equilibrium margin min over deviations to C and D (positive iff strict NE)."""
    return -np.maximum(gain_D(gA, cA, gB, cB, rA, rB, stage), gain_C(gA, cA, gB, cB, rA, rB, stage))


def idle_params(t, T1, T2):
    """(g, c) of a qubit idling for time t."""
    return 1 - np.exp(-t / T1), np.exp(-t / T2)


def critical_idle_time(T1A, T2A, T1B, T2B, stage="pre", t0=0.0):
    """Idle time at which (Q,Q) stops being a Nash equilibrium.  t0 is an idle time already present
    (e.g. accumulated during gates) that is added to the swept delay."""
    def f(t):
        gA, cA = idle_params(t + t0, T1A, T2A)
        gB, cB = idle_params(t + t0, T1B, T2B)
        return 7 * cA * cB - b_term(gA, gB, stage)
    if f(0) < 0:
        return 0.0
    hi = 1e-6
    while f(hi) > 0:
        hi *= 2
        if hi > 1e3:
            return np.inf
    return brentq(f, 0, hi, xtol=1e-12)


def critical_idle_time_dephasing_limit(T2A, T2B):
    """T1 -> infinity: 7 exp(-t(1/T2A + 1/T2B)) = 1."""
    return np.log(7) / (1 / T2A + 1 / T2B)


AD_THRESHOLD_SYMMETRIC = 1 - (9 - np.sqrt(73)) / 4   # pure amplitude damping on the entangled state
