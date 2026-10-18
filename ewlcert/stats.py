"""Finite-shot estimation of the incentive to deviate, readout mitigation and bootstrap bounds.

Circuit set (11 circuits): index 0 is (Q,Q); then Alice's five deviations against Q and Bob's five
deviations against Q, in the order of game.design() with the Q basis vector removed.
"""
from __future__ import annotations

import numpy as np

from . import game as G

DESIGN = G.design(G.RESTRICTED)                 # C, D, Q, (C+D)/r2, (C+Q)/r2, (D+Q)/r2
DEV = [k for k in range(6) if k != 2]           # deviations other than Q itself


def circuit_strategies():
    """List of (x_Alice, x_Bob) for the 11 circuits."""
    out = [(G.XQ, G.XQ)]
    out += [(DESIGN[k], G.XQ) for k in DEV]
    out += [(G.XQ, DESIGN[k]) for k in DEV]
    return out


def forms_from_probs(P: np.ndarray) -> dict:
    """P: (11, 4) outcome distributions -> quadratic forms of A and B."""
    out = {}
    for pl, off in (("A", 1), ("B", 6)):
        vals = np.empty(6)
        vals[2] = G.PAYOFF[pl] @ P[0]
        for j, k in enumerate(DEV):
            vals[k] = G.PAYOFF[pl] @ P[off + j]
        out[pl] = G.form_from_payoffs(vals)
    return out


def eps_from_probs(P):
    f = forms_from_probs(P)
    return max(G.incentive(f["A"]), G.incentive(f["B"]))


def signed_gap(P):
    """max over players of max(M_C, M_D) - M_Q: negative = strict margin, positive = violation by a pure
    deviation.  Used to locate thresholds (it changes sign where the equilibrium is lost)."""
    f = forms_from_probs(P)
    return max(max(f[p][0, 0], f[p][1, 1]) - f[p][2, 2] for p in "AB")


def mean_D_gain(P):
    """Average over both players of M_DD - M_QQ.  Linear in the outcome distributions, hence unbiased
    under shot noise (unlike signed_gap, which is a maximum).  By Theorem 4 both players have the same
    D-gain under relaxation-dephasing-depolarizing noise, and its zero is the stability threshold."""
    f = forms_from_probs(P)
    return 0.5 * sum(f[p][1, 1] - f[p][2, 2] for p in "AB")


def max_offdiag(P):
    f = forms_from_probs(P)
    return max(np.abs(f[p] - np.diag(np.diag(f[p]))).max() for p in "AB")


def vertex_eps_from_probs(P):
    """Shortcut that only compares Q with the pure deviations C and D (valid only if M is diagonal)."""
    f = forms_from_probs(P)
    return max(max(0.0, max(f[p][0, 0], f[p][1, 1]) - f[p][2, 2]) for p in "AB")


# ------------------------------------------------------------------ readout mitigation
def confusion_from_cal(n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """Tensor-product confusion matrix from calibration counts of |00> (n0) and |11> (n1)
    (4 outcome counts each, order CC, CD, DC, DD = 00, 01, 10, 11 as (a, b))."""
    def marg(n, q):
        n = np.asarray(n, float)
        return np.array([n[0] + n[1], n[2] + n[3]]) if q == 0 else np.array([n[0] + n[2], n[1] + n[3]])
    mats = []
    for q in (0, 1):
        m0, m1 = marg(n0, q), marg(n1, q)
        mats.append(np.column_stack([m0 / m0.sum(), m1 / m1.sum()]))   # P(meas | prep)
    return np.kron(mats[0], mats[1])


def mitigate(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, P.T).T


# ------------------------------------------------------------------ sampling and bootstrap
def sample_counts(P: np.ndarray, shots: int, rng) -> np.ndarray:
    P = np.clip(P, 0, None)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.stack([rng.multinomial(shots, p) for p in P])


def estimate(counts, cal=None):
    """Point estimate of epsilon from counts (11, 4); cal = (n0, n1) for readout mitigation."""
    P = counts / counts.sum(axis=1, keepdims=True)
    if cal is not None:
        P = mitigate(P, confusion_from_cal(*cal))
    return eps_from_probs(P), P


def bootstrap(counts, cal=None, B=500, rng=None, level=0.95, stat=eps_from_probs):
    """Parametric (multinomial) bootstrap of a statistic of the outcome distributions.
    Returns (point estimate, lower, upper) percentile bounds (one-sided bounds at `level`)."""
    rng = rng or np.random.default_rng()
    counts = np.asarray(counts)
    shots = counts.sum(axis=1)
    P_hat = counts / shots[:, None]
    A_hat = confusion_from_cal(*cal) if cal is not None else None
    point = stat(mitigate(P_hat, A_hat) if cal is not None else P_hat)
    draws = np.empty(B)
    for b in range(B):
        Pb = np.stack([rng.multinomial(s, p) / s for s, p in zip(shots, P_hat)])
        if cal is not None:
            c0 = rng.multinomial(int(np.sum(cal[0])), np.asarray(cal[0]) / np.sum(cal[0]))
            c1 = rng.multinomial(int(np.sum(cal[1])), np.asarray(cal[1]) / np.sum(cal[1]))
            Pb = mitigate(Pb, confusion_from_cal(c0, c1))
        draws[b] = stat(Pb)
    lo, hi = np.quantile(draws, [1 - level, level])
    return point, lo, hi


def deviation_gains(P):
    """Signed payoff gains of each tested deviation over Q (10 values: 5 for A, 5 for B)."""
    g = []
    for pl, off in (("A", 1), ("B", 6)):
        q = G.PAYOFF[pl] @ P[0]
        g += [G.PAYOFF[pl] @ P[off + j] - q for j in range(5)]
    return np.array(g)


def violation_test(counts, cal=None, alpha=0.05):
    """Declares a violation when some tested deviation beats Q significantly (one-sided z-tests,
    Bonferroni over the 10 deviations).  Unbiased: each gain is linear in the distributions."""
    counts = np.asarray(counts, float)
    shots = counts.sum(axis=1)
    Pm = counts / shots[:, None]                      # measured distributions
    Ainv = np.linalg.inv(confusion_from_cal(*cal)) if cal is not None else np.eye(4)
    gains = deviation_gains(Pm @ Ainv.T)
    se = []
    for pl, off in (("A", 1), ("B", 6)):
        m = Ainv.T @ G.PAYOFF[pl]                     # payoff vector acting on measured outcomes
        var = lambda p, n: max((p @ m**2 - (p @ m) ** 2) / n, 1e-15)
        vq = var(Pm[0], shots[0])
        se += [np.sqrt(var(Pm[off + j], shots[off + j]) + vq) for j in range(5)]
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / len(gains))
    return bool(np.any(gains - z * np.array(se) > 0)), gains
