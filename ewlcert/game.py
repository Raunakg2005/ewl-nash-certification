"""Exact density-matrix model of the Eisert-Wilkens-Lewenstein (EWL) Prisoner's Dilemma with noise.

Conventions
-----------
* Basis index k = 2*a + b, with a (b) Alice's (Bob's) bit; bit 0 = cooperate (C), 1 = defect (D).
* Entangler J(gamma) = exp(i gamma/2 D(x)D) with D = U(pi, 0) = i sigma_y.  J commutes with every
  product of classical moves, so the classical game is embedded for every gamma (EWL 1999).
* A strategy is written U(x) = x0 I + i x1 sigma_x + i x2 sigma_y + i x3 sigma_z with |x| = 1.
  The EWL two-parameter family U(theta, phi) is x = (cos(theta/2) cos(phi), 0, sin(theta/2),
  cos(theta/2) sin(phi)); C = e0, D = e2, Q = e3.
* Noise: an optional channel acts on the entangled state (stage "pre", between J and the moves)
  and/or after the moves (stage "post", before J^dagger).  Readout errors act on the outcome
  distribution through a confusion matrix.
"""
from __future__ import annotations

import itertools
import numpy as np

T_, R_, P_, S_ = 5.0, 3.0, 1.0, 0.0
PAYOFF = {"A": np.array([R_, S_, T_, P_]),   # outcomes CC, CD, DC, DD
          "B": np.array([R_, T_, S_, P_])}

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def U_x(x) -> np.ndarray:
    x0, x1, x2, x3 = x
    return x0 * I2 + 1j * x1 * X + 1j * x2 * Y + 1j * x3 * Z


def U_ewl(theta: float, phi: float) -> np.ndarray:
    return np.array([[np.exp(1j * phi) * np.cos(theta / 2), np.sin(theta / 2)],
                     [-np.sin(theta / 2), np.exp(-1j * phi) * np.cos(theta / 2)]], dtype=complex)


def x_ewl(theta: float, phi: float) -> np.ndarray:
    c = np.cos(theta / 2)
    return np.array([c * np.cos(phi), 0.0, np.sin(theta / 2), c * np.sin(phi)])


E = np.eye(4)
XC, XD, XQ = E[0], E[2], E[3]
C, D, Q = U_x(XC), U_x(XD), U_x(XQ)


def J(gamma: float) -> np.ndarray:
    DD = np.kron(D, D)
    return np.cos(gamma / 2) * np.eye(4) + 1j * np.sin(gamma / 2) * DD


# ---------------------------------------------------------------- channels (Kraus lists, single qubit)
def thermal_relaxation(g: float, c: float) -> list[np.ndarray]:
    """Phase-covariant qubit channel: |1> population decays with prob g, coherence scaled by c.
    Valid (CPTP) iff 0 <= g <= 1 and c <= sqrt(1-g).  For idle time t: g = 1-exp(-t/T1),
    c = exp(-t/T2)."""
    if c > np.sqrt(1 - g) + 1e-12:
        raise ValueError("c must not exceed sqrt(1-g) (T2 <= 2 T1)")
    ad = [np.array([[1, 0], [0, np.sqrt(1 - g)]], complex), np.array([[0, np.sqrt(g)], [0, 0]], complex)]
    s = np.sqrt(1 - g)
    lam = c / s if s > 0 else 0.0
    pd = [np.sqrt((1 + lam) / 2) * I2, np.sqrt((1 - lam) / 2) * Z]
    return [b @ a for a in ad for b in pd]


def depolarizing(r: float) -> list[np.ndarray]:
    """rho -> (1-r) rho + r I/2."""
    return [np.sqrt(1 - 3 * r / 4) * I2, np.sqrt(r / 4) * X, np.sqrt(r / 4) * Y, np.sqrt(r / 4) * Z]


def compose(k2: list[np.ndarray], k1: list[np.ndarray]) -> list[np.ndarray]:
    """Kraus list of k2 o k1."""
    return [b @ a for a in k1 for b in k2]


def local(kA: list[np.ndarray], kB: list[np.ndarray]) -> list[np.ndarray]:
    return [np.kron(a, b) for a in kA for b in kB]


def apply(rho: np.ndarray, kraus: list[np.ndarray] | None) -> np.ndarray:
    if kraus is None:
        return rho
    return sum(k @ rho @ k.conj().T for k in kraus)


def zz_idle(zeta_t: float) -> list[np.ndarray]:
    """Coherent static ZZ interaction exp(-i zeta t Z(x)Z / 2) (two-qubit unitary, as a 1-element Kraus list)."""
    return [np.diag(np.exp(-1j * zeta_t / 2 * np.array([1, -1, -1, 1])))]


# ---------------------------------------------------------------- the game
def probabilities(uA, uB, gamma=np.pi / 2, pre=None, post=None, confusion=None) -> np.ndarray:
    """Outcome distribution (CC, CD, DC, DD).  pre/post: two-qubit Kraus lists; confusion: 4x4
    column-stochastic matrix P(measured | true)."""
    Jg = J(gamma)
    rho = np.zeros((4, 4), complex)
    rho[0, 0] = 1
    rho = Jg @ rho @ Jg.conj().T
    rho = apply(rho, pre)
    W = np.kron(uA, uB)
    rho = W @ rho @ W.conj().T
    rho = apply(rho, post)
    rho = Jg.conj().T @ rho @ Jg
    p = np.clip(np.real(np.diag(rho)), 0, None)
    p = p / p.sum()
    if confusion is not None:
        p = confusion @ p
    return p


def payoff(player: str, uA, uB, **kw) -> float:
    return float(PAYOFF[player] @ probabilities(uA, uB, **kw))


# ---------------------------------------------------------------- quadratic-form certification
RESTRICTED = (0, 2, 3)          # coordinates (x0, x2, x3) spanned by the EWL two-parameter family
FULL = (0, 1, 2, 3)


def design(coords=RESTRICTED) -> list[np.ndarray]:
    """Deviation strategies whose payoffs determine the quadratic form: basis vectors and normalised
    pairwise sums (6 for the two-parameter family, 10 for full SU(2)).  All lie in the strategy set."""
    vecs = [E[i] for i in coords]
    vecs += [(E[i] + E[j]) / np.sqrt(2) for i, j in itertools.combinations(coords, 2)]
    return vecs


def form_from_payoffs(values, coords=RESTRICTED) -> np.ndarray:
    """Reconstruct the symmetric matrix M (in the given coordinates) from the payoffs of design()."""
    n = len(coords)
    M = np.zeros((n, n))
    M[np.diag_indices(n)] = values[:n]
    for k, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        M[i, j] = M[j, i] = values[n + k] - (M[i, i] + M[j, j]) / 2
    return M


def max_on_orthant(M: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact maximum of x^T M x over the unit sphere intersected with the non-negative orthant.
    At a maximiser the active coordinates form a principal submatrix whose top eigenvector is the
    maximiser (KKT); we enumerate all supports."""
    n = M.shape[0]
    best, arg = -np.inf, None
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            w, v = np.linalg.eigh(M[np.ix_(S, S)])
            for k in range(len(S)):
                vec = v[:, k] * (np.sign(v[:, k].sum()) or 1)
                if np.all(vec >= -1e-12):
                    x = np.zeros(n)
                    x[list(S)] = np.clip(vec, 0, None)
                    x /= np.linalg.norm(x)
                    val = x @ M @ x
                    if val > best:
                        best, arg = val, x
    return float(best), arg


def max_on_sphere(M: np.ndarray) -> tuple[float, np.ndarray]:
    w, v = np.linalg.eigh(M)
    return float(w[-1]), v[:, -1]


def incentive(M: np.ndarray, coords=RESTRICTED) -> float:
    """Largest gain from a unilateral deviation away from Q (>= 0; 0 iff Q is a best response)."""
    iq = coords.index(3)
    best = max_on_orthant(M)[0] if coords == RESTRICTED else max_on_sphere(M)[0]
    return max(0.0, best - M[iq, iq])


def exact_forms(coords=RESTRICTED, **kw) -> dict:
    """Quadratic forms of both players against an opponent playing Q, from exact payoffs."""
    out = {}
    for player in "AB":
        vals = []
        for x in design(coords):          # 4-vectors supported on coords
            u = U_x(x)
            vals.append(payoff(player, u, Q, **kw) if player == "A" else payoff(player, Q, u, **kw))
        out[player] = form_from_payoffs(np.array(vals), coords)
    return out


def epsilon(coords=RESTRICTED, **kw) -> float:
    f = exact_forms(coords, **kw)
    return max(incentive(f["A"], coords), incentive(f["B"], coords))
