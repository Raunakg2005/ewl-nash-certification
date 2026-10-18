"""E4: coherent (unitary) errors make the quadratic form non-diagonal.  How often does a test that only
compares Q with the pure deviations C and D reach the wrong verdict, compared with the exact
quadratic-form test?  Exact model; random two-qubit coherent errors exp(-i s H), H from the GUE
normalised to unit operator norm, acting on the entangled state and after the moves."""
import numpy as np

import common as C
from ewlcert import game as G
from ewlcert import stats as S

STRENGTHS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
DRAWS = 4000
TOL = 0.05            # epsilon tolerance used to call a profile an (approximate) equilibrium


def gue(rng):
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (A + A.conj().T) / 2
    return H / np.abs(np.linalg.eigvalsh(H)).max()


def unitary(H, s):
    w, v = np.linalg.eigh(H)
    return v @ np.diag(np.exp(-1j * s * w)) @ v.conj().T


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    out = []
    for s in STRENGTHS:
        eps, veps, offd, qcomp = [], [], [], []
        for _ in range(DRAWS):
            pre, post = [unitary(gue(rng), s)], [unitary(gue(rng), s)]
            P = np.array([G.probabilities(G.U_x(a), G.U_x(b), pre=pre, post=post) for a, b in S.circuit_strategies()])
            eps.append(S.eps_from_probs(P))
            veps.append(S.vertex_eps_from_probs(P))
            offd.append(S.max_offdiag(P))
            for M in S.forms_from_probs(P).values():          # where does the best deviation lie?
                val, x = G.max_on_orthant(M)
                if val - M[2, 2] > TOL:
                    qcomp.append(x[2])
        eps, veps = np.array(eps), np.array(veps)
        false_cert = np.mean((veps <= TOL) & (eps > TOL))
        row = dict(s=s, draws=DRAWS, frac_eps_gt_tol=float(np.mean(eps > TOL)),
                   false_certification_vertex=float(false_cert),
                   false_cert_given_violation=float(false_cert / max(np.mean(eps > TOL), 1e-12)),
                   median_underestimate=float(np.median(eps - veps)),
                   p95_underestimate=float(np.quantile(eps - veps, 0.95)),
                   median_offdiag=float(np.median(offd)),
                   n_violating_players=len(qcomp),
                   q_component_min=float(min(qcomp)) if qcomp else None,
                   q_component_median=float(np.median(qcomp)) if qcomp else None)
        out.append(row)
        print(row, flush=True)
    C.save("e4_coherent.json", dict(tol=TOL, rows=out))
