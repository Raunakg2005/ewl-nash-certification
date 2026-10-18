"""E3: zero-noise extrapolation targets a different estimand: the equilibrium of the IDEAL game.

For the typical pair of each device and idle delays tau/tau_c in TAUS, the 11 circuits are folded
(G (G^dagger G)^k, noise scales 1, 3, 5; folding also stretches the idle delay) and simulated exactly
(readout excluded, i.e. assumed perfectly mitigated).  Extrapolations: linear Richardson (scales 1, 3)
and quadratic Richardson (1, 3, 5), applied to each outcome distribution.  Finite shots: a fixed
budget of N_TOT shots per circuit is split evenly over the scales used."""
import time
import warnings
from multiprocessing import Pool

import numpy as np

import common as C
from ewlcert import game as G
from ewlcert import stats as S

warnings.filterwarnings("ignore")
TAUS = [0.0, 0.25, 0.5, 1.0, 1.5]
FOLDS = [0, 1, 2]                          # scales 1, 3, 5
W = {"none": {1: 1.0},
     "linear": {1: 1.5, 3: -0.5},
     "quadratic": {1: 15 / 8, 3: -5 / 4, 5: 3 / 8}}
N_TOT, REPS, BOOT, TOL = 30000, 200, 200, 0.05

P_IDEAL = np.array([G.probabilities(G.U_x(a), G.U_x(b)) for a, b in S.circuit_strategies()])


def combine(Ps, method):
    return sum(w * Ps[s] for s, w in W[method].items())


def exact(arg):
    name, pick = arg
    C.memory_guard()
    be = C.Dv.backend_by_name(name)
    dp = C.Dv.DevicePair(be, pick["pair"])
    out = {}
    for f in TAUS:
        out[f] = {}
        for k in FOLDS:
            circs = [dp.physical(a, b, f * pick["tc_theory"], fold=k) for a, b in S.circuit_strategies()]
            out[f][2 * k + 1] = dp.run(circs)
    return name, pick, out


def shots_job(arg):
    name, f, Ps, seed = arg
    rng = np.random.default_rng(seed)
    res = {}
    for method in W:
        scales = list(W[method])
        n = N_TOT // len(scales)
        pts, his, gaps = [], [], []
        for _ in range(REPS):
            counts = {s: S.sample_counts(Ps[s], n, rng) for s in scales}
            freq = combine({s: counts[s] / n for s in scales}, method)
            gaps.append(S.signed_gap(freq))
            pts.append(S.eps_from_probs(freq))
            # bootstrap: resample every scale, recombine
            draws = []
            for _ in range(BOOT):
                fb = combine({s: np.stack([rng.multinomial(n, np.clip(p, 0, 1) / np.clip(p, 0, 1).sum()) / n
                                          for p in counts[s] / n]) for s in scales}, method)
                draws.append(S.eps_from_probs(fb))
            his.append(np.quantile(draws, 0.95))
        res[method] = dict(certify_ideal_rate=float(np.mean(np.array(his) <= TOL)), mean_eps=float(np.mean(pts)),
                           gap_mean=float(np.mean(gaps)), gap_sd=float(np.std(gaps)))
    return dict(backend=name, f=f, shots=res)


if __name__ == "__main__":
    C.memory_guard()
    t0 = time.time()
    picks = []
    for name in C.BACKENDS:
        be = C.Dv.backend_by_name(name)
        ps, _ = C.select_pairs(be)
        picks.append((name, [p for p in ps if p["label"] == "typical"][0]))
    with Pool(6) as pool:
        exact_res = pool.map(exact, picks)
    rows, jobs, seed = [], [], 5000
    ideal_forms = S.forms_from_probs(P_IDEAL)
    for name, pick, out in exact_res:
        for f in TAUS:
            Ps = out[f]
            row = dict(backend=name, pair=pick["pair"], f=f, tau=f * pick["tc_theory"])
            for method in W:
                Pm = combine(Ps, method)
                fm = S.forms_from_probs(Pm)
                row[method] = dict(eps=S.eps_from_probs(Pm), gap=S.signed_gap(Pm),
                                   bias=max(np.abs(fm[p] - ideal_forms[p]).max() for p in "AB"))
            rows.append(row)
            seed += 1
            jobs.append((name, f, Ps, seed))
    print(f"exact part done in {time.time()-t0:.0f}s", flush=True)
    with Pool(6) as pool:
        shot_rows = pool.map(shots_job, jobs, chunksize=1)
    C.save("e3_zne.json", dict(taus=TAUS, n_tot=N_TOT, reps=REPS, boot=BOOT, tol=TOL, weights=W,
                               exact=rows, shots=shot_rows))
    print(f"done in {time.time()-t0:.0f}s")
