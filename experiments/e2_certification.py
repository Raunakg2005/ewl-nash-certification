"""E2: finite-shot certification on device-calibrated distributions.

Estimand: the incentive to deviate epsilon of the game as implemented on the device (gate errors and
relaxation included, readout excluded: readout is part of the measurement, not of the game).
For each typical pair and idle delay, R synthetic experiments with N shots per circuit (and N shots
per readout-calibration circuit) are drawn from the exact distributions.  Each experiment yields a
bootstrap upper bound on epsilon (certify an epsilon-equilibrium if the bound is <= TOL) and the
Bonferroni violation test, both with and without readout mitigation."""
import time
import warnings
from multiprocessing import Pool

import numpy as np

import common as C
from ewlcert import stats as S

warnings.filterwarnings("ignore")
TAUS = [0.0, 0.5, 0.9, 1.1, 1.5]          # in units of the predicted critical idle time
SHOTS = [500, 2000, 8000, 32000]
REPS, BOOT, TOL = 200, 200, 0.05


def exact_probs(name, pick):
    be = C.Dv.backend_by_name(name)
    dp = C.Dv.DevicePair(be, pick["pair"])
    Ps = {}
    for f in TAUS:
        circs = [dp.physical(a, b, f * pick["tc_theory"]) for a, b in S.circuit_strategies()]
        Ps[f] = dp.run(circs)
    return Ps, dp.confusion


def experiment(arg):
    name, label, pair, f, P, conf, N, seed = arg
    rng = np.random.default_rng(seed)
    Pm = P @ conf.T
    eps_true, gap_true = S.eps_from_probs(P), S.signed_gap(P)
    eps_ro = S.eps_from_probs(Pm)
    res = {"raw": [], "mit": []}
    for _ in range(REPS):
        counts = S.sample_counts(Pm, N, rng)
        cal = (rng.multinomial(N, conf[:, 0]), rng.multinomial(N, conf[:, 3]))
        for key, c in (("raw", None), ("mit", cal)):
            pt, lo, hi = S.bootstrap(counts, cal=c, B=BOOT, rng=rng)
            viol, _ = S.violation_test(counts, cal=c)
            res[key].append((pt, hi, viol))
    summary = dict(backend=name, label=label, pair=pair, f=f, shots=N, eps_true=eps_true, gap_true=gap_true,
                   eps_with_readout=eps_ro)
    for key in ("raw", "mit"):
        a = np.array(res[key], dtype=float)
        summary[key] = dict(certify_rate=float(np.mean(a[:, 1] <= TOL)), violation_rate=float(np.mean(a[:, 2])),
                            coverage=float(np.mean(eps_true <= a[:, 1] + 1e-12)), mean_point=float(a[:, 0].mean()),
                            mean_upper=float(a[:, 1].mean()))
    return summary


if __name__ == "__main__":
    C.memory_guard()
    t0 = time.time()
    jobs, seed = [], 1000
    for name in C.BACKENDS:
        be = C.Dv.backend_by_name(name)
        picks, _ = C.select_pairs(be)
        pick = [p for p in picks if p["label"] == "typical"][0]
        Ps, conf = exact_probs(name, pick)
        for f in TAUS:
            for N in SHOTS:
                seed += 1
                jobs.append((name, "typical", pick["pair"], f, Ps[f], conf, N, seed))
    print(f"prepared {len(jobs)} jobs in {time.time()-t0:.0f}s", flush=True)
    with Pool(6) as pool:
        out = pool.map(experiment, jobs, chunksize=1)
    C.save("e2_certification.json", dict(taus=TAUS, shots=SHOTS, reps=REPS, boot=BOOT, tol=TOL, rows=out))
    print(f"done in {time.time()-t0:.0f}s")
