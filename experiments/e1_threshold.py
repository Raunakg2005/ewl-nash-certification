"""E1: does the closed-form T1/T2 condition predict where (Q,Q) stops being an equilibrium on
device-calibrated noise models (gate errors, relaxation during gates, readout)?

For each selected qubit pair: exact outcome distributions of the 11 certification circuits over a
grid of idle delays inserted after the entangler; the simulated critical delay is the root of the
signed vertex gap (with and without readout error)."""
import time
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

import common as C
from ewlcert import stats as S

warnings.filterwarnings("ignore")
GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]


def probs(dp, tau):
    circs = [dp.physical(a, b, tau) for a, b in S.circuit_strategies()]
    return dp.run(circs)


def job(arg):
    name, pick = arg
    C.memory_guard()
    t0 = time.time()
    be = C.Dv.backend_by_name(name)
    dp = C.Dv.DevicePair(be, pick["pair"])
    tc = pick["tc_theory"]
    cache = {}

    def P_of(tau):
        if tau not in cache:
            cache[tau] = probs(dp, tau)
        return cache[tau]

    rows = []
    for f in GRID:
        P = P_of(f * tc)
        Pr = dp.with_readout(P)
        rows.append(dict(f=f, tau=f * tc, eps=S.eps_from_probs(P), gap=S.signed_gap(P),
                         eps_ro=S.eps_from_probs(Pr), gap_ro=S.signed_gap(Pr), offdiag=S.max_offdiag(P),
                         forms={k: v.tolist() for k, v in S.forms_from_probs(P).items()}))

    def root(key):
        g = [r[key] for r in rows]
        for i in range(len(g) - 1):
            if g[i] < 0 <= g[i + 1]:
                fun = (lambda t: S.signed_gap(P_of(t))) if key == "gap" else (lambda t: S.signed_gap(dp.with_readout(P_of(t))))
                return brentq(fun, rows[i]["tau"], rows[i + 1]["tau"], xtol=1e-3 * tc)
        return np.nan

    out = dict(backend=name, label=pick["label"], pair=pick["pair"], T1=pick["T1"], T2=pick["T2"],
               T2eff=pick["T2eff"], readout=pick["readout"], err2q=pick["err2q"], tc_theory=tc,
               tc_sim=root("gap"), tc_sim_readout=root("gap_ro"),
               confusion=dp.confusion.tolist(), grid=rows, seconds=time.time() - t0)
    print(f"{name} {pick['label']} {pick['pair']}: theory {tc*1e6:.2f} us, sim {out['tc_sim']*1e6:.2f} us, "
          f"sim+readout {out['tc_sim_readout']*1e6:.2f} us ({out['seconds']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    C.memory_guard()
    jobs, meta = [], {}
    for name in C.BACKENDS:
        be = C.Dv.backend_by_name(name)
        picks, info = C.select_pairs(be)
        meta[name] = info
        jobs += [(name, p) for p in picks]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    C.save("e1_threshold.json", dict(meta=meta, grid=GRID, pairs=res))
