"""Diagnose the hardware data with an explicit idle-noise model.

During the idle delay each qubit undergoes (i) energy relaxation with the backend's calibrated T1,
(ii) free-induction dephasing with an unknown T2* (no echo), and (iii) a coherent phase rotation
exp(-i Delta t Z/2) from an unknown frequency offset Delta; a constant depolarizing term r accounts for
gate errors.  The parameters (Delta_A, Delta_B, T2*_A, T2*_B, r) are fitted by weighted least squares to
the readout-mitigated outcome distributions of all game circuits at all delays.  A second fit without the
coherent term (Delta = 0) tests whether decoherence alone can explain the data.

  python hardware_fit.py hardware_job_<tag>.json
"""
import sys

import numpy as np
from scipy.optimize import least_squares

import common as C
import hardware_analyze as H
from ewlcert import game as G
from ewlcert import stats as S
from ewlcert import theory as Th


def rz(phi):
    return [np.diag(np.exp(-1j * phi / 2 * np.array([1, -1])))]


def model_probs(params, taus, T1, coherent=True, shape="exp"):
    dA, dB, t2a, t2b, r = params if coherent else (0.0, 0.0, *params)
    out = {}
    for tau in taus:
        ks = []
        for d, T1q, T2s in ((dA, T1[0], t2a), (dB, T1[1], t2b)):
            g = 1 - np.exp(-tau / T1q)
            deph = np.exp(-tau / T2s) if shape == "exp" else np.exp(-(tau / T2s) ** 2)
            c = np.sqrt(1 - g) * deph                     # relaxation part times pure-dephasing part
            ks.append(G.compose(G.depolarizing(r), G.compose(rz(d * tau), G.thermal_relaxation(g, c))))
        pre = G.local(ks[0], ks[1])
        out[tau] = np.array([G.probabilities(G.U_x(a), G.U_x(b), pre=pre) for a, b in S.circuit_strategies()])
    return out


def fit(recs, coherent=True, shape="exp"):
    """Joint fit to one or several hardware records of the same qubit pair."""
    recs = recs if isinstance(recs, list) else [recs]
    T1, T2 = recs[0]["pick"]["T1"], recs[0]["pick"]["T2"]
    obs, wts, taus = [], [], []
    for rec in recs:
        data = H.by_delay(rec)
        for f in rec["fractions"]:
            g, c0, c1 = data[f]
            P = H.probs(g, (c0, c1))
            n = g.sum(axis=1, keepdims=True)
            obs.append(P)
            wts.append(1 / np.sqrt(np.clip(P * (1 - P), 1e-4, None) / n))
            taus.append(f * rec["tc_pred"])
    uniq = sorted(set(taus))

    def resid(p):
        m = model_probs(p, uniq, T1, coherent, shape)
        return np.concatenate([((m[t] - o) * w).ravel() for t, o, w in zip(taus, obs, wts)])

    if coherent:
        # stage 1: common offset (Delta_A = Delta_B) from a coarse grid of starts; stage 2: free offsets
        def resid_common(q):
            return resid([q[0], q[0], q[1], q[2], q[3]])
        best1 = None
        for d0 in np.linspace(-1.2e5, 1.2e5, 9):           # rad/s, about +-19 kHz
            sol = least_squares(resid_common, [d0, T2[0], T2[1], 0.02],
                                bounds=([-3e5, 2e-6, 2e-6, 0], [3e5, 2e-3, 2e-3, 0.5]),
                                x_scale=[1e4, 1e-5, 1e-5, 0.01], max_nfev=60)
            if best1 is None or sol.cost < best1.cost:
                best1 = sol
        d, t2a, t2b, r = best1.x
        best = None
        for split in (-0.5, 0.0, 0.5):                     # try unequal offsets with the same sum
            x0 = [d * (1 + split), d * (1 - split), t2a, t2b, r]
            sol = least_squares(resid, x0, bounds=([-3e5, -3e5, 2e-6, 2e-6, 0], [3e5, 3e5, 2e-3, 2e-3, 0.5]),
                                x_scale=[1e4, 1e4, 1e-5, 1e-5, 0.01], max_nfev=200)
            if best is None or sol.cost < best.cost:
                best = sol
    else:
        best = least_squares(resid, [T2[0], T2[1], 0.02], bounds=([2e-6, 2e-6, 0], [2e-3, 2e-3, 0.5]),
                             x_scale=[1e-5, 1e-5, 0.01])
    dof = len(best.fun) - len(best.x)
    return best, 2 * best.cost / dof, dof


if __name__ == "__main__":
    # usage: python hardware_fit.py <out_name> <record1.json> [<record2.json> ...]
    out_name, files = sys.argv[1], sys.argv[2:]
    recs = [C.load(f) for f in files]
    rec = recs[0]
    out = {"files": files}
    for name, coh, shape in (("coherent_exp", True, "exp"), ("decoherence_only", False, "exp")):
        sol, chi2, dof = fit(recs, coh, shape)
        row = {"chi2_per_dof": chi2, "dof": dof, "params": sol.x.tolist()}
        if coh:
            row.update(detuning_kHz=[sol.x[0] / (2 * np.pi) / 1e3, sol.x[1] / (2 * np.pi) / 1e3],
                       T2star_us=[sol.x[2] * 1e6, sol.x[3] * 1e6], r=sol.x[4])
            # critical idle time implied by the fitted model (signed D-gain root, fine grid)
            T1 = rec["pick"]["T1"]
            ts = np.linspace(1e-6, 1.6 * rec["tc_pred"], 321)
            gains = [S.mean_D_gain(model_probs(sol.x, [t], T1, True, shape)[t]) for t in ts]
            row["tc_model_fit_us"] = H.interp_root(list(ts * 1e6), gains)
            row["gain_curve"] = {"t_us": (ts * 1e6).tolist(), "dgain": gains}
        else:
            row.update(T2star_us=[sol.x[0] * 1e6, sol.x[1] * 1e6], r=sol.x[2])
            # what tau_c would the decoherence-only model predict with the fitted dephasing?
            T1 = rec["pick"]["T1"]
            # total coherence time 1/T2 = 1/(2 T1) + 1/T_phi with the fitted pure-dephasing times
            T2a = 1 / (1 / sol.x[0] + 1 / (2 * T1[0]))
            T2b = 1 / (1 / sol.x[1] + 1 / (2 * T1[1]))
            row["tc_decoherence_fit_us"] = Th.critical_idle_time(T1[0], T2a, T1[1], T2b) * 1e6
        out[name] = row
        print(name, {k: (np.round(v, 3).tolist() if isinstance(v, (list, np.ndarray)) else (round(v, 4) if isinstance(v, float) else v))
                     for k, v in row.items() if k not in ("params", "gain_curve")}, flush=True)
    out["calibrated_T2_us"] = [t * 1e6 for t in rec["pick"]["T2"]]
    C.save(out_name, out)
