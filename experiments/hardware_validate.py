"""Coverage check of the hardware threshold estimator on synthetic data drawn from the device model.

For the dry-run pair, exact outcome distributions at the hardware delays are sampled REPS times with the
hardware shot number; each synthetic data set is analysed exactly like real data (mean D-gain, linear
interpolation of the sign change, bootstrap interval).  Reports bias and the coverage of the 95% interval
for the device-model critical time.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np

import common as C
import hardware_analyze as H
from ewlcert import stats as S

warnings.filterwarnings("ignore")
REPS, BOOT = 200, 300


def one(arg):
    seed, Ps, conf, shots, fs = arg
    rng = np.random.default_rng(seed)
    curves = []
    data = {}
    for f in fs:
        g = np.stack([rng.multinomial(shots, np.clip(p, 0, None) / np.clip(p, 0, None).sum()) for p in Ps[f] @ conf.T])
        c0, c1 = rng.multinomial(shots, conf[:, 0]), rng.multinomial(shots, conf[:, 3])
        data[f] = (g, c0, c1)
    point = H.interp_root(fs, [S.mean_D_gain(H.probs(*([data[f][0]] + [(data[f][1], data[f][2])]))) for f in fs])
    boot = []
    for _ in range(BOOT):
        curve = []
        for f in fs:
            g, c0, c1 = data[f]
            curve.append(S.mean_D_gain(H.probs(H.resample(g, rng), (H.resample(c0, rng), H.resample(c1, rng)))))
        boot.append(H.interp_root(fs, curve))
    boot = np.array(boot)
    ok = np.isfinite(boot)
    lo, hi = (np.quantile(boot[ok], [0.025, 0.975]) if ok.any() else (np.nan, np.nan))
    return point, lo, hi


if __name__ == "__main__":
    rec = C.load(sys.argv[1] if len(sys.argv) > 1 else "hardware_dryrun_fake_fez.json")
    be = C.Dv.backend_by_name(rec["backend"])
    dp = C.Dv.DevicePair(be, rec["pair"])
    fs, tc = rec["fractions"], rec["tc_pred"]
    Ps = {f: dp.run([dp.physical(a, b, f * tc) for a, b in S.circuit_strategies()]) for f in fs}
    f_true = H.interp_root(fs, [S.mean_D_gain(Ps[f]) for f in fs])        # same interpolation, no noise
    # exact root on a fine grid, for the interpolation bias
    fine = np.linspace(0.85, 1.15, 31)
    gf = [S.mean_D_gain(dp.run([dp.physical(a, b, f * tc) for a, b in S.circuit_strategies()])) for f in fine]
    f_exact = H.interp_root(list(fine), gf)
    with Pool(6) as pool:
        res = np.array(pool.map(one, [(1000 + i, Ps, dp.confusion, rec["shots"], fs) for i in range(REPS)]))
    pts, lo, hi = res[:, 0], res[:, 1], res[:, 2]
    out = dict(reps=REPS, boot=BOOT, shots=rec["shots"], tc_exact_us=f_exact * tc * 1e6,
               tc_interp_noiseless_us=f_true * tc * 1e6,
               mean_estimate_us=float(np.nanmean(pts) * tc * 1e6), sd_estimate_us=float(np.nanstd(pts) * tc * 1e6),
               coverage_exact=float(np.mean((lo <= f_exact) & (f_exact <= hi))),
               coverage_interp=float(np.mean((lo <= f_true) & (f_true <= hi))))
    C.save("hardware_estimator_validation.json", out)
    print(out)
