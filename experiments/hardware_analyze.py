"""Analyse a hardware (or dry-run) record written by hardware_run.py.

For every idle delay: readout-mitigated and raw estimates of the incentive to deviate (bootstrap 95%
upper bound), the signed gap delta with a bootstrap 95% interval, and the Bonferroni violation test.
The critical idle time is estimated by linear interpolation of the mitigated delta between the delays
that bracket its sign change, with a bootstrap interval.  Writes results/<name>_analysis.json and
figures/fig_hardware.{pdf,png}.

  python hardware_analyze.py results/hardware_job_<tag>.json
"""
import os
import sys

import numpy as np

import common as C
from ewlcert import stats as S

B = 600


def by_delay(rec):
    fr = rec["fractions"]
    out = {}
    for f in fr:
        idx = [i for i, m in enumerate(rec["meta"]) if m["f"] == f]
        game = sorted([i for i in idx if rec["meta"][i]["kind"] == "game"], key=lambda i: rec["meta"][i]["index"])
        c0 = [i for i in idx if rec["meta"][i]["kind"] == "cal00"][0]
        c1 = [i for i in idx if rec["meta"][i]["kind"] == "cal11"][0]
        out[f] = (np.array([rec["counts"][i] for i in game]), np.array(rec["counts"][c0]), np.array(rec["counts"][c1]))
    return out


def resample(counts, rng):
    n = counts.sum(axis=-1)
    if counts.ndim == 1:
        return rng.multinomial(n, counts / n)
    return np.stack([rng.multinomial(k, c / k) for k, c in zip(n, counts)])


def probs(counts, cal=None):
    P = counts / counts.sum(axis=1, keepdims=True)
    return S.mitigate(P, S.confusion_from_cal(*cal)) if cal is not None else P


def interp_root(fs, gaps):
    for i in range(len(fs) - 1):
        if gaps[i] < 0 <= gaps[i + 1]:
            return fs[i] + (fs[i + 1] - fs[i]) * (-gaps[i]) / (gaps[i + 1] - gaps[i])
    return np.nan


def main(path):
    rec = C.load(os.path.basename(path)) if not os.path.isabs(path) else __import__("json").load(open(path))
    rng = np.random.default_rng(2027)
    data = by_delay(rec)
    fs = rec["fractions"]
    rows = []
    boot_gaps = np.empty((B, len(fs)))
    for j, f in enumerate(fs):
        g, c0, c1 = data[f]
        row = {"f": f, "tau_us": f * rec["tc_pred"] * 1e6}
        for key, cal in (("mit", (c0, c1)), ("raw", None)):
            pt, lo, hi = S.bootstrap(g, cal=cal, B=B, rng=rng)
            viol, gains = S.violation_test(g, cal=cal)
            gap_draws, dg_draws = [], []
            for b in range(B):
                gb = resample(g, rng)
                cb = (resample(c0, rng), resample(c1, rng)) if cal is not None else None
                Pb = probs(gb, cb)
                gap_draws.append(S.signed_gap(Pb))
                dg_draws.append(S.mean_D_gain(Pb))
                if key == "mit":
                    boot_gaps[b, j] = dg_draws[-1]
            P0 = probs(g, cal)
            row[key] = dict(eps=pt, eps_upper95=hi, gap=S.signed_gap(P0),
                            gap_ci95=np.quantile(gap_draws, [0.025, 0.975]).tolist(),
                            dgain=S.mean_D_gain(P0), dgain_ci95=np.quantile(dg_draws, [0.025, 0.975]).tolist(),
                            violation=viol, certified=hi <= 0.05)
        rows.append(row)
    # threshold from the unbiased mean D-gain (Theorem 4), not from the max-based signed gap
    f_hat = interp_root(fs, [r["mit"]["dgain"] for r in rows])
    f_boot = np.array([interp_root(fs, boot_gaps[b]) for b in range(B)])
    ok = np.isfinite(f_boot)
    out = dict(source=os.path.basename(path), backend=rec["backend"], pair=rec["pair"], shots=rec["shots"],
               tc_pred_us=rec["tc_pred"] * 1e6, tc_model_us=rec.get("model_tc", np.nan) * 1e6,
               tc_hw_us=f_hat * rec["tc_pred"] * 1e6,
               tc_hw_ci95_us=(np.quantile(f_boot[ok], [0.025, 0.975]) * rec["tc_pred"] * 1e6).tolist() if ok.any() else None,
               frac_bootstrap_with_root=float(ok.mean()), rows=rows, calibration=rec["calibration"],
               job_id=rec.get("job_id"), dry_run=rec.get("dry_run"))
    name = os.path.splitext(os.path.basename(path))[0]
    C.save(f"{name}_analysis.json", out)
    for r in rows:
        print(f"tau={r['tau_us']:7.1f} us  Dgain(mit)={r['mit']['dgain']:+.3f} CI{np.round(r['mit']['dgain_ci95'],3)} "
              f"eps_U95={r['mit']['eps_upper95']:.3f} viol={r['mit']['violation']}  | raw gap={r['raw']['gap']:+.3f}")
    print(f"tau_c: predicted {out['tc_pred_us']:.1f} us, device model {out['tc_model_us']:.1f} us, "
          f"measured {out['tc_hw_us']:.1f} us, 95% CI {out['tc_hw_ci95_us']}")
    figure(rec, out)


def figure(rec, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_figures as F
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), constrained_layout=True)
    ax = axs[0]
    mf = [m["f"] for m in rec["model"]]
    ax.plot(mf, [m.get("dgain", m["gap"]) for m in rec["model"]], color=F.INK2, ls="--", label="device model")
    x = [r["f"] for r in out["rows"]]
    y = [r["mit"]["dgain"] for r in out["rows"]]
    lo = [r["mit"]["dgain"] - r["mit"]["dgain_ci95"][0] for r in out["rows"]]
    hi = [r["mit"]["dgain_ci95"][1] - r["mit"]["dgain"] for r in out["rows"]]
    lab = "hardware, readout mitigated" if not out["dry_run"] else "synthetic data, mitigated"
    ax.errorbar(x, y, yerr=[lo, hi], color=F.BLUE, marker="o", ms=4, capsize=2, ls="none", label=lab)
    ax.plot(x, [r["raw"]["dgain"] for r in out["rows"]], color=F.ORANGE, marker="s", ms=3.5, ls="none", label="unmitigated")
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.axvline(1, color=F.INK2, lw=0.8, ls=":")
    ax.set_xlabel(r"idle delay $\tau/\tau_c^{\mathrm{pred}}$")
    ax.set_ylabel(r"mean gain of deviating to $D$")
    ax.legend(loc="lower right", fontsize=7)
    F.panel(ax, "a")
    ax = axs[1]
    ax.plot(x, [r["mit"]["eps_upper95"] for r in out["rows"]], color=F.BLUE, marker="o", ms=4, label="95% upper bound, mitigated")
    ax.plot(x, [r["mit"]["eps"] for r in out["rows"]], color=F.AQUA, marker="^", ms=4, ls="--", label="point estimate")
    ax.axhline(0.05, color=F.RED, lw=0.9, ls=":", label=r"tolerance $\epsilon_0$")
    ax.set_xlabel(r"idle delay $\tau/\tau_c^{\mathrm{pred}}$")
    ax.set_ylabel(r"incentive to deviate $\epsilon$")
    ax.legend(loc="upper left", fontsize=7)
    F.panel(ax, "b")
    F.save(fig, "fig_hardware" if not out["dry_run"] else "fig_hardware_dryrun")


if __name__ == "__main__":
    main(sys.argv[1])
