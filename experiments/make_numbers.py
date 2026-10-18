"""Writes paper/numbers.tex (one macro per number quoted in the text) and paper/tab_pairs.tex from
results/*.json and the closed-form theory.  No number in the manuscript is typed by hand."""
import os

import numpy as np

import common as C
from ewlcert import theory as Th

PAPER = os.path.join(C.ROOT, "paper")
os.makedirs(PAPER, exist_ok=True)
M = {}


def put(name, value, fmt="{}"):
    s = fmt.format(value)
    if s.startswith("-"):                       # true minus sign in text and in math mode
        s = r"\ensuremath{-}" + s[1:]
    M[name] = s


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


# ---------------------------------------------------------------- theory constants
a_star = Th.AD_THRESHOLD_SYMMETRIC
put("ADthr", a_star, "{:.3f}")
put("ADtcOverTone", -np.log(1 - a_star), "{:.2f}")
put("lnSeven", np.log(7), "{:.3f}")

# ---------------------------------------------------------------- E1
if os.path.exists(os.path.join(C.RESULTS, "e1_threshold.json")):
    d = C.load("e1_threshold.json")
    p = d["pairs"]
    rel = np.array([(x["tc_sim"] - x["tc_theory"]) / x["tc_theory"] for x in p])
    ro = np.array([(x["tc_sim_readout"] - x["tc_sim"]) / x["tc_sim"] for x in p])
    tcs = np.array([x["tc_theory"] for x in p]) * 1e6
    put("EoneNpairs", len(p))
    put("EoneNdev", len(d["meta"]))
    put("EoneTcMin", tcs.min(), "{:.0f}")
    put("EoneTcMax", tcs.max(), "{:.0f}")
    put("EoneMedAbsErr", pct(np.median(np.abs(rel))))
    put("EoneMaxAbsErr", pct(np.abs(rel).max()))
    put("EoneNwithinOnePct", int(np.sum(np.abs(rel) <= 0.013)))
    put("EoneROshiftMean", pct(ro.mean()))
    put("EoneROshiftMin", pct(ro.min()))
    put("EoneROshiftMax", pct(ro.max()))
    put("EoneMaxOffdiag", max(r["offdiag"] for x in p for r in x["grid"]), "{:.4f}")
    m0 = [-x["grid"][0]["gap"] for x in p]
    put("EoneMarginMin", min(m0), "{:.2f}")
    put("EoneMarginMax", max(m0), "{:.2f}")
    put("EoneNcouplersMin", min(v["n_couplers"] for v in d["meta"].values()))
    put("EoneNcouplersMax", max(v["n_couplers"] for v in d["meta"].values()))
    # table of pairs
    lines = []
    for x in sorted(p, key=lambda r: (C.BACKENDS.index(r["backend"]), r["tc_theory"])):
        dev = x["backend"].replace("fake_", "").capitalize()
        lines.append(" & ".join([dev, x["label"][0].upper(), f"{x['pair'][0]}, {x['pair'][1]}",
                                 f"{x['T1'][0]*1e6:.0f}/{x['T1'][1]*1e6:.0f}",
                                 f"{x['T2eff'][0]*1e6:.0f}/{x['T2eff'][1]*1e6:.0f}",
                                 f"{x['err2q']*1e3:.1f}", f"{max(x['readout'])*100:.1f}",
                                 f"{x['tc_theory']*1e6:.1f}", f"{x['tc_sim']*1e6:.1f}",
                                 f"{x['tc_sim_readout']*1e6:.1f}"]) + r" \\")
    with open(os.path.join(PAPER, "tab_pairs.tex"), "w") as f:
        f.write("\n".join(lines) + "\n\\botrule\n")

if os.path.exists(os.path.join(C.RESULTS, "durations.json")):
    dur = np.array(list(C.load("durations.json").values())) * 1e6
    put("DurMin", dur.min(), "{:.1f}")
    put("DurMax", dur.max(), "{:.1f}")

# ---------------------------------------------------------------- E4
if os.path.exists(os.path.join(C.RESULTS, "e4_coherent.json")):
    d = C.load("e4_coherent.json")
    r = {row["s"]: row for row in d["rows"]}
    put("EfourDraws", d["rows"][0]["draws"])
    put("EfourTol", d["tol"])
    for s, tag in ((0.1, "One"), (0.2, "Two"), (0.3, "Three"), (0.5, "Five")):
        put(f"EfourViol{tag}", pct(r[s]["frac_eps_gt_tol"]))
        put(f"EfourMissed{tag}", pct(r[s]["false_cert_given_violation"], 0))
        put(f"EfourPninetyfive{tag}", r[s]["p95_underestimate"], "{:.3f}")
        put(f"EfourMedian{tag}", r[s]["median_underestimate"], "{:.3f}")
    qmins = [row["q_component_min"] for row in d["rows"] if row.get("q_component_min") is not None]
    qmeds = [row["q_component_median"] for row in d["rows"] if row.get("q_component_median") is not None]
    if qmins:
        put("EfourQcompMin", min(qmins), "{:.2f}")
        put("EfourQcompMedMin", min(qmeds), "{:.2f}")

# ---------------------------------------------------------------- E2
if os.path.exists(os.path.join(C.RESULTS, "e2_certification.json")):
    d = C.load("e2_certification.json")
    rows = d["rows"]
    put("EtwoReps", d["reps"])
    put("EtwoBoot", d["boot"])
    put("EtwoTol", d["tol"])

    def agg(f, N, key, metric):
        return np.mean([x[key][metric] for x in rows if x["f"] == f and x["shots"] == N])

    def tag(v):
        return {0.0: "Zero", 0.5: "Half", 0.9: "NineTenths", 1.1: "ElevenTenths", 1.5: "OneHalf"}[v]

    NW = {500: "fivehundred", 2000: "twothousand", 8000: "eightthousand", 32000: "thirtytwothousand"}
    for f in d["taus"]:
        for N in d["shots"]:
            for key in ("raw", "mit"):
                for metric, short in (("certify_rate", "Cert"), ("violation_rate", "Viol"), ("coverage", "Cov")):
                    put(f"Etwo{short}{key.capitalize()}{tag(f)}N{NW[N]}", pct(agg(f, N, key, metric), 1))
    covs = [x["mit"]["coverage"] for x in rows]
    put("EtwoCovMitMin", pct(min(covs), 1))
    put("EtwoCovMitMean", pct(np.mean(covs), 1))
    ne = [x for x in rows if x["gap_true"] < 0]
    put("EtwoFalseAlarmMax", pct(max(max(x["raw"]["violation_rate"], x["mit"]["violation_rate"]) for x in ne), 1))
    mant, expo = "{:.1e}".format(max(x["eps_true"] for x in ne)).split("e")
    put("EtwoEpsAtNEmax", rf"{mant}\times10^{{{int(expo)}}}")
    covr = [x["raw"]["coverage"] for x in rows]
    put("EtwoCovRawMin", pct(min(covr), 0))
    eps_t = {f: np.mean([x["eps_true"] for x in rows if x["f"] == f]) for f in d["taus"]}
    for f in d["taus"]:
        put(f"EtwoEpsTrue{tag(f)}", eps_t[f], "{:.3f}")
        put(f"EtwoGapTrue{tag(f)}", np.mean([x["gap_true"] for x in rows if x["f"] == f]), "{:.2f}")

# ---------------------------------------------------------------- E3
if os.path.exists(os.path.join(C.RESULTS, "e3_zne.json")):
    d = C.load("e3_zne.json")
    put("EthreeNtot", d["n_tot"])
    tg = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 1.0: "One", 1.5: "OneHalf"}
    for f in d["taus"]:
        for method in ("none", "linear", "quadratic"):
            ex = [r[method] for r in d["exact"] if r["f"] == f]
            sh = [s["shots"][method] for s in d["shots"] if s["f"] == f]
            mt = method.capitalize()
            put(f"EthreeGap{mt}{tg[f]}", np.mean([e["gap"] for e in ex]), "{:.2f}")
            put(f"EthreeBias{mt}{tg[f]}", np.mean([e["bias"] for e in ex]), "{:.3f}")
            put(f"EthreeSd{mt}{tg[f]}", np.mean([s["gap_sd"] for s in sh]), "{:.3f}")
            put(f"EthreeCert{mt}{tg[f]}", pct(np.mean([s["certify_ideal_rate"] for s in sh]), 0))
    sd0 = {m: np.mean([s["shots"][m]["gap_sd"] for s in d["shots"] if s["f"] == 0.0]) for m in ("none", "linear", "quadratic")}
    put("EthreeSdRatioQuad", sd0["quadratic"] / sd0["none"], "{:.1f}")
    put("EthreeSdRatioLin", sd0["linear"] / sd0["none"], "{:.1f}")
    put("EthreeShotFactorQuad", (sd0["quadratic"] / sd0["none"]) ** 2, "{:.0f}")
    put("EthreeShotFactorLin", (sd0["linear"] / sd0["none"]) ** 2, "{:.0f}")
    # equal-variance prediction for an even split of the budget over m scales: sqrt(m * sum w^2)
    for m_name, tagn in (("linear", "Lin"), ("quadratic", "Quad")):
        w = np.array(list(d["weights"][m_name].values()), float)
        put(f"EthreeSdRatioTheory{tagn}", np.sqrt(len(w) * np.sum(w ** 2)), "{:.2f}")

# ---------------------------------------------------------------- hardware (ibm_kingston)
HW1, HW2 = "hardware_job_20261018_111455", "hardware_job_20261018_113856"
if os.path.exists(os.path.join(C.RESULTS, HW2 + "_analysis.json")):
    r1, r2 = C.load(HW1 + ".json"), C.load(HW2 + ".json")
    a1, a2 = C.load(HW1 + "_analysis.json"), C.load(HW2 + "_analysis.json")
    f1, f2 = C.load(HW1 + "_fit.json"), C.load(HW2 + "_fit.json")
    vt = C.load("hardware_vertex.json")
    put("HWbackend", r2["backend"].replace("_", r"\_"))
    put("HWpair", f"({r2['pair'][0]}, {r2['pair'][1]})")
    put("HWdate", "18 October 2026")
    put("HWTone", "/".join(f"{t*1e6:.0f}" for t in r2["pick"]["T1"]))
    put("HWTtwo", "/".join(f"{t*1e6:.0f}" for t in r2["pick"]["T2"]))
    put("HWtcPred", r2["tc_pred"] * 1e6, "{:.1f}")
    put("HWnOne", len(r1["meta"]))
    put("HWnTwo", len(r2["meta"]))
    put("HWshotsOne", r1["shots"])
    put("HWshotsTwo", r2["shots"])
    put("HWqpuOne", r1["usage_seconds"], "{:.0f}")
    put("HWqpuTwo", r2["usage_seconds"], "{:.0f}")
    put("HWdelaysOne", len(r1["fractions"]))
    put("HWdelaysTwo", len(r2["fractions"]))
    put("HWtcOne", a1["tc_hw_us"], "{:.1f}")
    put("HWtcTwo", a2["tc_hw_us"], "{:.1f}")
    put("HWtcTwoLo", a2["tc_hw_ci95_us"][0], "{:.1f}")
    put("HWtcTwoHi", a2["tc_hw_ci95_us"][1], "{:.1f}")
    rows2 = {round(r["tau_us"], 1): r for r in a2["rows"]}
    peak = max(a2["rows"], key=lambda r: r["mit"]["dgain"])
    put("HWpeakGain", peak["mit"]["dgain"], "{:.2f}")
    put("HWpeakGainLo", peak["mit"]["dgain_ci95"][0], "{:.2f}")
    put("HWpeakTau", peak["tau_us"], "{:.0f}")
    put("HWgainZero", a2["rows"][0]["mit"]["dgain"], "{:.2f}")
    v2 = {round(v["tau_us"], 1): v for v in vt if v["file"].startswith(HW2)}
    for t, tg in ((5.0, "Five"), (10.0, "Ten")):
        put(f"HWeps{tg}", v2[t]["eps"], "{:.2f}")
        put(f"HWepsUB{tg}", rows2[t]["mit"]["eps_upper95"], "{:.2f}")
        put(f"HWepsCD{tg}", v2[t]["eps_vertex"], "{:.2f}")
        put(f"HWgain{tg}", v2[t]["dgain"], "{:.2f}")
        put(f"HWqcomp{tg}", v2[t]["q_component_best"], "{:.2f}")
        put(f"HWoffdiag{tg}", v2[t]["offdiag"], "{:.2f}")
    put("HWviolTen", "yes" if rows2[10.0]["mit"]["violation"] else "no")
    rest = [v2[t] for t in (50.0, 55.0, 60.0, 65.0)]
    put("HWepsRestMin", min(v["eps"] for v in rest), "{:.2f}")
    put("HWepsRestMax", max(v["eps"] for v in rest), "{:.2f}")
    put("HWrestoreLo", 50)
    put("HWrestoreHi", 65)
    put("HWfirstViolTau", min(t for t, r in rows2.items() if r["mit"]["violation"]), "{:.0f}")
    put("HWgainFortyRunOne", [r for r in a1["rows"] if abs(r["tau_us"] - 39.7) < 0.2][0]["mit"]["dgain"], "{:.2f}")
    put("HWgainFortyRunTwo", rows2[40.0]["mit"]["dgain"], "{:.2f}")
    for f, tg in ((f1, "One"), (f2, "Two")):
        ce, do = f["coherent_exp"], f["decoherence_only"]
        put(f"HWchiCoh{tg}", ce["chi2_per_dof"], "{:.1f}")
        put(f"HWchiDec{tg}", do["chi2_per_dof"], "{:.1f}")
        put(f"HWdetA{tg}", ce["detuning_kHz"][0], "{:.1f}")
        put(f"HWdetB{tg}", ce["detuning_kHz"][1], "{:.1f}")
        put(f"HWdetSum{tg}", abs(sum(ce["detuning_kHz"])), "{:.1f}")
        put(f"HWTphiA{tg}", ce["T2star_us"][0], "{:.0f}")
        put(f"HWTphiB{tg}", ce["T2star_us"][1], "{:.0f}")
        put(f"HWtcFit{tg}", ce["tc_model_fit_us"], "{:.1f}")
        put(f"HWTphiDecB{tg}", min(do["T2star_us"]), "{:.0f}")
    put("HWperiod", 1e3 / abs(sum(f2["coherent_exp"]["detuning_kHz"])), "{:.0f}")
    val = C.load("hardware_estimator_validation.json")
    put("HWvalBias", abs(val["mean_estimate_us"] - val["tc_exact_us"]), "{:.1f}")
    put("HWvalSd", val["sd_estimate_us"], "{:.1f}")
    put("HWvalCov", pct(val["coverage_exact"], 1))

with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
    f.write("% generated by experiments/make_numbers.py -- do not edit\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(M)} macros")
