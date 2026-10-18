"""Hardware figure: ibm_kingston, qubits (16, 23); coarse run (job 1) and dense run (job 2)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
import make_figures as F

RUN1, RUN2 = "hardware_job_20261018_111455", "hardware_job_20261018_113856"

if __name__ == "__main__":
    a1, a2 = C.load(RUN1 + "_analysis.json"), C.load(RUN2 + "_analysis.json")
    r2 = C.load(RUN2 + ".json")
    fit2 = C.load(RUN2 + "_fit.json")
    vert = [v for v in C.load("hardware_vertex.json") if v["file"].startswith(RUN2)]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)

    ax = axs[0]
    tc = r2["tc_pred"] * 1e6
    ax.plot([m["f"] * tc for m in r2["model"]], [m["dgain"] for m in r2["model"]], color=F.INK2, ls="--",
            lw=1.2, label=r"$T_1/T_2$ device model")
    gc = fit2["coherent_exp"]["gain_curve"]
    ax.plot(gc["t_us"], gc["dgain"], color=F.BLUE, lw=1.0, alpha=0.8, label="fit with frequency offset (run 2)")
    for a, col, mk, lab in ((a2, F.BLUE, "o", "run 2 (dense)"), (a1, F.ORANGE, "s", "run 1 (coarse)")):
        x = [r["tau_us"] for r in a["rows"]]
        y = [r["mit"]["dgain"] for r in a["rows"]]
        lo = [r["mit"]["dgain"] - r["mit"]["dgain_ci95"][0] for r in a["rows"]]
        hi = [r["mit"]["dgain_ci95"][1] - r["mit"]["dgain"] for r in a["rows"]]
        ax.errorbar(x, y, yerr=[lo, hi], color=col, marker=mk, ms=3.5, capsize=1.5, ls="none", label=lab)
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.axhline(0.5, color=F.INK2, lw=0.9, ls="-.", label=r"maximum under $T_1/T_2$ noise")
    ax.set_xlim(-5, 245)
    ax.set_ylim(-3.2, 2.6)
    ax.set_xlabel(r"idle delay $\tau$ ($\mu$s)")
    ax.set_ylabel(r"mean gain of deviating to $D$")
    F.panel(ax, "a")

    ax = axs[1]
    x = [v["tau_us"] for v in vert]
    ub = {round(r["tau_us"], 3): r["mit"]["eps_upper95"] for r in a2["rows"]}
    ax.plot(x, [ub[round(t, 3)] for t in x], color=F.BLUE, marker="o", ms=3.5, label="six-circuit test, 95% bound")
    ax.plot(x, [v["eps"] for v in vert], color=F.AQUA, marker="^", ms=3.5, ls="--", label="six-circuit test, estimate")
    ax.plot(x, [v["eps_vertex"] for v in vert], color=F.VIOLET, marker="D", ms=3.2, ls=":", label="test with $C$, $D$ only")
    ax.axhline(0.05, color=F.RED, lw=0.9, ls=":", label=r"tolerance $\epsilon_0$")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.05, 2.45)
    ax.set_xlabel(r"idle delay $\tau$ ($\mu$s), run 2")
    ax.set_ylabel(r"incentive to deviate $\epsilon$")
    F.panel(ax, "b")

    h0, l0 = axs[0].get_legend_handles_labels()
    h1, l1 = axs[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="outside lower center", ncol=3, fontsize=6.8, columnspacing=1.0, handlelength=1.8)
    F.save(fig, "fig_hardware")
