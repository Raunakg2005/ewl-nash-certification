"""All figures of the paper (PDF + PNG), from results/*.json and the closed-form theory.
After drawing, every figure is checked automatically for overlapping text and for text outside the
canvas (check_layout)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
from ewlcert import theory as Th

FIG = os.path.join(C.ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 5,
    "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150})
DEV_MARK = {"fake_torino": "o", "fake_fez": "s", "fake_marrakesh": "D",
            "fake_brisbane": "^", "fake_sherbrooke": "v", "fake_kyiv": "P"}
DEV_NAME = {k: k.replace("fake_", "").capitalize() for k in DEV_MARK}
LAB_COL = {"short": ORANGE, "typical": BLUE, "long": AQUA}


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")


def check_layout(fig, name):
    """Fail loudly if any two text boxes overlap or a text box leaves the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    hidden = set()                         # tick labels outside the view limits are never drawn
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if not (lo - 1e-9 * abs(hi - lo) <= loc <= hi + 1e-9 * abs(hi - lo)):
                    hidden.update({id(tick.label1), id(tick.label2)})
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    problems = []
    for t, b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            problems.append(f"outside canvas: {t.get_text()!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ov = max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
            if ov > 2:
                problems.append(f"overlap: {boxes[i][0].get_text()!r} / {boxes[j][0].get_text()!r}")
    if problems:
        raise RuntimeError(f"{name}: " + "; ".join(problems))


def save(fig, name):
    check_layout(fig, name)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ Fig. 1 theory
def fig_theory():
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.5), constrained_layout=True)
    ax = axs[0]
    g = np.linspace(0, 1, 401)
    c_pre = np.sqrt(((1 - g) ** 2 + g ** 2) / 7)
    c_post = (1 - g) / np.sqrt(7)
    c_max = np.sqrt(1 - g)
    ax.fill_between(g, c_pre, c_max, where=c_max >= c_pre, color="#cde2fb", lw=0, label="NE (noise on entangled state)")
    ax.plot(g, c_max, color=INK2, lw=1.0, ls=":", label=r"physical limit $c=\sqrt{1-g}$")
    ax.plot(g, c_pre, color=BLUE, label=r"boundary, entangled state")
    ax.plot(g, c_post, color=ORANGE, ls="--", label=r"boundary, after moves")
    ax.set_xlabel(r"relaxed population $g$ (both qubits)")
    ax.set_ylabel(r"remaining coherence $c$ (both qubits)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7, columnspacing=1.0)
    panel(ax, "a")
    ax = axs[1]
    t = np.linspace(0, 3, 600)                 # in units of T1
    for ratio, col, ls, mk in [(0.25, ORANGE, "-", "o"), (1.0, BLUE, "--", "s"), (2.0, AQUA, "-.", "^")]:
        gg, cc = Th.idle_params(t, 1.0, ratio)
        m = Th.margin(gg, cc, gg, cc)
        ax.plot(t, m, color=col, ls=ls, label=rf"$T_2/T_1={ratio:g}$")
        tc = Th.critical_idle_time(1.0, ratio, 1.0, ratio)
        ax.plot([tc], [0], marker=mk, color=col, ms=6, mec="white", mew=0.8, zorder=5, ls="none")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(r"idle time on the entangled state $t/T_1$")
    ax.set_ylabel(r"equilibrium margin $\mu$")
    ax.set_xlim(0, 3)
    ax.legend(loc="upper right")
    panel(ax, "b")
    save(fig, "fig_theory")


# ------------------------------------------------------------------ Fig. E1
def fig_threshold():
    d = C.load("e1_threshold.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)
    ax = axs[0]
    lo, hi = np.inf, 0
    for p in d["pairs"]:
        x, y = p["tc_theory"] * 1e6, p["tc_sim"] * 1e6
        lo, hi = min(lo, x, y), max(hi, x, y)
        ax.plot(x, y, marker=DEV_MARK[p["backend"]], color=LAB_COL[p["label"]], ls="none", mec="white", mew=0.6, ms=6)
    ax.plot([0.8 * lo, 1.15 * hi], [0.8 * lo, 1.15 * hi], color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [30, 50, 100, 200, 400]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"predicted $\tau_c$ from $T_1,T_2$ ($\mu$s)")
    ax.set_ylabel(r"simulated $\tau_c$, device model ($\mu$s)")
    hs = [plt.Line2D([], [], marker=m, color=INK2, ls="none", label=DEV_NAME[k]) for k, m in DEV_MARK.items()]
    hs += [plt.Line2D([], [], marker="o", color=c, ls="none", label=f"{k} pair") for k, c in LAB_COL.items()]
    ax.legend(handles=hs, loc="upper left", fontsize=6.5, ncol=1, handletextpad=0.3)
    panel(ax, "a")
    ax = axs[1]
    for p in d["pairs"]:
        f = [r["f"] for r in p["grid"]]
        gap = [r["gap"] for r in p["grid"]]
        ax.plot(f, gap, color=LAB_COL[p["label"]], lw=0.9, alpha=0.75, marker=DEV_MARK[p["backend"]], ms=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel(r"idle delay $\tau/\tau_c^{\mathrm{pred}}$")
    ax.set_ylabel(r"$\max(M_C,M_D)-M_Q$ (device model)")
    panel(ax, "b")
    save(fig, "fig_threshold")


# ------------------------------------------------------------------ Fig. E2
def fig_certification():
    d = C.load("e2_certification.json")
    rows = d["rows"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9), constrained_layout=True, sharex=True)
    cols = {0.0: BLUE, 0.5: AQUA, 0.9: YELLOW, 1.1: ORANGE, 1.5: RED}
    marks = {0.0: "o", 0.5: "s", 0.9: "D", 1.1: "^", 1.5: "v"}
    for f in d["taus"]:
        for key, ls in (("mit", "-"), ("raw", ":")):
            for ax, metric in zip(axs, ("certify_rate", "violation_rate", "coverage")):
                ys = [np.mean([r[key][metric] for r in rows if r["f"] == f and r["shots"] == N]) for N in d["shots"]]
                ax.plot(d["shots"], ys, color=cols[f], ls=ls, marker=marks[f], ms=3.5)
    for ax, title in zip(axs, ("certified $\\epsilon$-NE rate", "violation detected", "coverage of 95% bound")):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(d["shots"]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{n:,}" for n in d["shots"]]))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("shots per circuit")
        ax.set_ylabel(title)
        ax.set_ylim(-0.03, 1.03)
    axs[2].set_ylim(0.6, 1.01)
    axs[2].axhline(0.95, color=INK2, lw=0.8, ls="--")
    handles = [plt.Line2D([], [], color=cols[f], marker=marks[f], ms=3.5, label=(r"$\tau=0$" if f == 0 else rf"$\tau={f:g}\,\tau_c$"))
               for f in d["taus"]]
    handles += [plt.Line2D([], [], color=INK2, ls="-", label="readout mitigated"),
                plt.Line2D([], [], color=INK2, ls=":", label="unmitigated")]
    fig.legend(handles=handles, loc="outside upper center", ncol=7, fontsize=7, columnspacing=0.9, handlelength=1.8)
    for ax, l in zip(axs, "abc"):
        panel(ax, l)
    save(fig, "fig_certification")


# ------------------------------------------------------------------ Fig. E3
def fig_zne():
    d = C.load("e3_zne.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    styles = {"none": (INK2, "-", "o", "unmitigated"), "linear": (ORANGE, "--", "s", "ZNE linear (1,3)"),
              "quadratic": (BLUE, "-.", "^", "ZNE quadratic (1,3,5)")}
    for method, (col, ls, mk, lab) in styles.items():
        xs = d["taus"]
        ys = [np.mean([r[method]["gap"] for r in d["exact"] if r["f"] == f]) for f in xs]
        axs[0].plot(xs, ys, color=col, ls=ls, marker=mk, ms=4, label=lab)
        sd = [np.mean([s["shots"][method]["gap_sd"] for s in d["shots"] if s["f"] == f]) for f in xs]
        axs[1].plot(xs, sd, color=col, ls=ls, marker=mk, ms=4, label=lab)
    axs[0].axhline(-2, color=GREEN, lw=1.0, ls=":", label="ideal game")
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].set_xlabel(r"idle delay $\tau/\tau_c^{\mathrm{pred}}$")
    axs[0].set_ylabel(r"signed gap $\delta$ (no shot noise)")
    axs[0].set_ylim(-2.15, 0.4)
    axs[0].legend(loc="upper left", fontsize=7)
    axs[1].set_xlabel(r"idle delay $\tau/\tau_c^{\mathrm{pred}}$")
    axs[1].set_ylabel(r"s.d. of $\hat\delta$, %s shots/circuit" % f"{d['n_tot']:,}")
    axs[1].set_ylim(0, None)
    for ax, l in zip(axs, "ab"):
        panel(ax, l)
    save(fig, "fig_zne")


# ------------------------------------------------------------------ Fig. E4
def fig_coherent():
    d = C.load("e4_coherent.json")
    rows = d["rows"]
    s = [r["s"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    axs[0].plot(s, [r["frac_eps_gt_tol"] for r in rows], color=BLUE, marker="o",
                label="quadratic-form test (6 circuits/player)")
    axs[0].plot(s, [r["frac_eps_gt_tol"] - r["false_certification_vertex"] for r in rows], color=ORANGE, ls="--",
                marker="s", label="pure-deviation test (C, D only)")
    axs[0].set_xlabel(r"coherent error strength $s$")
    axs[0].set_ylabel(r"violations detected ($\epsilon>0.05$)")
    axs[0].legend(loc="upper left", fontsize=7)
    axs[1].plot(s, [r["median_underestimate"] for r in rows], color=BLUE, marker="o", label="median")
    axs[1].plot(s, [r["p95_underestimate"] for r in rows], color=ORANGE, ls="--", marker="s", label="95th percentile")
    axs[1].set_xlabel(r"coherent error strength $s$")
    axs[1].set_ylabel(r"$\epsilon-\epsilon_{\mathrm{pure}}$ (missed incentive)")
    axs[1].legend(loc="upper left", fontsize=7)
    for ax, l in zip(axs, "ab"):
        panel(ax, l)
    save(fig, "fig_coherent")


if __name__ == "__main__":
    which = sys.argv[1:] or ["theory", "threshold", "certification", "zne", "coherent"]
    for w in which:
        try:
            globals()[f"fig_{w}"]()
        except FileNotFoundError as e:
            print("skip", w, e)
