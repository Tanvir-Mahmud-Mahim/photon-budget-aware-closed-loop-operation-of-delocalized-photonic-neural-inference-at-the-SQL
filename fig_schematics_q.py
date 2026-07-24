"""PILOT-Q Fig. 1 (concept) and Fig. 2 (architecture) — optical substrate.

Reuses the lane/slot drawing system of the PILOT (RF) schematics.
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # shared figure toolchain ships in this repo
import figstyle as fs
from figstyle import C
import fig_schematics as B          # helpers: card/stage/chip/arrow/... + AUTOFIT

fs.use_style()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results", "results.json")
os.makedirs(FIG, exist_ok=True)
TINT, EDGE = B.TINT, B.EDGE


# ------------------------------------------------------------ new glyphs ---
def photodiode(ax, x, y, s=0.05, color=None, zorder=6):
    """Balanced photodiode pair glyph."""
    col = color or C["ink2"]
    for dy in (+0.55 * s, -0.55 * s):
        ax.add_patch(Polygon([(x - s * 0.5, y + dy + s * 0.35),
                              (x - s * 0.5, y + dy - s * 0.35),
                              (x + s * 0.25, y + dy)], closed=True,
                     fill=False, ec=col, lw=1.0, zorder=zorder))
        ax.plot([x + s * 0.25, x + s * 0.25], [y + dy - s * 0.35, y + dy + s * 0.35],
                color=col, lw=1.0, zorder=zorder)


def modulator(ax, x, y, w=0.075, h=0.10, color=None, zorder=6):
    """Mach-Zehnder modulator glyph: lens-shaped split/recombine."""
    col = color or C["violet"]
    th = np.linspace(0, np.pi, 40)
    xs = x - w / 2 + w * (1 - np.cos(th)) / 2
    ax.plot(xs, y + h / 2 * np.sin(th) * 0.9, color=col, lw=1.2, zorder=zorder)
    ax.plot(xs, y - h / 2 * np.sin(th) * 0.9, color=col, lw=1.2, zorder=zorder)
    ax.plot([x - w / 2 - 0.018, x - w / 2], [y, y], color=col, lw=1.2, zorder=zorder)
    ax.plot([x + w / 2, x + w / 2 + 0.018], [y, y], color=col, lw=1.2, zorder=zorder)


def laser(ax, x, y, s=0.06, color=None, zorder=6):
    col = color or "#b57b00"
    ax.add_patch(FancyBboxPatch((x - s, y - s * 0.5), 1.4 * s, s,
                 boxstyle="round,pad=0.003,rounding_size=0.012",
                 fc="#fdf3dd", ec=col, lw=1.1, zorder=zorder))
    for i in range(3):
        yy = y - s * 0.25 + i * s * 0.25
        ax.plot([x + 0.5 * s, x + 1.05 * s], [yy, yy], color=col, lw=1.0,
                zorder=zorder, solid_capstyle="round")


def beam(ax, p1, p2, color="#b57b00", lw=1.6, zorder=4, ms=9):
    B.arrow(ax, p1, p2, color=color, lw=lw, ms=ms, zorder=zorder)


# ------------------------------------------------------------- panel (a) ---
def panel_system(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    B.panel_title(ax, "delocalized photonic in-physics inference")
    B.card(ax, 0.025, 0.22, 0.26, 0.62, TINT["panel"], C["ink2"], lw=0.9)
    ax.text(0.155, 0.795, "central light server", ha="center",
            va="center", fontsize=5.8, fontweight="bold", color=C["ink"])
    B.stage(ax, 0.155, 0.655, 0.21, 0.155, "pre-trained model",
            sub=r"$\mathbf{W}^{(1)},\dots,\mathbf{W}^{(L)}$", tint="blue",
            fsz=5.8, shadow=False, accent=False)
    B.stage(ax, 0.155, 0.46, 0.21, 0.155, "weight intensity\nencoding (TDM)",
            tint="violet", fsz=5.8, shadow=False, accent=False)
    laser(ax, 0.13, 0.30, s=0.055)
    ax.text(0.155, 0.185, r"weight-encoded light", ha="center",
            fontsize=5.5, color=C["ink2"])
    # splitter node + beams
    ax.plot([0.285, 0.36], [0.50, 0.50], color="#b57b00", lw=1.6)
    ax.add_patch(Circle((0.375, 0.50), 0.014, fc="white", ec="#b57b00",
                 lw=1.2, zorder=6))
    ax.text(0.375, 0.415, "1:U\nsplitter", ha="center", fontsize=5.0,
            color=C["ink2"])
    for yy in [0.735, 0.50, 0.265]:
        beam(ax, (0.39, 0.50 + 0.35 * (yy - 0.50)), (0.505, yy))
    B.chip(ax, 0.44, 0.062,
           "link loss + background light + trim drift", tint="red", fsz=5.4)
    icons = [B.drone, B.camera, B.sensor]
    names = ["drone", "camera", "IoT node"]
    for i, yc in enumerate([0.735, 0.50, 0.265]):
        B.card(ax, 0.52, yc - 0.096, 0.462, 0.192, TINT["aqua"], EDGE["aqua"],
               lw=0.9)
        icons[i](ax, 0.578, yc + 0.024, s=0.072)
        ax.text(0.578, yc - 0.058, names[i], ha="center", fontsize=5.1,
                color=C["ink2"])
        modulator(ax, 0.70, yc + 0.024, w=0.06, h=0.075)
        ax.text(0.70, yc - 0.058, "modulator +\ndetectors", ha="center",
                fontsize=4.6, color=C["ink2"])
        B.arrow(ax, (0.745, yc + 0.024), (0.772, yc + 0.024), lw=1.0, ms=6)
        B.stage(ax, 0.872, yc, 0.175, 0.115,
                r"local $\mathbf{y}=\mathbf{W}\mathbf{x}$", tint="gray",
                fsz=5.6, accent=False, shadow=False, bold=False)


# ------------------------------------------------------------- panel (b) ---
def panel_chain(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.53, 0.955,
            r"one in-physics layer:  $\mathbf{y}^{(l)} = g_\xi(\mathbf{W}^{(l)}"
            r"\mathbf{x}^{(l)})$,   $\xi = (\bar{n},\, \rho_d,\, "
            r"\sigma_{cal},\, b)$",
            ha="center", va="center", fontsize=6.8, fontweight="bold",
            color=C["ink"])
    yl = 0.485
    xs = [0.062, 0.20, 0.35, 0.51, 0.655, 0.792, 0.928]
    B.stage(ax, xs[0], yl, 0.10, 0.21, "request",
            sub=r"$\mathbf{x} \geq 0$", tint="blue", fsz=5.7)
    B.stage(ax, xs[1], yl, 0.105, 0.21, "input\nmodulator", tint="violet",
            fsz=5.6)
    modulator(ax, xs[2], yl, w=0.075, h=0.10)
    ax.text(xs[2] - 0.012, yl - 0.125, "weight\nmodulation", ha="center",
            fontsize=4.8, color=C["violet"])
    photodiode(ax, xs[3], yl, s=0.05)
    ax.text(xs[3] - 0.005, yl - 0.160, "balanced\ndetection\n($\\pm$ rails)",
            ha="center", fontsize=4.8, color=C["ink2"])
    B.stage(ax, xs[4], yl, 0.085, 0.21, "ADC", tint="gray", fsz=6.0)
    B.stage(ax, xs[5], yl, 0.09, 0.21, "ReLU", tint="gray", fsz=6.0)
    B.stage(ax, xs[6], yl, 0.115, 0.21, "digital", sub=r"$\mathbf{y}$",
            tint="aqua", fsz=5.7)
    halfw = [0.054, 0.057, 0.048, 0.045, 0.0465, 0.049, 0.062]
    for i, (a, b) in enumerate(zip(xs[:-1], xs[1:])):
        B.arrow(ax, (a + halfw[i], yl), (b - halfw[i + 1], yl), lw=1.1, ms=7)
    ax.text(xs[6], yl - 0.165, "next layer /\nclass scores", ha="center",
            fontsize=5.1, color=C["ink2"])
    # weight light from the fiber
    beam(ax, (xs[2], 0.80), (xs[2], yl + 0.075), lw=1.5)
    ax.text(xs[2] - 0.042, 0.70, "weight-encoded\nlight", ha="right",
            fontsize=5.5, color="#b57b00")
    # impairment callouts
    B.chip(ax, 0.76, 0.83, r"(i) trim error:  $\mathbf{W}\odot(1+\delta)$",
           tint="red", fsz=5.4)
    B.arrow(ax, (0.655, 0.795), (xs[2] + 0.045, yl + 0.09), color=C["red"],
            lw=0.9, ls=(0, (2.5, 1.6)), ms=6)
    B.chip(ax, 0.235, 0.10,
           r"(ii) shot noise (SQL) $\propto 1/\sqrt{\bar{n}}$", tint="red",
           fsz=5.4)
    B.arrow(ax, (0.345, 0.135), (xs[3] - 0.043, yl - 0.068), color=C["red"],
            lw=0.9, ls=(0, (2.5, 1.6)), ms=6)
    B.chip(ax, 0.585, 0.10, r"(iii) dark counts $\rho_d$", tint="red", fsz=5.4)
    B.arrow(ax, (0.585, 0.145), (xs[3] + 0.043, yl - 0.068), color=C["red"],
            lw=0.9, ls=(0, (2.5, 1.6)), ms=6)
    B.chip(ax, 0.845, 0.10, r"(iv) $b$-bit ADC", tint="red", fsz=5.4)
    B.arrow(ax, (0.80, 0.145), (xs[4] + 0.005, yl - 0.112), color=C["red"],
            lw=0.9, ls=(0, (2.5, 1.6)), ms=6)


# ------------------------------------------------------------- panel (c) ---
def panel_training(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    B.panel_title(ax, "photon-budget-aware (PILOT-Q) training")
    yl = 0.585
    B.stage(ax, 0.107, yl, 0.17, 0.27, "weights",
            sub=r"$\theta=\{\mathbf{W}^{(l)}\}$", tint="blue", fsz=6.2)
    B.stage(ax, 0.402, yl, 0.245, 0.27, "stochastic photonics" + r" $g_\xi$",
            sub="shot + dark + trim + ADC", tint="violet", fsz=6.0)
    B.stage(ax, 0.678, yl, 0.16, 0.27, "task loss",
            sub=r"$\mathcal{L}_{CE}$", tint="yellow", fsz=6.2)
    B.stage(ax, 0.897, yl, 0.155, 0.27, "Adam\nupdate", tint="gray", fsz=6.2)
    B.arrow(ax, (0.196, yl), (0.276, yl), lw=1.4)
    B.arrow(ax, (0.528, yl), (0.594, yl), lw=1.4)
    B.arrow(ax, (0.762, yl), (0.816, yl), lw=1.4)
    B.stage(ax, 0.775, 0.84, 0.30, 0.175, r"sample impairments $\xi$",
            sub=r"$\bar{n}\sim\log\mathcal{U}$,  $\rho_d$,  $\sigma_{cal}$,  $b$",
            tint="red", fsz=5.6)
    B.arrow(ax, (0.625, 0.84), (0.402, 0.84), color=C["red"], lw=1.2,
            style="-", ms=1, shrinkA=0, shrinkB=0)
    B.arrow(ax, (0.402, 0.84), (0.402, yl + 0.142), color=C["red"], lw=1.2, ms=8)
    gy = 0.245
    B.arrow(ax, (0.678, yl - 0.142), (0.678, gy), color=C["orange"],
            lw=1.6, style="-", ms=1)
    B.arrow(ax, (0.678, gy), (0.107, gy), color=C["orange"], lw=1.6,
            style="-", ms=1)
    B.arrow(ax, (0.107, gy), (0.107, yl - 0.146), color=C["orange"], lw=1.6,
            ms=9)
    ax.text(0.39, 0.155, r"pathwise gradient "
            r"$\nabla_\theta\, \mathbb{E}_\xi[\mathcal{L}(g_\xi(\theta))]$"
            r" through the chain", ha="center", fontsize=5.1,
            color=C["orange"], fontweight="bold")
    uy = 0.06
    B.arrow(ax, (0.897, yl - 0.142), (0.897, uy), color=C["ink2"], lw=0.9,
            style="-", ms=1)
    B.arrow(ax, (0.897, uy), (0.055, uy), color=C["ink2"], lw=0.9,
            style="-", ms=1)
    B.arrow(ax, (0.055, uy), (0.055, yl - 0.148), color=C["ink2"], lw=0.9, ms=7)
    ax.text(0.868, 0.115, r"$\theta \leftarrow \theta - \eta_{lr}\hat{g}$",
            ha="right", fontsize=5.4, color=C["ink2"])


# ------------------------------------------------------------- panel (d) ---
def panel_controller(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    B.panel_title(ax, "closed-loop photon-budget operation")
    yt, yb = 0.67, 0.245
    B.stage(ax, 0.152, yt, 0.245, 0.29, "measure @ low\nbudget " + r"$\bar{n}_{lo}$",
            sub=r"photons $\propto \bar{n}$", tint="aqua", fsz=6.0)
    B.stage(ax, 0.483, yt, 0.25, 0.29, "confidence margin",
            sub=r"$m = p_{(1)}-p_{(2)}$", tint="yellow", fsz=6.2)
    B.stage(ax, 0.862, yt, 0.19, 0.29, "commit",
            sub=r"$m \geq \tau$", tint="green", fsz=6.4)
    B.stage(ax, 0.483, yb, 0.30, 0.27, r"re-expose @ $\bar{n}_{hi}$",
            sub="pool photon counts (ML)", tint="red", fsz=6.0)
    B.arrow(ax, (0.278, yt), (0.355, yt), lw=1.4)
    B.arrow(ax, (0.611, yt), (0.764, yt), color=EDGE["green"], lw=1.4)
    B.arrow(ax, (0.483, yt - 0.152), (0.483, yb + 0.142), color=C["red"], lw=1.4)
    B.chip(ax, 0.585, 0.455, r"$m < \tau$", tint="red", fsz=5.8)
    B.arrow(ax, (0.636, yb), (0.862, yb), color=EDGE["green"], lw=1.4,
            style="-", ms=1)
    B.arrow(ax, (0.862, yb), (0.862, yt - 0.152), color=EDGE["green"], lw=1.4,
            ms=9)
    B.chip(ax, 0.152, 0.35, r"$\bar{\Phi} = \Phi_{lo} + p_{esc}\, \Phi_{hi}$",
           tint="gray", fsz=6.1, ec=C["ink2"])
    ax.text(0.152, 0.16, "photon budget prices the\nserver laser (link budget)\n"
            "and integration time", ha="center", fontsize=5.2, color=C["ink2"],
            style="italic")


def fig_architecture():
    B.AUTOFIT.clear()
    fig = plt.figure(figsize=(fs.DOUBLE_COL, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.06, 1.0], hspace=0.09,
                          wspace=0.045, left=0.010, right=0.990, top=0.988,
                          bottom=0.010)
    panels = [panel_system, panel_chain, panel_training, panel_controller]
    for i, (fn, lab) in enumerate(zip(panels, "abcd")):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.add_patch(FancyBboxPatch((0.002, 0.002), 0.996, 0.996,
                     boxstyle="round,pad=0.002,rounding_size=0.02",
                     fc="white", ec="#e2e1dd", lw=0.8, zorder=0))
        fn(ax)
        ax.text(0.022, 0.955, f"({lab})", fontsize=8.5, fontweight="bold",
                va="center", color=C["ink"])
    B.autofit(fig)
    fig.savefig(os.path.join(FIG, "fig_architecture_q.pdf"))
    plt.close(fig)


# ---------------------------------------------------------- abstract fig ---
def headline_numbers(r):
    from physicsq import model_budget
    dims = [(784, 300), (300, 100), (100, 10)]
    macs = model_budget(1, dims)["macs"]

    def nbar_at(curve, target):
        ks = sorted(curve.keys(), key=float)
        x = np.array([float(k) for k in ks])
        y = np.array([curve[k]["mean"] for k in ks])
        lx = np.log(x)
        for i in range(1, len(x)):
            if y[i] >= target and y[i - 1] < target:
                t = (target - y[i - 1]) / (y[i] - y[i - 1])
                return float(np.exp(lx[i - 1] + t * (lx[i] - lx[i - 1])))
        return float(x[-1])

    target = r["mnist"]["digital"]["digital_acc"] - 0.02
    nb_dig = nbar_at(r["mnist"]["digital"]["acc_vs_nbar"], target)
    nb_pit = nbar_at(r["mnist"]["pitlq"]["acc_vs_nbar"], target)
    best = None
    for cfg in r["mnist"]["closed_loop_pitlq"].values():
        for p in cfg["points"]:
            if p["acc_mean"] >= target and (best is None or
                                            p["photons"] < best):
                best = p["photons"]
    return {"Phi_dig": nb_dig * macs / 1e6, "Phi_pit": nb_pit * macs / 1e6,
            "Phi_cl": best / 1e6, "x_total": nb_dig * macs / best}


def fig_abstract():
    B.AUTOFIT.clear()
    with open(RES) as f:
        r = json.load(f)
    hl = headline_numbers(r)
    fig = plt.figure(figsize=(5.6, 2.3))
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.035, 0.945, "photonic edge inference at the quantum limit:\n"
            "trained through, and operated on, its own shot noise",
            fontsize=7.2, fontweight="bold", color=C["ink"], va="top")
    yl = 0.44
    B.stage(ax, 0.14, yl, 0.225, 0.31, "train through\nthe shot noise",
            tint="violet", fsz=6.3)
    ax.text(0.018, yl - 0.27, "Poisson (SQL), dark counts, trim error,\n"
            r"quantization sampled in training",
            ha="left", fontsize=5.2, color=C["ink2"])
    laser(ax, 0.375, 0.70, s=0.05)
    beam(ax, (0.405, 0.665), (0.405, yl + 0.075), lw=1.4)
    modulator(ax, 0.405, yl, w=0.07, h=0.09)
    B.arrow(ax, (0.258, yl), (0.352, yl), lw=1.2)
    B.arrow(ax, (0.462, yl), (0.548, yl), lw=1.2)
    ax.text(0.405, yl - 0.185, "in-physics MVM\n(photonic, photon-starved)",
            ha="center", fontsize=5.2, color=C["ink2"])
    B.stage(ax, 0.652, yl, 0.17, 0.30, "confidence\ngate", tint="yellow", fsz=6.2)
    B.stage(ax, 0.885, yl, 0.165, 0.30, "commit", tint="green", fsz=6.3)
    B.arrow(ax, (0.742, yl), (0.798, yl), color=EDGE["green"], lw=1.2)
    B.arrow(ax, (0.652, yl - 0.158), (0.652, 0.115), color=C["red"], lw=1.1, ms=7)
    ax.text(0.685, 0.115, "re-expose only\nuncertain inputs", fontsize=5.2,
            color=C["red"], ha="left", va="center")
    ax2 = fig.add_axes([0.775, 0.20, 0.205, 0.57])
    vals = [hl["Phi_dig"], hl["Phi_pit"], hl["Phi_cl"]]
    names = ["static\nconv.", "static\nPILOT-Q", "closed\nloop"]
    cols = [C["muted"], C["blue"], "#128a60"]
    ax2.bar(range(3), vals, 0.62, color=cols, zorder=3)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(names, fontsize=5.4)
    ax2.set_ylabel("photon budget /\ninference (Mphotons)", fontsize=5.7)
    ax2.set_title("@ iso-accuracy", fontsize=5.7, color=C["ink2"], pad=3)
    for i, v in enumerate(vals):
        ax2.text(i, v + vals[0] * 0.035, f"{v:.2f}", ha="center", fontsize=6.0,
                 color=C["ink"], fontweight="bold")
    ax2.text(2.0, vals[0] * 0.90, f"{hl['x_total']:.1f}\u00d7\nfewer",
             ha="center", va="bottom", fontsize=6.2, color=C["red"],
             fontweight="bold")
    ax2.set_ylim(0, vals[0] * 1.30)
    ax2.tick_params(length=2, labelsize=5.6)
    ax2.grid(True, axis="y", alpha=0.5)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    B.autofit(fig)
    fig.savefig(os.path.join(FIG, "fig_abstract_q.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    print("fig_architecture_q done")
    if os.path.exists(RES):
        try:
            fig_abstract()
            print("fig_abstract_q done")
        except Exception as e:
            print("abstract deferred:", e)
