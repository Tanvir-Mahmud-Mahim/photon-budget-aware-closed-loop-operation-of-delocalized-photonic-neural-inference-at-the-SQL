"""PILOT-Q results figures, generated from results/results.json."""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # shared figure toolchain ships in this repo
import figstyle as fs
from figstyle import C
from physicsq import model_budget, H_NU_1550, E_ADC, E_DIG

fs.use_style()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results", "results.json")

MODE_STYLE = {
    "digital":    dict(color=C["muted"], marker="o", label="conventional (digital) training"),
    "pitlq":      dict(color=C["blue"],  marker="s", label="PILOT-Q (shot-noise-aware)"),
    "pitlq_full": dict(color=C["aqua"],  marker="^", label="PILOT-Q-full (hardware-aware)"),
}
DS_TITLE = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "fsdd": "FSDD (spoken digits)"}
DIMS = {"mnist": [(784, 300), (300, 100), (100, 10)],
        "fashion": [(784, 300), (300, 100), (100, 10)],
        "fsdd": [(4000, 300), (300, 100), (100, 10)]}


def plabel(ax, s, x=0.03, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="top", color=C["ink"],
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))


def curve(d):
    ks = sorted(d.keys(), key=float)
    x = np.array([float(k) for k in ks])
    m = np.array([d[k]["mean"] for k in ks]) * 100
    s = np.array([d[k]["std"] for k in ks]) * 100
    return x, m, s


def load():
    with open(RES) as f:
        return json.load(f)


# ------------------------------------------------- validation + budget -----
def fig_validation(r):
    fig, axes = plt.subplots(2, 1, figsize=(fs.SINGLE_COL, 4.6))
    ax = axes[0]
    for key, col, mk, lab in [("ip_validation", C["blue"], "o", "twin, $N{=}4096$"),
                              ("ip_validation_n32768", C["aqua"], "s",
                               "twin, $N{=}32{,}768$")]:
        v = r[key]
        nb = np.array(sorted(float(k) for k in v))
        kk = lambda s: str(int(s)) if s == int(s) else str(s)
        mc = np.array([v[kk(s)]["rmse"] for s in nb])
        sql = np.array([v[kk(s)]["sql"] for s in nb])
        ax.loglog(nb, mc, mk, color=col, ms=4.5, mfc="white", mew=1.2,
                  label=lab)
        if key == "ip_validation":
            ax.loglog(nb, sql, "-", color=C["ink2"], lw=1.0,
                      label="analytic SQL ($\\propto 1/\\sqrt{\\bar{n}}$)")
    ax.set_xlabel(r"photon budget $\bar{n}$ (photons/MAC)")
    ax.set_ylabel("normalized IP RMSE")
    ax.legend(loc="lower left", handlelength=1.5, fontsize=6.0)
    plabel(ax, "(a)", x=0.90)
    # budget accounting
    ax = axes[1]
    nb_f = np.logspace(-2, 6, 200)
    b = model_budget(1.0, DIMS["mnist"])
    E_opt = nb_f * b["macs"] * H_NU_1550
    E_laser = E_opt * 1e3
    ax.loglog(nb_f, E_opt * 1e9, color="#b57b00", lw=1.3,
              label=r"received optical, $\Phi h\nu$")
    ax.loglog(nb_f, E_laser * 1e9, color=C["red"], lw=1.3,
              label="server laser (30 dB link)")
    ax.axhline(b["E_elec"] * 1e9, color=C["violet"], lw=1.2, ls=(0, (4, 2)),
               label="client electronics (ADC+dig.)")
    cross = b["E_elec"] / (b["macs"] * H_NU_1550 * 1e3)
    ax.axvline(cross, color=C["ink2"], lw=0.7, ls=(0, (3, 2)))
    ax.text(cross * 1.7, 6e-5, f"laser $>$ electronics\nabove "
            f"$\\bar{{n}} \\approx {cross:.0f}$", fontsize=5.6,
            color=C["ink2"], ha="left")
    ax.axvspan(0.25, 8, color=C["blue"], alpha=0.06)
    ax.text(1.4, 2e-4, "operating\nrange", fontsize=5.6, color=C["blue"],
            ha="center")
    ax.set_xlabel(r"photon budget $\bar{n}$ (photons/MAC)")
    ax.set_ylabel("energy / inference (nJ)")
    ax.set_xlim(1e-2, 1e6)
    ax.legend(loc="upper left", handlelength=1.5, fontsize=6.0)
    plabel(ax, "(b)", x=0.90)
    for a in axes:
        a.grid(True, which="major", axis="both", alpha=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_validation_q.pdf"))
    plt.close(fig)


# ------------------------------------------------- accuracy envelopes ------
def fig_accuracy(r):
    fig, axes = plt.subplots(2, 3, figsize=(fs.DOUBLE_COL, 3.6), sharex=True)
    for j, ds in enumerate(["mnist", "fashion", "fsdd"]):
        for row, key in enumerate(["acc_vs_nbar", "acc_vs_nbar_impaired"]):
            ax = axes[row, j]
            for mode in ["digital", "pitlq", "pitlq_full"]:
                if key not in r[ds][mode]:
                    continue
                x, m, s = curve(r[ds][mode][key])
                st = MODE_STYLE[mode]
                ax.semilogx(x, m, color=st["color"], marker=st["marker"],
                            ms=3, lw=1.3, label=st["label"])
                ax.fill_between(x, m - s, m + s, color=st["color"],
                                alpha=0.18, lw=0)
            ax.axhline(r[ds]["digital"]["digital_acc"] * 100, color=C["ink2"],
                       lw=0.8, ls=":", zorder=1)
            if row == 0:
                ax.set_title(DS_TITLE[ds], fontsize=8)
            if j == 0:
                ax.set_ylabel("accuracy (%)\n" +
                              ("shot noise only" if row == 0 else
                               r"+ dark 20% + trim 15% + 4-bit"), fontsize=7)
            if row == 1:
                ax.set_xlabel(r"photon budget $\bar{n}$ (photons/MAC)")
            ax.set_ylim(8, 102)
            ax.grid(True, axis="y", alpha=0.6)
    axes[0, 0].legend(loc="lower right", fontsize=5.8, handlelength=1.5)
    for ax, lab in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
        plabel(ax, lab, x=0.03, y=0.96)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_accuracy_q.pdf"))
    plt.close(fig)


# ------------------------------------------------- impairments -------------
def fig_robustness(r):
    fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE_COL, 1.95))
    panels = [("dark", r"dark-count fraction $\rho_d$", 100),
              ("cal", r"trim error $\sigma_{cal}$ (%)", 100),
              ("adc", "ADC resolution (bits)", 1)]
    for k, (key, xlabel, mul) in enumerate(panels):
        ax = axes[k]
        for mode in ["digital", "pitlq", "pitlq_full"]:
            d = r["mnist"][mode]["impair_2ph"][key]
            if key == "adc":
                x = sorted(int(q) for q in d)
                m = [d[str(q)]["mean"] * 100 for q in x]
            else:
                x = np.array(sorted(float(q) for q in d)) * mul
                m = [d[q]["mean"] * 100 for q in sorted(d, key=float)]
            st = MODE_STYLE[mode]
            ax.plot(x, m, color=st["color"], marker=st["marker"], ms=3,
                    lw=1.3, label=st["label"])
        ax.set_xlabel(xlabel)
        if k == 0:
            ax.set_ylabel("accuracy (%)")
            ax.legend(fontsize=5.2, loc="lower left", handlelength=1.3)
    ax = axes[3]
    for mode, col, mk, lab in [
            ("pitlq", C["blue"], "s", r"PILOT-Q, $\bar{n}\sim\log\mathcal{U}[0.25,8]$"),
            ("pitlq_fixed1", C["yellow"], "v", r"PILOT-Q, fixed $\bar{n}=1$"),
            ("digital", C["muted"], "o", "conventional")]:
        if mode not in r["mnist"]:
            continue
        x, m, s = curve(r["mnist"][mode]["acc_vs_nbar"])
        ax.semilogx(x, m, color=col, marker=mk, ms=3, lw=1.3, label=lab)
    ax.set_xlabel(r"$\bar{n}$ (photons/MAC)")
    ax.legend(fontsize=5.2, loc="lower right", handlelength=1.3)
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        plabel(ax, lab, x=0.04, y=0.96)
        ax.grid(True, axis="y", alpha=0.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_robustness_q.pdf"))
    plt.close(fig)


# ------------------------------------------------- closed loop -------------
def static_points(r, ds, mode, key="acc_vs_nbar"):
    x, m, _ = curve(r[ds][mode][key])
    macs = model_budget(1, DIMS[ds])["macs"]
    return x * macs, m


def cl_points(r, ds, which="closed_loop_pitlq"):
    pts = []
    for cfg in r[ds][which].values():
        for p in cfg["points"]:
            pts.append((p["photons"], p["acc_mean"] * 100))
    return pts


def pareto(pts):
    pts = sorted(pts)
    out, best = [], -1
    for e, a in pts:
        if a > best:
            out.append((e, a)); best = a
    return np.array(out)


def fig_budget(r):
    fig = plt.figure(figsize=(fs.DOUBLE_COL, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.42,
                          left=0.075, right=0.992, top=0.92, bottom=0.19)
    ax = fig.add_subplot(gs[0])
    for mode in ["digital", "pitlq"]:
        E, m = static_points(r, "mnist", mode)
        st = MODE_STYLE[mode]
        ax.semilogx(E / 1e6, m, color=st["color"], marker=st["marker"], ms=3,
                    lw=1.3, label=f"static, {st['label'].split(' (')[0]}")
    pp = pareto(cl_points(r, "mnist", "closed_loop_pitlq"))
    ax.semilogx(pp[:, 0] / 1e6, pp[:, 1], color=C["red"], lw=1.6, marker="*",
                ms=5, label="closed loop + PILOT-Q")
    orc = [(o["photons"], o["acc"] * 100) for o in r["mnist"]["oracle"].values()]
    oo = pareto(orc)
    ax.semilogx(oo[:, 0] / 1e6, oo[:, 1], color=C["ink"], lw=1.0,
                ls=(0, (3, 2)), marker="D", ms=3, mfc="white",
                label="oracle escalation (bound)")
    ax.set_xlabel("photon budget per inference (Mphotons)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(55, 100)
    ax.set_xlim(3e-2, 3)
    ax.legend(fontsize=5.6, loc="lower right", handlelength=1.5)
    ax.grid(True, axis="y", alpha=0.6)
    plabel(ax, "(a)")
    ax2 = fig.add_subplot(gs[1])
    cl = r["mnist"]["closed_loop_pitlq"]
    for key, col in zip(["0.25_1", "0.5_2", "1_4"],
                        [C["red"], C["blue"], C["aqua"]]):
        p = cl[key]["points"]
        ax2.plot([q["tau"] for q in p], [q["esc_frac"] * 100 for q in p],
                 color=col, marker="o", ms=2.6, lw=1.2,
                 label=f"({key.split('_')[0]}, {key.split('_')[1]}) ph")
    ax2.set_xlabel(r"confidence threshold $\tau$")
    ax2.set_ylabel("re-exposed inputs (%)")
    ax2.legend(fontsize=5.6, handlelength=1.4, title=r"($\bar{n}_{lo}, \bar{n}_{hi}$)",
               title_fontsize=5.6, loc="upper left")
    ax2.grid(True, axis="y", alpha=0.6)
    plabel(ax2, "(b)", x=0.84)
    # (c) impaired-chain closed loop
    ax3 = fig.add_subplot(gs[2])
    x, m, s = curve(r["mnist"]["pitlq"]["acc_vs_nbar_impaired"])
    macs = model_budget(1, DIMS["mnist"])["macs"]
    ax3.semilogx(x * macs / 1e6, m, "-o", color=C["muted"], ms=3, lw=1.3,
                 label="static, PILOT-Q")
    cfg = r["mnist"]["closed_loop_impaired"]
    pts = [(p["photons"], p["acc_mean"] * 100) for p in cfg["points"]]
    pp = pareto(pts)
    ax3.semilogx(pp[:, 0] / 1e6, pp[:, 1], "-*", color=C["red"], ms=5, lw=1.5,
                 label="closed loop")
    ax3.set_xlabel("photon budget (Mphotons)")
    ax3.set_ylabel("accuracy (%)")
    ax3.set_title("impaired chain", fontsize=6.4)
    ax3.legend(fontsize=5.6, loc="lower right", handlelength=1.4)
    ax3.grid(True, axis="y", alpha=0.6)
    plabel(ax3, "(c)")
    fig.savefig(os.path.join(FIG, "fig_budget_q.pdf"))
    plt.close(fig)


# ------------------------------------------------- SOTA positioning --------
def nbar_at(d, target):
    """Photon budget at which the accuracy curve crosses target (log interp)."""
    x, m, _ = curve(d)
    t = target * 100
    for i in range(1, len(x)):
        if m[i - 1] < t <= m[i]:
            f = (t - m[i - 1]) / (m[i] - m[i - 1])
            return float(np.exp(np.log(x[i - 1]) +
                                f * (np.log(x[i]) - np.log(x[i - 1]))))
    return None


def fig_sota(r):
    fig = plt.figure(figsize=(fs.DOUBLE_COL, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.02, 1.18], wspace=0.44,
                          left=0.078, right=0.985, top=0.885, bottom=0.205)
    ax = fig.add_subplot(gs[0])
    ax.axvspan(1e-2, 1.0, color=C["blue"], alpha=0.055, zorder=0)
    ax.text(0.027, 85.5, "sub-photon regime", fontsize=5.2, color=C["ink2"],
            style="italic", ha="center", va="center", rotation=90)
    for mode in ["digital", "pitlq"]:
        x, m, s = curve(r["mnist"][mode]["acc_vs_nbar"])
        st = MODE_STYLE[mode]
        ax.semilogx(x, m, color=st["color"], marker=st["marker"], ms=2.8,
                    lw=1.5, zorder=4,
                    label=("this work, conventional" if mode == "digital"
                           else "this work, PILOT-Q"))
        ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.18, lw=0)
    y0 = curve(r["mnist"]["digital"]["acc_vs_nbar"])[1][2]
    y1 = curve(r["mnist"]["pitlq"]["acc_vs_nbar"])[1][2]
    ax.annotate("", xy=(0.25, y1 - 0.4), xytext=(0.25, y0 + 0.4),
                arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.3))
    ax.text(0.085, 88.5, f"+{y1-y0:.1f} pp\ntraining through\nthe SQL",
            fontsize=5.0, color=C["orange"], va="center", ha="center")
    tgt = r["mnist"]["digital"]["digital_acc"] - 0.02
    macs = model_budget(1, DIMS["mnist"])["macs"]
    best = min((p for cfg in r["mnist"]["closed_loop_pitlq"].values()
                for p in cfg["points"] if p["acc_mean"] >= tgt),
               key=lambda p: p["photons"])
    ax.plot(best["photons"] / macs, best["acc_mean"] * 100, marker="*", ms=10,
            color=C["red"], mec="white", mew=0.5, zorder=6,
            label="this work, closed loop")
    LIT = [
        ("Netcast '22 (86 km fiber)", 0.9, 98.8, C["violet"], "D", False),
        ("Wang '22", 3.2, 99.0, C["aqua"], "o", True),
        ("Wang '22", 0.64, 90.0, C["aqua"], "o", True),
        ("Ma '25 (SPD)", 0.038, 98.0, C["yellow"], "s", True),
    ]
    seen = set()
    for name, nb, acc, col, mk, aware in LIT:
        lab = name if name not in seen else None
        seen.add(name)
        ax.plot(nb, acc, marker=mk, ms=5, color=col, lw=0,
                mfc=col if aware else "white", mew=1.3, mec=col,
                zorder=5, label=lab)
    ax.set_xlim(2e-2, 9)
    ax.set_ylim(76, 101)
    ax.set_xlabel(r"photon budget $\bar{n}$ (photons/MAC)")
    ax.set_ylabel("MNIST accuracy (%)")
    ax.legend(fontsize=4.9, loc="lower right", handlelength=1.2,
              borderpad=0.5, labelspacing=0.35)
    ax.grid(True, axis="y", alpha=0.55)
    plabel(ax, "(a)")
    ax2 = fig.add_subplot(gs[1])
    rows, vals, cols, xlabels = [], [], [], []
    XMIN = 3e-2
    schemes = [("static conv.", C["muted"]), ("static PILOT-Q", C["blue"]),
               ("closed loop", "#128a60")]
    oracle_pts = []
    y = 0
    ylab, ypos = [], []
    for ds in ["mnist", "fashion", "fsdd"]:
        tgt = r[ds]["digital"]["digital_acc"] - 0.02
        macs = model_budget(1, DIMS[ds])["macs"]
        nd = nbar_at(r[ds]["digital"]["acc_vs_nbar"], tgt) * macs / 1e6
        np_ = nbar_at(r[ds]["pitlq"]["acc_vs_nbar"], tgt) * macs / 1e6
        cl = min((p for cfg in r[ds]["closed_loop_pitlq"].values()
                  for p in cfg["points"] if p["acc_mean"] >= tgt),
                 key=lambda p: p["photons"])["photons"] / 1e6
        orc = min((o for o in r[ds]["oracle"].values() if o["acc"] >= tgt),
                  key=lambda o: o["photons"])["photons"] / 1e6
        for v, (nm, col) in zip([nd, np_, cl], schemes):
            rows.append(y); vals.append(v); cols.append(col)
            xlabels.append("" if nm == "static conv." else
                           f"{nd/v:.1f}$\\times$")
            y -= 1
        oracle_pts.append((orc, y + 1))
        ylab.append(DS_TITLE[ds].replace(" (spoken digits)", ""))
        ypos.append(y + 2)
        y -= 0.8
    ax2.barh(rows, vals, height=0.82, color=cols, left=0, zorder=3)
    for yy, v, xl in zip(rows, vals, xlabels):
        if xl:
            ax2.text(v * 1.25, yy, xl, va="center", fontsize=5.2,
                     color=C["ink"], fontweight="bold")
    for orc, yy in oracle_pts:
        ax2.plot(orc, yy, marker="D", ms=4, color=C["ink"], mfc="white",
                 mew=1.1, zorder=5)
    ax2.plot([], [], marker="D", ms=4, color=C["ink"], mfc="white", mew=1.1,
             lw=0, label="oracle bound")
    for nm, col in schemes:
        ax2.plot([], [], marker="s", ms=5, color=col, lw=0, label=nm)
    ax2.set_xscale("log")
    ax2.set_xlim(XMIN, 60)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(ylab, fontsize=5.8)
    ax2.set_xlabel("photon budget per inference\nat dataset target (Mphotons)")
    ax2.legend(fontsize=4.9, loc="upper right", handlelength=1.0,
               borderpad=0.45, labelspacing=0.3)
    ax2.grid(True, axis="x", alpha=0.55)
    ax2.tick_params(axis="y", length=0)
    plabel(ax2, "(b)", x=0.035, y=0.985)
    ax3 = fig.add_subplot(gs[2])
    caps = ["photon-\nstarved", "deloca-\nlized", "physics-\ntrained",
            "adaptive", "regime\nanalysis", "open\nbench."]
    schemes3 = [
        ("Shen '17",      [0, 0, 0, 0, 0, 0]),
        ("HWA training",  [0, 0, 1, 0, 0, 0]),
        ("Netcast '22",   [1, 1, 0, 0, 0, 0]),
        ("Wang '22",      [1, 0, 1, 0, 0, 0]),
        ("PAT '22",       [0, 0, 1, 0, 0, 0]),
        ("Garg '23",      [0, 0, 1, 0.5, 0, 0]),
        ("Ma '25",        [1, 0, 1, 0, 0, 0]),
        ("WISE '26",      [0, 1, 0, 0, 0, 0]),
        ("PILOT-Q (this work)", [1, 1, 1, 1, 1, 1]),
    ]
    ny = len(schemes3)
    ax3.axhspan(-0.5, 0.5, color="#128a60", alpha=0.10, zorder=0)
    for i, (nm, v) in enumerate(schemes3):
        yy = ny - 1 - i
        for j, x in enumerate(v):
            if x == 1:
                ax3.plot(j, yy, "o", ms=5.2,
                         color="#128a60" if i == ny - 1 else C["blue"],
                         zorder=4)
            elif x == 0.5:
                ax3.plot(j, yy, "o", ms=5.2, color=C["yellow"], mfc="white",
                         mew=1.3, zorder=4)
                ax3.plot(j, yy, marker=".", ms=2.6, color=C["yellow"], lw=0,
                         zorder=5)
            else:
                ax3.plot(j, yy, "o", ms=5.2, color="#c9ced6", mfc="white",
                         mew=1.0, zorder=3)
    ax3.set_xticks(range(len(caps)))
    ax3.set_xticklabels(caps, fontsize=4.8, rotation=32, ha="right",
                        rotation_mode="anchor")
    ax3.set_yticks(range(ny))
    ax3.set_yticklabels([s[0] for s in schemes3][::-1], fontsize=5.4)
    ax3.set_xlim(-0.6, len(caps) - 0.4)
    ax3.set_ylim(-0.7, ny - 0.3)
    ax3.tick_params(length=0)
    for s in ["top", "right", "left", "bottom"]:
        ax3.spines[s].set_visible(False)
    ax3.set_title("capability fingerprint", fontsize=6.4, pad=4)
    plabel(ax3, "(c)", x=-0.30, y=1.06)
    fig.savefig(os.path.join(FIG, "fig_sota_q.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    r = load()
    fig_validation(r); print("validation done")
    fig_accuracy(r); print("accuracy done")
    fig_robustness(r); print("robustness done")
    fig_budget(r); print("budget done")
    fig_sota(r); print("sota done")
