"""Generate numbers.tex + tables.tex for PILOT-Q from results.json."""
import json
import math
import os
import numpy as np
from physicsq import model_budget, H_NU_1550

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "results.json")
OUT = os.path.join(HERE, "..", "latex", "numbers.tex")
TAB = os.path.join(HERE, "..", "latex", "tables.tex")

DIMS = {"mnist": [(784, 300), (300, 100), (100, 10)],
        "fashion": [(784, 300), (300, 100), (100, 10)],
        "fsdd": [(4000, 300), (300, 100), (100, 10)]}
DSN = {"mnist": "Mnist", "fashion": "Fashion", "fsdd": "Fsdd"}
DS_TITLE = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "fsdd": "FSDD"}
MODE_TITLE = {"digital": "conventional", "pitlq": "PILOT-Q",
              "pitlq_full": "PILOT-Q-full"}

macros = {}


def M(name, val):
    assert name.isalpha(), name
    macros[name] = val


def fmt_pct(x, d=1):
    return f"{100*x:.{d}f}"


def curve_xy(d):
    ks = sorted(d.keys(), key=float)
    return (np.array([float(k) for k in ks]),
            np.array([d[k]["mean"] for k in ks]))


def nbar_at(d, target):
    """Photon budget where accuracy first crosses target (log-interp)."""
    x, y = curve_xy(d)
    lx = np.log(x)
    for i in range(1, len(x)):
        if y[i] >= target and y[i - 1] < target:
            t = (target - y[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(lx[i - 1] + t * (lx[i] - lx[i - 1])))
    if y[0] >= target:
        return float(x[0])
    return None


def main():
    with open(RES) as f:
        r = json.load(f)

    # ---------------- twin validation ----------------
    v = r["ip_validation"]
    M("valRmseOne", f"{v['1']['rmse']:.3f}")
    M("valSqlOne", f"{v['1']['sql']:.3f}")
    M("valRmseQuarter", f"{v['0.25']['rmse']:.3f}")
    M("valSqlQuarter", f"{v['0.25']['sql']:.3f}")
    devs = [abs(v[k]["rmse"] / v[k]["sql"] - 1) for k in v]
    M("valMaxDev", f"{100*max(devs):.0f}")
    M("valRmseOneBig", f"{r['ip_validation_n32768']['1']['rmse']:.4f}")

    # ---------------- budget accounting reference ----------------
    b1 = model_budget(1.0, DIMS["mnist"])
    M("photPerInf", f"{b1['photons']:.2e}".replace("e+05", r"\times 10^{5}"))
    M("eOptOne", f"{b1['E_opt']*1e15:.1f}")
    M("eLaserOne", f"{b1['E_laser']*1e12:.1f}")
    M("eElecLenet", f"{b1['E_elec']*1e9:.2f}")
    M("macsLenet", f"{b1['macs']:,}".replace(",", "{,}"))
    b4 = model_budget(1.0, DIMS["fsdd"])
    M("macsFsdd", f"{b4['macs']:,}".replace(",", "{,}"))
    M("photonEnergy", f"{H_NU_1550*1e19:.2f}")

    summary = {}
    for ds in ["mnist", "fashion", "fsdd"]:
        n = DSN[ds]
        rd = r[ds]
        ceiling = rd["digital"]["digital_acc"]
        target = ceiling - 0.02
        M("accDig" + n, fmt_pct(ceiling, 2 if ds != "fsdd" else 1))
        M("accPitlq" + n, fmt_pct(rd["pitlq"]["digital_acc"], 2 if ds != "fsdd" else 1))
        M("accPitlqFull" + n, fmt_pct(rd["pitlq_full"]["digital_acc"], 2 if ds != "fsdd" else 1))
        M("target" + n, fmt_pct(target))
        M("nTrain" + n, f"{rd['n_train']:,}".replace(",", "{,}"))
        M("nTest" + n, f"{rd['n_test']:,}".replace(",", "{,}"))

        # accuracy at 1 photon/MAC, clean and impaired
        for mode, mm in [("digital", "Dig"), ("pitlq", "Pitlq"),
                         ("pitlq_full", "PitlqFull")]:
            M(f"acc{mm}OnePh{n}", fmt_pct(rd[mode]["acc_vs_nbar"]["1"]["mean"]))
            M(f"acc{mm}OnePhImp{n}",
              fmt_pct(rd[mode]["acc_vs_nbar_impaired"]["1"]["mean"]))

        # photon budget at target
        nb_dig = nbar_at(rd["digital"]["acc_vs_nbar"], target)
        nb_pit = nbar_at(rd["pitlq"]["acc_vs_nbar"], target)
        M("nbarTgtDig" + n, f"{nb_dig:.2f}")
        M("nbarTgtPitlq" + n, f"{nb_pit:.2f}" if nb_pit else "--")
        if nb_pit:
            M("xPhStatic" + n, f"{nb_dig/nb_pit:.1f}")
            M("xPhStaticDb" + n, f"{10*math.log10(nb_dig/nb_pit):.1f}")
        # impaired condition (PILOT-Q is the featured protocol; see ablation)
        ti = target - 0.02
        ndi = nbar_at(rd["digital"]["acc_vs_nbar_impaired"], ti)
        npi = nbar_at(rd["pitlq"]["acc_vs_nbar_impaired"], ti)
        if npi:
            M("nbarTgtImpPitlq" + n, f"{npi:.2f}")
        if ndi and npi:
            M("xPhImp" + n, f"{ndi/npi:.1f}")
            M("nbarTgtImpDig" + n, f"{ndi:.2f}")
        # digital impaired accuracy at the largest studied budget
        kmax = max(rd["digital"]["acc_vs_nbar_impaired"], key=float)
        M("accDigMaxImp" + n,
          fmt_pct(rd["digital"]["acc_vs_nbar_impaired"][kmax]["mean"]))
        M("nbarMax" + n, kmax)

        macs = model_budget(1, DIMS[ds])["macs"]
        Phi_dig = nb_dig * macs
        Phi_pit = nb_pit * macs if nb_pit else None
        M("phiTgtDig" + n, f"{Phi_dig/1e6:.2f}")
        if Phi_pit:
            M("phiTgtPitlq" + n, f"{Phi_pit/1e6:.2f}")

        # closed loop: photon-cheapest point reaching target (PILOT-Q model)
        best = None
        for cfg in rd["closed_loop_pitlq"].values():
            for p in cfg["points"]:
                if p["acc_mean"] >= target and (best is None or
                                                p["photons"] < best[0]):
                    best = (p["photons"], p, cfg)
        if best:
            Phi_cl, p, cfg = best
            M("phiTgtCl" + n, f"{Phi_cl/1e6:.2f}")
            M("clBase" + n, f"{cfg['nb_lo']:g}")
            M("clHi" + n, f"{cfg['nb_hi']:g}")
            M("clTau" + n, f"{p['tau']:g}")
            M("clEsc" + n, f"{100*p['esc_frac']:.1f}")
            M("clAcc" + n, fmt_pct(p["acc_mean"]))
            M("xPhCl" + n, f"{Phi_dig/Phi_cl:.1f}")
            M("nbarEffCl" + n, f"{Phi_cl/macs:.2f}")
            M("eLaserCl" + n, f"{Phi_cl*H_NU_1550*1e3*1e12:.1f}")  # pJ at 30 dB
            M("eLaserDig" + n, f"{Phi_dig*H_NU_1550*1e9*1e12:.1f}")
        bestd = None
        for cfg in rd["closed_loop_digital"].values():
            for p in cfg["points"]:
                if p["acc_mean"] >= target and (bestd is None or
                                                p["photons"] < bestd[0]):
                    bestd = (p["photons"], p, cfg)
        if bestd:
            M("xPhClDig" + n, f"{Phi_dig/bestd[0]:.1f}")
        summary[ds] = dict(target=target, nb_dig=nb_dig, nb_pit=nb_pit,
                           Phi_dig=Phi_dig, Phi_pit=Phi_pit, cl=best,
                           macs=macs)

    # impairment recovery (MNIST at 2 photons/MAC; PILOT-Q featured)
    d0 = r["mnist"]["digital"]["impair_2ph"]
    dp = r["mnist"]["pitlq"]["impair_2ph"]
    M("accDarkDig", fmt_pct(d0["dark"]["0.5"]["mean"]))
    M("accDarkPitlq", fmt_pct(dp["dark"]["0.5"]["mean"]))
    M("accCalDig", fmt_pct(d0["cal"]["0.3"]["mean"]))
    M("accCalPitlq", fmt_pct(dp["cal"]["0.3"]["mean"]))
    M("accAdcThreeDig", fmt_pct(d0["adc"]["3"]["mean"]))
    M("accAdcThreePitlq", fmt_pct(dp["adc"]["3"]["mean"]))
    # ablation: explicit impairment sampling vs shot-only (impaired chain, 1 ph)
    for ds in ["mnist", "fashion", "fsdd"]:
        n = DSN[ds]
        M("impOnePhPitlq" + n,
          fmt_pct(r[ds]["pitlq"]["acc_vs_nbar_impaired"]["1"]["mean"]))
        M("impOnePhPitlqFull" + n,
          fmt_pct(r[ds]["pitlq_full"]["acc_vs_nbar_impaired"]["1"]["mean"]))

    # oracle escalation bound (per dataset): best photon budget achieving target
    for ds in ["mnist", "fashion", "fsdd"]:
        if "oracle" not in r.get(ds, {}):
            continue
        n = DSN[ds]
        t = summary[ds]["target"]
        besto = None
        for o in r[ds]["oracle"].values():
            if o["acc"] >= t and (besto is None or o["photons"] < besto):
                besto = o["photons"]
        if besto:
            M("xPhOracle" + n, f"{summary[ds]['Phi_dig']/besto:.1f}")
            M("phiOracle" + n, f"{besto/1e6:.2f}")

    if "pitlq_fixed1" in r["mnist"]:
        M("accFixedOneDig", fmt_pct(r["mnist"]["pitlq_fixed1"]["digital_acc"], 2))
        M("accFixedOnePh",
          fmt_pct(r["mnist"]["pitlq_fixed1"]["acc_vs_nbar"]["1"]["mean"]))

    with open(OUT, "w") as f:
        f.write("% Auto-generated from results/results.json -- do not edit.\n")
        for k, v_ in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v_}}}\n")
    print(f"wrote {len(macros)} macros")

    # ------------- master table -------------
    with open(TAB, "w") as f:
        f.write("% Auto-generated tables -- do not edit.\n")
        f.write("\\newcommand{\\tablemaster}{%\n")
        f.write("\\begin{tabular}{llccccccc}\n\\hline\\hline\n")
        f.write("Dataset & Training & Ideal & \\multicolumn{2}{c}{Shot-limited"
                " acc.\\ (\\%)} & \\multicolumn{2}{c}{Impaired acc.\\ (\\%)}"
                " & $\\bar{n}_{tgt}$ & $\\Phi_{tgt}$ \\\\\n")
        f.write(" & & acc.\\ (\\%) & 0.5\\,ph & 2\\,ph & 0.5\\,ph & 2\\,ph &"
                " (ph/MAC) & (Mphotons) \\\\\n\\hline\n")
        prev = None
        for ds in ["mnist", "fashion", "fsdd"]:
            rd = r[ds]
            t = summary[ds]["target"]
            macs = summary[ds]["macs"]
            for mode in ["digital", "pitlq", "pitlq_full"]:
                m = rd[mode]
                nb_t = nbar_at(m["acc_vs_nbar"], t)
                row = [DS_TITLE[ds] if mode == "digital" else "",
                       MODE_TITLE[mode],
                       fmt_pct(m["digital_acc"]),
                       fmt_pct(m["acc_vs_nbar"]["0.5"]["mean"]),
                       fmt_pct(m["acc_vs_nbar"]["2"]["mean"]),
                       fmt_pct(m["acc_vs_nbar_impaired"]["0.5"]["mean"]),
                       fmt_pct(m["acc_vs_nbar_impaired"]["2"]["mean"]),
                       f"{nb_t:.2f}" if nb_t else "--",
                       f"{nb_t*macs/1e6:.2f}" if nb_t else "--"]
                if row[0] and prev is not None:
                    f.write("\\hline\n")
                if row[0]:
                    prev = row[0]
                f.write(" & ".join(row) + " \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}}\n")

        # ------------- closed-loop table -------------
        f.write("\\newcommand{\\tableclosedloop}{%\n")
        f.write("\\begin{tabular}{lccccccc}\n\\hline\\hline\n")
        f.write("Dataset & \\multicolumn{3}{c}{Photon budget at target"
                " (Mphotons/inf.)} & red. & Server laser & Eff.\\ $\\bar{n}$ &"
                " Esc. \\\\\n")
        f.write(" & conv. & PILOT-Q & closed loop & (conv.\\ base) &"
                " (pJ @ 30\\,dB) & (ph/MAC) & (\\%) \\\\\n\\hline\n")
        for ds in ["mnist", "fashion", "fsdd"]:
            s = summary[ds]
            if not s["cl"]:
                continue
            Phi_cl, p, cfg = s["cl"]
            f.write(f"{DS_TITLE[ds]} & {s['Phi_dig']/1e6:.2f} & "
                    f"{s['Phi_pit']/1e6:.2f} & {Phi_cl/1e6:.2f} & "
                    f"{s['Phi_dig']/Phi_cl:.1f}$\\times$ & "
                    f"{Phi_cl*H_NU_1550*1e3*1e12:.1f} & "
                    f"{Phi_cl/s['macs']:.2f} & "
                    f"{100*p['esc_frac']:.1f} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}}\n")
    print("wrote tables.tex")


if __name__ == "__main__":
    main()
