"""Generate supplementary tables (supp_tables.tex) for PILOT-Q."""
import json
import os
from physicsq import model_budget, H_NU_1550

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "results.json")
OUT = os.path.join(HERE, "..", "latex", "supp_tables.tex")

DS_TITLE = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "fsdd": "FSDD"}
MODE_TITLE = {"digital": "conventional", "pitlq": "PILOT-Q",
              "pitlq_full": "PILOT-Q-full", "pitlq_fixed1": "PILOT-Q (fixed 1 ph)"}
DIMS = {"mnist": [(784, 300), (300, 100), (100, 10)],
        "fashion": [(784, 300), (300, 100), (100, 10)],
        "fsdd": [(4000, 300), (300, 100), (100, 10)]}


def pm(d, dec=1):
    return f"{100*d['mean']:.{dec}f} $\\pm$ {100*d['std']:.{dec}f}"


def main():
    with open(RES) as f:
        r = json.load(f)
    L = []
    A = L.append

    # ---- S-I: twin validation ----
    A("\\begin{table}[h]\n\\caption{\\label{tab:s_val}Twin validation: "
      "Monte-Carlo normalized inner-product RMSE versus the analytic "
      "standard-quantum-limit prediction (400 trials at $N{=}4096$; 60 at "
      "$N{=}32{,}768$).}\n\\begin{ruledtabular}\n\\begin{tabular}{ccccc}\n"
      "$\\bar{n}$ (ph/MAC) & MC, $N{=}4096$ & SQL, $N{=}4096$ & "
      "MC, $N{=}32{,}768$ & SQL, $N{=}32{,}768$ \\\\\n\\hline")
    v1, v2 = r["ip_validation"], r["ip_validation_n32768"]
    for k in sorted(v1, key=float):
        A(f"{k} & {v1[k]['rmse']:.4f} & {v1[k]['sql']:.4f} & "
          f"{v2[k]['rmse']:.4f} & {v2[k]['sql']:.4f} \\\\")
    A("\\end{tabular}\n\\end{ruledtabular}\n\\end{table}\n")

    # ---- S-II: budget accounting ----
    A("\\begin{table}[h]\n\\caption{\\label{tab:s_budget}Photon-budget "
      "accounting per inference [Sec.~III C of the main text]: received "
      "photons and optical energy, server laser energy through a 30\\,dB "
      "link, and the photon-independent client electronics, for the two "
      "model geometries.}\n\\begin{ruledtabular}\n"
      "\\begin{tabular}{ccccccc}\n"
      " & \\multicolumn{3}{c}{$784$-input model} & "
      "\\multicolumn{3}{c}{$4000$-input model} \\\\\n"
      "$\\bar{n}$ & $\\Phi$ (M) & laser (pJ) & elec.\\ (nJ) & "
      "$\\Phi$ (M) & laser (pJ) & elec.\\ (nJ) \\\\\n\\hline")
    for nb in [0.25, 0.5, 1, 2, 4, 8]:
        b1 = model_budget(nb, DIMS["mnist"])
        b2 = model_budget(nb, DIMS["fsdd"])
        A(f"{nb:g} & {b1['photons']/1e6:.3f} & {b1['E_laser']*1e12:.1f} & "
          f"{b1['E_elec']*1e9:.2f} & {b2['photons']/1e6:.3f} & "
          f"{b2['E_laser']*1e12:.1f} & {b2['E_elec']*1e9:.2f} \\\\")
    A("\\end{tabular}\n\\end{ruledtabular}\n\\end{table}\n")

    # ---- S-III..V: accuracy sweeps ----
    for ds in ["mnist", "fashion", "fsdd"]:
        modes = [m for m in ["digital", "pitlq", "pitlq_full", "pitlq_fixed1"]
                 if m in r[ds] and "acc_vs_nbar" in r[ds][m]]
        A(f"\\begin{{table}}[h]\n\\caption{{\\label{{tab:s_{ds}}}"
          f"{DS_TITLE[ds]}: accuracy (\\%, mean $\\pm$ s.d.\\ over five "
          "noise seeds) versus photon budget on the shot-limited chain and "
          "under compound impairments ($\\rho_d{=}20\\%$, "
          "$\\sigma_{cal}{=}15\\%$, 4-bit ADC).}\n"
          "\\begin{ruledtabular}\n\\begin{tabular}{c" +
          "c" * (2 * len(modes)) + "}")
        A(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{MODE_TITLE[m]}}}"
                             for m in modes) + " \\\\")
        A("$\\bar{n}$ (ph/MAC)" + " & shot & impaired" * len(modes) +
          " \\\\\n\\hline")
        for k in sorted(r[ds]["digital"]["acc_vs_nbar"], key=float):
            row = [f"{float(k):g}"]
            for m in modes:
                row.append(pm(r[ds][m]["acc_vs_nbar"][k]))
                row.append(pm(r[ds][m]["acc_vs_nbar_impaired"][k])
                           if "acc_vs_nbar_impaired" in r[ds][m] else "---")
            A(" & ".join(row) + " \\\\")
        A("ideal & " + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{100*r[ds][m]['digital_acc']:.2f}}}"
            for m in modes) + " \\\\")
        A("\\end{tabular}\n\\end{ruledtabular}\n\\end{table}\n")

    # ---- S-VI: impairment sweeps (one table per impairment) ----
    imp_specs = [
        ("dark", "s_imp_dark", "dark-count fraction $\\rho_d$",
         "0 & 0.1 & 0.2 & 0.3 & 0.5 & 1.0", float),
        ("cal", "s_imp_cal", "trim error $\\sigma_{cal}$",
         "0 & 0.05 & 0.10 & 0.15 & 0.20 & 0.30", float),
        ("adc", "s_imp_adc", "ADC resolution (bits)",
         "2 & 3 & 4 & 5 & 6 & 8", int),
    ]
    for key, lab, title, cols, srt in imp_specs:
        A(f"\\begin{{table}}[h]\n\\caption{{\\label{{tab:{lab}}}"
          f"Impairment sweep at $\\bar{{n}}{{=}}2$ photons/MAC: accuracy "
          f"(\\%) versus {title}.}}\n"
          "\\begin{ruledtabular}\n\\begin{tabular}{llcccccc}")
        A(f"Dataset & Training & {cols} \\\\\n\\hline")
        for ds in ["mnist", "fashion", "fsdd"]:
            for m in ["digital", "pitlq", "pitlq_full"]:
                d = r[ds][m]["impair_2ph"][key]
                row = [DS_TITLE[ds] if m == "digital" else "", MODE_TITLE[m]]
                row += [f"{100*d[k]['mean']:.1f}" for k in sorted(d, key=srt)]
                A(" & ".join(row) + " \\\\")
        A("\\end{tabular}\n\\end{ruledtabular}\n\\end{table}\n")

    # ---- quantitative state-of-the-art comparison ----
    A(r"""\begin{table}[h]
\caption{\label{tab:s_sota}Quantitative figure-of-merit comparison for
photon-starved and in-physics inference. Bracketed numbers refer to the
main-text bibliography.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllll}
\hline\hline
Scheme & Architecture & Task & Acc.\ (\%) & Photons/MAC (or energy) & Key figure of merit \\
\hline
Digital GPU (H100) [4] & digital electronics & MNIST & 98.1 & 70\,fJ/MAC & baseline \\
Netcast [1] (2022) & delocalized fiber, balanced det. & MNIST & 98.8 & $<$1 (40\,aJ optical) & 86\,km deployed link \\
Wang \emph{et al.}\ [2] (2022) & free-space, noise-aware training & MNIST & 99 / 90 & 3.2 / 0.64 & sub-photon multiplication \\
Ma \emph{et al.}\ [3] (2025) & SPD, stochastic binary act. & MNIST & 98 & 0.038 & SNR $\approx 1$ operation \\
WISE [4] (2026) & wireless RF mixer & MNIST & 95.7 & 6.0\,fJ/MAC & disaggregated RF MVMs \\
\hline
Conventional practice (twin) & delocalized twin (this work) & MNIST & \targetMnist & \nbarTgtDigMnist & digitally trained, static budget \\
\textbf{PILOT-Q closed loop} & delocalized twin (this work) & MNIST & \clAccMnist & \nbarEffClMnist{} & \xPhClMnist$\times$ fewer photons @ iso-acc. \\
\textbf{PILOT-Q closed loop} & & Fashion-MNIST & \clAccFashion & \nbarEffClFashion{} & \xPhClFashion$\times$ fewer photons @ iso-acc. \\
\textbf{PILOT-Q closed loop} & & FSDD & \clAccFsdd & \nbarEffClFsdd{} & \xPhClFsdd$\times$ fewer photons @ iso-acc. \\
\hline\hline
\end{tabular}}
\end{table}
""")

    # ---- closed-loop Pareto per dataset ----
    for ds in ["mnist", "fashion", "fsdd"]:
        A(f"\\begin{{table}}[h]\n\\caption{{\\label{{tab:s_cl_{ds}}}"
          f"Photon-budget Pareto points of the closed-loop controller on "
          f"{DS_TITLE[ds]} (PILOT-Q model), thinned to $\\geq 0.25$ "
          "percentage-point accuracy increments (complete sweeps in the "
          "released record). $\\Phi$: photons per inference.}\n"
          "\\begin{ruledtabular}\n\\begin{tabular}{cccccc}\n"
          "$\\bar{n}_{lo}$ & $\\bar{n}_{hi}$ & $\\tau$ & acc.\\ (\\%) & "
          "esc.\\ (\\%) & $\\Phi$ (Mphotons) \\\\\n\\hline")
        pts = []
        for cfg in r[ds]["closed_loop_pitlq"].values():
            for p in cfg["points"]:
                pts.append((p["photons"], p["acc_mean"], cfg["nb_lo"],
                            cfg["nb_hi"], p["tau"], p["esc_frac"]))
        pts.sort()
        best, kept = -1, -1
        for (ph, acc, lo, hi, tau, esc) in pts:
            if acc > best:
                best = acc
                if acc >= kept + 0.0025:
                    kept = acc
                    A(f"{lo:g} & {hi:g} & {tau:g} & {100*acc:.2f} & "
                      f"{100*esc:.1f} & {ph/1e6:.3f} \\\\")
        A("\\end{tabular}\n\\end{ruledtabular}\n\\end{table}\n")

    # ---- oracle bounds ----
    A("\\begin{table}[h]\n\\caption{\\label{tab:s_oracle}Oracle escalation "
      "bounds (PILOT-Q models): accuracy and photon budget when exactly the "
      "inputs mispredicted at $\\bar{n}_{lo}$ are re-exposed at "
      "$\\bar{n}_{hi}$ --- the upper envelope of any per-input policy.}\n"
      "\\begin{ruledtabular}\n\\begin{tabular}{lccccc}\n"
      "Dataset & $\\bar{n}_{lo}$ & $\\bar{n}_{hi}$ & acc.\\ (\\%) & "
      "esc.\\ (\\%) & $\\Phi$ (Mphotons) \\\\\n\\hline")
    for ds in ["mnist", "fashion", "fsdd"]:
        if "oracle" not in r[ds]:
            continue
        first = True
        for key in sorted(r[ds]["oracle"], key=lambda s: float(s.split("_")[0])):
            o = r[ds]["oracle"][key]
            A(f"{DS_TITLE[ds] if first else ''} & {o['nb_lo']:g} & "
              f"{o['nb_hi']:g} & {100*o['acc']:.2f} & "
              f"{100*o['esc_frac']:.1f} & {o['photons']/1e6:.3f} \\\\")
            first = False
        if ds != "fsdd":
            A("\\hline")
    A("\\end{tabular}\n\\end{ruledtabular}\n\\end{table}\n")

    # ---- hyperparameters ----
    A("""\\begin{table}[h]
\\caption{\\label{tab:s_hyper}Training protocols and hyperparameters. All
protocols share: Adam, learning rate $10^{-3}$ decayed $\\times 0.3$ at
70\\% of training, batch 256, training seed 42, evaluation seeds
\\{101, 202, 303, 404, 505\\}.}
\\begin{ruledtabular}
\\begin{tabular}{lcccc}
 & conventional & PILOT-Q & PILOT-Q-full & PILOT-Q (fixed) \\\\
\\hline
forward physics & clean & twin & twin & twin \\\\
$\\bar{n}$ (ph/MAC)\\footnote{$\\log\\mathcal{U}[0.5,8]$ on
Fashion-MNIST, whose harder decision boundary tolerates less training
noise (main-text Sec.~IV\\,B).} & --- & $\\log\\mathcal{U}[0.25,8]$ & $\\log\\mathcal{U}[0.25,8]$ & $1$ \\\\
dark fraction $\\rho_d$ & --- & $0$ & $\\mathcal{U}[0,0.20]$ & $0$ \\\\
trim error $\\sigma_{cal}$ & --- & $0$ & $\\mathcal{U}[0,0.15]$ & $0$ \\\\
ADC bits $b$ & --- & $\\infty$ & $\\{4,6,8,\\infty\\}$ & $\\infty$ \\\\
clean warmup (epochs) & --- & 15\\% & 15\\% & 15\\% \\\\
clean batch anchor & --- & 35\\% & 35\\% & 35\\% \\\\
epochs (image / audio) & 25 / 100 & 35 / 140 & 35 / 140 & 35 / --- \\\\
\\end{tabular}
\\end{ruledtabular}
\\end{table}
""")
    with open(OUT, "w") as f:
        f.write("% Auto-generated from results/results.json -- do not edit.\n")
        f.write("\n".join(L))
    print("wrote supp_tables.tex")


if __name__ == "__main__":
    main()
