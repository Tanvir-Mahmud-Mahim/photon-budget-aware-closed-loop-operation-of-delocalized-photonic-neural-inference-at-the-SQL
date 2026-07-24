"""Assemble the PILOT-Q Zenodo upload package (data only, no code)."""
import csv
import json
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ZEN = os.path.join(HERE, "..", "zenodo")
PKG = os.path.join(ZEN, "pilotq-benchmark-v1")

README = """# PILOT-Q benchmark: photon-budget-aware training and closed-loop
# operation of delocalized photonic neural inference at the SQL

Complete data underlying every figure and number in the article
"PILOT-Q: physics-in-the-loop training and photon-budget-aware
closed-loop operation of delocalized photonic neural inference at the
standard quantum limit"
(T. M. Mahim, M. N. Islam, M. M. Rahman, and A. S. M. Mohsin).

## Contents
- `results.json` -- all computed results: twin validation against the
  analytic shot-noise (SQL) law (N = 4096 and 32768), and for each dataset
  (MNIST, Fashion-MNIST, FSDD) and protocol (conventional, PILOT-Q,
  PILOT-Q-full, fixed-budget ablation): clean ceilings, accuracy vs photon
  budget on the shot-limited and compound-impaired chains (mean +/- s.d.
  over five seeds), dark/trim/ADC sweeps, full closed-loop controller
  sweeps, and oracle escalation bounds.
- `checkpoints/` -- trained model weights (PyTorch state_dicts).
- `csv/` -- flat exports of the headline curves.

## Provenance
Models are real-valued fully connected networks on three public datasets:
MNIST, Fashion-MNIST, and the Free Spoken Digit Dataset
(https://doi.org/10.5281/zenodo.1342401). The stochastic twin models the
delocalized photonic architecture of Sludds et al., Science 378, 270
(2022), https://doi.org/10.1126/science.abq8271, with exact standard-
quantum-limit shot-noise statistics; electronics constants follow Gao et
al., Sci. Adv. 12, eadz0817 (2026).

## License
Creative Commons Attribution 4.0 International (CC BY 4.0).
"""

INSTRUCTIONS = """ZENODO UPLOAD INSTRUCTIONS (PILOT-Q)
====================================
1. zenodo.org -> New upload -> "Get a DOI now!" (reserve BEFORE publishing).
2. Paste the DOI into latex/main.tex:  \\newcommand{\\datasetdoi}{10.5281/zenodo.NNNNNNN}
3. Upload pilotq-benchmark-v1.zip; Resource type: Dataset;
   Title: PILOT-Q benchmark: photon-budget-aware training and closed-loop
   operation of delocalized photonic neural inference at the SQL;
   Authors: Mahim, Tanvir M.; Islam, Md Nahin; Rahman, M. Mosaddequr;
   Mohsin, Abu S. M.; License: CC BY 4.0; Description: paste the README.
4. Publish, then recompile the manuscript.
"""


def export_csvs(r, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "accuracy_vs_photons.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "training", "chain", "photons_per_mac",
                    "acc_mean", "acc_std"])
        for ds in ["mnist", "fashion", "fsdd"]:
            for mode in ["digital", "pitlq", "pitlq_full", "pitlq_fixed1"]:
                if mode not in r[ds]:
                    continue
                for key, ch in [("acc_vs_nbar", "shot"),
                                ("acc_vs_nbar_impaired", "impaired")]:
                    if key not in r[ds][mode]:
                        continue
                    for k in sorted(r[ds][mode][key], key=float):
                        d = r[ds][mode][key][k]
                        w.writerow([ds, mode, ch, k, f"{d['mean']:.6f}",
                                    f"{d['std']:.6f}"])
    with open(os.path.join(outdir, "closed_loop.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "nb_lo", "nb_hi", "tau", "acc_mean",
                    "acc_std", "esc_frac", "photons"])
        for ds in ["mnist", "fashion", "fsdd"]:
            for which in ["closed_loop_pitlq", "closed_loop_digital"]:
                for cfg in r[ds][which].values():
                    for p in cfg["points"]:
                        w.writerow([ds, which.replace("closed_loop_", ""),
                                    cfg["nb_lo"], cfg["nb_hi"], p["tau"],
                                    f"{p['acc_mean']:.6f}",
                                    f"{p['acc_std']:.6f}",
                                    f"{p['esc_frac']:.6f}",
                                    f"{p['photons']:.1f}"])
    with open(os.path.join(outdir, "oracle_bounds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "nb_lo", "nb_hi", "acc", "esc_frac", "photons"])
        for ds in ["mnist", "fashion", "fsdd"]:
            for o in r[ds].get("oracle", {}).values():
                w.writerow([ds, o["nb_lo"], o["nb_hi"], f"{o['acc']:.6f}",
                            f"{o['esc_frac']:.6f}", f"{o['photons']:.1f}"])
    with open(os.path.join(outdir, "twin_validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "photons_per_mac", "rmse_mc", "rmse_sql"])
        for key, n in [("ip_validation", 4096), ("ip_validation_n32768", 32768)]:
            for k in sorted(r[key], key=float):
                w.writerow([n, k, f"{r[key][k]['rmse']:.6f}",
                            f"{r[key][k]['sql']:.6f}"])


def main():
    with open(os.path.join(RES, "results.json")) as f:
        r = json.load(f)
    if os.path.exists(PKG):
        shutil.rmtree(PKG)
    os.makedirs(os.path.join(PKG, "checkpoints"))
    shutil.copy(os.path.join(RES, "results.json"), PKG)
    for fn in os.listdir(RES):
        if fn.endswith(".pt"):
            shutil.copy(os.path.join(RES, fn), os.path.join(PKG, "checkpoints", fn))
    export_csvs(r, os.path.join(PKG, "csv"))
    with open(os.path.join(PKG, "README.md"), "w") as f:
        f.write(README)
    with open(os.path.join(ZEN, "INSTRUCTIONS.md"), "w") as f:
        f.write(INSTRUCTIONS)
    zf = os.path.join(ZEN, "pilotq-benchmark-v1.zip")
    if os.path.exists(zf):
        os.remove(zf)
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _d, files in os.walk(PKG):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, ZEN))
    print("zenodo package:", zf, f"{os.path.getsize(zf)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
