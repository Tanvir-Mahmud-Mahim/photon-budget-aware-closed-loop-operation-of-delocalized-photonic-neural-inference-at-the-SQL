# PILOT-Q — code base

**PILOT-Q: physics-in-the-loop training and photon-budget-aware closed-loop
operation of delocalized photonic neural inference at the standard quantum
limit.**

T. M. Mahim, M. N. Islam, M. M. Rahman, and A. S. M. Mohsin
(BRAC University / University of Memphis).

This repository contains the complete code base of the manuscript. Every
number, table, and figure in the paper is generated programmatically from a
single results record (`results/results.json`) produced by this code — no
quantitative value is typed by hand.

## Repository structure

| File | Purpose |
|---|---|
| `physicsq.py` | Differentiable quantum-limited digital twin: differential-rail shot noise at the analytic SQL, dark counts, trim error, b-bit ADC (straight-through estimator); photon/energy accounting; validation against the closed-form shot-noise law |
| `modelsq.py` | Thin-client fully connected networks (intensity inputs, ReLU) |
| `trainq.py` | Training protocols: conventional / PILOT-Q (shot-noise-aware) / PILOT-Q-full / fixed-budget ablation — clean warmup 15%, clean anchor 35%, lr decay, normalized logits γ=8 |
| `experimentsq.py` | Photon sweeps, impairment sweeps, confidence-gated closed-loop controller with maximum-likelihood photon pooling, oracle escalation bound |
| `data.py` | Dataset loaders: MNIST, Fashion-MNIST, FSDD (auto-download; STFT amplitude front-end for audio) |
| `run_all_q.py` | Master experiment driver — `python3 run_all_q.py all` reproduces the full campaign (checkpoints `results/results.json`) |
| `make_numbers_q.py` / `make_supp_q.py` | Auto-generate the paper's `numbers.tex`, `tables.tex`, `supp_tables.tex` |
| `fig_schematics_q.py` / `fig_results_q.py` | All manuscript figures (vector PDF) |
| `figstyle.py` / `fig_schematics.py` | Shared publication figure style and schematic drawing toolchain |
| `package_zenodo_q.py` | Assembles the Zenodo data package (results, checkpoints, CSV exports) |

## Reproduction

```bash
pip install -r requirements.txt
python3 run_all_q.py all        # full campaign (CPU, deterministic)
python3 make_numbers_q.py       # regenerate the paper's numbers/tables
python3 fig_schematics_q.py     # regenerate schematic figures
python3 fig_results_q.py        # regenerate result figures
```

Datasets download automatically into `data/` on first run. All experiments
are deterministic: training seed 42; evaluation seeds {101, 202, 303, 404,
505}. Every reported quantity is exactly reproducible from this code.

## Data

The benchmark data package (results record, all trained checkpoints, CSV
exports) is archived on Zenodo under CC-BY 4.0; see the manuscript's Data
Availability statement for the DOI. MNIST, Fashion-MNIST, and the Free
Spoken Digit Dataset (doi:10.5281/zenodo.1342401) are publicly available
from their original distributions.

## License

This code is released under the Apache License 2.0 — see [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite the PILOT-Q manuscript (citation details
will be updated upon publication).
