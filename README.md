# PILOT-Q: Code Base

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository holds the code for our manuscript **"PILOT-Q: physics-in-the-loop training and
photon-budget-aware closed-loop operation of delocalized photonic neural
inference at the standard quantum limit"**
by T. M. Mahim, M. N. Islam, M. M. Rahman, and A. S. M. Mohsin
(BRAC University / University of Memphis).

The supplementary material in this repository (`supplemental_material.pdf`)
carries a different title, **"Photon-efficient neural inference over optical
links: training and adaptive operation at the quantum limit"**. It lists
Tanvir M. Mahim, M. Mosaddequr Rahman and Abu S. M. Mohsin (Department of
Electrical and Electronic Engineering, BRAC University, Dhaka) and Md Nahin
Islam (University of Memphis). The manuscript is not yet published. Citation
details will be added on publication.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/photon-budget-aware-closed-loop-operation-of-delocalized-photonic-neural-inference-at-the-SQL
- Supplementary material: [`supplemental_material.pdf`](supplemental_material.pdf) (18 pages)
- Archived data: the code prepares a Zenodo data package
  (`package_zenodo_q.py`), but the DOI of that package is not recorded in
  this repository.

According to the scripts and the supplementary material, this code generates
every number, table and figure of the manuscript from one results file,
`results.json`. No value is typed by hand.

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](#5-the-scripts-step-by-step)
6. [Which script makes which figure and table](#6-which-script-makes-which-figure-and-table)
7. [The Python modules](#7-the-python-modules)
8. [Where the numbers come from](#8-where-the-numbers-come-from)
9. [Built-in checks](#9-built-in-checks)
10. [Notes on the calculations](#10-notes-on-the-calculations)
11. [Version history](#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

---

## 1. The idea in one minute

A neural network spends most of its effort on **multiply-accumulate
operations (MACs)**: multiply two numbers and add the result to a running sum.
Think of a *delocalized* photonic system (the "Netcast" design of Sludds et
al., Science 378, 270, 2022). There, a central server sends the network's
weights as light in an optical fiber to small edge devices (for example a
camera, drone or sensor). Each device does the multiplications by modulating
that light with its own input data. It then measures the result with
photodetectors.

Light arrives in single particles, **photons**. When very few photons are
used per MAC, the count at the detector fluctuates at random. This unavoidable
counting noise is called **shot noise**, and its size sets the **standard
quantum limit (SQL)** of ordinary light detection. Fewer photons save laser
power, but they add noise and lower the accuracy.

In this project we ask how few photons per MAC are really needed. Our answer
uses two ideas:

- **Physics-in-the-loop training (PILOT-Q).** The network is trained while a
  simulated copy of the hardware (a "digital twin") adds the same shot noise,
  plus other hardware errors, in every training step. This way the network
  learns to cope with the noise it will meet in use.
- **Closed-loop operation.** Each input is first measured with a small photon
  budget. If the network is confident, the answer is kept. If not, the input
  is measured again with more photons, and the two measurements are combined.
  Only the uncertain inputs pay for extra light.

The code builds the twin and checks it against the exact shot-noise formula.
It trains networks on three public datasets (handwritten digits, clothing
images and spoken digits) and measures accuracy against photon budget. It
also runs the closed-loop controller and writes all tables and figures.

---

## 2. What is in this repository

```
photon-budget-aware-closed-loop-operation-of-delocalized-photonic-neural-inference-at-the-SQL/
|-- README.md                  this guide
|-- CHANGELOG.md               what changed, from the git history
|-- CITATION.cff               citation details (drives the "Cite this repository" button)
|-- LICENSE                    Apache-2.0 license
|-- requirements.txt           Python packages to install (torch, numpy, matplotlib)
|-- .gitignore                 files git ignores
|-- supplemental_material.pdf  supplementary material of the manuscript (18 pages)
|-- physicsq.py                the digital twin of the photonic link; photon and energy accounting; twin check
|-- modelsq.py                 the neural network (fully connected, runs on the twin)
|-- trainq.py                  the training protocols (conventional, PILOT-Q, PILOT-Q-full, fixed budget)
|-- experimentsq.py            evaluation: photon sweeps, hardware-error sweeps, closed-loop controller, oracle bound
|-- data.py                    dataset loaders (MNIST, Fashion-MNIST, FSDD)
|-- run_all_q.py               master driver: runs every experiment and writes results.json
|-- make_numbers_q.py          writes the manuscript's LaTeX numbers and main tables
|-- make_supp_q.py             writes the supplementary LaTeX tables
|-- fig_schematics_q.py        draws the schematic figures
|-- fig_results_q.py           draws the result figures
|-- fig_schematics.py          shared drawing helpers for the schematics
|-- figstyle.py                shared figure style and colours
`-- package_zenodo_q.py        assembles the data package for Zenodo
```

**All files are read and written inside the repository folder.** No
script writes outside it. After a run, the repository also contains the
folders below. None of them is stored on GitHub (they are listed in
`.gitignore`):

```
photon-budget-aware-closed-loop-...-SQL/
|-- data/       datasets; you put them here (Section 3)
|-- results/    results.json and trained models (*.pt), made by run_all_q.py;
|               the LaTeX output of make_numbers_q.py and make_supp_q.py
|-- figures/    PDF figures, made by fig_schematics_q.py and fig_results_q.py
`-- zenodo/     data package, made by package_zenodo_q.py
```

Every script creates the folder it writes to. The LaTeX generators write to
`results/`, which already holds their input `results.json`. You fill `data/`
yourself. (Before 30 September 2026, these folders were expected *next to*
the repository folder, and the LaTeX output went to a folder `latex/` there;
see [CHANGELOG.md](CHANGELOG.md).)

---

## 3. Installation

The repository does not state a required Python version. I checked the
commands in this guide with **Python 3.11**, PyTorch 2.14, NumPy 2.4 and
Matplotlib 3.10, on CPU only.

```
pip install -r requirements.txt
```

This installs `torch` (version 2.3 or newer; 2.4.1 or newer on Windows),
`numpy` (1.24 or newer) and `matplotlib` (3.7 or newer). A CPU-only build of
PyTorch is enough, since the code never uses a GPU.

**Why PyTorch 2.3.** The code itself runs with NumPy 1.24 and with NumPy 2.
It uses no function that exists in only one of them. Note that pip may
install NumPy 2. PyTorch releases before 2.3 were built for NumPy 1 only.
With NumPy 2, `torch.from_numpy` stops with "Numpy is not available" (I
checked this with PyTorch 2.2.2 and NumPy 2.0.0). On Windows, this was fixed
only in PyTorch 2.4.1.

One caution about Matplotlib. Releases 3.7.0, 3.7.1 and 3.7.2 were built for
NumPy 1, but they do not say so in their package data. So pip can install
them next to NumPy 2, and then `import matplotlib.pyplot` fails (I checked
this with Matplotlib 3.7.0 and NumPy 2.0.0). Releases 3.7.3 to 3.8.3 declare
`numpy<2`, and 3.8.4 works with NumPy 2. A fresh
`pip install -r requirements.txt` installs current versions and is not
affected. If you pin Matplotlib 3.7.0, 3.7.1 or 3.7.2, also pin `numpy<2`.

**Checked with the minimum versions** (30 September 2026, Python 3.11, CPU).
I ran `run_all_q.py validate`, all dataset loaders, a short training and
evaluation run, and the five output scripts. I did this twice: with
torch 2.3.0, numpy 1.24.0 and matplotlib 3.7.0, and again with
torch 2.3.0, numpy 2.0.0 and matplotlib 3.8.4. Everything ran (on synthetic
stand-in data; see the note under the table in Section 5). The three LaTeX
files came out identical in the two setups.

**Extra package for the spoken-digit dataset.** `data.py` reads the FSDD
sound files with `soundfile`, which is not in `requirements.txt`. Install it
before you run the FSDD stage:

```
pip install soundfile
```

**Datasets (manual step).** The loaders do **not** download anything. They
read files from `data/` inside the repository folder, so you need to put the
files there:

| Dataset | Files the code expects | Source recorded in `data.py` |
|---|---|---|
| MNIST (handwritten digits) | `data/mnist_repo/train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` | LeCun et al., CC BY-SA 3.0, "GitHub mirror of the canonical distribution" (the mirror is not named) |
| Fashion-MNIST (clothing images) | the same four file names in `data/fmnist_repo/data/fashion/` | Xiao et al. 2017, MIT license, official Zalando repository |
| FSDD, Free Spoken Digit Dataset | `data/fsdd/recordings/<digit>_<speaker>_<index>.wav` | v1.0.10 (Jackson et al.), CC BY-SA 4.0, official repository; doi:10.5281/zenodo.1342401 |

The Fashion-MNIST path matches the folder layout of
https://github.com/zalandoresearch/fashion-mnist (`data/fashion/`), and the
FSDD path matches https://github.com/Jakobovski/free-spoken-digit-dataset
(`recordings/`). I checked both while writing this guide. I downloaded the
four Fashion-MNIST files from that repository's `data/fashion/` folder and
read them successfully with `data.py`. I also found the FSDD file
`recordings/0_jackson_0.wav` in the FSDD repository.

**Fonts.** Figures use the DejaVu Sans font that ships with Matplotlib, so
you do not need any extra fonts.

---

## 4. Quick start: three ways to use the code

Run all commands from inside the repository folder.

### Way A: check the digital twin (about 5 seconds)

```
python3 run_all_q.py validate
```

This compares the simulated shot noise with the exact formula (Section 9) and
writes the result to `results/results.json`. It needs no datasets. It
prints `== Twin validation vs analytic SQL ==` and a total time.

You can also draw the architecture schematic, which needs no results (2 to 5
seconds):

```
python3 fig_schematics_q.py
```

This writes `figures/fig_architecture_q.pdf` and prints
`fig_architecture_q done`. If `results.json` exists but holds only the twin
check, it also prints `abstract deferred: 'mnist'`. This means the overview
figure is skipped until the dataset results exist. This is expected.

### Way B: redraw figures and tables from an existing results file

For this you need a complete `results.json`. To skip training, you also need
the trained models `*.pt`. The contents of the data package made by
`package_zenodo_q.py` would do (`pilotq-benchmark-v1/results.json` and
`pilotq-benchmark-v1/checkpoints/*.pt`). No such package is stored in the
repository, so I could not test this path with the real results (see the
note under the table in Section 5).

1. Copy `results.json` (and the `.pt` files) into `results/` (create the
   folder if needed).
2. Run:

```
python3 fig_schematics_q.py     # schematics -> figures/
python3 fig_results_q.py        # result figures -> figures/
python3 make_numbers_q.py       # results/numbers.tex and results/tables.tex
python3 make_supp_q.py          # results/supp_tables.tex
```

If the `.pt` files are present, `python3 run_all_q.py all` skips all
training and only repeats the evaluations. It still needs the datasets. It
also overwrites the evaluation entries of `results.json`, and the twin check
gives slightly different values on every run (see Section 10).

### Way C: recompute everything from scratch (long)

Put the datasets in `data/` (Section 3), install `soundfile`, and then run:

```
python3 run_all_q.py all
```

This trains ten networks and evaluates them. I did not re-time this step.
To give you a sense of scale: two training epochs of the PILOT-Q protocol on
Fashion-MNIST took 64 seconds when I measured them while writing this guide.
I used a shared 2-core machine, CPU only. The full campaign trains for 605
epochs in total (95 for each image dataset, 380 for the audio dataset and 35
for the ablation). After that, continue with the
commands of Way B.

---

## 5. The scripts, step by step

`run_all_q.py` takes one stage name (default `all`). Every stage adds its
results to `results/results.json` and keeps what is already there. If a
trained model file already exists, the script loads it instead of
retraining.

| Step | Command | What it does | Time* | Results |
|---|---|---|---|---|
| 1 | `python3 run_all_q.py validate` | Checks the twin's shot noise against the exact formula at 8 photon budgets (0.25 to 64 photons/MAC), for vectors of length 4,096 (400 trials) and 32,768 (60 trials) | 5 s | `ip_validation`, `ip_validation_n32768` in `results.json` |
| 2 | `python3 run_all_q.py mnist` | Trains three networks (conventional 25 epochs; PILOT-Q and PILOT-Q-full 35 epochs each); measures accuracy at 10 photon budgets with and without extra hardware errors; hardware-error sweeps at 2 photons/MAC; closed-loop sweeps (6 budget pairs x 12 thresholds) for the conventional and PILOT-Q networks, plus one on the impaired chain; oracle bound | long (not re-timed) | `mnist_digital.pt`, `mnist_pitlq.pt`, `mnist_pitlq_full.pt`; `mnist` in `results.json` |
| 3 | `python3 run_all_q.py fashion` | Same for Fashion-MNIST (PILOT-Q training band 0.5 to 8 photons/MAC instead of 0.25 to 8) | long (not re-timed) | `fashion_*.pt`; `fashion` in `results.json` |
| 4 | `python3 run_all_q.py fsdd` | Same for spoken digits (100 / 140 / 140 epochs); needs `soundfile` | long (not re-timed) | `fsdd_*.pt`; `fsdd` in `results.json` |
| 5 | `python3 run_all_q.py ablation` | Trains one extra MNIST network at a single fixed budget of 1 photon/MAC (35 epochs) and measures its accuracy curve | long (not re-timed) | `mnist_pitlq_fixed1.pt`; `mnist/pitlq_fixed1` in `results.json` |
| - | `python3 run_all_q.py all` | Steps 1 to 5 in this order | long (not re-timed) | all of the above |
| 6 | `python3 make_numbers_q.py` | Writes every number quoted in the manuscript as a LaTeX command, plus the two main tables | not run here** | `results/numbers.tex`, `results/tables.tex` |
| 7 | `python3 make_supp_q.py` | Writes the supplementary tables | not run here** | `results/supp_tables.tex` |
| 8 | `python3 fig_schematics_q.py` | Draws the architecture schematic; also the overview figure if `results.json` exists | 2 to 5 s (schematic only) | `figures/fig_architecture_q.pdf`, `figures/fig_abstract_q.pdf` |
| 9 | `python3 fig_results_q.py` | Draws the five result figures | not run here** | `figures/fig_*_q.pdf` (Section 6) |
| 10 | `python3 package_zenodo_q.py` | Copies `results.json` and all `.pt` files, exports four CSV tables, writes a README and upload instructions, and zips everything | not run here** | `zenodo/pilotq-benchmark-v1/`, `zenodo/pilotq-benchmark-v1.zip`, `zenodo/INSTRUCTIONS.md` |

\*I measured these on a shared 2-core machine.
\*\*These steps need the complete `results.json` from steps 1 to 5, which is
not stored in the repository. Without it, they stop with "No such file or
directory". On 30 September 2026, I ran them on a `results.json` and
model files made by `run_all_q.py all` from synthetic stand-in datasets.
The datasets were random images and tones in the expected file formats, and training was cut to
2 epochs. I did this only to check that they run and where they write. The
numbers in those files mean nothing. `make_numbers_q.py`, `make_supp_q.py`,
`fig_schematics_q.py` and `package_zenodo_q.py` finished.
`fig_results_q.py` drew four of the five result figures. It then stopped in
the last one (`fig_sota_q.pdf`) with `ValueError: min() arg is an empty
sequence`. This happened because in the stand-in results, no closed-loop
point reaches the target accuracy. The previous version of the script stops
at the same point with the same input. Every file written was identical to
the file the previous version (with the folders next to the repository)
wrote from the same input. (The PDFs were written with a fixed timestamp,
`SOURCE_DATE_EPOCH=0`, and the zip archive was compared file by file.)

All training uses the fixed seed 42, and all evaluations use the fixed seeds
101, 202, 303, 404 and 505. PyTorch is limited to 2 threads (`trainq.py`).

---

## 6. Which script makes which figure and table

I matched the supplementary figure numbers below by comparing the panels,
axis labels and captions in `supplemental_material.pdf` with the plotting
code. The main-text figure numbers are not recorded consistently in the
repository. The header of `fig_schematics_q.py` says "Fig. 1 (concept) and
Fig. 2 (architecture)", while the supplementary material refers to the
accuracy sweeps as "Fig. 2 of the Letter". So I left them out.

| Output file | Content | Data needed | Drawn by | In the supplement |
|---|---|---|---|---|
| `fig_architecture_q.pdf` | Four schematic panels: (a) server and edge clients, (b) one layer with its noise sources, (c) PILOT-Q training loop, (d) closed-loop controller | none | `fig_schematics_q.py` | - |
| `fig_abstract_q.pdf` | Overview: train through the noise, confidence gate, and a bar chart of the MNIST photon budget per inference at equal accuracy | step 2 (MNIST results only) | `fig_schematics_q.py` | - |
| `fig_validation_q.pdf` | (a) twin noise versus exact formula; (b) energy per inference versus photon budget | step 1 | `fig_results_q.py` | Fig. S1 |
| `fig_accuracy_q.pdf` | Accuracy versus photon budget, 3 datasets x (shot noise only; shot noise + hardware errors) | steps 2 to 4 | `fig_results_q.py` | - |
| `fig_robustness_q.pdf` | MNIST at 2 photons/MAC: (a) dark counts, (b) trim error, (c) ADC bits; (d) fixed-budget training ablation | steps 2, 5 | `fig_results_q.py` | Fig. S2 |
| `fig_budget_q.pdf` | MNIST closed loop: (a) accuracy versus photons per inference with oracle bound, (b) share of re-measured inputs versus threshold, (c) closed loop on the impaired chain | step 2 | `fig_results_q.py` | - |
| `fig_sota_q.pdf` | (a) MNIST accuracy versus photons/MAC with published operating points, (b) photon budget at the target accuracy for all datasets, (c) capability overview | steps 2 to 4 | `fig_results_q.py` | Fig. S3 |

| LaTeX output | Content | In the supplement |
|---|---|---|
| `numbers.tex` | every number quoted in the text, as `\newcommand` macros | - |
| `tables.tex` | `\tablemaster` (accuracy summary) and `\tableclosedloop` (photon budget at target) | same columns as Tables SI and SII |
| `supp_tables.tex` | twin check; photon and energy accounting; full accuracy sweeps; dark, trim and ADC sweeps; literature comparison; closed-loop Pareto points; oracle bounds; hyperparameters | same order as Tables SIII to SXVI |

The LaTeX source of the manuscript itself is not in this repository.

---

## 7. The Python modules

| File | What it contains |
|---|---|
| `physicsq.py` | `PhotonChain`: the noisy photonic matrix-vector product (shot noise, dark counts, trim error, ADC); `model_budget`: photons and energy per inference; `validate_ip_rmse`: the twin check |
| `modelsq.py` | `RealFC`: fully connected network 784-300-100-10 (images) or 4000-300-100-10 (audio); ReLU between layers, linear output |
| `trainq.py` | `train_model` with the four protocols `digital` (conventional), `pitlq` (PILOT-Q), `pitlq_full` (PILOT-Q-full) and `pitlq_fixed1` (fixed-budget ablation); `accuracy` |
| `experimentsq.py` | `acc_vs_nbar` (accuracy versus photon budget), `impairment_sweeps`, `closed_loop` (confidence-gated re-measurement), `oracle_bound`; the evaluation grids and seeds |
| `data.py` | `load_mnist`, `load_fsdd` (with the spectrogram front-end) and the `DATASETS` table |
| `figstyle.py` | Figure style (font sizes, colors, APL column widths) |
| `fig_schematics.py` | Drawing helpers (boxes, arrows, icons) and an automatic text-fitting pass |

---

## 8. Where the numbers come from

**Physical constants** (`physicsq.py`):

| Value | Used | Source as recorded in the code |
|---|---|---|
| Photon energy at 1550 nm | 1.2816e-19 J | stated as the photon energy at 1550 nm (equals hc / 1550 nm) |
| Energy per ADC sample | 1 pJ | "same constant as Ref. WISE"; the Zenodo README text in `package_zenodo_q.py` cites Gao et al., Sci. Adv. 12, eadz0817 (2026) for the electronics constants |
| Energy per digital operation | 1 pJ | as above |
| Optical link loss (server to client) | 30 dB | default of `model_budget`, described as representative in the supplement |
| System being modeled | - | Sludds et al., Science 378, 270 (2022), https://doi.org/10.1126/science.abq8271 |

**Training settings** (`trainq.py`, `run_all_q.py`): Adam optimizer,
learning rate 0.001, reduced to 0.3 times that at 70% of the epochs; batch
256; first 15% of epochs without noise ("clean warmup"); 35% of later batches
also without noise ("clean anchor"). In the loss, each output vector is
divided by its length (Euclidean norm) and multiplied by 8 ("normalized
logits", gamma = 8).
Epochs: 25 (conventional) and 35 (physics-in-the-loop) for images, 100 and
140 for audio. Photon budgets drawn in training: log-uniform between 0.25 and
8 photons/MAC (0.5 to 8 for Fashion-MNIST). PILOT-Q-full also draws dark
fraction 0 to 0.20, trim error 0 to 0.15 and ADC bits from {4, 6, 8, ideal}.

**Evaluation settings** (`experimentsq.py`, `run_all_q.py`): photon budgets
0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 64 photons/MAC. The "impaired"
chain has dark fraction 0.20, trim error 0.15 and a 4-bit ADC. Hardware-error
sweeps run at 2 photons/MAC. Closed-loop budget pairs (low, high) = (0.25, 1),
(0.5, 2), (1, 4), (2, 8), (0.5, 4), (4, 16); thresholds 0 to 0.8.

**Model sizes** (from the code; I checked them with `model_budget`): 266,200
MACs per image inference and 1,231,000 per audio inference; 1.23 nJ of client
electronics per inference for both.

**Literature values** in the comparison table (`make_supp_q.py`) and in
panel (a) of `fig_sota_q.pdf` (`fig_results_q.py`) are typed into the code.
The supplement describes them as "as reported in the cited works". They
refer to the manuscript's reference list, which is not in this repository.
Some of these typed-in rows in the current `make_supp_q.py` differ from the
comparison table printed in `supplemental_material.pdf` (Table SXI). So the
PDF was made with a slightly different version of that script.

---

## 9. Built-in checks

There is no separate test suite. The one built-in check is the twin
validation (`python3 run_all_q.py validate`, function `validate_ip_rmse`):

- It draws random inputs (0 to 1) and random signed weights (-1 to 1), passes
  them through the twin at 8 photon budgets, and measures the error of the
  result (the normalized root-mean-square error).
- It compares this with the exact shot-noise prediction for the same inputs.
  The prediction falls as one over the square root of the photon budget.

I ran it repeatedly while writing this guide. The largest difference
between simulation and prediction was 4% to 7% for vectors of length 4,096
(400 trials; three runs). For length 32,768 it was about 13% to 21% (60
trials; seven runs). The simulated column changes from
run to run (see Section 10). The "within 9%" figure quoted in the supplement
is the macro `valMaxDev` of `make_numbers_q.py`. It is computed from the
length-4,096 check (`ip_validation`) only. In the supplement's own Table
SIII, the length-32,768 column also stays within 9% for that run.

The training scripts print loss and training accuracy every 5 epochs, and
`run_all_q.py` prints the clean test accuracy of each trained network.

---

## 10. Notes on the calculations

- **Photon budget.** `n_bar` is the mean number of photons per MAC, averaged
  over a layer. Photons per inference = `n_bar` x number of MACs. Received
  optical energy = photons x photon energy; server laser energy = optical
  energy x 10^(30 dB / 10). Client electronics = 1 pJ per ADC sample (two
  per output, one for each signal rail) + 1 pJ per output. This part does
  not depend on the photon budget.
- **Noise model.** Signed weights are carried on two optical "rails" (one for
  positive, one for negative values). Photon counts follow Poisson
  statistics. The twin uses the Gaussian approximation, which keeps the
  chain differentiable, so training can pass gradients through the noise. The
  code comments call this "accurate to a few percent" at the photon counts
  involved. Shot-noise variance of output i = `a_i * a_mean / (n_bar * N)`
  with `a_i = sum_j |W_ij x_j|` and N the input length.
- **Hardware errors.** *Dark counts* are detector clicks with no signal
  photon, set as a fraction of the signal budget. *Trim error* is a random
  relative error on each received weight, drawn anew each time a layer is
  computed. The *ADC* (analog-to-digital converter) clips at plus or minus 4
  times the batch RMS and rounds to `b` bits. Training passes gradients
  straight through the rounding.
- **Training versus testing.** The normalized logits (output vector divided
  by its Euclidean norm, then multiplied by 8) are used only in the training
  loss. Test accuracy uses the largest raw output.
- **Closed loop.** Each input is measured at the low budget. Then the code
  looks at the gap between the two highest softmax probabilities. If it is
  below the threshold `tau`, the input is measured again at the high budget,
  and the outputs are combined as `(n_lo*y1 + n_hi*y2)/(n_lo + n_hi)`. The
  code simulates both measurements for every input, but it charges the high
  budget only for the re-measured share. The *oracle bound* re-measures
  exactly the inputs that were wrong at the low budget. No real controller
  can know these. The oracle bound runs on the noise-only chain.
- **Target accuracy.** Each dataset's target is the conventional network's
  noise-free test accuracy minus 2 percentage points (minus 4 on the
  impaired chain). The budget that reaches it is found by interpolating the
  accuracy curve on a logarithmic photon axis.
- **FSDD split and features.** Recordings with index 0 to 4 are the test set
  and 5 to 49 the training set. Each recording is cut or padded to its
  middle 4,000 samples (0.5 s at 8 kHz) and split into 200 frames of 20
  samples. It is then turned into the magnitude of a Fourier transform: a
  4,000-number vector scaled to a maximum of 1.
- **Reproducibility caveat.** Training (seed 42) and evaluation (seeds 101 to
  505) set their random seeds. The twin check in `validate_ip_rmse` seeds
  only the random inputs and weights, not the noise draws. So its simulated
  error column differs slightly on every run.
- **Paths.** `data.py`, `run_all_q.py` and all generators use folders inside
  the repository (`data/`, `results/`, `figures/`, `zenodo/`), as shown in
  Section 2. The LaTeX output goes to `results/`, next to `results.json`.
- **Upload instructions.** `package_zenodo_q.py` writes an `INSTRUCTIONS.md`
  file next to the zip, not inside it. It says to paste the dataset DOI into
  the manuscript's `\datasetdoi` macro and to recompile the manuscript. The
  manuscript is not part of this repository, and no script reads or writes
  it.

---

## 11. Version history

The repository has no tagged releases. You can find the changes, taken from
the git history, in [CHANGELOG.md](CHANGELOG.md).

| Date | Change |
|---|---|
| 30 Sep 2026 | Fixes: all folders moved inside the repository (LaTeX output to `results/`); PyTorch minimum raised to 2.3 |
| 30 Sep 2026 | Documentation rewritten (this guide, CHANGELOG, CITATION.cff); code unchanged |
| 1 Aug 2026 | Supplementary material added, then replaced by an updated copy |
| 25 Jul 2026 | Code base and shared figure toolchain added |

---

## 12. How to cite

Our manuscript is not yet published, and the repository states that citation
details will be updated on publication. Until then, please cite this
repository. GitHub shows a **"Cite this repository"** button in the
right-hand column, which reads `CITATION.cff`.

> T. M. Mahim, M. N. Islam, M. M. Rahman, and A. S. M. Mohsin, "PILOT-Q:
> physics-in-the-loop training and photon-budget-aware closed-loop operation
> of delocalized photonic neural inference at the standard quantum limit
> (code)", https://github.com/Tanvir-Mahmud-Mahim/photon-budget-aware-closed-loop-operation-of-delocalized-photonic-neural-inference-at-the-SQL

Please cite the datasets as their authors request: MNIST (LeCun et al.),
Fashion-MNIST (Xiao et al. 2017) and the Free Spoken Digit Dataset
(doi:10.5281/zenodo.1342401).

---

## 13. License and contact

Code: Apache License 2.0 (see `LICENSE`). The data package prepared by
`package_zenodo_q.py` is labeled CC BY 4.0.

If you have questions or find a bug, please open an issue on this
repository, or contact me, Tanvir M. Mahim, BRAC University
(tanvir.mahim@bracu.ac.bd).
