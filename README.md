# PILOT-Q: Code Base

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository holds the code for our manuscript **"PILOT-Q: physics-in-the-loop training and
photon-budget-aware closed-loop operation of delocalized photonic neural
inference at the standard quantum limit"**
by T. M. Mahim, M. N. Islam, M. M. Rahman, and A. S. M. Mohsin
(BRAC University / University of Memphis).

The manuscript is not yet published. Citation details will be added on
publication. The supplementary material carries a different title; see
[DETAILS.md](DETAILS.md#more-on-the-introduction-the-supplements-title-and-authors).

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
5. [The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step)
6. [Which script makes which figure and table](DETAILS.md#6-which-script-makes-which-figure-and-table)
7. [The Python modules](DETAILS.md#7-the-python-modules)
8. [Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from)
9. [Built-in checks](DETAILS.md#9-built-in-checks)
10. [Notes on the calculations](DETAILS.md#10-notes-on-the-calculations)
11. [Version history](DETAILS.md#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

Extra notes for the introduction and Sections 1 to 4 are in
[DETAILS.md](DETAILS.md#extra-notes-for-sections-1-to-4).

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

The most important files are:

- **physicsq.py**: the digital twin of the photonic link; photon and energy accounting; twin check
- **modelsq.py**: the neural network (fully connected, runs on the twin)
- **trainq.py**: the training protocols (conventional, PILOT-Q, PILOT-Q-full, fixed budget)
- **experimentsq.py**: evaluation: photon sweeps, hardware-error sweeps, closed-loop controller, oracle bound
- **run_all_q.py**: master driver: runs every experiment and writes results.json
- **make_numbers_q.py** and **make_supp_q.py**: write the LaTeX numbers and tables
- **fig_schematics_q.py** and **fig_results_q.py**: draw the figures
- **supplemental_material.pdf**: supplementary material of the manuscript (18 pages)

The full annotated file tree is in
[DETAILS.md](DETAILS.md#more-on-section-2-the-full-file-tree).

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
yourself.

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

Why PyTorch 2.3, a caution about Matplotlib 3.7.0 to 3.7.2, and my checks with
the minimum versions are in
[DETAILS.md](DETAILS.md#more-on-section-3-versions-checks-and-dataset-sources).

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
(`recordings/`). How I checked both is in
[DETAILS.md](DETAILS.md#more-on-section-3-versions-checks-and-dataset-sources).

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
`fig_architecture_q done`. A message you may see here is explained in
[DETAILS.md](DETAILS.md#more-on-section-4-outputs-timings-and-the-data-package).

### Way B: redraw figures and tables from an existing results file

For this you need a complete `results.json`. To skip training, you also need
the trained models `*.pt`. Where these could come from is in
[DETAILS.md](DETAILS.md#more-on-section-4-outputs-timings-and-the-data-package).

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
training and only repeats the evaluations. It still needs the datasets.

### Way C: recompute everything from scratch (long)

Put the datasets in `data/` (Section 3), install `soundfile`, and then run:

```
python3 run_all_q.py all
```

This trains ten networks and evaluates them. After that, continue with the
commands of Way B. My timing notes are in
[DETAILS.md](DETAILS.md#more-on-section-4-outputs-timings-and-the-data-package).

---

## More details (Sections 5 to 11)

The full notes are in [DETAILS.md](DETAILS.md). There you find:

- [5. The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step):
  each stage of `run_all_q.py` and each output script, with times, results
  and my test-run notes.
- [6. Which script makes which figure and table](DETAILS.md#6-which-script-makes-which-figure-and-table):
  each figure and LaTeX file, its content, the data it needs and its place in
  the supplement.
- [7. The Python modules](DETAILS.md#7-the-python-modules):
  what each module contains.
- [8. Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from):
  physical constants, training and evaluation settings, model sizes and
  literature values.
- [9. Built-in checks](DETAILS.md#9-built-in-checks):
  the twin validation and what I saw when I ran it.
- [10. Notes on the calculations](DETAILS.md#10-notes-on-the-calculations):
  photon budget, noise model, hardware errors, closed loop, targets, FSDD
  features and other notes.
- [11. Version history](DETAILS.md#11-version-history):
  a dated list of changes.

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
