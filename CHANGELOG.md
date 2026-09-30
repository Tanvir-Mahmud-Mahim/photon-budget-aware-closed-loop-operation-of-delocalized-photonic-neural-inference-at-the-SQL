# Changelog

All notable changes to this repository are listed here, newest first.
The repository has no tagged releases, so entries are dated from the git
history (commit dates in UTC+6).

## Unreleased: fixes (30 September 2026)

- No script writes outside the repository any more. Before, every script
  built its paths as `<repository>/../<name>`, so the datasets, results,
  figures, LaTeX output and Zenodo package lived in folders next to the
  repository, and the LaTeX output went to a manuscript folder `../latex/`
  that had to be created by hand. Now:
  - `data.py` reads the datasets from `data/` (was `../data/`);
  - `run_all_q.py` writes `results/results.json` and the `.pt` models to
    `results/` (was `../results/`);
  - `fig_schematics.py`, `fig_schematics_q.py` and `fig_results_q.py` write
    to `figures/` (was `../figures/`); `fig_results_q.py` now creates that
    folder itself (before, it stopped if the folder was missing);
  - `make_numbers_q.py` writes `results/numbers.tex` and
    `results/tables.tex`, and `make_supp_q.py` writes
    `results/supp_tables.tex` (were `../latex/...`), next to their input
    `results/results.json`;
  - `package_zenodo_q.py` reads `results/` and writes `zenodo/` (were
    `../results/` and `../zenodo/`).
  - the upload instructions that `package_zenodo_q.py` writes to
    `zenodo/INSTRUCTIONS.md` no longer name a manuscript folder: step 2
    said "Paste the DOI into latex/main.tex" and now says "Paste the DOI
    into the manuscript". This file is written next to the zip, not into
    it, so the zip is unchanged.
  File names and contents are unchanged: from the same input, every output
  file was identical to the one the previous version wrote (PDFs written
  with a fixed timestamp; the zip archive compared file by file).
  Anyone with the old layout can move the folders `data/`, `results/`,
  `figures/` and `zenodo/` into the repository folder, and `latex/*.tex`
  into `results/`.
- `.gitignore`: `results/*.pt` replaced by `results/`; `figures/` and
  `zenodo/` added (`data/` was already there), so generated files are not
  committed.
- `requirements.txt`: `torch>=2.0` changed to `torch>=2.3` (and
  `torch>=2.4.1` on Windows). The code runs with NumPy 1.24 and with
  NumPy 2, and pip may install NumPy 2, but PyTorch releases before 2.3
  cannot exchange arrays with NumPy 2 ("Numpy is not available"; checked
  with PyTorch 2.2.2), and on Windows this was fixed only in PyTorch 2.4.1.
  `numpy>=1.24` and `matplotlib>=3.7` are unchanged.
- README: new folder layout, removed the `mkdir ../latex` step, new
  PyTorch minimum with the reason, and what was checked with the minimum
  versions.

## Unreleased: documentation (30 September 2026)

Documentation only; code, data and results are unchanged.

- README rewritten as a step-by-step guide: the idea in plain language, the
  folder layout the scripts expect (`../data`, `../results`, `../figures`,
  `../latex`, `../zenodo`), installation including the extra `soundfile`
  package and the manual dataset step, three ways to use the code, a table of
  stages with measured run times where they were measured, which script makes
  which figure and table, parameter sources, the built-in twin check, and
  notes on the calculations.
- Added this CHANGELOG.
- Added CITATION.cff.

## 1 August 2026

- Added `supplemental_material.pdf`, the supplementary material of the
  manuscript (commit b58a991), and replaced it the same day with an updated
  copy (commit 6574b10).

## 25 July 2026

- Added the PILOT-Q code base (commit c0b3c68): the digital twin
  (`physicsq.py`), network (`modelsq.py`), training protocols (`trainq.py`),
  evaluation and closed-loop controller (`experimentsq.py`), dataset loaders
  (`data.py`), master driver (`run_all_q.py`), LaTeX generators
  (`make_numbers_q.py`, `make_supp_q.py`), schematic figures
  (`fig_schematics_q.py`), Zenodo packager (`package_zenodo_q.py`), README,
  LICENSE (Apache-2.0), `requirements.txt` and `.gitignore`.
- Added the shared figure toolchain (commit 4fd298c): `fig_results_q.py`,
  `fig_schematics.py` and `figstyle.py`.
