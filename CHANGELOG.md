# Changelog

All notable changes to this repository are listed here, newest first.
The repository has no tagged releases, so entries are dated from the git
history (commit dates in UTC+6).

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
