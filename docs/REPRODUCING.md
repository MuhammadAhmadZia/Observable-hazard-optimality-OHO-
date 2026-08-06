# Reproducing the results

Every result file in `results/` was produced by the experiment suite. This note
records what each experiment does and which paper element it feeds.

## Verification battery (Sections 4.2 to 4.7)

The battery runs in a single notebook with a FAST flag. FAST=True gives a quick
smoke test of every section. FAST=False produces publication-grade runs at ten
seeds and five segments.

Sections are independent after setup, so each can run in its own session. Each
writes its CSV on completion, and the observability section writes incrementally
per corruption level.

## Mapping

- A_proposition.csv    Theorem 1 numerical checks: exhaustive, local search, LP.
- C_dfi_sensitivity.csv Depth index across 405 metric settings.
- D_battery.csv         Six re-ranking methods, two pools, four cut-offs.
- E_steelman.csv        Four learned formulations on identical held-out folds.
- E_theta.csv           Learned penalty weights per segment.
- F_observability.csv   Fixed-policy comparison across masking and value noise.
- B_decomposition.csv   Withdrawn decomposition diagnostic (reported honestly).
- G_parity.csv          Withdrawn representation hypothesis.
- H_domains.csv         Second hazard axis, conditionality of drift.

## Figures

Run `code/make_figures.py` pointed at `results/`. It writes PNG at 600 dpi and
editable SVG, plus table CSVs, into `figures/`.
