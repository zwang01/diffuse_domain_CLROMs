# Code for the diffuse-domain CL-ROM paper

This package reproduces the numerical experiments in the manuscript

> X. Zhang, A. Vasudevan, L. Ju, and Z. Wang, *A Compact Residual-Screened Local
> Reduced-Order Model for Diffuse-Domain Shape Optimization*.

It contains the three diffuse-domain (DDM) design problems of the paper, the
compact residual-screened local reduced-order model (CL-ROM), and the drivers
that produce every table and figure of Sections 4 and Appendices D and E:

| problem | section | state system |
|---|---|---|
| shape identification | 4.2 | scalar diffusion with a variational surface penalty |
| drag minimization in a channel | 4.3 | phi-weighted Navier–Stokes on a MAC grid |
| compliance minimization of a cantilever | 4.4 | phi-weighted Q1 plane-stress elasticity |

Each problem is solved twice with the same design update: by the
preconditioned full-order optimization and by CL-ROM.

## Requirements and setup

- MATLAB R2024b (base MATLAB; no toolboxes are needed).
- Python 3 with NumPy, SciPy and Matplotlib, for one refinement study and
  the figures.
- About two hours of otherwise idle machine time for the complete set of
  published runs on the paper's workstation (single-threaded MATLAB on an
  Intel Core i9-10900F). Wall times and speedups vary by machine.

In MATLAB, from this directory:

```matlab
setup_paths
```

Output is written to `results/`. Saved results are generated outputs and are
not part of the source package.

## Reproduce the manuscript

```matlab
run_tests                 % code checks quoted in Sections 4.3 and 4.4 (about a minute)
run_paper_experiments     % all published optimization runs and Table 7 (MATLAB part)
```

Then, from this directory:

```sh
python experiments/run_scalar_manufactured_refinement.py   # Table 7, shape-identification row
python figures/make_paper_figures.py                       # Figures 3-8 -> results/figures
python figures/make_ddm_schematic.py                       # Figure 1    -> results/figures
python figures/make_ddm_settings.py                        # Figure 2    -> results/figures
```

and in MATLAB:

```matlab
export_paper_numbers      % prints Tables 3-8 and writes results/paper_numbers.json
```

Then compare with the values reported in the paper:

```sh
python experiments/compare_paper_numbers.py
```

It lists every quantity that differs from `reference/published_numbers.json`
and the timings separately. Start MATLAB with `-singleCompThread` for timings
comparable with the paper.

### Expected agreement

A complete rerun of this package on a second machine (Windows, MATLAB R2024b,
three problems run in parallel) gave the following agreement with the
published runs, which were made on Linux:

- All full-order optimizations: identical update, solve, and factorization
  counts, objectives agreeing to about `1e-9`, and identical thresholded
  designs. The forcing coefficients, Table 7, the shape error, and the
  objective reductions are identical.
- CL-ROM for the cantilever: identical counts, objectives, and audit.
- CL-ROM for shape identification and the flow: the same thresholded
  designs, mean-objective gaps of the same order as published (at most
  `1.5e-5`), and audit maxima within 25% of the published values. The counts of
  corrections and full-order events can differ: for example, 977 instead of
  821 corrections for shape identification and 53 instead of 54 for the
  flow. Screening decisions close to a threshold depend on round-off, which
  differs between platforms.
- Wall times and speedups depend on the machine and its load.

`run_paper_experiments` calls the drivers below in this order. Every driver
saves each completed leg and skips saved legs, so an interrupted run can be
restarted. The timed legs run one after another; avoid other heavy work on
the machine while they run.

| driver | output in `results/` | paper |
|---|---|---|
| `run_common_balance({'scalar','cant'})` | `common_balance_r03/scalar.mat`, `cant.mat` | Figures 3, 4; Tables 3, 6 |
| `continue_cant_fom` | `common_balance_r03/cant_fom_continuation.mat` | cantilever full order beyond 12,000 updates |
| `run_cant_converged_comparison` | `common_balance_r03/cant_converged.mat` | Figures 7, 8; Tables 5, 6 |
| `run_submission_flow` | `flow_submission_traction/flow.mat`, `flow_submission_traction_tau05/flow.mat` | Figures 5, 6; Tables 4, 6, 8 |
| `run_scalar_flow_tau05_common('scalar')` | `common_balance_r03/scalar_tau05.mat` | Table 8 |
| `run_cant_tau05_common` | `common_balance_r03/cant_tau05.mat` | Table 8 |
| `run_flow_manufactured_refinement(...)` | `flow_mesh_fixed_epsilon.csv` | Table 7 |
| `run_elastic_manufactured_refinement` | `elastic_manufactured_2d.csv` | Table 7 |
| `run_scalar_manufactured_refinement.py` | `scalar_manufactured_2d.csv` | Table 7 |
| `export_paper_fields` | `figure_data.mat` | data of Figures 3–8 |

For a quick workflow check, run `run_paper_experiments(120)` in a fresh copy
of the package. It caps every optimization at 120 updates and does not
reproduce the paper.

## Published settings

All three problems use the common design settings of Section 4.1, applied by
`balanced_ddm_problem`: interface width `epsilon = 4h`, design step
`tau = 0.01`, `K = 5` semi-implicit substeps, normalization after gating,
and one initial full-order solve that fixes the forcing coefficient by the
balance rule with `rho = 0.3`. The resulting coefficients are 0.70, 0.13 and
0.72.

Each full-order run continues until its objective stagnation test; CL-ROM is
then run for the same number of updates. CL-ROM uses two warm-up full-order
updates, windows and ranks capped at 8, a POD cutoff of `1e-10` (`1e-12` for
the flow), a screening tolerance that starts at `1e-5` within
`[1e-9, 3e-5]`, objective tolerance `1e-4`, and a full-order backstop every
400 updates. After a failed screen, the linear problems try up to two
corrections with the retained factorization and the flow tries one inexact
Newton correction (`correction_method = 'gmres'`, at most 8 GMRES
iterations); after an objective trigger all three allow up to two further
corrections. Full-order solves, fallbacks and verifications use Krylov
solvers preconditioned by the retained factors, refactoring beyond 8
iterations (40 for the flow).

The cantilever full-order run reached the initial 12,000-update cap of
`run_common_balance`. `continue_cant_fom` restarts it from the saved design
until the stopping test is met (15,581 updates in total), and
`run_cant_converged_comparison` joins the two records and runs CL-ROM for
that length. The same-design audits (`compact_own` legs and the flow
`audit` leg) re-solve sampled screened candidates at full order. They do
not change the trajectories and are excluded from the timings.

The problem adapters contain options that the published runs do not enable,
for example adaptive design steps, alternative gates and the cantilever's
`'inverter'` and `'target'` variants. Their defaults are off or are
overridden by `balanced_ddm_problem`.

## Code layout

| directory | contents |
|---|---|
| `src/core` | `tr_optimize` (the optimization loop with CL-ROM), residual indicators and calibration, POD windows, dual-weighted objective estimate, factorizations and preconditioned CG/GMRES, stopping rule, `balanced_ddm_problem` |
| `src/scalar` | `scalar_surface_problem`: shape identification with the variational surface penalty (`scalar_problem` supplies its grid and data) |
| `src/flow` | `flow_problem`: phi-weighted MAC Navier–Stokes, boundary momentum-flux drag, volume-constrained Allen–Cahn update |
| `src/elast` | `cantilever_problem`: Q1 plane-stress elasticity, perforated-plate initial design, Allen–Cahn update |
| `experiments` | published run drivers, `run_paper_experiments`, `export_paper_numbers`, `compare_paper_numbers.py` |
| `figures` | `export_paper_fields`, `make_paper_figures.py`, `make_ddm_schematic.py`, `make_ddm_settings.py` |
| `tests` | `run_tests` and the four checks it calls |
| `reference` | `published_numbers.json` (values in the paper), `source_sha256.txt` |

## Provenance

`reference/published_numbers.json` was produced by `export_paper_numbers`
from the saved results behind the manuscript. Those runs used these drivers
with one difference: `run_common_balance` and `run_cant_converged_comparison`
also ran a local ROM without corrections, which the paper does not report.
That leg was removed; each leg is an independent run, so the removal does not
change the reported legs. Exploratory drivers, pilot studies and their
outputs were removed from this package.

## License

This package is released under the [MIT License](LICENSE).
