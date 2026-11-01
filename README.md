# Diffuse-domain CL-ROM shape optimization

This package reproduces the numerical results in

> X. Zhang, A. Vasudevan, L. Ju, and Z. Wang, *A Compact Residual-Screened Local
> Reduced-Order Model for Diffuse-Domain Shape Optimization*.

The experiments compare full-order optimization with the compact
residual-screened local reduced-order model (CL-ROM) for shape identification,
flow drag minimization, and cantilever compliance minimization.

## Requirements

- MATLAB R2024b; base MATLAB is sufficient.
- Python 3 with NumPy, SciPy, and Matplotlib for the scalar refinement study
  and plotting.

## Reproduce the numerical results

Run the following in MATLAB from the package directory:

```matlab
setup_paths
run_tests
run_paper_experiments
```

Then run the scalar refinement study from the same directory in a terminal:

```sh
python experiments/run_scalar_manufactured_refinement.py
```

Export the numerical results in MATLAB:

```matlab
export_paper_numbers
```

Compare the exported results with the supplied reference values:

```sh
python experiments/compare_paper_numbers.py
```

Outputs are written to `results/`, including `paper_numbers.json`.
Completed experiment runs are saved and reused, so interrupted execution can
resume. Use a fresh package copy when changing the experiment settings.
Plots can be regenerated with `python figures/make_paper_figures.py`.

For timing comparisons, start MATLAB with `-singleCompThread` and run the
experiments sequentially on an otherwise idle machine. A complete run takes
about two hours on the workstation used in the paper. Runtimes depend on
hardware and load; screening decisions and correction counts can also vary
with floating-point differences.

## Numerical schemes and settings

| Problem | State discretization | Main mesh size |
|---|---|---|
| Shape identification | Flux-form finite differences with a variational diffuse Dirichlet penalty | `h = 0.002` |
| Drag minimization | MAC staggered finite differences for the phase-field-weighted Navier–Stokes equations | `h = 0.025` |
| Cantilever compliance | Bilinear quadrilateral finite elements, plane stress, and SIMP stiffness with exponent 3 | `h = 0.01` |

All problems use a gated Allen–Cahn design update with interface width
`epsilon = 4h`, time step `0.01`, and five semi-implicit substeps per design
iteration. The sensitivity stays fixed during these substeps. The gated
sensitivity is normalized, and the initial balance rule with `rho = 0.3`
sets a forcing coefficient that remains fixed throughout each run. Drag
minimization enforces the prescribed volume through a multiplier and
bound projection; the cantilever uses a volume penalty in the objective.

The symmetric state systems use Galerkin reduction and CG with retained
Cholesky factors. Flow uses least-squares Petrov–Galerkin reduction with at
most two Gauss–Newton steps per reduced state solve, and Newton–GMRES with
retained LU factors for full-order solves. CL-ROM starts with two full-order
updates and caps its snapshot windows and reduced ranks at eight. Residual
screening, retained-factor corrections, full-order fallback, and objective
verification control acceptance of reduced candidates.

Timed CL-ROM runs use the same update budget as the full-order stopping
length. Separate accuracy audits re-solve sampled candidates at full order
and are excluded from the timings. The drivers also include the larger
`0.05` design step and spatial refinement at fixed interface width.

## License

Released under the [MIT License](LICENSE).
