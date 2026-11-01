"""Panels of Figures 3-8 from results/figure_data.mat (export_paper_fields.m).

Usage:  python make_paper_figures.py [--data FILE] [--out DIR]
Writes PDF and PNG files named as included by the manuscript; the default
output directory is results/figures.  Figures 1 and 2 are drawn by
make_ddm_schematic.py and make_ddm_settings.py.
"""
from pathlib import Path
import argparse

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKG = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--data', default=PKG / 'results' / 'figure_data.mat')
parser.add_argument('--out', default=PKG / 'results' / 'figures')
args = parser.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 10,
                     'pdf.fonttype': 42, 'legend.fontsize': 9})
M = loadmat(args.data, squeeze_me=True, struct_as_record=False)
sc, fl, ca = M['scalar'], M['flow'], M['cant']


def save(fig, stem):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=180)
    plt.close(fig)


def field(x, y, z, title, stem, cmap='viridis', phase=True, scalar=False, limits=None):
    fig, ax = plt.subplots(figsize=(5, 1.22) if scalar else (4.6, 2.6), constrained_layout=True)
    kw = {'vmin': 0, 'vmax': 1} if phase else {'vmin': limits[0], 'vmax': limits[1]}
    im = ax.pcolormesh(x, y, z, shading='nearest', cmap=cmap, rasterized=True, **kw)
    if phase:
        ax.contour(x, y, z, [.5], colors='#d62728', linewidths=.85)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    if scalar:
        fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
    save(fig, stem)


def histories(r, stem, label):
    """Objective histories (left) and calibrated indicator (right)."""
    fig, ax = plt.subplots(figsize=(4.54, 2.875), constrained_layout=True)
    for z, c, ls, lw, lab in [(r.J_fom, 'k', '-', 1.4, 'Full order'),
                              (r.J_rom, '#d62728', (0, (6, 2)), 1.8, 'CL-ROM')]:
        ax.semilogy(np.arange(len(z)), z, color=c, ls=ls, lw=lw, label=lab)
    ax.set(xlabel=r'Iteration $k$', ylabel=label); ax.grid(alpha=.25)
    ax.legend(loc='upper right', fancybox=False)
    save(fig, stem + '_Jhist')
    eta, src, tol = r.eta, r.source, r.tol; k = np.arange(len(eta))
    accepted = np.isin(src, [0, 2]) & np.isfinite(eta) & (eta > 0); full = np.isin(src, [1, 3])
    # full-order updates without a reduced indicator (the two warm-up updates)
    # are drawn below the smallest plotted value
    floor = min(eta[accepted].min(), tol.min()) / 2
    full_eta = eta[full].copy(); full_eta[~np.isfinite(full_eta) | (full_eta <= 0)] = floor
    fig, ax = plt.subplots(figsize=(4.54, 2.875), constrained_layout=True)
    ax.semilogy(k[accepted], eta[accepted], 'b.', ms=2, label='CL-ROM accepted')
    ax.semilogy(k[full], full_eta, 'rv', ms=3, label='Full-order solve')
    ax.semilogy(k, tol, 'k-.', lw=1.1, label=r'Tolerance $\tau_\eta$')
    ax.set(xlabel=r'Iteration $k$', ylabel=r'$\tilde\eta_k$'); ax.grid(alpha=.25)
    ax.legend(loc='lower right', fancybox=False)
    save(fig, stem + '_eta_compact')


# ---- Figure 3: shape identification
x, y = sc.x, sc.y
limits = (min(sc.target.min(), sc.observed.min()), max(sc.target.max(), sc.observed.max()))
for z, title, stem in [(sc.phi0.T, r'Initial $\phi$', 'fig_phi_init'),
                       (sc.phi.T, r'Final $\phi$ (CL-ROM)', 'fig_phi_final'),
                       (sc.domain.T, r'Target domain $\Omega_{\rm ref}$', 'fig_phi_target')]:
    field(x, y, z, title, stem, scalar=True)
for z, title, stem in [(sc.observed.T, r'Observed state $\phi u$', 'fig_state_final'),
                       (sc.target.T, r'Target $\bar u$', 'fig_state_target')]:
    field(x, y, z, title, stem, phase=False, scalar=True, limits=limits)
# the initial state is deeper than the target and final states, so its panel
# keeps its own lower colour limit; the upper limit is common
z = sc.observed_init.T
field(x, y, z, r'Observed state $\phi u$ (initial)', 'fig_state_init', phase=False,
      scalar=True, limits=(z.min(), limits[1]))

# ---- Figure 5: drag minimization
x, y, xn, yn = fl.xc, fl.yc, fl.x, fl.y
mask = (x >= -.8) & (x <= 1.2)
for i, (phi, suffix, title) in enumerate([(fl.phi0.T, 'init', 'Initial disk'),
                                          (fl.phi.T, 'final', 'Final design (CL-ROM)')]):
    fig, ax = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    ax.pcolormesh(x[mask], y, phi[:, mask], shading='nearest', cmap='viridis',
                  vmin=0, vmax=1, rasterized=True)
    ax.contour(x[mask], y, phi[:, mask], [.5], colors='#d62728', linewidths=.85)
    ax.set(aspect='equal', title=title, xlim=(x[mask].min(), x[mask].max()),
           ylim=(y.min(), y.max()))
    save(fig, f'cmp2_N80_phi_{suffix}')
    # contours of the weighted flux, with extra levels near the stagnation
    # streamline; the same levels in both panels
    psi = fl.psi[:, :, i].T
    flux = fl.psi[0, -1, 0]; middle = flux / 2
    levels = np.unique(np.r_[np.linspace(.01, .99, 56) * flux,
                             middle + flux * np.r_[-.007, -.004, -.002, -.001, .001, .002, .004, .007]])
    xx, yy = np.meshgrid(xn, yn)
    pn = RegularGridInterpolator((y, x), phi, bounds_error=False, fill_value=1)(
        np.c_[yy.ravel(), xx.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    ax.contour(xn, yn, np.ma.masked_where(pn < .5, psi), levels=levels,
               colors='#252525', linewidths=.48)
    ax.contourf(x, y, phi, levels=[0, .5], colors=['#d4d4d4'])
    ax.contour(x, y, phi, levels=[.5], colors='#d62728', linewidths=1)
    ax.set(xlim=(-.8, 1.2), ylim=(-.5, .5), xticks=[-.5, 0, .5, 1], yticks=[-.4, 0, .4],
           title='Flow around the ' + ('initial disk' if i == 0 else 'final design'))
    ax.set_aspect('equal'); save(fig, f'cmp2_N80_str_{suffix}')

# ---- Figure 7: cantilever designs
x = (np.arange(200) + .5) * .01; y = (np.arange(100) + .5) * .01
for z, suffix, title in [(ca.phi0.T, 'init', 'Initial design'),
                         (ca.phi.T, 'final', f'CL-ROM: {len(ca.J_rom):,} updates')]:
    field(x, y, z, title, f'cmp2_cant_{suffix}', cmap='gray_r')

# ---- Figures 4, 6, 8: histories
histories(sc, 'fig', r'$J(\phi_k)$')
histories(fl, 'cmp2_N80', r'Drag $\mathcal{D}(\phi_k)$')
histories(ca, 'cmp2_cant', r'$J(\phi_k)$')
print(f'Wrote the panels of Figures 3-8 to {OUT}')
