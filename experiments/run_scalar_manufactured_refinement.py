"""Two-dimensional disk benchmark for the scalar surface-penalty DDM.

The sharp Dirichlet solution is T=R^2-x^2-y^2 on the disk, with f=4 and
T=0 on its boundary.  The box solve uses the same cell-centred face weights,
discrete phase gradient, surface penalty, and void floor as the optimization.
Run from anywhere with: python run_scalar_manufactured_refinement.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


HALF_BOX = 0.12
RADIUS = 0.075
H_LEVELS = (0.004, 0.002, 0.001, 0.0005)
EPS_EXPERIMENT = 0.008
KMIN = 1e-6


def solve(h: float, epsilon: float):
    n = round(2 * HALF_BOX / h)
    assert abs(n * h - 2 * HALF_BOX) < 1e-12
    x = -HALF_BOX + (np.arange(n) + 0.5) * h
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rad = np.hypot(xx, yy)
    phi = 0.5 * (1 - np.tanh(3 * (rad - RADIUS) / epsilon))
    p = phi.ravel()
    # This reproduces the central difference D_x,D_y in scalar_surface_problem.
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * h)
    gy[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * h)
    sigma = 1e-5 / epsilon
    delta = np.sqrt(gx**2 + gy**2 + sigma**2) - sigma
    beta = 0.1 * epsilon**0.75
    diag = (delta.ravel() / beta).copy()
    rhs = 4 * p
    rows = []
    cols = []
    data = []
    idx = np.arange(n * n).reshape(n, n)
    for axis in (0, 1):
        if axis == 0:
            left, right = idx[:-1, :].ravel(), idx[1:, :].ravel()
        else:
            left, right = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        face = (KMIN + (1 - KMIN) * 0.5 * (p[left] + p[right])) / h**2
        diag[left] += face
        diag[right] += face
        rows.extend((left, right))
        cols.extend((right, left))
        data.extend((-face, -face))
    # The manuscript's scalar G gives one boundary-edge contribution per cell.
    edge = np.concatenate((idx[0, :], idx[-1, :], idx[:, 0], idx[:, -1]))
    np.add.at(diag, edge, (KMIN + (1 - KMIN) * p[edge]) / h**2)
    rows.append(np.arange(n * n))
    cols.append(np.arange(n * n))
    data.append(diag)
    matrix = coo_matrix((np.concatenate(data),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n * n, n * n)).tocsr()
    state = spsolve(matrix, rhs).reshape(n, n)
    exact = np.maximum(RADIUS**2 - rad**2, 0)
    return x, rad, state, exact


def sample_fine(xfine: np.ndarray, fine: np.ndarray, xcoarse: np.ndarray):
    # Coarse cell centres are halfway between two fine centres at every level.
    h = xfine[1] - xfine[0]
    t = (xcoarse - xfine[0]) / h
    lo = np.floor(t).astype(int)
    alpha = t - lo
    assert np.all(lo >= 0) and np.all(lo + 1 < len(xfine))
    a = fine[np.ix_(lo, lo)]
    b = fine[np.ix_(lo + 1, lo)]
    c = fine[np.ix_(lo, lo + 1)]
    d = fine[np.ix_(lo + 1, lo + 1)]
    wx = alpha[:, None]
    wy = alpha[None, :]
    return (1 - wx) * (1 - wy) * a + wx * (1 - wy) * b + (1 - wx) * wy * c + wx * wy * d


def relative_error(state: np.ndarray, reference: np.ndarray, mask: np.ndarray):
    return float(np.linalg.norm((state - reference)[mask]) / np.linalg.norm(reference[mask]))


def main():
    records = []
    for study in ("fixed-epsilon", "epsilon=4h"):
        fields = []
        for h in H_LEVELS:
            eps = EPS_EXPERIMENT if study == "fixed-epsilon" else 4 * h
            x, rad, state, exact = solve(h, eps)
            fields.append((x, rad, state, exact))
        xref, _, uref, _ = fields[-1]
        for h, (x, rad, state, exact) in zip(H_LEVELS, fields):
            eps = EPS_EXPERIMENT if study == "fixed-epsilon" else 4 * h
            mask = rad < RADIUS
            sharp_error = relative_error(state, exact, mask)
            diffuse_error = float("nan") if h == H_LEVELS[-1] else relative_error(
                state, sample_fine(xref, uref, x), mask)
            record = dict(model="scalar", study=study, h=h, epsilon=eps,
                          eps_over_h=eps / h, sharp_error=sharp_error,
                          fixed_epsilon_error=diffuse_error)
            records.append(record)
            print(record, flush=True)
    out = Path(__file__).resolve().parents[1] / "results" / "scalar_manufactured_2d.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
