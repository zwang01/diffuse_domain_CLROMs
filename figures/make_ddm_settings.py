"""Figure 2: boundary conditions and phase-field conventions of the three problems.

Run from any directory with: python make_ddm_settings.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt

DARK = "#263746"
RED = "#bd5a2b"
CMAP = LinearSegmentedColormap.from_list(
    "phase", ["#f8fafc", "#d5e5e9", "#76bac3", "#22758b"])
X0, X1 = -1.05, 8.25          # common horizontal range of every panel
XT = 4.75                     # left edge of the condition lists
WIDTH = 8.0                   # figure width in inches
EPS = 0.16                    # interface width of the drawing (not to scale)


def phase(inside, lx, ly, n=900):
    """Equilibrium profile of a region given as a vectorized predicate."""
    nx, ny = n, int(n * ly / lx)
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    m = inside(X, Y)
    dx = lx / (nx - 1)
    gamma = (distance_transform_edt(~m) - distance_transform_edt(m)) * dx
    return X, Y, 0.5 * (1 - np.tanh(3 * gamma / EPS))


def draw_phase(ax, X, Y, phi, lx, ly):
    ax.imshow(phi, extent=(0, lx, 0, ly), origin="lower", cmap=CMAP,
              vmin=0, vmax=1, interpolation="bilinear")
    ax.contour(X, Y, phi, levels=[0.5], colors=[RED], linewidths=1.5)
    ax.add_patch(Rectangle((0, 0), lx, ly, facecolor="none",
                           edgecolor=DARK, linewidth=1.3))


def conditions(ax, y0, lines, dy=0.24):
    for k, s in enumerate(lines):
        ax.text(XT, y0 - k * dy, s, color=DARK, va="center", ha="left")


def arrow(ax, text, xy, xytext, color=DARK, **kw):
    ax.annotate(text, xy=xy, xytext=xytext, color=color, ha="center",
                va="center", arrowprops={"arrowstyle": "->", "color": color,
                                         "lw": 1.1, "shrinkA": 2,
                                         "shrinkB": 1}, **kw)


def shape_identification(ax):
    lx, ly = 4.0, 1.0
    t = np.linspace(0, 2 * np.pi, 400)
    bx = 2.0 + 1.3 * np.cos(t) * (1 + 0.06 * np.sin(2 * t))
    by = 0.5 + 0.27 * np.sin(t) * (1 + 0.15 * np.cos(3 * t))
    blob = MplPath(np.column_stack([bx, by]))

    def inside(X, Y):
        pts = np.column_stack([X.ravel(), Y.ravel()])
        return blob.contains_points(pts).reshape(X.shape)

    X, Y, phi = phase(inside, lx, ly)
    draw_phase(ax, X, Y, phi, lx, ly)
    ax.text(0.08, 0.83, r"$D$", fontsize=13, color=DARK)
    ax.text(1.55, 0.5, r"$\Omega$:  $\phi\approx1$", color="white",
            va="center")
    ax.text(3.45, 0.12, r"$\phi\approx0$", color=DARK)
    arrow(ax, r"$\Gamma_f$", xy=(2.95, 0.66), xytext=(3.45, 0.86))
    arrow(ax, r"$\partial D$", xy=(4.0, 0.62), xytext=(4.35, 0.85))
    conditions(ax, 0.78, [
        r"$\Gamma_f=\partial\Omega$:  $u=0$, imposed by the",
        r"      surface penalty $\lambda_\varepsilon\delta_\sigma(\phi)\,u$",
        r"$\partial D$ (all four edges):  $u=0$",
    ])
    return -0.15, ly + 0.42, "(a) Shape identification"


def channel_flow(ax):
    lx, ly = 4.0, 1.0
    cx, cy, r = 1.0, 0.5, 0.23
    X, Y, phi = phase(lambda X, Y: np.hypot(X - cx, Y - cy) > r, lx, ly)
    draw_phase(ax, X, Y, phi, lx, ly)
    # parabolic inflow profile
    ys = np.linspace(0.08, 0.92, 7)
    for yy in ys:
        L = 0.42 * (1 - (2 * yy - 1) ** 2)
        ax.annotate("", xy=(-0.04, yy), xytext=(-0.04 - L, yy),
                    arrowprops={"arrowstyle": "->", "color": DARK,
                                "lw": 0.9, "shrinkA": 0, "shrinkB": 0})
    yp = np.linspace(0, 1, 100)
    ax.plot(-0.04 - 0.42 * (1 - (2 * yp - 1) ** 2), yp, color=DARK, lw=0.8)
    ax.text(-0.78, 0.5, r"$\Gamma_i$", color=DARK, va="center", ha="center")
    ax.text(4.2, 0.5, r"$\Gamma_o$", color=DARK, va="center", ha="center")
    ax.text(3.3, ly + 0.11, r"$\Gamma_w$", color=DARK, va="center",
            ha="center")
    ax.text(3.3, -0.12, r"$\Gamma_w$", color=DARK, va="center", ha="center")
    ax.text(cx, cy, r"$\phi\approx0$", color=DARK, fontsize=8.5,
            va="center", ha="center")
    ax.text(2.55, 0.42, r"$\Omega$ (fluid):  $\phi\approx1$", color="white",
            va="center", ha="center")
    arrow(ax, r"$\Gamma_f$", xy=(cx + 0.17, cy + 0.17), xytext=(1.6, 0.82),
          color="white")
    conditions(ax, 0.88, [
        r"$\Gamma_i$ (left edge):  $\mathbf{u}=\mathbf{g}$",
        r"$\Gamma_w$ (top and bottom edges):  $\mathbf{u}=\mathbf{0}$",
        r"$\Gamma_o$ (right edge):  $\nu\partial_{\mathbf{n}}\mathbf{u}-p\,\mathbf{n}=\mathbf{0}$",
        r"$\Gamma_f$:  $\mathbf{u}=\mathbf{0}$, imposed by the",
        r"      volume penalty $\alpha_\varepsilon(1-\phi)^2\mathbf{u}$",
    ])
    return -0.26, ly + 0.45, "(b) Drag minimization in a channel"


def cantilever(ax):
    lx, ly = 2.6, 1.3
    holes = [(0.95, 0.36), (1.75, 0.36), (1.35, 0.94), (2.15, 0.94)]
    r = 0.17

    def inside(X, Y):
        m = np.ones_like(X, dtype=bool)
        for hx, hy in holes:
            m &= np.hypot(X - hx, Y - hy) > r
        return m

    X, Y, phi = phase(inside, lx, ly)
    draw_phase(ax, X, Y, phi, lx, ly)
    # clamp
    ax.plot([0, 0], [0, ly], color=DARK, lw=2.2)
    for yy in np.linspace(0.0, ly - 0.1, 9):
        ax.plot([-0.1, 0], [yy, yy + 0.1], color=DARK, lw=0.9)
    ax.text(-0.42, ly / 2, r"$\Gamma_D$", color=DARK, va="center",
            ha="center")
    # point load at the midpoint of the right edge
    ax.annotate("", xy=(lx, ly / 2), xytext=(lx, ly / 2 + 0.42),
                arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 1.6,
                            "shrinkA": 0, "shrinkB": 0})
    ax.plot([lx], [ly / 2], "o", color=DARK, ms=4.5, zorder=5)
    ax.text(lx + 0.12, ly / 2 + 0.3, r"$\Gamma_N$", color=DARK, va="center",
            ha="left")
    ax.text(lx + 0.12, ly / 2 + 0.05, r"$\mathbf{t}$", color=DARK,
            va="center", ha="left")
    ax.text(0.38, ly / 2, r"$\Omega$", color="white", va="center",
            ha="center", fontsize=13)
    ax.text(0.38, ly / 2 - 0.3, r"$\phi\approx1$", color="white",
            va="center", ha="center", fontsize=9.5)
    arrow(ax, r"$\phi\approx0$", xy=(0.95, 0.30), xytext=(0.55, -0.17))
    arrow(ax, r"$\Gamma_f$", xy=(1.35 + 0.12, 0.94 - 0.12),
          xytext=(1.62, 0.64), color="white")
    conditions(ax, 1.1, [
        r"$\Gamma_D$ (left edge):  $\mathbf{u}=\mathbf{0}$",
        r"$\Gamma_N$ (midpoint of right edge):  point load $\mathbf{t}$",
        r"$\Gamma_f$:  traction free, through the",
        r"      stiffness weight $k(\phi)$; no penalty",
        r"other edges of $D$:  traction free",
    ])
    return -0.3, ly + 0.42, "(c) Compliance minimization of a cantilever"


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
    })
    panels = [shape_identification, channel_flow, cantilever]
    scale = WIDTH / (X1 - X0)                 # inches per drawing unit
    fig = plt.figure(figsize=(WIDTH, 1.0))
    axes, spans = [], []
    for draw in panels:
        ax = fig.add_axes((0, 0, 1, 1))
        y0, y1, title = draw(ax)
        ax.text(X0 + 0.02, y1 - 0.08, title, color=DARK, va="top",
                ha="left", fontsize=11.5, weight="bold")
        ax.set_xlim(X0, X1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        axes.append(ax)
        spans.append((y1 - y0) * scale)
    gap = 0.12
    height = sum(spans) + gap * (len(spans) - 1)
    fig.set_size_inches(WIDTH, height)
    top = height
    for ax, h in zip(axes, spans):
        top -= h
        ax.set_position((0, top / height, 1, h / height))
        top -= gap
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ddm_settings.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
