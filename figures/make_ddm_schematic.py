"""Figure 1: the diffuse-domain schematic of Section 2.1.

Run from any directory with: python make_ddm_schematic.py
Writes results/figures/ddm_schematic.pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.25),
                                 gridspec_kw={"width_ratios": [1.18, 1]})
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.17, top=0.93,
                        wspace=0.25)

    # A circular example makes gamma the exact signed distance to Gamma.
    x = np.linspace(0, 2.75, 700)
    y = np.linspace(0, 1.8, 460)
    X, Y = np.meshgrid(x, y)
    cx, cy, radius, eps = 1.28, 0.90, 0.57, 0.19
    gamma = np.hypot(X - cx, Y - cy) - radius
    phi = 0.5 * (1 - np.tanh(3 * gamma / eps))
    cmap = LinearSegmentedColormap.from_list(
        "phase", ["#f8fafc", "#d5e5e9", "#76bac3", "#22758b"])
    ax.imshow(phi, extent=(0, 2.75, 0, 1.8), origin="lower", cmap=cmap,
              vmin=0, vmax=1, interpolation="bilinear")
    ax.contour(X, Y, phi, levels=[0.5], colors=["#bd5a2b"], linewidths=1.65)
    ax.add_patch(Rectangle((0, 0), 2.75, 1.8, facecolor="none",
                           edgecolor="#263746", linewidth=1.3))
    ax.text(0.12, 1.63, r"$D$", fontsize=15, color="#263746")
    ax.text(cx - 0.17, cy - 0.07, r"$\Omega$", fontsize=16,
            color="white", weight="bold")
    ax.text(2.14, 0.32, r"$\phi\approx 0$", color="#263746")
    ax.text(1.04, 0.57, r"$\phi\approx 1$", color="white")
    ax.annotate(r"$\Gamma:\ \phi=\frac{1}{2}$", xy=(cx + radius * 0.72,
                cy + radius * 0.69), xytext=(1.92, 1.52),
                color="#873a19", ha="left",
                arrowprops={"arrowstyle": "->", "color": "#873a19",
                            "lw": 1.25})
    ax.set_xlim(-0.03, 2.78)
    ax.set_ylim(-0.03, 1.83)
    ax.set_aspect("equal")
    ax.axis("off")

    s = np.linspace(-1.5, 1.5, 500)
    p = 0.5 * (1 - np.tanh(3 * s))
    bx.axvspan(-0.5, 0.5, facecolor="#deedf0", alpha=0.9, zorder=0)
    bx.plot(s, p, color="#176c80", lw=2.3)
    bx.axvline(0, color="#bd5a2b", lw=1.2, ls="--")
    bx.axhline(0.5, color="#7a8790", lw=0.85, ls=":")
    bx.annotate("", xy=(-0.5, 0.14), xytext=(0.5, 0.14),
                arrowprops={"arrowstyle": "<->", "lw": 1.2,
                            "color": "#263746"})
    bx.text(0, 0.19, r"$\varepsilon$", ha="center", va="bottom",
            color="#263746")
    bx.text(-1.30, 0.88, r"$\Omega$", color="#263746")
    bx.text(0.78, 0.88, r"outside $\Omega$", color="#263746")
    bx.set_xlim(-1.5, 1.5)
    bx.set_ylim(-0.02, 1.04)
    bx.set_xticks([-1, -0.5, 0, 0.5, 1])
    bx.set_xticklabels([r"$-1$", r"$-\frac{1}{2}$", r"$0$",
                        r"$\frac{1}{2}$", r"$1$"])
    bx.set_yticks([0, 0.5, 1])
    bx.set_xlabel(r"signed distance $\gamma/\varepsilon$")
    bx.set_ylabel(r"$\phi$")
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ddm_schematic.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
