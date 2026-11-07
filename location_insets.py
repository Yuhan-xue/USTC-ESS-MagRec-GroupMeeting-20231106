"""English location keys for the edited deck; coloured regions are sampling keys."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

OUT = Path("output/figs")
X = np.linspace(-45, 22, 671)
Z = np.linspace(-30, 30, 601)
ORANGE = "#d45d21"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def background(ax, plane):
    d = np.load(f"output/cache/avg_{plane}.npz")
    ax.pcolormesh(X, Z, d["N"].T, cmap="Greys", norm=LogNorm(.08, 25),
                  alpha=.40, shading="auto", rasterized=True)
    theta = np.linspace(-2.2, 2.2, 400)
    r0, alpha = (11.50, .411) if plane == "eq" else (10.84, .454)
    r = r0 * (2 / (1 + np.cos(theta))) ** alpha
    ax.plot(r * np.cos(theta), r * np.sin(theta), color="#6e7c89", lw=.65)
    ax.add_patch(Circle((0, 0), 2.1, fc="#22384b", ec="white", lw=.5, zorder=5))
    ax.annotate("", (12, 21), (21, 21), arrowprops={"arrowstyle": "->", "lw": .7})
    ax.text(21, 23, "SW", ha="right", fontsize=6)
    ax.text(-42, -24, "Tail", fontsize=6, color="#526272")
    ax.text(1, -7, "Earth", fontsize=6, color="#22384b")
    ax.set(xlim=(-43, 22), ylim=(-26, 27), aspect="equal")
    ax.axis("off")
    return d


def make(n):
    fig, ax = plt.subplots(figsize=(2.3, 1.22))
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.17)
    plane = "mer" if n in (15, 17, 19) else "eq"
    d = background(ax, plane)
    label = ""
    if n == 7:
        ax.plot([-42, 21], [0, 0], color=ORANGE, lw=2)
        label = "Equatorial | Sun-Earth cut"
    elif n == 8:
        ax.plot([4, 21], [0, 0], color=ORANGE, lw=2.3)
        ax.scatter([11.5, 14.2], [0, 0], color=ORANGE, s=12)
        ax.text(9, 8, "MP / BS", fontsize=7, color=ORANGE, ha="center")
        label = "Equatorial | dayside nose"
    elif n == 11:
        # Two planes are shown separately below.
        plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(2.3, 1.22))
        fig.subplots_adjust(left=.01, right=.99, top=.83, bottom=.27, wspace=.06)
        for a, pl in zip(axs, ("eq", "mer")):
            dd = background(a, pl)
            xx, zz = np.meshgrid(X, Z, indexing="ij")
            mask = ((dd["N"] > 1.5 * 4.01) & (dd["T"] > 4 * 6.96)
                    & (np.hypot(xx, zz) > 8) & (xx > -10) & (xx < 20))
            a.contourf(X, Z, mask.T.astype(float), levels=[.5, 1.5],
                       colors=[ORANGE], alpha=.7)
            a.set_title("Equatorial" if pl == "eq" else "Meridian", fontsize=6.5, pad=0)
        label = "Orange: magnetosheath samples"
    elif n == 14:
        for s in (-1, 1):
            ax.plot([-20, 4], [s * 18, s * 12], color=ORANGE, lw=3, alpha=.8)
        ax.text(-18, 22, "dusk (+y)", fontsize=6, color=ORANGE)
        ax.text(-18, -18, "dawn (-y)", fontsize=6, color=ORANGE)
        label = "Equatorial | flank strips (schematic)"
    elif n == 16:
        ax.plot([-43, -6], [0, 0], color=ORANGE, lw=2.4)
        ax.scatter([-17], [0], color=ORANGE, marker="s", s=15)
        ax.text(-16, 8, "braking", color=ORANGE, fontsize=6)
        label = "Tail-axis cuts | eq. + mer."
    elif n in (15, 19):
        ax.add_patch(Rectangle((-33, -4), 13, 8, fc=ORANGE, alpha=.65))
        ax.text(-26, 10, "X-line window", color=ORANGE, fontsize=6, ha="center")
        label = "Meridian | local tail window"
    elif n == 17:
        ax.add_patch(Rectangle((-40, -2), 28, 4, fc=ORANGE, alpha=.75))
        ax.text(-27, 8, "X-line search", color=ORANGE, fontsize=6, ha="center")
        label = "Meridian | -40 < x < -12"
    fig.text(.5, .04, label, ha="center", va="bottom", fontsize=6.8, color="#22384b")
    fig.savefig(OUT / f"loc_s{n:02d}.png", dpi=320, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for slide in (7, 8, 11, 14, 15, 16, 17, 19):
        make(slide)
