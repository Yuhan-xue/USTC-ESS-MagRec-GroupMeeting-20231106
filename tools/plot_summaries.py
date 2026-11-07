"""Plot two small diagnostics using only the published reduced tables."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "output/figs"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "processed/eq_timeseries.npz", allow_pickle=False) as series:
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        ax.plot(series["t"], series["mp"], label="Magnetopause: Bz sign")
        ax.plot(series["t"], series["bs"], label="Bow shock: density threshold")
        ax.set(xlabel="Simulation time (code units)",
               ylabel="Subsolar distance (Earth radii)",
               title="Reduced equatorial boundary time series")
        ax.legend()
        ax.grid(alpha=.25)
        fig.savefig(out / "summary_boundaries.png", dpi=150)
        plt.close(fig)
    with (ROOT / "processed/reconnection_candidates.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r["cls"] == "clean"]
    keys = ("EN", "ES", "TN", "TS")
    mean = [np.mean([float(row[k]) for row in selected]) for k in keys]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axs[0].bar(keys, mean, color=["#b94332", "#427a9b", "#427a9b", "#b94332"])
    axs[0].axhline(0, color="black", lw=.8)
    axs[0].set(ylabel="Background-subtracted By (nT)",
               title="28 flow-selected frames: quadrant means")
    fractions = [
        sum(r["full"] == "True" for r in selected) / len(selected),
        sum(r["full"] == "True" for r in rows) / len(rows),
    ]
    axs[1].bar(["Selected: 21/28", "All candidates: 27/90"], fractions)
    axs[1].set(ylim=(0, 1), ylabel="Fraction with all four expected signs",
               title="Correlated frames, not independent events")
    fig.savefig(out / "summary_hall.png", dpi=150)
    plt.close(fig)
    print("Wrote output/figs/summary_boundaries.png and summary_hall.png; no raw input.")


if __name__ == "__main__":
    main()
