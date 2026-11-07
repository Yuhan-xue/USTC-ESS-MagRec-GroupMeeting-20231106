"""Presentation figures from the cached, independently flow-selected X-line sample."""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

C = pickle.load(open("output/cache/recon_redo.pkl", "rb"))
S = json.loads(Path("output/recon_redo.json").read_text())
OUT = Path("output/figs")
gx, gz = np.linspace(-6, 6, 121), np.linspace(-4, 4, 81)
GX, GZ = np.meshgrid(gx, gz, indexing="ij")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "font.family": "DejaVu Sans"})
clean = [r for r in C["rows"] if r["cls"] == "clean"]
assert len(clean) == S["classes"]["clean"]["n"]
assert sum(r["full"] for r in clean) == S["classes"]["clean"]["full_quadrupole"]


def bar(ax, im, label):
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=.08)
    plt.colorbar(im, cax=cax).set_label(label)


def lines(ax, x, z, bx, bz):
    ax.streamplot(x, z, bx.T, bz.T, density=.85, color="#34404c", linewidth=.5, arrowsize=.5)


def hall_maps():
    fig = plt.figure(figsize=(14.2, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .95],
                          left=.05, right=.985, bottom=.18, top=.87, wspace=.48)
    for i, d in enumerate((C["snap"]["w"], C["comp"])):
        ax = fig.add_subplot(gs[i])
        by = d["By"].copy()
        if i == 0:
            by -= np.median(by[(abs(GX) < 5) & (abs(GZ) < 3)])
        im = ax.pcolormesh(gx, gz, by.T, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -1.5, 1.5), shading="auto")
        lines(ax, gx, gz, d["Bx"], d["Bz"])
        ax.plot(0, 0, "x", color="#f0c400", ms=12, mew=3)
        # Boxes are exactly the regions used for the numerical sign test.
        for sx in (-1, 1):
            for sz in (-1, 1):
                x, z = (.5 if sx == 1 else -3), (.2 if sz == 1 else -1.5)
                ax.add_patch(Rectangle((x, z), 2.5, 1.3, fc="none",
                                       ec="#273747", ls=":", lw=.9))
        ax.set(xlim=(-6, 6), ylim=(-4, 4), aspect="equal",
               xlabel=r"$x-x_X$ ($R_E$)     Earthward $\rightarrow$",
               ylabel=r"$z-z_X$ ($R_E$)")
        ax.set_title("(a) t = 122: example snapshot" if i == 0 else
                     f"(b) {len(clean)} flow-selected frames", loc="left")
        bar(ax, im, r"$B_y-B_{y,bg}$ (nT)")
    ax = fig.add_subplot(gs[2])
    keys = ["EN", "ES", "TN", "TS"]
    snap = S["snapshot_t122"]
    vals = [snap[k] for k in keys]
    means = [S["classes"]["clean"]["By_mean"][k] for k in keys]
    x = np.arange(4)
    ax.bar(x - .18, vals, .34, color="#c44b32", label="t = 122")
    ax.bar(x + .18, means, .34, color="#547f9f", label="selected mean")
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(x, ["E/N\n+", "E/S\n-", "T/N\n-", "T/S\n+"])
    ax.set_ylim(-1.6, 1.8)
    ax.set_ylabel(r"quadrant mean $B_y-B_{y,bg}$ (nT)")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("(c) measured quadrant means", loc="left")
    ax.grid(axis="y", alpha=.2)
    fig.text(.73, .035, "E/T = Earthward/Tailward; N/S = North/South\n"
             "Signs under labels: ideal Hall prediction", fontsize=9)
    fig.savefig(OUT / "hall_quadrupole_v2.png", dpi=220)
    plt.close(fig)


def anatomy():
    snap = C["snap"]
    x = np.linspace(-45, 22, 671)[snap["sx"]]
    z = np.linspace(-30, 30, 601)[snap["sz"]]
    d = snap["f"]
    x0, z0 = snap["r"]["x0"], snap["r"]["z0"]
    fig, axs = plt.subplots(3, 1, figsize=(8.6, 5.45),
                            gridspec_kw={"height_ratios": [1, 1, .8]})
    fig.subplots_adjust(left=.105, right=.87, bottom=.09, top=.94, hspace=.45)
    for ax, key, lim, title in zip(axs[:2], ["Vx", "Jy"], [800, 3],
                                   [r"(a) $V_x$ (km/s): two-sided jets, t = 122",
                                    r"(b) $J_y$ (nA/m$^2$): current sheet"]):
        im = ax.pcolormesh(x, z, d[key].T, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
        lines(ax, x, z, d["Bx"], d["Bz"])
        ax.plot(x0, z0, "x", color="#c59c00", ms=12, mew=2.5)
        ax.axvline(x0, color="#886800", ls="--", lw=.8)
        ax.set(xlim=(-42, -8), ylim=(-4, 4), ylabel=r"z ($R_E$)")
        ax.set_title(title, loc="left")
        ax.tick_params(labelbottom=False)
        bar(ax, im, "")
    ax = axs[2]
    xx = np.linspace(-45, 22, 671)
    m = (xx > -42) & (xx < -8)
    ax.plot(xx[m], snap["bz"][m], "k", lw=1.6)
    ax.axhline(0, color=".5", lw=.6)
    ax.axvline(x0, color="#886800", ls="--", lw=1)
    ax.set(xlim=(-42, -8), ylim=(-8, 14), ylabel=r"$B_z$ (nT)",
           xlabel=r"x ($R_E$)     Earthward $\rightarrow$")
    twin = ax.twinx()
    twin.plot(xx[m], snap["vx"][m], color="#bd3426", lw=1.4)
    twin.set(ylim=(-800, 800), ylabel=r"$V_x$ (km/s)")
    twin.tick_params(axis="y", colors="#bd3426", labelsize=9)
    twin.yaxis.label.set_color("#bd3426")
    ax.set_title(r"(c) along $B_x=0$: $B_z$ (black), $V_x$ (red)", loc="left")
    fig.savefig(OUT / "recon_anatomy.png", dpi=220)
    plt.close(fig)


def diagnostics():
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.2))
    fig.subplots_adjust(left=.05, right=.985, bottom=.2, top=.88, wspace=.30)
    ax = axs[0]
    ax.axvspan(-30, -20, color="#eee0a1", alpha=.6, label="Nagai: -30 to -20")
    bins = np.arange(-41, -11, 1)
    ax.hist([r["x0"] for r in C["rows"]], bins=bins, color=".75", label="all candidates (90)")
    ax.hist([r["x0"] for r in clean], bins=bins, color="#bd482d", label="flow-selected (28)")
    ax.set(xlabel=r"X-line x ($R_E$)", ylabel="frames", title="(a) tail X-line location")
    ax.legend(fontsize=9, frameon=False)
    ax = axs[1]
    ey = np.array([r["Ey"] for r in clean])
    ax.hist(ey, bins=np.linspace(min(0, ey.min()), max(2.6, ey.max()), 15), color="#bd482d")
    ax.axvline(np.median(ey), color="k", ls="--", label=f"median {np.median(ey):.2f}")
    ax.set(xlabel=r"sheet-centred $E_y$ (mV/m)", ylabel="selected frames",
           title="(b) tail reconnection electric field")
    ax.legend(fontsize=10, frameon=False)
    ax = axs[2]
    vals = [S["classes"][k]["n"] for k in ("clean", "inactive", "other")]
    ax.bar(range(3), vals, color=["#bd482d", ".65", "#547f9f"])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha="center")
    ax.set_xticks(range(3), ["selected:\ntwo jets, NX<=2", "one-sided /\nweak jets", "two jets,\nNX>=3"])
    ax.set(ylabel="frames", ylim=(0, 65), title="(c) selection before inspecting By")
    ax.tick_params(axis="x", labelsize=9)
    fig.savefig(OUT / "recon_rate_v2.png", dpi=220)
    plt.close(fig)


def hall_timeseries():
    rows = C["rows"]
    t = np.array([r["t"] for r in rows])
    mask = np.array([r["cls"] == "clean" for r in rows])
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.subplots_adjust(left=.09, right=.97, top=.92, bottom=.12, hspace=.33)
    for ax, keys, title in zip(axs, [("EN", "ES"), ("TN", "TS")],
                               ["Earthward quadrants", "Tailward quadrants"]):
        for k, color in zip(keys, ("#b84132", "#286a9c")):
            v = np.array([r[k] for r in rows])
            ax.plot(t, v, color=color, alpha=.35, lw=1)
            ax.scatter(t[mask], v[mask], color=color, s=22,
                       label=k + (" (expected +)" if k in ("EN", "TS") else " (expected -)"))
        ax.axhline(0, color="k", lw=.65)
        ax.set(ylabel=r"$B_y-B_{y,bg}$ (nT)", title=title)
        ax.legend(ncol=2, fontsize=9, frameon=False)
        ax.grid(alpha=.2)
    axs[-1].set_xlabel(r"time ($\Omega_i^{-1}$); coloured dots = flow-selected frames")
    fig.savefig(OUT / "hall_quadrant_ts_v2.png", dpi=220)
    plt.close(fig)


def dayside():
    import magviz
    p = magviz.Plane("mer")
    # Frame numbers are output indices, not normalized times.
    frames = [(f, float(np.load(p.dir + f"/f{f:05d}.npz")["time"])) for f in p.frames()]
    f, t = min(frames, key=lambda a: abs(a[1] - 125))
    d = p.load(f)
    sx, sz = (p.xu > 5) & (p.xu < 17), abs(p.tu) < 12
    x, z = p.xu[sx], p.tu[sz]
    fig, axs = plt.subplots(1, 3, figsize=(10, 5.3))
    fig.subplots_adjust(left=.065, right=.92, bottom=.12, top=.90, wspace=.55)
    for ax, k, lim, title in zip(axs, ["Bz", "Vz", "Jy"], [40, 350, 12],
                                 [r"(a) reversing $B_z$", r"(b) north/south $V_z$", r"(c) current layer $J_y$"]):
        a = d[k][np.ix_(sx, sz)]
        im = ax.pcolormesh(x, z, a.T, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
        lines(ax, x, z, d["Bx"][np.ix_(sx, sz)], d["Bz"][np.ix_(sx, sz)])
        # A Bz=0 contour marks a candidate boundary, not a measured 3D separator.
        ax.contour(x, z, d["Bz"][np.ix_(sx, sz)].T, levels=[0], colors="#b18c00", linewidths=1)
        ax.set(xlim=(5, 17), ylim=(-12, 12), xlabel=r"x ($R_E$)")
        ax.set_title(title, fontsize=11, loc="left")
        bar(ax, im, {"Bz": "nT", "Vz": "km/s", "Jy": r"nA/m$^2$"}[k])
    axs[0].set_ylabel(r"z ($R_E$)")
    fig.suptitle(rf"Dayside meridian plane, t = {t:.0f} $\Omega_i^{{-1}}$", fontsize=12)
    fig.savefig(OUT / "recon_dayside_v2.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    hall_maps()
    anatomy()
    diagnostics()
    hall_timeseries()
    dayside()
