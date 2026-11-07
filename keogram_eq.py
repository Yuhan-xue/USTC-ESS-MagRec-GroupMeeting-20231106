"""Equatorial Sun–Earth x–t keograms: stretched full run and a t = 125–175 zoom."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from magviz import Plane, RMIN

OUT = 'output/figs'
CACHE = 'output/eq_keogram.npz'
KEYS = ['N', 'Bz', 'Vx', 'T']

eq = Plane('eq')
if os.path.exists(CACHE):
    c = np.load(CACHE)
    ts, K = c['t'], {k: c[k] for k in KEYS}
else:
    m = np.abs(eq.tu) < 0.3
    ts, K = [], {k: [] for k in KEYS}
    for fr in eq.frames():
        d = eq.load(fr)
        ts.append(d['time'])
        for k in KEYS:
            K[k].append(d[k][:, m].mean(axis=1))
    ts = np.array(ts)
    K = {k: np.array(v) for k, v in K.items()}
    np.savez(CACHE, t=ts, **K)
b = np.load('output/eq_timeseries.npz')
mp, bs = b['mp'], b['bs']
assert len(mp) == len(ts)

SPECS = [('N', r'$N_i$ (cm$^{-3}$)', 'turbo', LogNorm(0.1, 40)),
         ('Bz', r'$B_z$ (nT)', 'RdBu_r', SymLogNorm(10, vmin=-200, vmax=200)),
         ('Vx', r'$V_x$ (km/s)', 'RdBu_r', TwoSlopeNorm(0, -600, 600)),
         ('T', r'$T_i$ (eV)', 'inferno', LogNorm(3, 5000))]


def keogram(t0, t1, ylim, figsize, fname, title, tick):
    sel = (ts >= t0) & (ts <= t1)
    t = ts[sel]
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
    for ax, (k, lab, cmap, norm) in zip(axs, SPECS):
        a = K[k][sel]
        if isinstance(norm, LogNorm):
            a = np.clip(a, norm.vmin, norm.vmax)
        im = ax.pcolormesh(t, eq.xu, a.T, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
        ax.axhspan(-RMIN, RMIN, color='0.6')
        ax.plot(t, mp[sel], 'w-', lw=0.9, label='MP')
        ax.plot(t, bs[sel], 'k-', lw=0.9, label='BS')
        ax.set_ylim(*ylim)
        ax.set_ylabel(r'$x$ ($R_E$)')
        ax.set_xticks(np.arange(np.ceil(t0 / tick) * tick, t1 + 1e-6, tick))
        ax.grid(axis='x', color='k', alpha=0.15, lw=0.6)
        fig.colorbar(im, ax=ax, pad=0.005, aspect=12).set_label(lab)
    axs[0].legend(fontsize=9, loc='lower left')
    axs[-1].set_xlabel(r'$t$ ($\Omega_i^{-1}$);  1 unit ≈ 15.7 s')
    axs[0].set_title(title)
    axs[-1].set_xlim(t0, t1)
    fig.savefig(fname, dpi=110)
    plt.close(fig)


keogram(0, ts[-1], (eq.xu[0], eq.xu[-1]), (22, 12.5), f'{OUT}/fig_keogram_eq_sunearth.png',
        'Equatorial plane, Sun–Earth line (y = 0): x–t diagram, t = 0–189', 10)
keogram(125, 175, (eq.xu[0], eq.xu[-1]), (22, 12.5), f'{OUT}/fig_keogram_eq_sunearth_t125_175.png',
        'Equatorial plane, Sun–Earth line (y = 0): x–t diagram, zoom t = 125–175', 5)

# quick numbers for the text: earthward/tailward fast-flow occupancy in the zoom window
sel = (ts >= 125) & (ts <= 175)
tail = (eq.xu < -12) & (eq.xu > -40)
vx = K['Vx'][sel][:, tail]
for lo, hi in ((-15, -25), (-25, -35)):
    w = (eq.xu[tail] <= lo) & (eq.xu[tail] > hi)
    v = vx[:, w]
    print(f'x in ({hi},{lo}]: frac Vx>+300 {np.mean(v > 300):.2f}, Vx<-300 {np.mean(v < -300):.2f}, max {v.max():.0f}, min {v.min():.0f}')
xe = eq.xu[tail]
fast = vx > 300
peaks = [xe[np.argmax(row)] if row.max() > 300 else np.nan for row in vx]
print('earthward-peak x: median %.1f, IQR %.1f..%.1f' % tuple(np.nanpercentile(peaks, [50, 25, 75])))
print('frames', sel.sum(), 'MP mean %.2f BS mean %.2f' % (np.nanmean(mp[sel]), np.nanmean(bs[sel])))
