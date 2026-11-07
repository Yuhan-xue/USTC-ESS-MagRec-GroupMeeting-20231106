"""Key-time snapshot figures for the report.  python snapshots.py eq|mer"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from magviz import Plane, render, draw_earth, PANELS, RMIN, DX

OUT = 'output/figs'
os.makedirs(OUT, exist_ok=True)
PANELS['Wz'] = (r'$\omega_z$ (10$^{-3}$ s$^{-1}$)', 'RdBu_r', lambda: TwoSlopeNorm(0, -20, 20))
PANELS['Wy'] = (r'$\omega_y$ (10$^{-3}$ s$^{-1}$)', 'RdBu_r', lambda: TwoSlopeNorm(0, -20, 20))
PANELS['Vx8'] = (r'$V_x$ (km/s)', 'RdBu_r', lambda: TwoSlopeNorm(0, -800, 800))
PANELS['Jy15'] = (r'$J_y$ (nA/m$^2$)', 'PuOr_r', lambda: TwoSlopeNorm(0, -15, 15))
PANELS['Bz50'] = (r'$B_z$ (nT)', 'RdBu_r', lambda: TwoSlopeNorm(0, -50, 50))
PANELS['Bx50'] = (r'$B_x$ (nT)', 'RdBu_r', lambda: TwoSlopeNorm(0, -50, 50))
PANELS['Bz10'] = (r'$B_z$ (nT)', 'RdBu_r', lambda: TwoSlopeNorm(0, -10, 10))
PANELS['Jy5'] = (r'$J_y$ (nA/m$^2$)', 'PuOr_r', lambda: TwoSlopeNorm(0, -5, 5))
PANELS['Epar'] = (r'$E_\parallel$ (mV/m)', 'PiYG_r', lambda: TwoSlopeNorm(0, -1.5, 1.5))
RE_KM = 6371.0


def nearest(pl, t):
    fr = np.array(pl.frames())
    tt = 0.01 + 0.25 * (fr - 1)
    return int(fr[np.argmin(abs(tt - t))])


def extra(pl, d):
    tn = pl.tname
    # vorticity component normal to the plane, (1/s) * 1e3
    from scipy.ndimage import gaussian_filter
    vt = gaussian_filter(d['V' + tn], 3)
    vx = gaussian_filter(d['Vx'], 3)
    dvt_dx = np.gradient(vt, DX, axis=0)
    dvx_dt = np.gradient(vx, DX, axis=1)
    w = (dvt_dx - dvx_dt) / RE_KM * 1e3
    d['Wz' if tn == 'y' else 'Wy'] = w if tn == 'y' else -w
    d['Vx8'] = d['Vx']; d['Jy15'] = d['Jy']; d['Bz50'] = d['Bz']; d['Bx50'] = d['Bx']; d['Bz10'] = d['Bz']; d['Jy5'] = d['Jy']
    B = np.stack([d['Bx'], d['By'], d['Bz']])
    E = np.stack([d['Ex'], d['Ey'], d['Ez']])
    d['Epar'] = (E * B).sum(0) / np.clip(d['Bmag'], 1e-3, None)
    return d


def sequence(pl, times, keys, fname, title, fieldlines=False, xlim=(-45, 22), tlim=(-30, 30)):
    """rows = keys, cols = times"""
    nc, nr = len(times), len(keys)
    w = 3.2
    asp = (tlim[1] - tlim[0]) / (xlim[1] - xlim[0])
    fig, axs = plt.subplots(nr, nc, figsize=(w * nc + 1.2, w * asp * nr + 0.8), constrained_layout=True, squeeze=False)
    inner = pl.R < RMIN
    ext = (pl.xu[0], pl.xu[-1], pl.tu[0], pl.tu[-1])
    for j, t in enumerate(times):
        fr = nearest(pl, t)
        d = extra(pl, pl.load(fr))
        lines = pl.trace_fieldlines(d) if fieldlines else None
        for i, k in enumerate(keys):
            ax = axs[i, j]
            lab, cmap, norm = PANELS[k]
            nrm = norm()
            a = d[k]
            if isinstance(nrm, LogNorm):
                a = np.clip(a, nrm.vmin, nrm.vmax)
            im = ax.imshow(np.ma.array(a, mask=inner).T, origin='lower', extent=ext, cmap=cmap, norm=nrm,
                           interpolation='nearest', aspect='equal')
            if lines is not None:
                for q in lines:
                    ax.plot(q[:, 0], q[:, 1], 'k', lw=0.3, alpha=0.7)
            draw_earth(ax)
            ax.set_xlim(*xlim); ax.set_ylim(*tlim)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(rf'$t$ = {d["time"]:.1f} $\Omega_i^{{-1}}$ (~{d["tsec"]/60:.1f} min)', fontsize=9)
            if j == 0:
                ax.set_ylabel(rf'${pl.tname}$ ($R_E$)', fontsize=8)
            if i == nr - 1:
                ax.set_xlabel(r'$x$ ($R_E$)', fontsize=8)
            if j == nc - 1:
                cb = fig.colorbar(im, ax=axs[i, :], shrink=0.9, pad=0.01)
                cb.set_label(lab, fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print('wrote', fname)


def zoom(pl, t, keys, ncols, fname, xlim, tlim, fieldlines=None, flowlines=None, title=None, panel_w=4.3):
    fr = nearest(pl, t)
    d = extra(pl, pl.load(fr))
    if title is None:
        title = ''
    title += rf'   $t$ = {d["time"]:.1f} $\Omega_i^{{-1}}$'
    render(pl, d, keys, ncols, fname, title=title, dpi=110, xlim=xlim, tlim=tlim, fieldlines=fieldlines,
           flowlines=flowlines, panel_w=panel_w)
    print('wrote', fname)


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'eq':
        eq = Plane('eq')
        sequence(eq, [0, 8, 16, 25, 50, 80, 110, 170], ['N', 'Bz', 'Vx', 'T'], f'{OUT}/seq_eq.png',
                 'Equatorial plane: formation and evolution of the magnetosphere')
        for t in (120, 150, 180):
            zoom(eq, t, ['N', 'Bz50', 'Vx', 'Wz', 'T', 'Tani'], 3, f'{OUT}/khi_eq_t{t}.png',
                 (-35, 8), (-24, 24), flowlines=['N'], title='Flank magnetopause / KHI (equatorial)')
        for t in (85, 125, 160):
            zoom(eq, t, ['N', 'Bz', 'Vx8', 'Vy', 'T', 'Ey'], 3, f'{OUT}/tail_eq_t{t}.png',
                 (-45, -5), (-20, 20), flowlines=['Vx8'], title='Tail plasma sheet (equatorial)')
        zoom(eq, 150, ['Tani', 'beta', 'T', 'N'], 2, f'{OUT}/sheath_eq_t150.png', (-20, 22), (-30, 30),
             title='Magnetosheath ion anisotropy (equatorial)')
    else:
        me = Plane('mer')
        sequence(me, [0, 8, 16, 25, 50, 80, 110, 170], ['N', 'Bx', 'Vx', 'T'], f'{OUT}/seq_mer.png',
                 'Noon–midnight meridian plane: formation and evolution', fieldlines=True)
        for t in [float(a) for a in sys.argv[2:]] or (100, 140, 170):
            zoom(me, t, ['N', 'Bx50', 'Bz50', 'Vz', 'Jy15', 'Tani'], 3, f'{OUT}/day_mer_t{t:.0f}.png',
                 (2, 18), (-15, 15), fieldlines=['N', 'Jy15', 'Bz50'], title='Dayside magnetopause (meridian)',
                 panel_w=3.6)
            zoom(me, t, ['N', 'Bz10', 'Vx8', 'Jy5', 'T', 'Vz'], 3, f'{OUT}/tail_mer_t{t:.0f}.png',
                 (-45, -5), (-15, 15), fieldlines=['N', 'Bz10', 'Jy5'], title='Magnetotail (meridian)')
        zoom(me, 150, ['N', 'Tani', 'beta', 'T', 'Vx', 'Bmag'], 3, f'{OUT}/global_mer_t150.png', (-45, 22), (-30, 30),
             fieldlines=['N', 'Bmag'], title='Meridian plane overview')
