"""Reconnection redo: active ("clean") tail X-lines, meridian plane, t >= 100.

Per frame
  * current-sheet centre z_c(x): Bx = 0 nearest to z = 0
  * X-line: Bz along the sheet turns from - (tailward) to + (earthward), -40 < x < -12; most earthward one kept
  * jets: max / min Vx along the sheet within 6 R_E earthward / tailward of the X-line
  * class  clean    : both jets faster than 300 km/s and at most two X-lines in -40..-12
           inactive : at least one jet slower than 300 km/s
           other    : both jets but >= 3 X-lines (plasmoid chains)
  * quadrant mean By (window median removed)
  * legacy proxy R = Ey / (B_L V_out), NOT the standard Alfven-normalized rate;
    do not compare this value directly with the conventional 0.1 reference
Composite of clean frames aligned on (x0, z0) in x AND z.
Exploratory Hall electric-field check: its run-specific scale factor has not been
independently verified. It is not a validated result in the public summary.

python recon_redo.py          analysis + figures (cache written to output/cache/recon_redo.pkl)
python recon_redo.py --fig    figures only, from the cache
"""
import json
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import map_coordinates, uniform_filter, uniform_filter1d, median_filter
import magviz

P = magviz.Plane('mer')
X, Z = P.xu, P.tu
DX = 0.1
SCALE = 6.01                      # ion-scale enlargement of the run
HALL = 1e-18 / (1e6 * 1.602e-19) * 1e3 * SCALE   # (nA/m2 * nT / cm-3) -> mV/m, scaled
OUT = 'output/figs'
CACHE = 'output/cache/recon_redo.pkl'

WX, WZ = 6.0, 4.0
gx = np.arange(-WX, WX + 1e-6, DX)
gz = np.arange(-WZ, WZ + 1e-6, DX)
GX, GZ = np.meshgrid(gx, gz, indexing='ij')
CKEYS = ['Bx', 'By', 'Bz', 'Vx', 'Vy', 'Ez', 'Ey', 'Jx', 'Jy', 'N']


def sheet(d):
    iz = np.where(np.abs(Z) < 5)[0]
    bx = uniform_filter(d['Bx'], (5, 3))[:, iz]
    zc = np.full(len(X), np.nan)
    for i in range(len(X)):
        s = np.where(np.diff(np.sign(bx[i])) != 0)[0]
        if len(s):
            zc[i] = Z[iz][s[np.argmin(np.abs(Z[iz][s]))]] + DX / 2
    ok = np.isfinite(zc)
    zc = np.interp(np.arange(len(X)), np.flatnonzero(ok), zc[ok])
    return median_filter(zc, 9)


def along(d, k, zc):
    iz = np.clip(np.round((zc - Z[0]) / DX).astype(int), 2, len(Z) - 3)
    a = d[k]
    return np.mean([a[np.arange(len(X)), iz + o] for o in (-2, -1, 0, 1, 2)], 0)


def analyse(d):
    zc = sheet(d)
    bz = uniform_filter1d(along(d, 'Bz', zc), 11)
    vx = uniform_filter1d(along(d, 'Vx', zc), 5)
    xs = [i for i in range(len(X) - 1, 0, -1) if -40 < X[i] < -12 and bz[i] > 0 and bz[i - 1] <= 0]
    if not xs:
        return None
    i0 = xs[0]
    x0 = X[i0 - 1] - bz[i0 - 1] * DX / (bz[i0] - bz[i0 - 1])
    z0 = zc[i0]
    e = (X > x0) & (X < x0 + 6)
    t = (X < x0) & (X > x0 - 6)
    r = dict(t=float(d['time']), x0=float(x0), z0=float(z0), nX=len(xs),
             vjet_e=float(vx[e].max()), vjet_t=float(vx[t].min()))
    XX, ZZ = np.meshgrid(X - x0, Z - z0, indexing='ij')
    win = (np.abs(XX) < 5) & (np.abs(ZZ) < 3)
    by = d['By'] - np.median(d['By'][win])
    q = lambda sx, sz: float(by[(np.sign(XX) == sx) & (np.sign(ZZ) == sz) & (np.abs(XX) > 0.5) & (np.abs(XX) < 3)
                                & (np.abs(ZZ) > 0.2) & (np.abs(ZZ) < 1.5)].mean())
    r.update(EN=q(1, 1), ES=q(1, -1), TN=q(-1, 1), TS=q(-1, -1))
    r['full'] = bool(r['EN'] > 0 and r['ES'] < 0 and r['TN'] < 0 and r['TS'] > 0)
    xl = np.abs(XX) < 0.5
    ey = float(d['Ey'][xl & (np.abs(ZZ) < 0.3)].mean())
    bl = float(np.nanmean(np.abs(d['Bx'][xl & (np.abs(ZZ) > 4) & (np.abs(ZZ) < 8)])))
    vout = 0.5 * (r['vjet_e'] - r['vjet_t'])
    r.update(Ey=ey, B_L=bl, V_out=vout, V_in=ey / bl * 1e3, R=ey / (bl * vout * 1e-3),
             N_x=float(d['N'][xl & (np.abs(ZZ) < 0.3)].mean()))
    jets = r['vjet_e'] > 300 and r['vjet_t'] < -300
    r['cls'] = 'clean' if jets and r['nX'] <= 2 else ('inactive' if not jets else 'other')
    return r, zc, bz, vx


def window(d, x0, z0, keys=CKEYS):
    ix = (GX + x0 - X[0]) / DX
    iz = (GZ + z0 - Z[0]) / DX
    return {k: map_coordinates(d[k], [ix, iz], order=1, mode='nearest') for k in keys}


def compute():
    rows, comp, ncomp, snap = [], {k: 0 for k in CKEYS}, 0, None
    hall_pts = []
    for f in P.frames():
        d = P.load(f)
        if d['time'] < 100:
            continue
        out = analyse(d)
        if out is None:
            continue
        r, zc, bz, vx = out
        if r['cls'] == 'clean':
            w = window(d, r['x0'], r['z0'])
            w['By'] = w['By'] - np.median(w['By'][(np.abs(GX) < 5) & (np.abs(GZ) < 3)])
            for k in CKEYS:
                comp[k] = comp[k] + w[k]
            ncomp += 1
            ezp = w['Ez'] + (w['Vx'] * w['By'] - w['Vy'] * w['Bx']) * 1e-3
            hz = HALL * (w['Jx'] * w['By'] - w['Jy'] * w['Bx']) / w['N']
            m = (w['N'] > 0.03) & (np.abs(GX) < 5) & (np.abs(GZ) < 3)
            hall_pts.append(np.c_[ezp[m], hz[m]])
        if abs(r['t'] - 122) < 0.5:
            sx, sz = (X > -43) & (X < -7), np.abs(Z) < 8.5
            snap = dict(r=r, zc=zc, bz=bz, vx=vx, sx=sx, sz=sz,
                        f={k: d[k][np.ix_(sx, sz)] for k in ('Vx', 'Jy', 'Bx', 'Bz', 'N')},
                        w=window(d, r['x0'], r['z0'], ['Bx', 'By', 'Bz', 'Ez']))
        rows.append(r)
    for k in CKEYS:
        comp[k] = comp[k] / ncomp
    H = np.vstack(hall_pts)
    H = H[np.all(np.isfinite(H), 1)]
    pickle.dump(dict(rows=rows, comp=comp, ncomp=ncomp, snap=snap, H=H), open(CACHE, 'wb'))


if '--fig' not in sys.argv:
    compute()
C = pickle.load(open(CACHE, 'rb'))
rows, comp, ncomp, snap, H = C['rows'], C['comp'], C['ncomp'], C['snap'], C['H']
corr = float(np.corrcoef(H[:, 0], H[:, 1])[0, 1])
slope = float(np.sum(H[:, 0] * H[:, 1]) / np.sum(H[:, 1] ** 2))


def stats(c):
    s = [r for r in rows if r['cls'] == c] if c != 'all' else rows
    med = lambda k: float(np.nanmedian([r[k] for r in s]))
    return dict(n=len(s), full_quadrupole=int(sum(r['full'] for r in s)),
                frac_full=float(np.mean([r['full'] for r in s])),
                By_mean={k: float(np.mean([r[k] for r in s])) for k in ('EN', 'ES', 'TN', 'TS')},
                R_median=med('R'), R_iqr=[float(np.nanpercentile([r['R'] for r in s], q)) for q in (25, 75)],
                V_in_median=med('V_in'), V_out_median=med('V_out'), Ey_median=med('Ey'), B_L_median=med('B_L'),
                vjet_e_median=med('vjet_e'), vjet_t_median=med('vjet_t'), xline_median=med('x0'))


cl = [r for r in rows if r['cls'] == 'clean']
S = {c: stats(c) for c in ('clean', 'inactive', 'other', 'all')}
amp = float(np.mean([(r['EN'] - r['ES'] - r['TN'] + r['TS']) / 4 for r in cl]))
blobe = S['clean']['B_L_median']
n_ps = float(np.median(comp['N'][(np.abs(GX) < 5) & (np.abs(GZ) < 1)]))
di = 0.2151 / np.sqrt(n_ps)        # R_E, scaled run: 6.01 * 228 km / sqrt(n)
ez_e = comp['Ez'][(gx > 1) & (gx < 3)].mean(0)
ez_t = comp['Ez'][(gx < -1) & (gx > -3)].mean(0)
pk = lambda p: float(abs(gz[np.argmax(p)] - gz[np.argmin(p)]))
x0s = np.array([r['x0'] for r in rows])
rs = snap['r']
res = dict(frames=len(rows), classes=S, composite_frames=ncomp,
           xline_all=dict(median=float(np.median(x0s)), p10=float(np.percentile(x0s, 10)),
                          p90=float(np.percentile(x0s, 90))),
           quadrupole_amp_clean_nT=amp, B_L_clean_median_nT=blobe, amp_over_BL=amp / blobe,
           N_plasma_sheet_composite=n_ps, d_i_plasma_sheet_RE=float(di),
           Ez_peak_separation_RE=dict(earthward=pk(ez_e), tailward=pk(ez_t)),
           Ez_peak_mVm=float(max(np.abs(ez_e).max(), np.abs(ez_t).max())),
           hall_check=dict(points=int(len(H)), corr=corr, slope=slope),
           snapshot_t122={k: rs[k] for k in ('t', 'x0', 'z0', 'nX', 'vjet_e', 'vjet_t', 'EN', 'ES', 'TN', 'TS',
                                             'Ey', 'B_L', 'V_out', 'V_in', 'R')})
json.dump(res, open('output/recon_redo.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

# =============================================================== figures
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})
RB = 'RdBu_r'
HALO = [pe.withStroke(linewidth=3, foreground='white')]


def cbar(ax, im, label=None, size='3%'):
    cax = make_axes_locatable(ax).append_axes('right', size=size, pad=0.08)
    cb = plt.colorbar(im, cax=cax)
    if label:
        cb.set_label(label)
    return cb


def flines(ax, x, z, bx, bz, dens=1.0, lw=0.55):
    ax.streamplot(x, z, bx.T, bz.T, color='k', density=dens, linewidth=lw, arrowsize=0.55)


def xmark(ax, x0, z0, ms=14):
    ax.plot(x0, z0, 'x', color='k', ms=ms, mew=3.4, zorder=5)
    ax.plot(x0, z0, 'x', color='#ffd400', ms=ms - 3, mew=1.6, zorder=6)


x0, z0 = rs['x0'], rs['z0']
xa, za = X[snap['sx']], Z[snap['sz']]
F = snap['f']
bzs = snap['bz']
# O-line (plasmoid centre) tailward of the X-line: Bz goes + (tailward) -> - (earthward)
oc = [i for i in range(len(X) - 1, 0, -1) if X[i] < x0 - 2 and X[i] > -42 and bzs[i] < 0 and bzs[i - 1] >= 0]
xo = X[oc[0]] if oc else None

# ---------------- F1 anatomy of the X-line (slide 15)
fig = plt.figure(figsize=(8.6, 5.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.9], hspace=0.38, left=0.075, right=0.985, top=0.95, bottom=0.085)
ax = fig.add_subplot(gs[0])
im = ax.pcolormesh(xa, za, F['Vx'].T, cmap=RB, norm=TwoSlopeNorm(0, -800, 800), shading='auto', rasterized=True)
flines(ax, xa, za, F['Bx'], F['Bz'], 1.4)
xmark(ax, x0, z0)
ax.annotate('', (x0 + 9, z0 - 2.9), (x0 + 1.2, z0 - 2.9), arrowprops=dict(arrowstyle='-|>', color='#b2182b', lw=2.4))
ax.annotate('', (x0 - 9, z0 - 2.9), (x0 - 1.2, z0 - 2.9), arrowprops=dict(arrowstyle='-|>', color='#2166ac', lw=2.4))
ax.text(x0 + 5, z0 - 2.55, 'earthward jet', color='#b2182b', ha='center', va='bottom', fontsize=10,
        fontweight='bold', path_effects=HALO)
ax.text(x0 - 5, z0 - 2.55, 'tailward jet', color='#2166ac', ha='center', va='bottom', fontsize=10,
        fontweight='bold', path_effects=HALO)
ax.text(x0, z0 + 1.15, 'X-line', ha='center', fontsize=10, fontweight='bold', path_effects=HALO)
if xo is not None:
    ax.text(xo, 2.9, 'plasmoid', ha='center', fontsize=10, fontweight='bold', color='#2166ac', path_effects=HALO)
ax.set_title(f'(a) $V_x$ (km/s) and magnetic field lines, t = {rs["t"]:.0f} $\\Omega_i^{{-1}}$', loc='left')
cbar(ax, im)
ax2 = fig.add_subplot(gs[1], sharex=ax)
im = ax2.pcolormesh(xa, za, F['Jy'].T, cmap='PuOr_r', norm=TwoSlopeNorm(0, -3, 3), shading='auto', rasterized=True)
flines(ax2, xa, za, F['Bx'], F['Bz'], 1.4)
xmark(ax2, x0, z0)
ax2.set_title('(b) cross-tail current $J_y$ (nA/m$^2$): a thin current sheet', loc='left')
cbar(ax2, im)
for a in (ax, ax2):
    a.set_aspect('equal')
    a.set_xlim(-42, -8)
    a.set_ylim(-4.5, 4.5)
    a.set_ylabel('z ($R_E$)')
    plt.setp(a.get_xticklabels(), visible=False)
ax3 = fig.add_subplot(gs[2], sharex=ax)
sp = make_axes_locatable(ax3).append_axes('right', size='3%', pad=0.08)
sp.axis('off')
m = (X > -42) & (X < -8)
ax3.plot(X[m], bzs[m], color='k', lw=1.8)
ax3.axhline(0, color='0.5', lw=0.7)
ax3.axvline(x0, color='#d4a000', lw=2, ls='--')
if xo is not None:
    ax3.axvline(xo, color='#2166ac', lw=1.2, ls=':')
ax3.set_ylabel('$B_z$ (nT)')
ax3.set_ylim(-6.5, 6.5)
ax4 = ax3.twinx()
ax4.plot(X[m], snap['vx'][m], color='#c0392b', lw=1.5)
ax4.set_ylim(-900, 900)
ax4.set_yticks([-500, 0, 500])
ax4.tick_params(axis='y', colors='#c0392b', labelsize=9, pad=1)
ax3.set_title('(c) along the sheet: $B_z$ (black) and $V_x$ (red, km/s) flip sign at the X-line', loc='left',
              fontsize=11)
ax3.set_xlabel('x ($R_E$)      Earth is to the right →')
fig.savefig(f'{OUT}/recon_anatomy.png', dpi=220)
plt.close(fig)

# ---------------- F2 Hall quadrupole (slide 19)
fig = plt.figure(figsize=(14.2, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.82], wspace=0.24, left=0.045, right=0.99, top=0.9, bottom=0.15)
w = snap['w']
SIG = [(2.6, 1.7, '+'), (2.6, -1.7, '−'), (-2.6, 1.7, '−'), (-2.6, -1.7, '+')]


def hallmap(ax, by, bx, bz, vm, title):
    im = ax.pcolormesh(gx, gz, by.T, cmap=RB, norm=TwoSlopeNorm(0, -vm, vm), shading='auto', rasterized=True)
    flines(ax, gx, gz, bx, bz, 1.0)
    for xx, zz, s in SIG:
        ax.text(xx, zz, s, ha='center', va='center', fontsize=30, fontweight='bold', color='k',
                path_effects=[pe.withStroke(linewidth=4, foreground='white')])
    xmark(ax, 0, 0)
    ax.set_aspect('equal')
    ax.set_xlim(-WX, WX)
    ax.set_ylim(-WZ, WZ)
    ax.set_xlabel('distance from X-line ($R_E$)     → Earth')
    ax.set_title(title, loc='left')
    cbar(ax, im, '$B_y$ (nT)', '4%')


ax = fig.add_subplot(gs[0])
w2 = dict(w)
w2['By'] = w['By'] - np.median(w['By'][(np.abs(GX) < 5) & (np.abs(GZ) < 3)])
hallmap(ax, w2['By'], w['Bx'], w['Bz'], 2.0, f'(a) one clean X-line, t = {rs["t"]:.0f}')
ax.set_ylabel('z ($R_E$)')
ax = fig.add_subplot(gs[1])
hallmap(ax, comp['By'], comp['Bx'], comp['Bz'], 0.8, f'(b) average of {ncomp} clean X-lines')
ax = fig.add_subplot(gs[2])
lab = ['earthward\nnorth', 'earthward\nsouth', 'tailward\nnorth', 'tailward\nsouth']
exp = [1, -1, -1, 1]
xx = np.arange(4)
for j, (c, col, nm) in enumerate([('clean', '#c0392b', 'clean X-line'), ('inactive', '0.62', 'inactive X-line')]):
    v = [S[c]['By_mean'][k] for k in ('EN', 'ES', 'TN', 'TS')]
    ax.bar(xx + (j - 0.5) * 0.36, v, 0.34, color=col, label=f'{nm} ({S[c]["n"]} frames)')
for k, e in enumerate(exp):
    ax.text(k, 1.55 * e, '+' if e > 0 else '−', ha='center', va='center', fontsize=20, fontweight='bold')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(xx)
ax.set_xticklabels(lab, fontsize=10)
ax.set_ylim(-1.9, 2.6)
ax.set_ylabel('mean $B_y$ in quadrant (nT)')
ax.legend(fontsize=9.5, loc='upper center', frameon=False, ncol=1)
ax.set_title('(c) quadrant means; bold sign = Hall prediction', loc='left')
ax.grid(axis='y', alpha=0.3)
fig.savefig(f'{OUT}/hall_quadrupole_v2.png', dpi=200)
plt.close(fig)

# ---------------- F3 Hall electric field / ion decoupling (slide 20)
fig = plt.figure(figsize=(14.2, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.5, 0.72], wspace=0.3, left=0.045, right=0.985, top=0.9, bottom=0.15)
ax = fig.add_subplot(gs[0])
im = ax.pcolormesh(gx, gz, w['Ez'].T, cmap=RB, norm=TwoSlopeNorm(0, -3, 3), shading='auto', rasterized=True)
flines(ax, gx, gz, w['Bx'], w['Bz'], 1.0)
for xa_ in (-3.5, 3.5):
    for s in (1, -1):
        ax.annotate('', (xa_, 0.2 * s), (xa_, 1.7 * s), arrowprops=dict(arrowstyle='-|>', color='k', lw=2.2))
xmark(ax, 0, 0)
ax.set_aspect('equal')
ax.set_xlim(-WX, WX)
ax.set_ylim(-WZ, WZ)
ax.set_xlabel('distance from X-line ($R_E$)     → Earth')
ax.set_ylabel('z ($R_E$)')
ax.set_title(f'(a) $E_z$ at t = {rs["t"]:.0f}: points toward the sheet', loc='left')
cbar(ax, im, '$E_z$ (mV/m)', '4%')
ax = fig.add_subplot(gs[1])
ax.plot(ez_e, gz, color='#c0392b', lw=2, label='earthward side')
ax.plot(ez_t, gz, color='#2166ac', lw=2, label='tailward side')
ax.axvline(0, color='k', lw=0.7)
ax.axhline(0, color='0.6', lw=0.7, ls=':')
ax.set_ylim(-WZ, WZ)
ax.set_xlim(-4.2, 4.2)
ax.set_xlabel('$E_z$ (mV/m)')
ax.set_ylabel('z ($R_E$)')
ax.text(-4.0, 2.2, '$E_z<0$\n(down)', fontsize=10, color='#2166ac')
ax.text(0.6, -2.8, '$E_z>0$\n(up)', fontsize=10, color='#c0392b')
ax.legend(fontsize=9, loc='upper right', frameon=False)
ax.set_title(f'(b) {ncomp} clean X-lines', loc='left')
ax = fig.add_subplot(gs[2])
lim = 4
ax.hexbin(H[:, 1], H[:, 0], gridsize=60, extent=(-lim, lim, -lim, lim), bins='log', cmap='Greys', mincnt=1)
ax.plot([-lim, lim], [-lim, lim], color='#c0392b', lw=1.8, ls='--', label='1 : 1')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.set_xlabel(r'Hall term  $(\mathbf{J}\times\mathbf{B})_z/ne$  (mV/m)')
ax.set_ylabel(r'$(\mathbf{E}+\mathbf{V}_i\times\mathbf{B})_z$  (mV/m)')
ax.text(0.04, 0.95, f'r = {corr:.2f}\nslope = {slope:.2f}', transform=ax.transAxes, va='top', fontsize=11)
ax.legend(fontsize=10, loc='lower right', frameon=False)
ax.set_title('(c) ions are not frozen in: Hall term', loc='left')
fig.savefig(f'{OUT}/hall_efield.png', dpi=200)
plt.close(fig)

# ---------------- F4 reconnection rate (slide 17)
rep = json.load(open('output/repro.json', encoding='utf-8'))['dayside_reconnection']
fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 0.85], wspace=0.25,
                                                                     left=0.05, right=0.99, top=0.9, bottom=0.17))
ax = axs[0]
bins = np.arange(-41, -11, 1)
ax.axvspan(-30, -20, color='#f2e394', alpha=0.6, lw=0, label='Nagai et al. 1998: −20 … −30 $R_E$')
ax.hist(x0s, bins=bins, color='0.72', label=f'all X-lines ({len(rows)} frames)')
ax.hist([r['x0'] for r in cl], bins=bins, color='#c0392b', label=f'clean X-lines ({len(cl)})')
ax.set_xlabel('X-line position x ($R_E$)')
ax.set_ylabel('frames')
ax.legend(fontsize=9, loc='upper left', frameon=False)
ax.set_ylim(0, 19)
ax.set_title('(a) where the tail X-line sits', loc='left')
ax = axs[1]
Rc = np.array([r['R'] for r in cl])
ax.hist(Rc, bins=np.arange(0, 0.42, 0.025), color='#c0392b', alpha=0.9)
ax.axvspan(0.05, 0.2, color='#f2e394', alpha=0.6, lw=0, zorder=0, label='fast reconnection, 0.05 … 0.2')
ax.axvline(np.median(Rc), color='k', lw=1.8, ls='--', label=f'median {np.median(Rc):.2f}')
ax.set_xlabel(r'$R = E_y\,/\,(B_L V_{out})$')
ax.set_ylabel('clean frames')
ax.legend(fontsize=9, loc='upper right', frameon=False)
ax.set_title('(b) tail reconnection rate', loc='left')
ax = axs[2]
vals = [rep['E_sw'], rep['E_measured_mVm'], rep['E_CS07_mVm']]
ax.bar([0, 1, 2], vals, color=['0.62', '#c0392b', '#2166ac'], width=0.62)
for k_, v in enumerate(vals):
    ax.text(k_, v + 0.04, f'{v:.2f}', ha='center', fontsize=11)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['solar wind\n$V_{sw}B_z$', 'measured $E_y$\nat dayside MP', 'Cassak–Shay\n2007 estimate'],
                   fontsize=10)
ax.set_ylabel('mV/m')
ax.set_ylim(0, 2.05)
ax.set_title('(c) dayside reconnection E field', loc='left')
fig.savefig(f'{OUT}/recon_rate_v2.png', dpi=200)
plt.close(fig)
print('done')
