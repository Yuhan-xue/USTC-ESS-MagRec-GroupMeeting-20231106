"""Quantitative diagnostics for the report. Run with system python after downloads."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from magviz import Plane, render, draw_earth, RMIN, DX, PANELS

OUT = 'output/figs'
os.makedirs(OUT, exist_ok=True)
MP = 1.6726e-27
res = {}


def strip(a, pl, axis_val=0.0, half=0.3):
    """Average a (nx, nt) map over |t - axis_val| < half -> profile along x."""
    m = np.abs(pl.tu - axis_val) < half
    return a[:, m].mean(axis=1)


def find_boundaries(xu, N, Bz, T, nsw):
    """Subsolar magnetopause (Bz sign change, outward from 3.5 Re) and bow shock (outermost N>2 nsw)."""
    day = xu > 3.5
    xs, b = xu[day], Bz[day]
    neg = np.where(b < 0)[0]
    mp = xs[neg[0]] if len(neg) else np.nan
    up = (xu > 4) & (xu < 21.5)
    xs2, n2 = xu[up], N[up]
    hi = np.where(n2 > 2.0 * nsw)[0]
    bs = xs2[hi[-1]] if len(hi) else np.nan
    return mp, bs


# ---------------------------------------------------------------- equatorial
eq = Plane('eq')
ef = eq.frames()
ie_up = np.argmin(abs(eq.xu - 20)), np.argmin(abs(eq.tu - 0))
keys_k = ['N', 'Bz', 'Vx', 'T', 'Vy']
keo = {k: [] for k in keys_k}
ts, tsec, up = [], [], {k: [] for k in ['N', 'Vx', 'Bz', 'T', 'Pdyn']}
mp_t, bs_t, flank_mp, flank_bs = [], [], [], []
ix0 = np.argmin(abs(eq.xu - 0))
late = []  # for Reynolds stress
for fr in ef:
    d = eq.load(fr)
    ts.append(d['time']); tsec.append(d['tsec'])
    for k in up:
        up[k].append(d[k][ie_up[0] - 5:ie_up[0] + 5, ie_up[1] - 5:ie_up[1] + 5].mean())
    for k in keys_k:
        keo[k].append(strip(d[k], eq))
    nsw = up['N'][-1]
    mp, bs = find_boundaries(eq.xu, keo['N'][-1], keo['Bz'][-1], keo['T'][-1], nsw)
    mp_t.append(mp); bs_t.append(bs)
    # dusk flank (y>0) at x=0: magnetopause = first Bz<0 outward along y, bow shock = outermost N > 2 nsw
    col_b, col_n = d['Bz'][ix0], d['N'][ix0]
    yy = eq.tu
    sel = yy > 3.5
    neg = np.where(col_b[sel] < 0)[0]
    flank_mp.append(yy[sel][neg[0]] if len(neg) else np.nan)
    hi = np.where(col_n[sel] > 2 * nsw)[0]
    flank_bs.append(yy[sel][hi[-1]] if len(hi) else np.nan)
    if d['time'] >= 100:
        late.append(fr)
ts, tsec = np.array(ts), np.array(tsec)
for k in keo:
    keo[k] = np.array(keo[k])
mp_t, bs_t = np.array(mp_t), np.array(bs_t)
flank_mp, flank_bs = np.array(flank_mp), np.array(flank_bs)
np.savez('output/eq_timeseries.npz', t=ts, tsec=tsec, mp=mp_t, bs=bs_t, flank_mp=flank_mp, flank_bs=flank_bs,
         **{'up_' + k: np.array(v) for k, v in up.items()})

# Upstream monitor + boundary positions
pd = np.array(up['Pdyn'])
shue = (10.22 + 1.29 * np.tanh(0.184 * (np.array(up['Bz']) + 8.14))) * np.clip(pd, 0.05, None) ** (-1 / 6.6)
shue_scaled = shue * (37200 / 31000) ** (1 / 3)
fig, axs = plt.subplots(5, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
axs[0].plot(ts, up['N']); axs[0].set_ylabel(r'$N_{sw}$ (cm$^{-3}$)')
axs[1].plot(ts, up['Vx']); axs[1].set_ylabel(r'$V_{x,sw}$ (km/s)')
axs[2].plot(ts, up['Bz']); axs[2].set_ylabel(r'$B_{z,sw}$ (nT)')
axs[3].plot(ts, pd); axs[3].set_ylabel(r'$P_{dyn}$ (nPa)')
axs[4].plot(ts, bs_t, label='bow shock (subsolar)')
axs[4].plot(ts, mp_t, label='magnetopause (subsolar)')
axs[4].plot(ts, flank_bs, '--', label='bow shock (dusk, x=0)')
axs[4].plot(ts, flank_mp, '--', label='magnetopause (dusk, x=0)')
axs[4].plot(ts, shue_scaled, 'k:', label='Shue98 (scaled to $B_0$=37200 nT)')
axs[4].set_ylabel(r'distance ($R_E$)'); axs[4].set_ylim(0, 32); axs[4].legend(fontsize=8, ncol=2)
axs[4].set_xlabel(r'$t$ ($\Omega_i^{-1}$);  1 unit ≈ 15.7 s (scaled)')
for a in axs:
    a.grid(alpha=0.3)
axs[0].set_title('Upstream solar wind at (20, 0, 0) $R_E$ and boundary positions')
fig.savefig(f'{OUT}/fig_upstream_boundaries.png', dpi=110)
plt.close(fig)

st = ts >= 100
res['upstream_late'] = {k: float(np.mean(np.array(v)[st])) for k, v in up.items()}
res['mp_subsolar_late'] = [float(np.nanmean(mp_t[st])), float(np.nanstd(mp_t[st])), float(np.nanmin(mp_t[st])), float(np.nanmax(mp_t[st]))]
res['bs_subsolar_late'] = [float(np.nanmean(bs_t[st])), float(np.nanstd(bs_t[st]))]
res['mp_flank_late'] = [float(np.nanmean(flank_mp[st])), float(np.nanstd(flank_mp[st]))]
res['bs_flank_late'] = [float(np.nanmean(flank_bs[st])), float(np.nanstd(flank_bs[st]))]
res['shue_scaled_late'] = float(np.mean(shue_scaled[st]))
res['shue_late'] = float(np.mean(shue[st]))
res['t_end'] = float(ts[-1]); res['tsec_end'] = float(tsec[-1])


def keogram(xu, ts, K, specs, fname, title, overlays=None, ylabel=r'$x$ ($R_E$)'):
    fig, axs = plt.subplots(len(specs), 1, figsize=(11, 3.0 * len(specs)), sharex=True, constrained_layout=True)
    for ax, (k, lab, cmap, norm) in zip(axs, specs):
        a = K[k]
        if isinstance(norm, LogNorm):
            a = np.clip(a, norm.vmin, norm.vmax)
        im = ax.pcolormesh(ts, xu, a.T, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
        ax.axhspan(-RMIN, RMIN, color='0.6')
        if overlays:
            for (yy, sty, lb) in overlays:
                ax.plot(ts, yy, sty, lw=0.8, label=lb)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, pad=0.01).set_label(lab)
    if overlays:
        axs[0].legend(fontsize=8, loc='lower left')
    axs[-1].set_xlabel(r'$t$ ($\Omega_i^{-1}$)')
    axs[0].set_title(title)
    fig.savefig(fname, dpi=110)
    plt.close(fig)


keogram(eq.xu, ts, keo, [
    ('N', r'$N_i$ (cm$^{-3}$)', 'turbo', LogNorm(0.1, 40)),
    ('Bz', r'$B_z$ (nT)', 'RdBu_r', SymLogNorm(10, vmin=-200, vmax=200)),
    ('Vx', r'$V_x$ (km/s)', 'RdBu_r', TwoSlopeNorm(0, -600, 600)),
    ('T', r'$T_i$ (eV)', 'inferno', LogNorm(3, 5000))],
    f'{OUT}/fig_keogram_eq_sunearth.png', 'Equatorial plane, Sun–Earth line (y = 0): x–t diagram',
    overlays=[(mp_t, 'w-', 'MP'), (bs_t, 'k-', 'BS')])

# ---------------------------------------------------------------- meridian
me = Plane('mer')
mf = me.frames()
keyM = ['N', 'Bz', 'Vx', 'T', 'Bx', 'Ey', 'Jy']
kM = {k: [] for k in keyM}
tm = []
xline, xline_all = [], []
cusp = []
lobe_B = []
izc = np.argmin(abs(me.tu))
for fr in mf:
    d = me.load(fr)
    tm.append(d['time'])
    for k in keyM:
        kM[k].append(strip(d[k], me))
    bz = kM['Bz'][-1]
    tail = me.xu < -6
    xs = me.xu[tail]
    neg = np.where(bz[tail] < -1.0)[0]
    # near-Earth X-line candidate: most earthward Bz<0 point in the tail current sheet
    xline.append(xs[neg[-1]] if len(neg) else np.nan)
    # lobe field at x=-20, |z| in 6..12
    i20 = np.argmin(abs(me.xu + 20))
    m = (np.abs(me.tu) > 6) & (np.abs(me.tu) < 12)
    lobe_B.append(np.abs(d['Bx'][i20, m]).mean())
tm = np.array(tm)
for k in kM:
    kM[k] = np.array(kM[k])
xline = np.array(xline); lobe_B = np.array(lobe_B)
np.savez('output/mer_timeseries.npz', t=tm, xline=xline, lobe_B=lobe_B)
keogram(me.xu, tm, kM, [
    ('Bz', r'$B_z$ (nT)', 'RdBu_r', SymLogNorm(2, vmin=-100, vmax=100)),
    ('Vx', r'$V_x$ (km/s)', 'RdBu_r', TwoSlopeNorm(0, -800, 800)),
    ('N', r'$N_i$ (cm$^{-3}$)', 'turbo', LogNorm(0.05, 40)),
    ('T', r'$T_i$ (eV)', 'inferno', LogNorm(3, 10000))],
    f'{OUT}/fig_keogram_mer_sunearth.png', 'Meridian plane, Sun–Earth line (z = 0): x–t diagram',
    overlays=[(xline, 'k.', 'most earthward $B_z<-1$ nT (tail)')])
sl = tm >= 100
if not sl.any():  # partial data (test run)
    sl = np.ones_like(tm, dtype=bool)
res['xline_late'] = [float(np.nanmedian(xline[sl])), float(np.nanpercentile(xline[sl], 10)), float(np.nanpercentile(xline[sl], 90))]
res['lobe_B'] = [float(np.mean(lobe_B[sl])), float(lobe_B[sl].min()), float(lobe_B[sl].max())]

# tail Vx statistics in plasma sheet: fraction of time with earthward / tailward fast flows at x=-15,-25
for xx in (-12, -15, -20, -25, -30):
    i = np.argmin(abs(me.xu - xx))
    v = kM['Vx'][sl, i]
    res[f'tailVx_{xx}'] = dict(mean=float(v.mean()), max=float(v.max()), min=float(v.min()),
                               frac_earthward_gt200=float((v > 200).mean()), frac_tailward_lt_m200=float((v < -200).mean()))

# ------------------------------------------------- time-averaged maps (t >= 100): Reynolds stress & J.E'
def tavg(pl, frames, comp):
    acc = None
    n = 0
    for fr in frames:
        d = pl.load(fr)
        rho = np.clip(d['N'], 0, None) * 1e6 * MP
        va, vb = d['Vx'], d['V' + comp]
        Ep = [d['E' + c] for c in 'xyz']
        V = [d['V' + c] * 1e3 for c in 'xyz']
        B = [d['B' + c] * 1e-9 for c in 'xyz']
        VxB = [V[1] * B[2] - V[2] * B[1], V[2] * B[0] - V[0] * B[2], V[0] * B[1] - V[1] * B[0]]
        Eprime = [Ep[i] * 1e-3 + VxB[i] for i in range(3)]
        JE = sum(d['J' + c] * 1e-9 * Eprime[i] for i, c in enumerate('xyz')) * 1e12  # pW/m^3
        JEt = sum(d['J' + c] * 1e-9 * Ep[i] * 1e-3 for i, c in enumerate('xyz')) * 1e12
        terms = dict(rho=rho, va=va, vb=vb, rva=rho * va, rvb=rho * vb, rvava=rho * va * va,
                     rvbvb=rho * vb * vb, rvavb=rho * va * vb, JE=JE, JEt=JEt, N=d['N'])
        if acc is None:
            acc = {k: np.zeros_like(v, dtype=float) for k, v in terms.items()}
        for k, v in terms.items():
            acc[k] += v
        n += 1
    for k in acc:
        acc[k] /= n
    rho = np.clip(acc['rho'], 1e-30, None)
    Ua, Ub = acc['rva'] / rho, acc['rvb'] / rho   # Favre-averaged mean flow (km/s)
    # Reynolds stress (density-weighted): <rho va' vb'> = <rho va vb> - rho_bar Ua Ub   [kg m^-3 (km/s)^2] -> nPa
    R_ab = (acc['rvavb'] - rho * Ua * Ub) * 1e6 * 1e9
    k_turb = 0.5 * ((acc['rvava'] - rho * Ua ** 2) + (acc['rvbvb'] - rho * Ub ** 2)) * 1e6 * 1e9
    return dict(Ua=Ua, Ub=Ub, R=R_ab, K=k_turb, JE=acc['JE'], JEt=acc['JEt'], N=acc['N'], n=n)


def plot_avg(pl, A, comp, fname, title):
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    inner = pl.R < RMIN
    ext = (pl.xu[0], pl.xu[-1], pl.tu[0], pl.tu[-1])
    items = [
        (A['N'], r'$\langle N_i\rangle$ (cm$^{-3}$)', 'turbo', LogNorm(0.1, 40)),
        (A['Ua'], r'$\tilde U_x$ (km/s)', 'RdBu_r', TwoSlopeNorm(0, -500, 500)),
        (A['Ub'], rf'$\tilde U_{comp}$ (km/s)', 'RdBu_r', TwoSlopeNorm(0, -300, 300)),
        (A['K'], r'$\frac{1}{2}\langle\rho\,|\delta \mathbf{v}_\parallel|^2\rangle$ in-plane (nPa)', 'magma', LogNorm(1e-3, 1)),
        (A['R'], rf'Reynolds stress $\langle\rho\,\delta v_x\delta v_{comp}\rangle$ (nPa)', 'RdBu_r', SymLogNorm(0.005, vmin=-0.3, vmax=0.3)),
        (A['JE'], r"$\langle\mathbf{J}\cdot\mathbf{E}'\rangle$, $\mathbf{E}'=\mathbf{E}+\mathbf{V}\times\mathbf{B}$ (pW/m$^3$)", 'PRGn', SymLogNorm(0.5, vmin=-30, vmax=30)),
    ]
    for ax, (a, lab, cmap, norm) in zip(axs.flat, items):
        if isinstance(norm, LogNorm):
            a = np.clip(a, norm.vmin, norm.vmax)
        im = ax.imshow(np.ma.array(a, mask=inner).T, origin='lower', extent=ext, cmap=cmap, norm=norm, aspect='equal')
        draw_earth(ax)
        ax.set_xlabel(r'$x$ ($R_E$)'); ax.set_ylabel(rf'${pl.tname}$ ($R_E$)')
        fig.colorbar(im, ax=ax, shrink=0.8).set_label(lab, fontsize=9)
    fig.suptitle(title)
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return A


Aeq = tavg(eq, [f for f in ef if f >= 401], 'y')
plot_avg(eq, Aeq, 'y', f'{OUT}/fig_avg_eq.png', f'Equatorial plane, average over t = 100–{ts[-1]:.0f} ($N$={Aeq["n"]} frames)')
Ame = tavg(me, [f for f in mf if f >= 401] or mf, 'z')
plot_avg(me, Ame, 'z', f'{OUT}/fig_avg_mer.png', f'Meridian plane, average over t = 100–{tm[-1]:.0f} ($N$={Ame["n"]} frames)')


# flank Reynolds stress: sample dusk/dawn flank cuts at x = -10 and -20
def flank_profile(pl, A, x0):
    i = np.argmin(abs(pl.xu - x0))
    return pl.tu, A['R'][i], A['K'][i], A['N'][i], A['Ua'][i]


fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
for ax, x0 in zip(axs, (-10, -20)):
    yy, R, K, N, U = flank_profile(eq, Aeq, x0)
    ax.plot(yy, R, label=r'$\langle\rho\delta v_x\delta v_y\rangle$ (nPa)')
    ax.plot(yy, K, label=r'$k_{turb}$ (nPa)')
    ax2 = ax.twinx(); ax2.plot(yy, U, 'k--', lw=0.8, label=r'$\tilde U_x$'); ax2.set_ylabel(r'$\tilde U_x$ (km/s)')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_title(f'Equatorial cut at x = {x0} $R_E$ (t ≥ 100 average)')
    ax.set_xlabel(r'$y$ ($R_E$)'); ax.legend(loc='upper left', fontsize=8)
fig.savefig(f'{OUT}/fig_flank_reynolds.png', dpi=110)
plt.close(fig)
for x0 in (-10, -20):
    yy, R, K, N, U = flank_profile(eq, Aeq, x0)
    dusk = (yy > 5) & (yy < 28); dawn = (yy < -5) & (yy > -28)
    res[f'R_flank_x{x0}'] = dict(dusk_min=float(R[dusk].min()), dusk_max=float(R[dusk].max()),
                                  dawn_min=float(R[dawn].min()), dawn_max=float(R[dawn].max()),
                                  dusk_int=float(np.trapezoid(R[dusk], yy[dusk])), dawn_int=float(np.trapezoid(R[dawn], yy[dawn])))

json.dump(res, open('output/diagnostics.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
