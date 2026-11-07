"""Classic-paper / textbook reproductions on the ear-Reynolds-S hybrid run (system python).
Needs output/cache/avg_*.npz from repro_cache.py. Writes output/figs/repro_*.png and output/repro.json."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.ndimage import map_coordinates, uniform_filter1d
from scipy.optimize import brentq, curve_fit
from magviz import Plane, draw_earth, RMIN, DX

OUT = 'output/figs'
MU0, MP, QE = 4e-7 * np.pi, 1.6726e-27, 1.602e-19
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
res = {}
diag = json.load(open('output/diagnostics.json'))
recon = json.load(open('output/recon.json'))
SW = diag['upstream_late']
nsw, vsw, bsw, tsw, pdsw = SW['N'], abs(SW['Vx']), abs(SW['Bz']), SW['T'], SW['Pdyn']
eq, me = Plane('eq'), Plane('mer')
AV = {p: dict(np.load(f'output/cache/avg_{p}.npz')) for p in ('eq', 'mer')}
PL = {'eq': eq, 'mer': me}


def samp(p, a, x, y):
    pl = PL[p]
    out = map_coordinates(a, [np.atleast_1d((np.asarray(x, float) - pl.xu[0]) / DX), np.atleast_1d((np.asarray(y, float) - pl.tu[0]) / DX)], order=1, mode='nearest')
    return out if np.ndim(x) else float(out[0])


def va(B_nT, n_cc):
    return B_nT * 1e-9 / np.sqrt(MU0 * np.clip(n_cc, 1e-4, None) * 1e6 * MP) / 1e3  # km/s


# --------------------------------------------------------------- upstream parameters
VA_sw = va(bsw, nsw)
CS_i = np.sqrt(5 / 3 * tsw * QE / MP) / 1e3
beta_i = 2 * MU0 * nsw * 1e6 * tsw * QE / (bsw * 1e-9) ** 2
MA = vsw / VA_sw
Mms = vsw / np.hypot(VA_sw, CS_i)
# effective dipole moment of the run: B_eq * r^3 on the dawn/dusk terminator
B0 = float(np.mean([samp('eq', AV['eq']['Bz'], 0, s * r) * r ** 3 for s in (1, -1) for r in (5, 6, 7)]))
res['upstream'] = dict(n=nsw, V=vsw, Bz=-bsw, Ti_eV=tsw, Pdyn=pdsw, VA=VA_sw, cs_i=CS_i, beta_i=beta_i,
                       MA=MA, Mms=Mms, B0_eff_nT=B0, Esw_mVm=vsw * bsw * 1e-3)
print('upstream', res['upstream'])


# --------------------------------------------------------------- R1: boundaries + Shue fit
def ray_boundaries(p, th_deg, sgn):
    A = AV[p]
    r = np.arange(4.0, 32, 0.05)
    t = np.radians(th_deg)
    x, y = r * np.cos(t), sgn * r * np.sin(t)
    ok = (x > -44.5) & (np.abs(y) < 29.9) & (x < 21.5)
    r, x, y = r[ok], x[ok], y[ok]
    n = samp(p, A['N'], x, y)
    hi = np.where(n > 2 * nsw)[0]
    bs = r[hi[-1]] if len(hi) and hi[-1] < len(r) - 3 else np.nan
    if p == 'eq':
        bz = samp(p, A['Bz'], x, y)
        neg = np.where(bz < 0)[0]
        mp = r[neg[0]] if len(neg) else np.nan
    else:  # inward scan from the sheath density peak to N < 0.3 * peak
        if np.isnan(bs):
            return np.nan, np.nan
        k = np.argmax(np.where(r <= bs, n, -1))
        below = np.where(n[:k] < 0.3 * n[k])[0]
        mp = r[below[-1]] if len(below) else np.nan
    return mp, bs


def shue(th, r0, a):
    return r0 * (2 / (1 + np.cos(th))) ** a


thetas = np.arange(0, 132.5, 2.5)
BND = {}
for p in ('eq', 'mer'):
    rows = []
    for th in thetas:
        for s in (1, -1):
            mp, bs = ray_boundaries(p, th, s)
            rows.append((th, s, mp, bs))
    BND[p] = np.array(rows, float)

shue98_r0 = (10.22 + 1.29 * np.tanh(0.184 * (-bsw + 8.14))) * pdsw ** (-1 / 6.6)
shue98_a = (0.58 - 0.007 * (-bsw)) * (1 + 0.024 * np.log(pdsw))
scale = (B0 / 31000.0) ** (1 / 3)
fits = {}
CUSP = (40, 85)
for p in ('eq', 'mer'):
    b = BND[p]
    th, mp = np.radians(b[:, 0]), b[:, 2]
    m = np.isfinite(mp) & (b[:, 0] <= 120)
    if p == 'mer':
        m &= ~((b[:, 0] > CUSP[0]) & (b[:, 0] < CUSP[1]))
    (r0, a), cov = curve_fit(shue, th[m], mp[m], p0=(11, 0.6))
    rms = float(np.sqrt(np.mean((shue(th[m], r0, a) - mp[m]) ** 2)))
    fits[p] = dict(r0=float(r0), alpha=float(a), r0_err=float(np.sqrt(cov[0, 0])), alpha_err=float(np.sqrt(cov[1, 1])), rms=rms, npts=int(m.sum()))
    # bow shock: Farris-type conic r' = L/(1+e cos th') about focus x0; fit in cartesian
    bs = b[:, 3]
    mb = np.isfinite(bs) & (b[:, 0] <= 110)
    X, Y = bs[mb] * np.cos(th[mb]), bs[mb] * np.sin(th[mb])

    def resid_conic(XY, L, e, x0):
        X, Y = XY
        rr = np.hypot(X - x0, Y)
        tt = np.arctan2(Y, X - x0)
        return rr - L / (1 + e * np.cos(tt))
    (L, e, x0), _ = curve_fit(resid_conic, (X, Y), np.zeros_like(X), p0=(25, 0.9, 3))
    fits[p].update(bs_L=float(L), bs_ecc=float(e), bs_x0=float(x0), bs_nose=float(x0 + L / (1 + e)))
fits['shue98'] = dict(r0=float(shue98_r0), alpha=float(shue98_a), r0_scaled=float(shue98_r0 * scale))
res['shue_fit'] = fits
print('fits', fits)

fig = plt.figure(figsize=(13.5, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
for k, p in enumerate(('eq', 'mer')):
    ax = fig.add_subplot(gs[k, 0])
    pl, A = PL[p], AV[p]
    im = ax.imshow(np.ma.array(np.clip(A['N'], 0.05, 40), mask=pl.R < RMIN).T, origin='lower', cmap='turbo',
                   norm=LogNorm(0.05, 40), extent=(pl.xu[0], pl.xu[-1], pl.tu[0], pl.tu[-1]))
    draw_earth(ax)
    b = BND[p]
    th = np.radians(b[:, 0])
    cm = (b[:, 0] > CUSP[0]) & (b[:, 0] < CUSP[1]) if p == 'mer' else np.zeros(len(b), bool)
    ax.plot(b[~cm, 2] * np.cos(th[~cm]), b[~cm, 1] * b[~cm, 2] * np.sin(th[~cm]), 'wo', ms=3, label='magnetopause (extracted)')
    if cm.any():
        ax.plot(b[cm, 2] * np.cos(th[cm]), b[cm, 1] * b[cm, 2] * np.sin(th[cm]), 'o', mfc='none', mec='w', ms=3, label='cusp (excluded)')
    ax.plot(b[:, 3] * np.cos(th), b[:, 1] * b[:, 3] * np.sin(th), 'k^', ms=3, label='bow shock (extracted)')
    tt = np.radians(np.linspace(-150, 150, 400))
    f = fits[p]
    rs = shue(tt, f['r0'], f['alpha'])
    ax.plot(rs * np.cos(tt), rs * np.sin(tt), 'w-', lw=1.5, label=f"Shue fit: $r_0$={f['r0']:.2f}, α={f['alpha']:.2f}")
    r98 = shue(tt, fits['shue98']['r0_scaled'], shue98_a)
    ax.plot(r98 * np.cos(tt), r98 * np.sin(tt), 'm--', lw=1.3, label=f"Shue98 empirical (dipole-scaled): $r_0$={fits['shue98']['r0_scaled']:.2f}, α={shue98_a:.2f}")
    tb = np.radians(np.linspace(-125, 125, 300))
    rb = f['bs_L'] / (1 + f['bs_ecc'] * np.cos(tb))
    ax.plot(f['bs_x0'] + rb * np.cos(tb), rb * np.sin(tb), 'k-', lw=1.2, label=f"conic BS fit: ε={f['bs_ecc']:.2f}, L={f['bs_L']:.1f}")
    ax.set_xlim(-30, 22); ax.set_ylim(-30, 30); ax.set_aspect('equal')
    ax.set_xlabel('$x$ ($R_E$)'); ax.set_ylabel(f'${pl.tname}$ ($R_E$)')
    ax.set_title(f"{'Equatorial' if p == 'eq' else 'Noon–midnight meridian'} plane, ⟨N⟩ for t ≥ 100")
    ax.legend(fontsize=7, loc='lower left', framealpha=0.7)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label(r'$\langle N\rangle$ (cm$^{-3}$)')
ax = fig.add_subplot(gs[:, 1])
for p, c in (('eq', 'C0'), ('mer', 'C3')):
    b = BND[p]
    ax.plot(b[:, 0], b[:, 2], '.', color=c, ms=4, label=f'{p}: extracted MP')
    tt = np.linspace(0, 130, 200)
    ax.plot(tt, shue(np.radians(tt), fits[p]['r0'], fits[p]['alpha']), '-', color=c, label=f"{p}: Shue fit")
ax.plot(tt, shue(np.radians(tt), fits['shue98']['r0_scaled'], shue98_a), 'm--', label='Shue98 (dipole-scaled)')
ax.plot(tt, shue(np.radians(tt), shue98_r0, shue98_a), 'm:', label='Shue98 (Earth dipole)')
ax.axvspan(*CUSP, color='0.85', zorder=-1, label='cusp sector (mer)')
ax.set_xlabel(r'angle from $+x$, $\theta$ (deg)'); ax.set_ylabel(r'$r_{mp}$ ($R_E$)')
ax.set_title('Magnetopause shape $r=r_0[2/(1+\\cos\\theta)]^{\\alpha}$')
ax.legend(fontsize=8)
fig.savefig(f'{OUT}/repro_shue_fit.png', dpi=110)
plt.close(fig)

# --------------------------------------------------------------- R2: Chapman-Ferraro pressure balance
A = AV['eq']
m0 = np.abs(eq.tu) < 0.3
line = {k: A[k][:, m0].mean(axis=1) for k in ('Pdyn', 'Pth', 'PB', 'Bmag', 'N', 'Vx')}
x = eq.xu
day = (x > 4) & (x < 21.5)
mp0, bs0 = fits['eq']['r0'], diag['bs_subsolar_late'][0]
ptot = line['Pdyn'] + line['Pth'] + line['PB']
i_in = (x > mp0 - 0.8) & (x < mp0 - 0.3)
B_in = float(line['Bmag'][i_in].mean())
B_dip = B0 / mp0 ** 3
k_gd = 0.881  # Spreiter et al. (1966): stagnation pressure / rho u^2 for gamma = 5/3, M >> 1
r_cf = lambda f: ((f * B0 * 1e-9) ** 2 / (2 * MU0 * k_gd * pdsw * 1e-9)) ** (1 / 6)
res['chapman_ferraro'] = dict(B_inside_mp_nT=B_in, B_dipole_at_mp_nT=B_dip, compression_f=B_in / B_dip,
                              PB_inside_over_Pdyn_sw=float((B_in * 1e-9) ** 2 / (2 * MU0) * 1e9 / pdsw),
                              r_cf_f2=float(r_cf(2.0)), r_cf_f244=float(r_cf(2.44)),
                              ptot_sw=float(ptot[(x > 17) & (x < 21)].mean()),
                              ptot_sheath=float(ptot[(x > mp0 + 0.5) & (x < bs0 - 0.8)].mean()),
                              ptot_inside_mp=float(ptot[i_in].mean()))
fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
ax = axs[0]
ax.plot(x[day], line['Pdyn'][day], label=r'dynamic $\rho V^2$')
ax.plot(x[day], line['Pth'][day], label=r'ion thermal $nk_BT_i$')
ax.plot(x[day], line['PB'][day], label=r'magnetic $B^2/2\mu_0$')
ax.plot(x[day], ptot[day], 'k', lw=2, label='sum')
for xx, lb in ((mp0, 'MP'), (bs0, 'BS')):
    ax.axvline(xx, color='0.4', ls='--'); ax.text(xx, 2.6, lb, ha='center')
ax.set_ylabel('pressure (nPa)'); ax.set_ylim(0, 2.8); ax.legend(fontsize=8, loc='upper left')
ax.set_title('Sun–Earth line (y = 0), t ≥ 100 average: momentum flux handed from flow → heat → field')
ax = axs[1]
ax.semilogy(x[day], line['Bmag'][day], 'k', lw=2, label='simulation |B|')
ax.semilogy(x[day], B0 / x[day] ** 3, 'C0--', label=f'dipole $B_0/r^3$ ($B_0$={B0:.0f} nT)')
ax.semilogy(x[day], 2 * B0 / x[day] ** 3, 'C1--', label='2 × dipole (planar image current)')
ax.axvline(mp0, color='0.4', ls='--'); ax.axvline(bs0, color='0.4', ls='--')
ax.set_ylabel('|B| (nT)'); ax.set_xlabel('x ($R_E$)'); ax.legend(fontsize=8); ax.set_ylim(3, 1000)
fig.savefig(f'{OUT}/repro_pressure_balance.png', dpi=110)
plt.close(fig)


# --------------------------------------------------------------- per-frame loops
def collect(p, t_min=0.0):
    pl = PL[p]
    m0 = np.abs(pl.tu) < 0.3
    i0 = np.argmin(abs(pl.xu))
    S = {k: [] for k in ('t', 'N', 'Bmag', 'Bz', 'Bx', 'Vx', 'Tpar', 'Tper', 'Ey', 'colN', 'colB')}
    lobe = {k: [] for k in ('B', 'N', 'Vin')}
    bz_pts = []
    rng = np.random.default_rng(1)
    frames = pl.frames()
    for j, f in enumerate(frames):
        d = pl.load(f)
        if d['time'] < t_min:
            continue
        S['t'].append(d['time'])
        for k in ('N', 'Bmag', 'Bz', 'Bx', 'Vx', 'Tpar', 'Tper', 'Ey'):
            S[k].append(d[k][:, m0].mean(axis=1))
        S['colN'].append(d['N'][i0]); S['colB'].append(d['Bmag'][i0])
        if p == 'mer':
            up_, dn_ = (pl.tu > 4) & (pl.tu < 8), (pl.tu < -4) & (pl.tu > -8)
            lobe['B'].append(0.5 * (np.abs(d['Bx'][:, up_]).mean(1) + np.abs(d['Bx'][:, dn_]).mean(1)))
            lobe['N'].append(0.5 * (d['N'][:, up_].mean(1) + d['N'][:, dn_].mean(1)))
            u2, d2 = (pl.tu > 2) & (pl.tu < 4), (pl.tu < -2) & (pl.tu > -4)
            lobe['Vin'].append(0.5 * (-d['Vz'][:, u2].mean(1) + d['Vz'][:, d2].mean(1)))
        if d['time'] >= 100 and j % 3 == 0:  # Brazil-plot sample of shocked solar wind
            bpar = 2 * MU0 * np.clip(d['N'], 1e-3, None) * 1e6 * np.clip(d['Tpar'], 1e-2, None) * QE / (d['Bmag'] * 1e-9) ** 2
            msk = (d['N'] > 1.5 * nsw) & (d['T'] > 4 * tsw) & (pl.R > 8) & (pl.XU > -10) & (pl.XU < 20)
            ii = np.flatnonzero(msk)
            ii = rng.choice(ii, min(len(ii), 15000), replace=False)
            bz_pts.append(np.stack([bpar.flat[ii], d['Tani'].flat[ii], pl.XU.flat[ii], pl.TU.flat[ii]], 1))
    S = {k: np.array(v) for k, v in S.items()}
    lobe = {k: np.array(v) for k, v in lobe.items()}
    return S, lobe, np.concatenate(bz_pts)


SE, _, BZ_eq = collect('eq')
SM, LOBE, BZ_mer = collect('mer')
print('loops done')

# --------------------------------------------------------------- R3: shock structure + Rankine-Hugoniot


def rh_perp(MA_, beta, g=5 / 3):
    p1 = beta / 2

    def F(r):
        u2 = MA_ / r
        p2 = MA_ ** 2 + p1 + 0.5 - MA_ ** 2 / r - r ** 2 / 2
        return (0.5 * MA_ ** 3 + g / (g - 1) * p1 * MA_ + MA_) - (0.5 * MA_ * u2 ** 2 + g / (g - 1) * p2 * u2 + r * MA_)
    try:
        return brentq(F, 1.0001, 3.9999)
    except ValueError:
        return np.nan


res['rh_check_textbook'] = dict(beta1_MA5=rh_perp(5, 1.0))
late = SE['t'] >= 100
xs = eq.xu
xp = np.arange(-2.8, 4.01, 0.1)
prof = {k: [] for k in ('N', 'Bmag', 'Vx', 'Tpar', 'Tper')}
for i in np.flatnonzero(late):
    hi = np.where((SE['N'][i] > 2 * nsw) & (xs > 5) & (xs < 21.5))[0]
    xb = xs[hi[-1]]
    for k in prof:
        prof[k].append(np.interp(xb + xp, xs, SE[k][i]))
prof = {k: np.array(v) for k, v in prof.items()}
up = (xp > 2) & (xp < 4)
dn = (xp > -1.0) & (xp < -0.5)
N1, N2 = prof['N'][:, up].mean(), prof['N'][:, dn].mean()
B1, B2 = prof['Bmag'][:, up].mean(), prof['Bmag'][:, dn].mean()
rN, rB = N2 / N1, B2 / B1
ov = prof['Bmag'].mean(0)
ramp = xp > -1
iov = np.flatnonzero(ramp)[np.argmax(ov[ramp])]
res['shock_subsolar'] = dict(r_N=float(rN), r_B=float(rB), r_RH_beta_i=rh_perp(MA, beta_i), r_RH_beta_2i=rh_perp(MA, 2 * beta_i),
                             B_overshoot_ratio=float(ov[iov] / B2), x_overshoot_rel=float(xp[iov]),
                             Tani_peak=float((prof['Tper'].mean(0) / prof['Tpar'].mean(0)).max()),
                             B_pileup_ratio_at_MP_side=float(prof['Bmag'][:, xp < -2.3].mean() / B1),
                             Tper_over_Tpar_down=float((prof['Tper'][:, dn] / prof['Tpar'][:, dn]).mean()),
                             gyro_radius_flow_RE=float(vsw * 1e3 / (QE * bsw * 1e-9 / MP) * 6.01 / 6.371e6))
fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
ax = axs[0]
for k, lb, c in (('N', '$N/N_1$', 'C0'), ('Bmag', '$|B|/B_1$', 'C3')):
    nrm = N1 if k == 'N' else B1
    mu, sd = prof[k].mean(0) / nrm, prof[k].std(0) / nrm
    ax.plot(xp, mu, color=c, lw=2, label=lb); ax.fill_between(xp, mu - sd, mu + sd, color=c, alpha=0.2)
ax.plot(xp, -prof['Vx'].mean(0) / vsw * 4, 'k--', label=r'$4\,|V_x|/V_{sw}$')
ax.plot(xp, prof['Tper'].mean(0) / prof['Tpar'].mean(0), 'C2:', lw=2, label=r'$T_\perp/T_\parallel$')
r1_, r2_ = rh_perp(MA, beta_i), rh_perp(MA, 2 * beta_i)
ax.axhline(r1_, color='0.3', ls=':', lw=1)
ax.text(1.9, r1_ - 0.22, f'R–H $r$ = {r1_:.2f} (β$_i$) / {r2_:.2f} ($T_e$=$T_i$)', fontsize=8, ha='left')
ax.axvspan(*xp[dn][[0, -1]], color='0.5', alpha=0.15)
ax.text(xp[dn].mean(), 0.25, 'downstream\nwindow', ha='center', fontsize=7)
ax.axvspan(*xp[up][[0, -1]], color='0.5', alpha=0.15)
ax.text(xp[up].mean(), 0.25, 'upstream window', ha='center', fontsize=7)
ax.axvspan(0, 0.9, color='y', alpha=0.2); ax.text(0.45, 6.1, 'foot', ha='center', fontsize=9)
ax.annotate('overshoot', (xp[iov], ov[iov] / B1), (xp[iov] - 1.1, 6.1), fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.text(-2.75, 6.1, '← toward MP\n(field pile-up)', fontsize=8, va='center')
ax.set_xlabel(r'$x - x_{BS}$ ($R_E$)  (upstream → right)'); ax.set_ylim(0, 6.6); ax.set_xlim(xp[0], xp[-1])
ax.set_title(f'Subsolar perpendicular shock, superposed over {late.sum()} frames')
ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 0.88))
ax = axs[1]
Ms = np.linspace(1.05, 14, 200)
for bt, ls in ((0.1, ':'), (beta_i, '-'), (2 * beta_i, '--'), (5, '-.')):
    ax.plot(Ms, [rh_perp(m, bt) for m in Ms], 'k', ls=ls, lw=1.2, label=f'β = {bt:.2g}')
ax.plot([MA], [rN], 'C0o', ms=9, label=f'simulation $N_2/N_1$ = {rN:.2f}')
ax.plot([MA], [rB], 'C3s', ms=9, label=f'simulation $B_2/B_1$ = {rB:.2f}')
ax.axhline(4, color='0.6', lw=0.8); ax.text(1.3, 4.05, 'strong-shock limit (γ=5/3)', fontsize=8)
ax.set_xlabel('Alfvén Mach number $M_A$'); ax.set_ylabel('compression ratio $r$')
ax.set_title('Perpendicular-shock Rankine–Hugoniot solution (γ = 5/3)')
ax.legend(fontsize=8, loc='lower right'); ax.set_ylim(1, 4.9)
fig.savefig(f'{OUT}/repro_shock_rh.png', dpi=110)
plt.close(fig)

# standoff (Spreiter 1966 / Farris & Russell 1994)
g = 5 / 3
for nm, M in (('Mms', Mms), ('MA', MA)):
    res.setdefault('standoff', {})[nm] = dict(M=M, spreiter=1.1 * ((g - 1) * M ** 2 + 2) / ((g + 1) * M ** 2),
                                              farris_russell=1.1 * ((g - 1) * M ** 2 + 2) / ((g + 1) * (M ** 2 - 1)))
fe = fits['eq']
Rc = fe['r0'] / (1 - fe['alpha'] / 2)
res['standoff']['sim_delta_over_Rmp'] = float((fe['bs_nose'] - fe['r0']) / fe['r0'])
res['standoff']['sim_delta_over_Rmp_direct'] = float((diag['bs_subsolar_late'][0] - diag['mp_subsolar_late'][0]) / diag['mp_subsolar_late'][0])
res['standoff']['Rc_nose_curvature'] = float(Rc)
res['standoff']['sim_delta_over_Rc'] = float((diag['bs_subsolar_late'][0] - diag['mp_subsolar_late'][0]) / Rc)

# --------------------------------------------------------------- R4: quasi-perpendicular vs quasi-parallel flank shock
Am = AV['mer']
gX, gT = np.gradient(np.log(np.clip(Am['N'], 0.05, None)), DX, DX)
bm = BND['mer']
thbn = []
for th, s, mp, bs in bm:
    if not np.isfinite(bs) or th > 120:
        continue
    t = np.radians(th)
    X, Z = bs * np.cos(t), s * bs * np.sin(t)
    nx_, nz_ = -samp('mer', gX, X, Z), -samp('mer', gT, X, Z)
    bxu, bzu = samp('mer', Am['Bx'], X + 2 * np.cos(t), Z + 2 * s * np.sin(t)), samp('mer', Am['Bz'], X + 2 * np.cos(t), Z + 2 * s * np.sin(t))
    c = abs(nx_ * bxu + nz_ * bzu) / np.hypot(nx_, nz_) / np.hypot(bxu, bzu)
    thbn.append((th, s, X, Z, np.degrees(np.arccos(np.clip(c, 0, 1)))))
thbn = np.array(thbn)
res['theta_Bn_mer'] = {f'{int(r[0])}deg_{"N" if r[1] > 0 else "S"}': float(r[4]) for r in thbn[::4]}
yy = eq.tu
for S, nm in ((SE, 'eq'), (SM, 'mer')):
    sel = S['t'] >= 60
    up_reg = (yy > 26) & (yy < 29.5)
    res.setdefault('flank_shock_x0', {})[nm] = dict(std_B_upstream_over_Bsw=float((S['colB'][sel][:, up_reg] / bsw).std(0).mean()),
                                                    std_N_upstream_over_nsw=float((S['colN'][sel][:, up_reg] / nsw).std(0).mean()))
# relative |B| fluctuation level just behind / ahead of the shock, along the same rays used for the boundary fit
fl = {}
for p in ('eq', 'mer'):
    A = AV[p]
    rel = A['std_Bmag'] / np.clip(A['Bmag'], 0.5, None)
    rows = []
    for th, s, mp, bs in BND[p]:
        if not (np.isfinite(bs) and np.isfinite(mp)) or th > 120:
            continue
        t = np.radians(th)
        rr_d = np.linspace(mp + 0.6 * (bs - mp), bs - 0.3, 6)
        rr_u = np.linspace(bs + 0.5, bs + 2.0, 4)
        dwn = samp(p, rel, rr_d * np.cos(t), s * rr_d * np.sin(t)).mean()
        upr = samp(p, rel, rr_u * np.cos(t), s * rr_u * np.sin(t)).mean()
        rows.append((th, s, dwn, upr))
    fl[p] = np.array(rows)
fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for ax, p in zip(axs[:2], ('eq', 'mer')):
    pl, A = PL[p], AV[p]
    rel = np.ma.array(A['std_Bmag'] / np.clip(A['Bmag'], 0.5, None), mask=pl.R < RMIN)
    im = ax.imshow(rel.T, origin='lower', cmap='magma', vmin=0, vmax=0.6, extent=(pl.xu[0], pl.xu[-1], pl.tu[0], pl.tu[-1]))
    draw_earth(ax)
    b = BND[p]
    th = np.radians(b[:, 0])
    if p == 'mer':
        sc = ax.scatter(thbn[:, 2], thbn[:, 3], c=thbn[:, 4], cmap='cool', vmin=30, vmax=90, s=18, edgecolor='k', lw=0.3, zorder=5)
        cax = ax.inset_axes([0.06, 0.06, 0.4, 0.035])
        fig.colorbar(sc, cax=cax, orientation='horizontal').ax.set_title(r'$\theta_{Bn}$ at bow shock (deg)', color='w', fontsize=8)
        cax.tick_params(colors='w', labelsize=7)
    else:
        ax.plot(b[:, 3] * np.cos(th), b[:, 1] * b[:, 3] * np.sin(th), 'o', ms=3, color='C0', mec='k', mew=0.3, label=r'bow shock, $\theta_{Bn}$ = 90° everywhere')
        ax.legend(fontsize=8, loc='lower left')
    ax.set_xlim(-25, 22); ax.set_ylim(-30, 30); ax.set_aspect('equal')
    ax.set_xlabel('$x$ ($R_E$)'); ax.set_ylabel(f'${pl.tname}$ ($R_E$)')
    ax.set_title(('Equatorial plane (IMF normal to plane)' if p == 'eq' else 'Meridian plane (IMF in plane)') + r': std$(|B|)/\langle|B|\rangle$, t ≥ 100', fontsize=10)
fig.colorbar(im, ax=axs[1], shrink=0.75).set_label(r'std$(|B|)/\langle|B|\rangle$')
ax = axs[2]
e = fl['eq']
for s, mk in ((1, '-'), (-1, '--')):
    r = e[e[:, 1] == s]
    ax.plot(r[:, 0], r[:, 2], 'C0' + mk, lw=2, label=f"eq {'dusk' if s > 0 else 'dawn'}: sheath (outer 40%)")
m = fl['mer']
for s, mk in ((1, '-'), (-1, '--')):
    r = m[(m[:, 1] == s) & (m[:, 0] <= 100)]
    ax.plot(r[:, 0], r[:, 2], 'C3' + mk, lw=2, label=f"mer {'north' if s > 0 else 'south'}: sheath (outer 40%)")
    ax.plot(r[:, 0], r[:, 3], 'C3' + mk, lw=1, alpha=0.5, label=f"mer {'north' if s > 0 else 'south'}: 0.5–2 $R_E$ upstream")
ee = e[(e[:, 1] > 0) & (e[:, 0] <= 100)]
ax.plot(ee[:, 0], ee[:, 3], 'C0-', lw=1, alpha=0.5, label='eq dusk: 0.5–2 $R_E$ upstream')
ax.set_xlabel(r'polar angle from $+x$, $\theta$ (deg)'); ax.set_ylabel(r'std$(|B|)/\langle|B|\rangle$'); ax.set_ylim(0, 0.34); ax.set_xlim(0, 100)
ax.axvspan(*CUSP, color='0.9', zorder=-2)
axt = ax.twinx(); axt.grid(False)
for s, mk in ((1, 'k-'), (-1, 'k--')):
    r = thbn[thbn[:, 1] == s]
    axt.plot(r[:, 0], r[:, 4], mk, lw=0.8, alpha=0.6)
axt.set_ylabel(r'meridian $\theta_{Bn}$ (deg, thin black)'); axt.set_ylim(0, 95)
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.set_title(r'Fluctuation level vs shock position / $\theta_{Bn}$', fontsize=10)
fig.savefig(f'{OUT}/repro_qperp_qpar.png', dpi=110)
plt.close(fig)
for p in ('eq', 'mer'):
    r = fl[p]
    res.setdefault('shock_fluct', {})[p] = {f'{lo}-{hi}deg': dict(sheath=float(r[(r[:, 0] >= lo) & (r[:, 0] < hi), 2].mean()), upstream=float(r[(r[:, 0] >= lo) & (r[:, 0] < hi), 3].mean()))
                                            for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 121))}

# --------------------------------------------------------------- R5: Brazil plot


def thr(bp, a, b, b0):
    return 1 + a / np.clip(bp - b0, 1e-6, None) ** b


HEL = {'proton cyclotron': (0.43, 0.42, -0.0004, 'C1'), 'mirror': (0.77, 0.76, -0.016, 'C3'),
       'parallel firehose': (-0.47, 0.53, 0.59, 'C2'), 'oblique firehose': (-1.4, 1.0, -0.11, 'C4')}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
for ax, (P, nm, p) in zip(axs, ((BZ_eq, 'equatorial sheath (q-perp)', 'eq'), (BZ_mer, 'meridian sheath', 'mer'))):
    bp, an = P[:, 0], P[:, 1]
    ok = (bp > 1e-2) & (bp < 1e2) & (an > 0.1) & (an < 10)
    # fractional depth in the sheath: 0 at the (time-averaged) MP, 1 at the bow shock
    b_ = BND[p]
    thb = np.arange(0, 125, 2.5)
    r_bs = np.array([np.nanmean(b_[b_[:, 0] == t, 3]) for t in thb])
    r_mp = shue(np.radians(thb), fits[p]['r0'], fits[p]['alpha'])
    thp = np.degrees(np.arctan2(np.abs(P[:, 3]), P[:, 2]))
    rp = np.hypot(P[:, 2], P[:, 3])
    fdep = (rp - np.interp(thp, thb, r_mp)) / (np.interp(thp, thb, r_bs) - np.interp(thp, thb, r_mp))
    h = ax.hist2d(np.log10(bp[ok]), np.log10(an[ok]), bins=[120, 100], range=[[-2, 2], [-1, 1]], cmap='Greys', norm=LogNorm(), rasterized=True)
    bb = np.logspace(-2, 2, 300)
    for k, (a, b, b0, c) in HEL.items():
        v = thr(bb, a, b, b0)
        good = v > 0
        ax.plot(np.log10(bb[good]), np.log10(v[good]), c, lw=2, label=k)
    ax.plot(np.log10(bb), np.log10(1 + 1 / bb), 'k:', lw=1.5, label=r'bi-Maxwellian mirror: $T_\perp/T_\parallel-1=1/\beta_\perp$ (approx.)')
    edges = np.linspace(-1.5, 1.5, 16)
    out = {}
    for lbl, sm, c in (('near shock (depth > 0.7)', fdep > 0.7, 'deepskyblue'), ('near MP (depth < 0.3)', (fdep < 0.3) & (fdep > -0.1), 'gold')):
        q = ok & sm
        lb, la = np.log10(bp[q]), np.log10(an[q])
        idx = np.digitize(lb, edges)
        med = [np.median(la[idx == k]) if (idx == k).sum() > 50 else np.nan for k in range(1, len(edges))]
        ax.plot(0.5 * (edges[1:] + edges[:-1]), med, 'o-', color=c, mec='k', lw=2, ms=5, label=f'median, {lbl}')
        out[lbl.split(' (')[0].replace(' ', '_')] = dict(npts=int(q.sum()), median_beta_par=float(np.median(bp[q])), median_Tani=float(np.median(an[q])),
                                                         frac_above_mirror=float((an[q] > thr(bp[q], *HEL['mirror'][:3])).mean()))
    ax.set_xlim(-2, 2); ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$\log_{10}\beta_{\parallel i}$'); ax.set_ylabel(r'$\log_{10}(T_\perp/T_\parallel)$')
    ax.set_title(f'{nm}, t ≥ 100')
    ax.legend(fontsize=7, loc='lower left')
    above_m = an[ok] > thr(bp[ok], *HEL['mirror'][:3])
    above_c = an[ok] > thr(bp[ok], *HEL['proton cyclotron'][:3])
    res.setdefault('brazil', {})[nm] = dict(npts=int(ok.sum()), median_beta_par=float(np.median(bp[ok])), median_Tani=float(np.median(an[ok])),
                                            frac_above_mirror=float(above_m.mean()), frac_above_cyclotron=float(above_c.mean()), **out)
fig.colorbar(h[3], ax=axs, shrink=0.8).set_label('counts')
fig.savefig(f'{OUT}/repro_brazil.png', dpi=110)
plt.close(fig)

# --------------------------------------------------------------- R6: KHI criterion along the flanks
A = AV['eq']
krow = []
for xx in np.arange(4, -20.5, -1.0):
    i = np.argmin(abs(eq.xu - xx))
    for s, side in ((1, 'dusk'), (-1, 'dawn')):
        yv = eq.tu * s
        sel = np.flatnonzero(yv > 4)
        sel = sel[np.argsort(yv[sel])]
        neg = [j for j in sel if A['Bz'][i, j] < 0]
        if not neg:
            continue
        ymp = yv[neg[0]]
        jo = np.flatnonzero((yv > ymp + 1.5) & (yv < ymp + 2.5))
        ji = np.flatnonzero((yv > ymp - 2.5) & (yv < ymp - 1.5))
        g_ = lambda k, J: float(A[k][i, J].mean())
        r1, r2 = g_('N', jo) * 1e6 * MP, g_('N', ji) * 1e6 * MP
        dv = np.hypot(g_('Vx', jo) - g_('Vx', ji), g_('Vy', jo) - g_('Vy', ji)) * 1e3
        lhs = r1 * r2 / (r1 + r2) * dv ** 2 * 1e9
        kx = np.array([g_('Vx', jo) - g_('Vx', ji), g_('Vy', jo) - g_('Vy', ji)]); kx /= np.linalg.norm(kx)
        bk1 = (g_('Bx', jo) * kx[0] + g_('By', jo) * kx[1]) * 1e-9
        bk2 = (g_('Bx', ji) * kx[0] + g_('By', ji) * kx[1]) * 1e-9
        rhs_k = (bk1 ** 2 + bk2 ** 2) / MU0 * 1e9
        B1t = np.sqrt(g_('Bx', jo) ** 2 + g_('By', jo) ** 2 + g_('Bz', jo) ** 2) * 1e-9
        B2t = np.sqrt(g_('Bx', ji) ** 2 + g_('By', ji) ** 2 + g_('Bz', ji) ** 2) * 1e-9
        rhs_t = (B1t ** 2 + B2t ** 2) / MU0 * 1e9
        cf1 = np.hypot(va(B1t * 1e9, g_('N', jo)), np.sqrt(5 / 3 * g_('T', jo) * QE / MP) / 1e3)
        krow.append((xx, s, ymp, lhs, rhs_k, rhs_t, dv / 1e3, dv / 1e3 / cf1,
                     float(A['std_N'][i, np.flatnonzero(np.abs(yv - ymp) < 1.5)].mean())))
krow = np.array(krow)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
for s, c, side in ((1, 'C3', 'dusk'), (-1, 'C0', 'dawn')):
    r = krow[krow[:, 1] == s]
    axs[0].semilogy(r[:, 0], r[:, 3], c + '-', lw=2, label=f'{side}: ' + r'$\frac{\rho_1\rho_2}{\rho_1+\rho_2}(\Delta v)^2$')
    axs[0].semilogy(r[:, 0], r[:, 4], c + ':', label=f'{side}: ' + r'$[(\mathbf{\hat k\cdot B_1})^2+(\mathbf{\hat k\cdot B_2})^2]/\mu_0$')
    axs[0].semilogy(r[:, 0], r[:, 5], c + '--', lw=1, label=f'{side}: ' + r'$(B_1^2+B_2^2)/\mu_0$ (k ∥ B worst case)')
    axs[1].plot(r[:, 0], r[:, 7], c, lw=2, label=f'{side}: ' + r'$\Delta v / c_{f,sheath}$')
    axs[2].plot(r[:, 0], r[:, 8], c, lw=2, label=f'{side}')
axs[0].set_xlabel('x ($R_E$)'); axs[0].set_ylabel('nPa'); axs[0].legend(fontsize=7); axs[0].invert_xaxis()
axs[0].set_title('Chandrasekhar (1961) KHI criterion, ±2 $R_E$ across the MP')
axs[1].axhline(2, color='k', ls=':'); axs[1].set_xlabel('x ($R_E$)'); axs[1].invert_xaxis(); axs[1].legend(fontsize=8)
axs[1].set_title('shear fast-Mach number (compressional stabilisation ≳ 2)')
axs[2].set_xlabel('x ($R_E$)'); axs[2].set_ylabel(r'std$(N)$ within ±1.5 $R_E$ of MP (cm$^{-3}$)'); axs[2].invert_xaxis(); axs[2].legend()
axs[2].set_title('measured boundary fluctuation level')
fig.savefig(f'{OUT}/repro_khi_criterion.png', dpi=110)
plt.close(fig)
for s, side in ((1, 'dusk'), (-1, 'dawn')):
    r = krow[(krow[:, 1] == s) & (krow[:, 0] <= 0) & (krow[:, 0] >= -20)]
    res.setdefault('khi', {})[side] = dict(lhs_nPa=float(r[:, 3].mean()), rhs_k_nPa=float(r[:, 4].mean()), rhs_total_nPa=float(r[:, 5].mean()),
                                           dv_kms=float(r[:, 6].mean()), Mf_shear=float(r[:, 7].mean()), stdN=float(r[:, 8].mean()))

# --------------------------------------------------------------- R7: tail flows, braking, X-line
selE, selM = SE['t'] >= 100, SM['t'] >= 100
tailx = (eq.xu < -6) & (eq.xu > -44)
fig, axs = plt.subplots(3, 1, figsize=(10, 10.5), constrained_layout=True)
for S, sel, nm, c in ((SE, selE, 'equatorial (y=0)', 'C0'), (SM, selM, 'meridian (z=0)', 'C3')):
    v = S['Vx'][sel][:, tailx]
    axs[0].plot(eq.xu[tailx], np.median(v, 0), c, lw=2, label=f'{nm}: median')
    axs[0].fill_between(eq.xu[tailx], np.percentile(v, 10, 0), np.percentile(v, 90, 0), color=c, alpha=0.2, label=f'{nm}: 10–90%')
    b = S['Bz'][sel][:, tailx]
    axs[1].semilogy(eq.xu[tailx], np.clip(np.median(b, 0), 0.1, None), c, lw=2, label=f'{nm}: median $B_z$')
    axs[2].plot(eq.xu[tailx], (v > 400).mean(0), c, lw=2, label=f'{nm}: P($V_x$ > 400 km/s), BBF')
    axs[2].plot(eq.xu[tailx], (v < -400).mean(0), c + '--', lw=1.5, label=f'{nm}: P($V_x$ < −400 km/s)')
axs[0].axhline(0, color='k', lw=0.6); axs[0].set_ylabel('$V_x$ (km/s)'); axs[0].legend(fontsize=8, ncol=2)
axs[0].axvspan(-30, -20, color='y', alpha=0.15); axs[0].text(-25, -700, 'Nagai et al. (1998) X-line range', ha='center', fontsize=8)
axs[0].set_title('Tail flows, t ≥ 100: earthward jets brake where the dipole field takes over')
xd = eq.xu[tailx]
axs[1].semilogy(xd, B0 / np.abs(xd) ** 3, 'k--', label='dipole $B_0/|x|^3$')
axs[1].set_ylabel('$B_z$ (nT)'); axs[1].legend(fontsize=8); axs[1].set_ylim(0.1, 100)
axs[2].set_ylabel('occurrence'); axs[2].set_xlabel('x ($R_E$)'); axs[2].legend(fontsize=8)
for ax in axs:
    ax.set_xlim(-44, -6)
fig.savefig(f'{OUT}/repro_tail_flows.png', dpi=110)
plt.close(fig)
vE = SE['Vx'][selE][:, tailx]
xb = eq.xu[tailx]
res['tail'] = dict(x_peak_median_Vx_eq=float(xb[np.argmax(np.median(vE, 0))]), peak_median_Vx_eq=float(np.median(vE, 0).max()),
                   x_where_median_Vx_drops_below_100_eq=float(xb[(np.median(vE, 0) > 100)].max()),
                   P_BBF_at_x20_eq=float((vE[:, np.argmin(abs(xb + 20))] > 400).mean()))

# X-line per frame (meridian): sign change of Bz on z=0 from + (earthward) to - (tailward), most earthward
xl, rate, rate_in, vout_ratio = [], [], [], []
xm = me.xu
for i in np.flatnonzero(selM):
    bz = uniform_filter1d(SM['Bz'][i], 5)
    cand = [j for j in range(len(xm) - 1, 0, -1) if xm[j] < -10 and bz[j] > 0 and bz[j - 1] < 0]
    if not cand:
        continue
    j = cand[0]
    xl.append(xm[j])
    win = np.abs(xm - xm[j]) < 0.5
    Ey = SM['Ey'][i][win].mean()
    BL, NL = LOBE['B'][i][win].mean(), LOBE['N'][i][win].mean()
    VAL = va(BL, NL)
    rate.append(Ey / (BL * VAL * 1e-3))
    rate_in.append(LOBE['Vin'][i][win].mean() / VAL)
    vx = SM['Vx'][i]
    w2 = (xm > xm[j] - 6) & (xm < xm[j] + 6)
    vout_ratio.append(np.max(np.abs(vx[w2])) / VAL)
xl, rate, rate_in, vout_ratio = map(np.array, (xl, rate, rate_in, vout_ratio))
res['tail_reconnection'] = dict(nframes=int(len(xl)), x_line_median=float(np.median(xl)), x_line_p10=float(np.percentile(xl, 10)),
                                x_line_p90=float(np.percentile(xl, 90)), rate_E_median=float(np.median(rate)),
                                rate_E_iqr=[float(np.percentile(rate, 25)), float(np.percentile(rate, 75))],
                                rate_inflow_median=float(np.median(rate_in)), vout_over_VA_median=float(np.median(vout_ratio)),
                                VA_lobe_median=float(np.median([va(b, n) for b, n in zip(LOBE['B'][selM].mean(1), LOBE['N'][selM].mean(1))])))

# dayside Cassak-Shay (2007) asymmetric reconnection estimate from mer average along z = 0
Am = AV['mer']
mz = np.abs(me.tu) < 0.5
lineM = {k: Am[k][:, mz].mean(1) for k in ('N', 'Bz', 'Ey')}
xmp = fits['eq']['r0']
o = (xm > xmp + 0.8) & (xm < xmp + 1.5)
iN = (xm > xmp - 1.5) & (xm < xmp - 0.8)
B1d, B2d = abs(lineM['Bz'][o].mean()), abs(lineM['Bz'][iN].mean())
n1d, n2d = lineM['N'][o].mean(), lineM['N'][iN].mean()
rho_out = (n1d * B2d + n2d * B1d) / (B1d + B2d) * 1e6 * MP
v_out = np.sqrt(B1d * B2d * 1e-18 / (MU0 * rho_out)) / 1e3
E_cs = 0.1 * 2 * B1d * B2d / (B1d + B2d) * v_out * 1e-3  # mV/m
E_meas = recon['Ey_dayMP']
res['dayside_reconnection'] = dict(B_sheath=float(B1d), B_msph=float(B2d), n_sheath=float(n1d), n_msph=float(n2d),
                                   v_out_CS=float(v_out), E_CS07_mVm=float(E_cs), E_measured_mVm=E_meas,
                                   ratio_meas_over_CS=float(E_meas / E_cs), E_sw=float(recon['Ey_sw']))
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
ax = axs[0]
ax.hist(xl, bins=np.arange(-44, -9, 1), color='C3', alpha=0.8)
ax.axvspan(-30, -20, color='y', alpha=0.25, label='Nagai et al. (1998, 2005): −20 to −30 $R_E$')
ax.set_xlabel('X-line x ($R_E$), meridian $B_z$ reversal on z = 0'); ax.set_ylabel('frames'); ax.legend(fontsize=8)
ax.set_title('Near-Earth neutral line location (t ≥ 100)')
ax = axs[1]
ax.hist(rate, bins=np.linspace(-0.2, 0.6, 33), color='C0', alpha=0.8, label=r'$E_y/(B_L V_{A,L})$')
ax.hist(rate_in, bins=np.linspace(-0.2, 0.6, 33), color='C2', alpha=0.5, label=r'$V_{in}/V_{A,L}$')
ax.axvline(0.1, color='k', ls='--', label='≈ 0.1 (Cassak et al. 2017)')
ax.set_xlabel('normalized reconnection rate'); ax.legend(fontsize=8); ax.set_title('Tail reconnection rate')
ax = axs[2]
vals = [recon['Ey_sw'], E_meas, E_cs]
ax.bar(['$E_{sw}=V_{sw}B_z$', 'measured $E_y$\nat dayside MP', 'Cassak–Shay\n(2007) estimate'], vals, color=['0.6', 'C3', 'C0'])
for k_, v in enumerate(vals):
    ax.text(k_, v + 0.03, f'{v:.2f}', ha='center')
ax.set_ylabel('mV/m'); ax.set_title('Dayside reconnection electric field')
fig.savefig(f'{OUT}/repro_recon_rate.png', dpi=110)
plt.close(fig)

# --------------------------------------------------------------- R8: BBF / dipolarization-front superposed epoch
pe = dict(np.load('output/cache/probe_eq.npz'))
tp = pe['t']
dt = np.median(np.diff(tp))
DFXS = ('x-16', 'x-18', 'x-20', 'x-22')
lag = max(1, int(round(2.0 / dt)))
W = int(round(10 / dt))
tt = np.arange(-W, W + 1) * dt
ev, evp = [], []
for DFX in DFXS:
    bz, vx = pe[f'{DFX}|Bz'], pe[f'{DFX}|Vx']
    dbz = np.r_[np.zeros(lag), bz[lag:] - bz[:-lag]]
    thr_ = max(2.0, 2 * np.std(dbz[tp > 40]))
    last = -1e9
    for i in np.flatnonzero((dbz > thr_) & (tp > 40)):
        j = i - lag + np.argmax(np.diff(bz[i - lag:i + 1]))  # steepest step
        if vx[max(0, j - lag):j + lag + 1].max() > 200 and tp[j] - last > 6 and W <= j < len(tp) - W:
            ev.append((DFX, j)); last = tp[j]
sea = {k: np.array([pe[f'{X}|{k}'][j - W:j + W + 1] for X, j in ev]) for k in ('Bz', 'Vx', 'N', 'T')}
pre = slice(W - 6, W - 1)
sea['dBz'] = sea['Bz'] - sea['Bz'][:, pre].mean(1, keepdims=True)
sea['Nrel'] = sea['N'] / sea['N'][:, pre].mean(1, keepdims=True)
res['df'] = dict(probes=list(DFXS), n_events=len(ev), events=[(X, float(tp[j])) for X, j in ev], dt=float(dt),
                 dBz_jump_mean=float(sea['dBz'][:, W + 2:W + 6].mean()), dip_before_mean=float(sea['dBz'][:, W - 2:W].mean()),
                 N_after_over_before=float(np.median(sea['Nrel'][:, W + 2:W + 8].mean(1))),
                 Vx_peak_mean=float(sea['Vx'][:, W - 2:W + 3].max(1).mean()),
                 T_after_over_before=float(np.median((sea['T'][:, W + 2:W + 8].mean(1) / sea['T'][:, pre].mean(1)))))
fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1])
labs = {'Bz': '$B_z$ (nT)', 'Vx': '$V_x$ (km/s)', 'N': '$N$ (cm$^{-3}$)'}
cols = {'x-16': 'C2', 'x-18': 'C3', 'x-20': 'C0', 'x-22': 'C4'}
for r_, k in enumerate(('Bz', 'Vx', 'N')):
    ax = fig.add_subplot(gs[r_, 0])
    for xx in DFXS:
        ax.plot(tp, pe[f'{xx}|{k}'], cols[xx], lw=0.9, label=f'x = {xx[1:]} $R_E$')
    for X, j in ev:
        ax.plot(tp[j], 1, 'v', color=cols[X], ms=6, mec='k', mew=0.4, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_ylabel(labs[k]); ax.set_xlim(40, tp[-1])
    if k == 'Bz':
        ax.set_ylim(-5, 20)
    if k == 'N':
        ax.set_ylim(0, 2.2)
    if k == 'Vx':
        ax.axhline(400, color='k', ls=':', lw=0.8); ax.text(41, 430, 'BBF threshold 400 km/s (Angelopoulos et al. 1992)', fontsize=8)
    if r_ == 0:
        ax.legend(fontsize=8, ncol=4, loc='upper left'); ax.set_title('Virtual probes on the tail axis (equatorial plane, y = 0); ▼ = detected fronts (colour = probe)')
ax.set_xlabel(r'$t$ ($\Omega_i^{-1}$)')
slabs = {'dBz': r'$B_z - B_{z,\rm pre}$ (nT)', 'Vx': '$V_x$ (km/s)', 'Nrel': r'$N/N_{\rm pre}$'}
for r_, k in enumerate(('dBz', 'Vx', 'Nrel')):
    ax = fig.add_subplot(gs[r_, 1])
    mu, q1, q3 = np.median(sea[k], 0), np.percentile(sea[k], 25, 0), np.percentile(sea[k], 75, 0)
    ax.plot(tt, mu, 'k', lw=2, label='median'); ax.fill_between(tt, q1, q3, color='0.75', label='IQR')
    ax.axvline(0, color='C3'); ax.set_ylabel(slabs[k])
    if k == 'Nrel':
        ax.set_ylim(0, 4)
    if r_ == 0:
        ax.set_title(f'Superposed epoch of {len(ev)} fronts (x = −16 … −22 $R_E$)'); ax.legend(fontsize=8)
ax.set_xlabel(r'$t - t_{DF}$ ($\Omega_i^{-1}$)')
fig.savefig(f'{OUT}/repro_df_bbf.png', dpi=110)
plt.close(fig)

# --------------------------------------------------------------- R9: FTE virtual probes (meridian magnetopause)
pm = dict(np.load('output/cache/probe_mer.npz'))
tm = pm['t']
fm = fits['mer']
fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
fte = {}
for ax, nm in zip(axs, ('mpN20', 'mpN30', 'mpS20', 'mpS30')):
    lat = int(nm[3:]); s = 1 if nm[2] == 'N' else -1
    th = np.radians(lat)
    rr = 11.6 * (2 / (1 + np.cos(th))) ** 0.6  # probe position used in repro_cache.py
    dr = rr * 0.6 * np.sin(th) / (1 + np.cos(th))
    rhat = np.array([np.cos(th), s * np.sin(th)]); thh = np.array([-np.sin(th), s * np.cos(th)])
    n = rhat - dr / rr * thh; n /= np.linalg.norm(n)
    Bn = pm[f'{nm}|Bx'] * n[0] + pm[f'{nm}|Bz'] * n[1]
    ax.plot(tm, Bn, 'C3', label='$B_n$ (outward normal)')
    ax.plot(tm, pm[f'{nm}|Bmag'], 'k', lw=0.8, label='|B|')
    ax.plot(tm, pm[f'{nm}|N'] * 5, 'C0', lw=0.8, alpha=0.7, label='5 × N')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_ylabel('nT'); ax.set_title(f'probe at ({rr * np.cos(th):.1f}, {s * rr * np.sin(th):.1f}) $R_E$, latitude {s * lat}°', fontsize=9)
    sel = tm >= 60
    b = Bn[sel] - uniform_filter1d(Bn[sel], 9)
    zc = np.flatnonzero(np.diff(np.sign(b)) != 0)
    fte[nm] = dict(std_Bn=float(Bn[sel].std()), std_Bn_detrended=float(b.std()),
                   n_bipolar_gt5nT=int(sum(1 for z in zc if z > 2 and z < len(b) - 3 and b[z - 2] * b[z + 3] < 0 and min(abs(b[z - 2]), abs(b[z + 3])) > 5)))
axs[0].legend(fontsize=8, ncol=3); axs[-1].set_xlabel(r'$t$ ($\Omega_i^{-1}$)')
fig.suptitle('Virtual magnetopause probes (meridian plane): search for Russell–Elphic bipolar $B_n$')
fig.savefig(f'{OUT}/repro_fte_probes.png', dpi=110)
plt.close(fig)
res['fte_probes'] = fte

# --------------------------------------------------------------- R10: Dungey/Axford-Hines flow pattern
A = AV['eq']
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
for ax, (p, comp) in zip(axs, (('eq', 'Vy'), ('mer', 'Vz'))):
    pl, A = PL[p], AV[p]
    vx_, vt_ = A['Vx'], A[comp]
    msk = pl.R < RMIN
    im = ax.imshow(np.ma.array(vx_, mask=msk).T, origin='lower', cmap='RdBu_r', norm=TwoSlopeNorm(0, -500, 500),
                   extent=(pl.xu[0], pl.xu[-1], pl.tu[0], pl.tu[-1]))
    U, V = np.where(msk, np.nan, vx_).T, np.where(msk, np.nan, vt_).T
    sp = np.hypot(U, V)
    ax.streamplot(pl.xu, pl.tu, U, V, density=2.2, color='k', linewidth=0.5, arrowsize=0.7)
    b = BND[p]; th = np.radians(b[:, 0])
    ax.plot(b[:, 2] * np.cos(th), b[:, 1] * b[:, 2] * np.sin(th), 'y.', ms=3)
    draw_earth(ax)
    ax.set_xlim(-40, 20); ax.set_ylim(-28, 28); ax.set_aspect('equal')
    ax.set_xlabel('$x$ ($R_E$)'); ax.set_ylabel(f'${pl.tname}$ ($R_E$)')
    ax.set_title(('Equatorial' if p == 'eq' else 'Meridian') + ' plane: ⟨V⟩ streamlines (t ≥ 100), colour = ⟨$V_x$⟩')
    fig.colorbar(im, ax=ax, shrink=0.8).set_label('$V_x$ (km/s)')
fig.savefig(f'{OUT}/repro_convection.png', dpi=110)
plt.close(fig)
A = AV['eq']
inside = (np.abs(eq.TU) < 6) & (eq.XU < -8) & (eq.XU > -14)
res['convection'] = dict(mean_Vx_inner_tail_eq=float(A['Vx'][inside].mean()),
                         mean_Vx_flank_LLBL_dusk=float(A['Vx'][(eq.XU > -10) & (eq.XU < 0) & (eq.TU > 12) & (eq.TU < 14)].mean()))

json.dump(res, open('output/repro.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
