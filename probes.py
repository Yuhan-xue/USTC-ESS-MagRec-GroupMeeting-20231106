"""Point-probe diagnostics (equatorial): flank KHI periodicity & tail flow reversal."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from magviz import Plane

OUT = 'output/figs'
eq = Plane('eq')
fr = eq.frames()
probes = {'dusk (-10, +16.5)': (-10, 16.5), 'dawn (-10, -16.5)': (-10, -16.5),
          'dusk (-20, +18)': (-20, 18), 'dawn (-20, -18)': (-20, -18)}
idx = {k: (np.argmin(abs(eq.xu - x)), np.argmin(abs(eq.tu - y))) for k, (x, y) in probes.items()}
ts = []
P = {k: {q: [] for q in ('N', 'Vx', 'Bz', 'Vy')} for k in probes}
xrev = []
ymid = np.abs(eq.tu) < 4
for f in fr:
    d = eq.load(f)
    ts.append(d['time'])
    for k, (i, j) in idx.items():
        for q in P[k]:
            P[k][q].append(d[q][i - 2:i + 3, j - 2:j + 3].mean())
    vx = d['Vx'][:, ymid].mean(axis=1)
    vx = uniform_filter1d(vx, 10)
    tail = (eq.xu < -8)
    xs, v = eq.xu[tail], vx[tail]
    # scanning tailward from x=-8: first point where Vx < -150 after having Vx > 150 earthward of it
    pos = np.where(v > 150)[0]
    cand = np.nan
    if len(pos):
        k0 = pos[-1]  # most earthward... xs ascending, so last index = most earthward positive
        k1 = pos[0]   # most tailward positive
        neg = np.where(v[:k1] < -150)[0]
        if len(neg):
            cand = 0.5 * (xs[neg[-1]] + xs[k1])
    xrev.append(cand)
ts = np.array(ts); xrev = np.array(xrev)

res = {}
fig, axs = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
for k in ('dusk (-10, +16.5)', 'dawn (-10, -16.5)'):
    axs[0].plot(ts, P[k]['N'], label=k)
    axs[1].plot(ts, P[k]['Vx'], label=k)
axs[0].set_ylabel(r'$N_i$ (cm$^{-3}$)'); axs[1].set_ylabel(r'$V_x$ (km/s)')
axs[0].legend(); axs[0].set_title('Flank magnetopause probes (equatorial plane)')
dt = ts[1] - ts[0]
late = ts >= 60
for k in probes:
    for q in ('N', 'Vx'):
        s = np.array(P[k][q])[late]
        s = s - s.mean()
        s = s * np.hanning(len(s))
        F = np.abs(np.fft.rfft(s)) ** 2
        fq = np.fft.rfftfreq(len(s), dt)
        if q == 'N':
            axs[2].semilogy(1 / fq[1:], F[1:], label=k)
        m = (1 / np.clip(fq, 1e-9, None) > 2) & (1 / np.clip(fq, 1e-9, None) < 40)
        res[f'{k} {q}'] = dict(std=float(np.array(P[k][q])[late].std()),
                               peak_period=float(1 / fq[m][np.argmax(F[m])]))
axs[2].set_xscale('log'); axs[2].set_xlabel(r'period ($\Omega_i^{-1}$)'); axs[2].set_ylabel('PSD of $N_i$ (t ≥ 60)')
axs[2].legend(fontsize=8)
axs[1].set_xlabel(r'$t$ ($\Omega_i^{-1}$)')
fig.savefig(f'{OUT}/fig_flank_probes.png', dpi=110)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
ax.plot(ts, xrev, '.', ms=3)
ax.set_ylabel(r'$x$ of $V_x$ reversal ($R_E$)'); ax.set_xlabel(r'$t$ ($\Omega_i^{-1}$)')
ax.set_title('Tail flow-reversal point (|y|<4, z=0): proxy for the near-Earth X-line')
ax.grid(alpha=.3)
fig.savefig(f'{OUT}/fig_tail_xline_eq.png', dpi=110)
plt.close(fig)
s = ts >= 80
res['xrev_eq_late'] = dict(median=float(np.nanmedian(xrev[s])), p10=float(np.nanpercentile(xrev[s], 10)),
                           p90=float(np.nanpercentile(xrev[s], 90)), frac_found=float(np.isfinite(xrev[s]).mean()),
                           first_t=float(ts[np.isfinite(xrev) & (ts > 30)][0]) if np.any(np.isfinite(xrev) & (ts > 30)) else None)
json.dump(res, open('output/probes.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
