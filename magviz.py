"""Shared loading / regridding / plotting helpers for the ear-Reynolds-S slices."""
import glob
import os
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from matplotlib.patches import Circle, Wedge

ROOT = os.path.dirname(os.path.abspath(__file__))
EV_K = 11604.518
RMIN = 3.0          # inner boundary of the simulation (rmin in fefi.input)
XLIM = (-45.0, 22.0)
TLIM = (-30.0, 30.0)
DX = 0.1


class Plane:
    """Holds the native grid of one plane plus a uniform 0.1 Re grid."""

    def __init__(self, plane):
        self.plane = plane
        raw_root = os.environ.get('MAG_RAW_DIR', os.path.join(ROOT, 'data'))
        self.dir = os.path.join(raw_root, plane)
        if not os.path.isfile(os.path.join(self.dir, 'coords.npz')):
            raise FileNotFoundError(
                f"Missing private raw coordinates: {self.dir}. "
                "Set MAG_RAW_DIR to the parent of eq/ and mer/; "
                "see docs/REPRODUCING.md. Do not commit raw inputs."
            )
        c = np.load(os.path.join(self.dir, 'coords.npz'))
        self.x = c['x'].astype(float)
        self.t = c['t'].astype(float)  # y (eq) or z (mer)
        self.u = {k[2:]: float(c[k]) for k in c.files if k.startswith('u_')}
        self.xu = np.arange(XLIM[0], XLIM[1] + 1e-6, DX)
        self.tu = np.arange(TLIM[0], TLIM[1] + 1e-6, DX)
        ix = np.interp(self.xu, self.x, np.arange(len(self.x)))
        it = np.interp(self.tu, self.t, np.arange(len(self.t)))
        IX, IT = np.meshgrid(ix, it, indexing='ij')
        self._idx = np.array([IX, IT])
        self.XU, self.TU = np.meshgrid(self.xu, self.tu, indexing='ij')
        self.R = np.hypot(self.XU, self.TU)
        self.tname = 'y' if plane == 'eq' else 'z'

    def frames(self):
        fs = sorted(glob.glob(os.path.join(self.dir, 'f[0-9][0-9][0-9][0-9][0-9].npz')))
        return [int(os.path.basename(f)[1:6]) for f in fs]

    def regrid(self, a):
        return map_coordinates(a, self._idx, order=1, mode='nearest')

    def load(self, frame, uniform=True):
        """Return dict of physical quantities (cm^-3, nT, km/s, mV/m, nA/m^2, eV, nPa)."""
        d = np.load(os.path.join(self.dir, f'f{frame:05d}.npz'))
        u = self.u
        g = self.regrid if uniform else (lambda a: a)
        out = {'time': float(d['time']), 'frame': int(d['frame'])}
        out['tsec'] = out['time'] * u['t']
        out['N'] = g(d['Ni']) * u['Ni'] * 1e-6
        for c in 'xyz':
            out['B' + c] = g(d['B' + c]) * u['B'] * 1e9
            out['V' + c] = g(d['Up' + c]) * u['Up'] * 1e-3
            out['E' + c] = g(d['E' + c]) * u['E'] * 1e3
            out['J' + c] = g(d['J' + c]) * u['J'] * 1e9
        out['Tpar'] = g(d['Ti1par']) * u['T'] / EV_K
        out['Tper'] = g(d['Ti1per']) * u['T'] / EV_K
        out['T'] = (out['Tpar'] + 2 * out['Tper']) / 3
        out['Bmag'] = np.sqrt(out['Bx']**2 + out['By']**2 + out['Bz']**2)
        out['Vmag'] = np.sqrt(out['Vx']**2 + out['Vy']**2 + out['Vz']**2)
        n = np.clip(out['N'], 1e-3, None)
        out['Pth'] = n * 1e6 * np.clip(out['T'], 0, None) * 1.602e-19 * 1e9       # nPa
        out['PB'] = (out['Bmag'] * 1e-9)**2 / (2 * 4e-7 * np.pi) * 1e9           # nPa
        out['Pdyn'] = n * 1e6 * 1.6726e-27 * (out['Vmag'] * 1e3)**2 * 1e9         # nPa
        out['beta'] = out['Pth'] / np.clip(out['PB'], 1e-6, None)
        out['Tani'] = out['Tper'] / np.clip(out['Tpar'], 1e-3, None)
        return out

    def trace_fieldlines(self, d, seeds=None, ds=0.1, nmax=1500):
        """Trace in-plane (Bx, B_t) field lines from fixed seeds, both directions.
        Returns list of (N,2) arrays. Projection of 3D lines onto the plane."""
        bx = d['Bx']
        bt = d['B' + self.tname]
        if seeds is None:
            seeds = default_seeds()
        seeds = np.asarray(seeds, float)
        x0, t0 = self.xu[0], self.tu[0]
        nx, nt = len(self.xu), len(self.tu)

        def field(p):
            ix = (p[:, 0] - x0) / DX
            it = (p[:, 1] - t0) / DX
            c = np.array([ix, it])
            fx = map_coordinates(bx, c, order=1, mode='nearest')
            ft = map_coordinates(bt, c, order=1, mode='nearest')
            m = np.hypot(fx, ft) + 1e-12
            return np.stack([fx / m, ft / m], axis=1)

        lines = []
        for sgn in (1, -1):
            p = seeds.copy()
            alive = np.ones(len(p), bool)
            path = [p.copy()]
            for _ in range(nmax):
                k1 = field(p)
                k2 = field(p + 0.5 * sgn * ds * k1)
                pn = p + sgn * ds * k2
                r = np.hypot(pn[:, 0], pn[:, 1])
                out = (r < RMIN) | (pn[:, 0] < XLIM[0]) | (pn[:, 0] > XLIM[1]) | \
                      (pn[:, 1] < TLIM[0]) | (pn[:, 1] > TLIM[1])
                alive &= ~out
                pn[~alive] = np.nan
                p = np.where(alive[:, None], pn, p)
                path.append(np.where(alive[:, None], pn, np.nan))
                if not alive.any():
                    break
            P = np.array(path)  # (steps, nseeds, 2)
            for i in range(P.shape[1]):
                q = P[:, i]
                q = q[~np.isnan(q[:, 0])]
                if len(q) > 1:
                    lines.append(q if sgn == 1 else q[::-1])
        return lines


def default_seeds():
    s = []
    for lat in np.arange(-84, 85, 6):          # dayside & nightside footpoints at r=3.2
        th = np.deg2rad(lat)
        s += [(3.2 * np.cos(th), 3.2 * np.sin(th)), (-3.2 * np.cos(th), 3.2 * np.sin(th))]
    s += [(x, zz) for x in np.arange(-43, 21, 4) for zz in (29.5, -29.5)]  # IMF from top/bottom
    s += [(-38.0, z) for z in np.arange(-18, 19, 2)]  # tail lobes / plasma sheet
    return s


def draw_earth(ax, rmin=RMIN):
    ax.add_patch(Circle((0, 0), rmin, fc='0.6', ec='k', lw=0.5, zorder=5))
    ax.add_patch(Wedge((0, 0), 1, -90, 90, fc='white', ec='k', lw=0.5, zorder=6))
    ax.add_patch(Wedge((0, 0), 1, 90, 270, fc='black', ec='k', lw=0.5, zorder=6))


def flux_levels(A, dA=8.0, n_lin=40, growth=1.12):
    """Uniform spacing for |A|<n_lin*dA, then geometric -> keeps near-Earth dipole readable."""
    amax = np.nanmax(np.abs(A))
    lv = [k * dA for k in range(1, n_lin + 1)]
    s = dA
    while lv[-1] < amax:
        s *= growth
        lv.append(lv[-1] + s)
    lv = np.array(lv)
    return np.concatenate([-lv[::-1], [0.0], lv])


# Panel specifications: key, label, cmap, norm factory
def _sym(v):
    return lambda: TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v)


PANELS = {
    'N':    (r'$N_i$ (cm$^{-3}$)', 'turbo', lambda: LogNorm(0.1, 40)),
    'Bz':   (r'$B_z$ (nT)', 'RdBu_r', lambda: SymLogNorm(10, vmin=-200, vmax=200, base=10)),
    'Bx':   (r'$B_x$ (nT)', 'RdBu_r', lambda: SymLogNorm(10, vmin=-200, vmax=200, base=10)),
    'By':   (r'$B_y$ (nT)', 'RdBu_r', _sym(30)),
    'Bmag': (r'$|B|$ (nT)', 'viridis', lambda: LogNorm(2, 300)),
    'Vx':   (r'$V_x$ (km/s)', 'RdBu_r', _sym(600)),
    'Vy':   (r'$V_y$ (km/s)', 'RdBu_r', _sym(400)),
    'Vz':   (r'$V_z$ (km/s)', 'RdBu_r', _sym(400)),
    'T':    (r'$T_i$ (eV)', 'inferno', lambda: LogNorm(3, 5000)),
    'Jy':   (r'$J_y$ (nA/m$^2$)', 'PuOr_r', _sym(30)),
    'Jz':   (r'$J_z$ (nA/m$^2$)', 'PuOr_r', _sym(30)),
    'Ey':   (r'$E_y$ (mV/m)', 'PiYG_r', _sym(6)),
    'beta': (r'$\beta_i$', 'cividis', lambda: LogNorm(0.01, 100)),
    'Pdyn': (r'$P_{dyn}$ (nPa)', 'magma', lambda: LogNorm(0.01, 3)),
    'Tani': (r'$T_\perp/T_\parallel$', 'BrBG', lambda: LogNorm(0.25, 4)),
}


def render(pl, d, keys, ncols, fname, title=None, dpi=90, panel_w=4.3, fieldlines=None,
           flowlines=None, xlim=XLIM, tlim=TLIM):
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_w * 0.95 * nrows),
                            constrained_layout=True, squeeze=False)
    inner = pl.R < RMIN
    ext = (pl.xu[0] - DX / 2, pl.xu[-1] + DX / 2, pl.tu[0] - DX / 2, pl.tu[-1] + DX / 2)
    lines = pl.trace_fieldlines(d) if fieldlines else None
    for ax, k in zip(axs.flat, keys):
        lab, cmap, norm = PANELS[k]
        nrm = norm()
        a = d[k]
        if isinstance(nrm, LogNorm):
            a = np.clip(a, nrm.vmin, nrm.vmax)
        a = np.ma.array(a, mask=inner)
        im = ax.imshow(a.T, origin='lower', extent=ext, cmap=cmap, norm=nrm,
                       interpolation='nearest', aspect='equal')
        if fieldlines and k in fieldlines:
            for q in lines:
                ax.plot(q[:, 0], q[:, 1], color='k', lw=0.4, alpha=0.75)
        if flowlines and k in flowlines:
            s = 6
            ax.streamplot(pl.xu[::s], pl.tu[::s], d['Vx'][::s, ::s].T, d['V' + pl.tname][::s, ::s].T,
                          density=1.2, color='k', linewidth=0.3, arrowsize=0.5)
        draw_earth(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*tlim)
        ax.set_xlabel(r'$x$ ($R_E$)')
        ax.set_ylabel(rf'${pl.tname}$ ($R_E$)')
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label(lab, fontsize=9)
        cb.ax.tick_params(labelsize=7)
    for ax in axs.flat[len(keys):]:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
