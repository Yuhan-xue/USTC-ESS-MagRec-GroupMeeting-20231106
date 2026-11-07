"""Build caches for reproduce.py: late-time (t >= 100) mean/std maps and virtual-probe time series."""
import os
import numpy as np
from magviz import Plane

C = 'output/cache'
os.makedirs(C, exist_ok=True)
Q = ['N', 'Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'Ex', 'Ey', 'Ez', 'T', 'Tpar', 'Tper', 'Bmag', 'Pdyn', 'Pth', 'PB']


def stats(pl, t0=100.0):
    acc, acc2, n = None, None, 0
    for f in pl.frames():
        d = pl.load(f)
        if d['time'] < t0:
            continue
        if acc is None:
            acc = {k: np.zeros_like(d[k], float) for k in Q}
            acc2 = {k: np.zeros_like(d[k], float) for k in ('N', 'Bmag', 'Bz', 'Vx')}
        for k in Q:
            acc[k] += d[k]
        for k in acc2:
            acc2[k] += d[k] ** 2
        n += 1
    out = {k: v / n for k, v in acc.items()}
    for k, v in acc2.items():
        out['std_' + k] = np.sqrt(np.clip(v / n - out[k] ** 2, 0, None))
    out['n'] = n
    return out


def probes(pl, pts, box=2):
    idx = {k: (np.argmin(abs(pl.xu - x)), np.argmin(abs(pl.tu - y))) for k, (x, y) in pts.items()}
    ts, P = [], {k: {q: [] for q in ('N', 'Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'Ey', 'T', 'Bmag')} for k in pts}
    for f in pl.frames():
        d = pl.load(f)
        ts.append(d['time'])
        for k, (i, j) in idx.items():
            for q in P[k]:
                P[k][q].append(d[q][i - box:i + box + 1, j - box:j + box + 1].mean())
    flat = {'t': np.array(ts)}
    for k in P:
        for q in P[k]:
            flat[f'{k}|{q}'] = np.array(P[k][q])
    return flat


if __name__ == '__main__':
    for p in ('eq', 'mer'):
        pl = Plane(p)
        s = stats(pl)
        np.savez_compressed(f'{C}/avg_{p}.npz', **s)
        print(p, 'avg frames', s['n'])
    eqpts = {f'x{x}': (x, 0.0) for x in (-12, -14, -16, -18, -20, -22, -25, -28)}
    np.savez_compressed(f'{C}/probe_eq.npz', **probes(Plane('eq'), eqpts))
    # meridian: tail line + dayside magnetopause latitudes (positions refined later in reproduce.py)
    mer = Plane('mer')
    merpts = {f'x{x}': (x, 0.0) for x in (-14, -16, -18, -20, -22, -25)}
    for lat in (20, 30, 40, 50):
        th = np.radians(lat)
        r = 11.6 * (2 / (1 + np.cos(th))) ** 0.6
        for s, nm in ((1, 'N'), (-1, 'S')):
            merpts[f'mp{nm}{lat}'] = (r * np.cos(th), s * r * np.sin(th))
    np.savez_compressed(f'{C}/probe_mer.npz', **probes(mer, merpts, box=1))
    print('done')
