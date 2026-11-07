"""Download noon-meridian (y=0) and equatorial (z=0) slices from the
ear-Reynolds-S global hybrid run via pyfefi2. Resumable.

Usage: python download.py eq|mer [step] [workers]
Output: data/<plane>/f#####.npz with variables in code units + time,
        data/<plane>/coords.npz with 1D axes (Re) and unit factors.
"""
import os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pyfefi2 import Data, Slice, Config

URL = 'https://workstation.huaji.dev/data/qnap1/jguo/ear-Reynolds-S'
VARS = ['Ni', 'Bx', 'By', 'Bz', 'Upx', 'Upy', 'Upz', 'Ex', 'Ey', 'Ez',
        'Jx', 'Jy', 'Jz', 'Ti1par', 'Ti1per']
P0, P1 = 24, 588   # x in [-49.8, 26.0] Re
Q0, Q1 = 12, 516   # transverse in [-32.8, 32.8] Re

if len(sys.argv) < 2 or sys.argv[1] not in ('eq', 'mer'):
    raise SystemExit('Usage: python download.py eq|mer [step] [workers]')
plane = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 2
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
if step < 1 or workers < 1:
    raise SystemExit('step and workers must both be positive integers')

cfg = Config(URL)
g = cfg.grid_size
if plane == 'eq':
    slc = Slice[P0:P1, Q0:Q1, g[2] // 2]
else:
    slc = Slice[P0:P1, g[1] // 2, Q0:Q1]

out = os.path.join(os.environ.get('MAG_RAW_DIR', 'data'), plane)
os.makedirs(out, exist_ok=True)

d0 = Data(cfg, slc, max_memory=2e9)
units = cfg.get_units()
cfile = os.path.join(out, 'coords.npz')
if not os.path.exists(cfile):
    x = d0.x[:, 0]
    t_ax = d0.y[0, :] if plane == 'eq' else d0.z[0, :]
    np.savez(cfile, x=x, t=t_ax, X=d0.x, Y=d0.y, Z=d0.z,
             **{f'u_{k}': units[k] for k in ['Ni', 'B', 'Up', 'E', 'J', 'T', 't', 'L']})

frames = [f for f in cfg.get_frames() if (f - 1) % step == 0 or f == 756]
todo = [f for f in frames if not os.path.exists(os.path.join(out, f'f{f:05d}.npz'))]
print(f'{plane}: {len(frames)} frames, {len(todo)} to download', flush=True)


def fetch(args):
    f, v = args
    for attempt in range(5):
        try:
            d = Data(cfg, slc, max_memory=2e9)
            a = d[f, v]
            if np.ndim(a) != 2:
                raise ValueError(f'bad array for {f} {v}: {np.shape(a)}')
            return a.astype(np.float32)
        except Exception as e:  # network hiccups
            err = e
            time.sleep(2 + 3 * attempt)
    raise err


t0 = time.time()
with ThreadPoolExecutor(workers) as ex:
    for n, f in enumerate(todo):
        arrs = list(ex.map(fetch, [(f, v) for v in VARS]))
        tm = 0.01 + 0.25 * (f - 1)
        for attempt in range(3):
            try:
                tm = float(Data(cfg, slc, max_memory=2e9).time(f))
                break
            except Exception:
                time.sleep(2)
        tmp = os.path.join(out, f'f{f:05d}.tmp.npz')
        np.savez(tmp, time=tm, frame=f, **dict(zip(VARS, arrs)))
        os.replace(tmp, os.path.join(out, f'f{f:05d}.npz'))
        if n % 10 == 0:
            el = time.time() - t0
            print(f'{plane} frame {f} t={tm:.2f} [{n+1}/{len(todo)}] '
                  f'{el:.0f}s elapsed, ETA {el/(n+1)*(len(todo)-n-1)/60:.1f} min', flush=True)
print('done', plane, time.time() - t0)
