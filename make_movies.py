"""Render animation frames and encode MP4 + GIF.
Usage: python make_movies.py eq|mer [nproc]
"""
import os, sys, subprocess, shutil
from multiprocessing import Pool
import numpy as np

FFMPEG = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
ROOT = os.path.dirname(os.path.abspath(__file__))

CFG = {
    'eq': dict(keys=['N', 'Bz', 'Vx', 'Vy', 'T', 'Jy', 'Ey', 'beta'], fieldlines=None, flowlines=['N'],
               name='Equatorial plane (z = 0)'),
    'mer': dict(keys=['N', 'Bx', 'Bz', 'By', 'Vx', 'Vz', 'T', 'Jy'], fieldlines=['N', 'Jy'], flowlines=None,
                name='Noon-midnight meridian plane (y = 0)'),
}


def work(args):
    plane, fr = args
    from magviz import Plane, render
    out = os.path.join(ROOT, 'frames', plane, f'{fr:05d}.png')
    if os.path.exists(out):
        return
    pl = Plane(plane)
    d = pl.load(fr)
    c = CFG[plane]
    title = (f"{c['name']}   t = {d['time']:6.2f} $\\Omega_i^{{-1}}$  "
             f"(≈ {d['tsec']:6.0f} s scaled)   frame {fr}")
    render(pl, d, c['keys'], 4, out + '.tmp.png', title=title, dpi=80,
           fieldlines=c['fieldlines'], flowlines=c['flowlines'])
    os.replace(out + '.tmp.png', out)


if __name__ == '__main__':
    if not FFMPEG or not os.path.isfile(FFMPEG):
        raise SystemExit('FFmpeg not found. Set FFMPEG to its executable or add it to PATH.')
    if len(sys.argv) < 2 or sys.argv[1] not in CFG:
        raise SystemExit('Usage: python make_movies.py eq|mer [nproc]')
    plane = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sys.path.insert(0, ROOT)
    from magviz import Plane
    frames = Plane(plane).frames()
    os.makedirs(os.path.join(ROOT, 'frames', plane), exist_ok=True)
    with Pool(nproc) as p:
        p.map(work, [(plane, f) for f in frames], chunksize=2)
    # sequential numbering for ffmpeg
    lst = os.path.join(ROOT, 'frames', plane, 'list.txt')
    fps = 15 if plane == 'eq' else 8
    with open(lst, 'w') as f:
        for fr in frames:
            f.write(f"file '{fr:05d}.png'\nduration {1/fps:.4f}\n")
    os.makedirs(os.path.join(ROOT, 'output'), exist_ok=True)
    mp4 = os.path.join(ROOT, 'output', f'magnetosphere_{plane}.mp4')
    gif = os.path.join(ROOT, 'output', f'magnetosphere_{plane}.gif')
    cwd = os.path.join(ROOT, 'frames', plane)
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'list.txt',
                    '-vf', f'fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '20', '-preset', 'slow', mp4], cwd=cwd, check=True)
    # GIF: every 2nd frame for eq, scaled down, palette-optimised
    step = 2 if plane == 'eq' else 1
    gfps = fps / step
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', mp4, '-vf',
                    f"select='not(mod(n\\,{step}))',setpts=N/({gfps}*TB),scale=1100:-1:flags=lanczos,"
                    f"split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    '-r', f'{gfps}', '-loop', '0', gif], check=True)
    for f in (mp4, gif):
        print(f, os.path.getsize(f) / 1e6, 'MB')
