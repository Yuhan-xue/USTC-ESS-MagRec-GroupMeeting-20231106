"""Annotated anatomy figure (English labels) for the slides: meridian density + field lines at t=150."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from magviz import Plane, draw_earth, RMIN
from snapshots import nearest

me = Plane('mer')
d = me.load(nearest(me, 150))
lines = me.trace_fieldlines(d)
ext = (me.xu[0], me.xu[-1], me.tu[0], me.tu[-1])
fig, ax = plt.subplots(figsize=(10, 7.2))
a = np.clip(d['N'], 0.1, 40)
im = ax.imshow(np.ma.array(a, mask=me.R < RMIN).T, origin='lower', extent=ext, cmap='turbo',
               norm=LogNorm(0.1, 40), interpolation='nearest', aspect='equal')
for q in lines:
    ax.plot(q[:, 0], q[:, 1], color='k', lw=0.35, alpha=0.75)
draw_earth(ax)
ax.set_xlim(-45, 22); ax.set_ylim(-28, 28)
ax.set_xlabel(r'$x$ ($R_E$)'); ax.set_ylabel(r'$z$ ($R_E$)')
from mpl_toolkits.axes_grid1 import make_axes_locatable
cax = make_axes_locatable(ax).append_axes('right', size='2.5%', pad=0.1)
cb = fig.colorbar(im, cax=cax); cb.set_label(r'$N_i$ (cm$^{-3}$)')
box = dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85)
kw = dict(fontsize=11, bbox=box, arrowprops=dict(arrowstyle='->', color='k', lw=1.1))
ax.text(17.8, 21.5, 'Solar wind', fontsize=11, ha='center', bbox=box)
ax.annotate('', xy=(15, 18.5), xytext=(21, 18.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('Bow shock', xy=(13.6, 6), xytext=(13, -4), **kw)
ax.annotate('Magnetosheath', xy=(9, 14), xytext=(5, 24), **kw)
ax.annotate('Magnetopause', xy=(10.8, -1.5), xytext=(11.5, -13), **kw)
ax.annotate('Cusp', xy=(4, 9.5), xytext=(-6, 20), **kw)
ax.annotate('Inner magnetosphere', xy=(-4.5, -3), xytext=(-14, -21), **kw)
ax.annotate('North lobe', xy=(-25, 9), xytext=(-27, 9), fontsize=11, bbox=box)
ax.annotate('South lobe', xy=(-25, -11), xytext=(-27, -11), fontsize=11, bbox=box)
ax.annotate('Hot plasma sheet', xy=(-15, 0.3), xytext=(-20, 4.2), **kw)
ax.annotate('Tail X-line', xy=(-26, 0), xytext=(-40, -6), **kw)
ax.set_title(r'Noon–midnight meridian, $N_i$ and field lines, $t$ = 150 $\Omega_i^{-1}$ (~39 min)', fontsize=12)
fig.savefig('output/figs/anatomy_mer.png', dpi=130, bbox_inches='tight')
print('ok')


