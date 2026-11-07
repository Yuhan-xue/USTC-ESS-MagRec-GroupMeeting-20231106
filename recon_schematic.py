"""Schematic of the two-scale (Hall) collisionless reconnection picture (Lu et al. 2022, CPB 31 089401)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.6, 5.6))
x = np.linspace(-6, 6, 400)
z = np.linspace(-3.6, 3.6, 300)
X, Z = np.meshgrid(x, z)
A = Z**2 - 0.12 * X**2
ax.contour(X, Z, A, levels=np.r_[np.linspace(-3.5, -0.3, 7), np.linspace(0.3, 11, 9)], colors='0.35', linewidths=0.9, linestyles='solid')
ax.contour(X, Z, A, levels=[0], colors='k', linewidths=1.6, linestyles='--')

ax.add_patch(Rectangle((-3.4, -0.95), 6.8, 1.9, fc='#cfe3f5', ec='#1f5fa8', lw=1.8, alpha=0.75, zorder=2))
ax.add_patch(Rectangle((-0.8, -0.22), 1.6, 0.44, fc='#f6c7a0', ec='#b5651d', lw=1.8, zorder=3))
ax.text(0, 0, 'EDR  ~ d$_e$', ha='center', va='center', fontsize=10, color='#7a3d00', zorder=4, weight='bold')
ax.text(-3.3, 0.72, 'Ion diffusion region (IDR)  ~ d$_i$', fontsize=10, color='#1f5fa8', zorder=4, weight='bold')
ax.text(-3.3, -0.85, 'ions demagnetized, electrons still frozen-in', fontsize=8.5, color='#1f5fa8', zorder=4)

kw = dict(arrowstyle='-|>', mutation_scale=20, lw=2.2)
for s in (1, -1):
    ax.add_patch(FancyArrowPatch((0, 3.3 * s), (0, 1.5 * s), color='#2c7a2c', **kw))
    ax.add_patch(FancyArrowPatch((3.6 * s, 0), (5.7 * s, 0), color='#c0392b', **kw))
ax.text(0.15, 3.0, 'inflow', color='#2c7a2c', fontsize=10, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5), zorder=6)
ax.text(4.0, 0.2, 'ion outflow ≈ V$_A$', color='#c0392b', fontsize=10, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5), zorder=6)
ax.text(-5.7, 0.2, 'ion outflow', color='#c0392b', fontsize=10, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5), zorder=6)

# electron flow: in along the separatrices, out along the midplane
ek = dict(arrowstyle='-|>', mutation_scale=13, lw=1.6, color='#6a1b9a', zorder=6)
for sx in (1, -1):
    for sz in (1, -1):
        xs = np.array([5.2, 2.0]) * sx
        ax.add_patch(FancyArrowPatch((xs[0], sz * np.sqrt(0.12) * abs(xs[0]) + 0.25 * sz),
                                     (xs[1], sz * np.sqrt(0.12) * abs(xs[1]) + 0.25 * sz), **ek))
    ax.add_patch(FancyArrowPatch((1.0 * sx, -0.45), (3.2 * sx, -0.45), **ek))
ax.text(1.3, 2.25, 'electron flow (schematic)', color='#6a1b9a', fontsize=9, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5), zorder=6)

for sx, sz, sym in [(1, 1, '⊗'), (1, -1, '⊙'), (-1, 1, '⊙'), (-1, -1, '⊗')]:
    ax.text(2.6 * sx, 1.55 * sz, sym, fontsize=26, ha='center', va='center', color='#b5651d', zorder=5)
ax.text(-6.0, -3.5, 'Ideal Hall sign pattern: theory, not simulation data\n'
        'B$_y$:  ⊗ into page = +y;  ⊙ out of page = −y', fontsize=9, color='0.2',
        bbox=dict(fc='white', ec='none', alpha=0.85))
ax.text(5.9, -2.75, '+x (Earthward in the tail) →', ha='right', fontsize=9, bbox=dict(fc='white', ec='none', alpha=0.85))
ax.set_xlim(-6.2, 6.2)
ax.set_ylim(-3.6, 3.6)
ax.set_aspect('equal')
ax.axis('off')
fig.savefig('output/figs/recon_two_scale_schematic.png', dpi=160, bbox_inches='tight')
