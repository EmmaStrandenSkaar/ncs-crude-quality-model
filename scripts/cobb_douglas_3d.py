"""
Cobb-Douglas produktfunksjon — interaktiv 3D-visualisering
med sliders for α (kapitalandel) og A (teknologinivå).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

K_max = 10
L_max = 10
grid_size = 50

K = np.linspace(0.1, K_max, grid_size)
L = np.linspace(0.1, L_max, grid_size)
K_grid, L_grid = np.meshgrid(K, L)

contour_levels = [1, 2, 3, 4, 5, 6, 7, 8]

fig = plt.figure(figsize=(14, 7))
fig.subplots_adjust(bottom=0.22)

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

def compute_Y(alpha, A):
    return K_grid**alpha * (A * L_grid)**(1 - alpha)

def draw(alpha, A):
    Y_grid = compute_Y(alpha, A)
    y_max = np.max(Y_grid)

    ax1.clear()
    surf = ax1.plot_surface(K_grid, L_grid, Y_grid,
                             cmap='viridis', alpha=0.85,
                             edgecolor='none', antialiased=True)
    ax1.plot_wireframe(K_grid, L_grid, Y_grid,
                        color='black', linewidth=0.3,
                        rstride=5, cstride=5, alpha=0.4)
    ax1.set_xlabel('K (kapital)', fontsize=11, labelpad=10)
    ax1.set_ylabel('L (arbeidskraft)', fontsize=11, labelpad=10)
    ax1.set_zlabel('Y (produksjon)', fontsize=11, labelpad=10)
    ax1.set_zlim(0, max(y_max * 1.05, 1))
    ax1.set_title(f'Cobb-Douglas: $Y = K^{{\\alpha}} (AL)^{{1-\\alpha}}$\n'
                  f'α = {alpha:.2f}, A = {A:.2f}',
                  fontsize=12, pad=15)
    ax1.view_init(elev=25, azim=-50)

    ax2.clear()
    valid_levels = [lv for lv in contour_levels if lv < y_max]
    if valid_levels:
        contour = ax2.contour(K_grid, L_grid, Y_grid,
                               levels=valid_levels, cmap='viridis', linewidths=2)
        ax2.clabel(contour, inline=True, fontsize=9, fmt='Y=%g')
    ax2.contourf(K_grid, L_grid, Y_grid, levels=20, cmap='viridis', alpha=0.25)
    t = np.linspace(0.1, 9, 100)
    ax2.plot(t, t, 'r--', linewidth=1.5, alpha=0.7,
              label='Stråle K=L (CRS-sjekk)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlabel('K (kapital)', fontsize=11)
    ax2.set_ylabel('L (arbeidskraft)', fontsize=11)
    ax2.set_title('Isokvanter', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    fig.canvas.draw_idle()

# Sliders
ax_alpha = fig.add_axes([0.15, 0.08, 0.30, 0.03])
ax_A     = fig.add_axes([0.15, 0.03, 0.30, 0.03])

slider_alpha = Slider(ax_alpha, 'α (kapitalandel)', 0.05, 0.95,
                       valinit=0.5, valstep=0.01, color='steelblue')
slider_A     = Slider(ax_A, 'A (teknologi)', 0.1, 5.0,
                       valinit=1.0, valstep=0.1, color='darkorange')

def on_update(val):
    draw(slider_alpha.val, slider_A.val)

slider_alpha.on_changed(on_update)
slider_A.on_changed(on_update)

draw(0.5, 1.0)
plt.show()
