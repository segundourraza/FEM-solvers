from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar

plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                     'mathtext.fontset': 'cm'})


# ── Domain ────────────────────────────────────────────────────────────────────
x_domain = [-1.0, 2.0]
y_domain = [-0.5, 1.5]
nx=ny =12

x_domain = [-0.5, 1.0]
y_domain = [-0.5, 1.5]
nx = ny = 4 # elements per direction
# factor = 4
# nx*= factor
# ny*= factor



origin = (x_domain[0], y_domain[0])

a = x_domain[1] - x_domain[0]
b = y_domain[1] - y_domain[0]



order  = 2             # Q9 elements

# ── Physics ───────────────────────────────────────────────────────────────────
Re = rho = 40.0
mu = 1.0


pref = 10
corner_id = 0

lam = Re/2 - np.sqrt((Re/2)**2 + 4*np.pi**2)
print(lam)
def vx_analytical(x, y):
    return 1 - np.exp(lam*x)*np.cos(2*np.pi*y)

def vy_analytical(x, y):
    return (lam/(2*np.pi))*np.exp(lam*x)*np.sin(2*np.pi*y)

def p_analytical(x, y):
    return pref - 0.5*np.exp(2*lam*x)


# ── Boundary conditions ───────────────────────────────────────────────────────
top = BoundaryCondition(
    name="dirichlet",
    boundary_key="top",
    type=BCType.DIRICHLET,
    variable=BCVar.VELOCITY,
    value= lambda x, y, t: (vx_analytical(x, y), vy_analytical(x,y)),
    apply_strong=True,
    metadata={"note": "no-slip top wall"},
)


bottom = top.copy(); bottom.boundary_key = 'bottom'
right = top.copy(); right.boundary_key = 'right'
left = top.copy(); left.boundary_key = 'left'

# ── Solve ─────────────────────────────────────────────────────────────────────
sol = NavierStokesSolver.uniform_rectangular_domain_rect(
    nx, ny, a, b, order=order, origin=origin,
)
sol.setup_physics(rho, mu)
sol.setup_boundary_conditions([bottom, top, left, right],
                              pref_corner_id=corner_id, pref_value=p_analytical(x_domain[0], y_domain[0]))
sol.solve_steadystate(u0=1, p0=pref)

sol_vx, sol_vy, sol_p = sol.get_solution()

factor = 2
ref = NavierStokesSolver.uniform_rectangular_domain_rect(
    nx*factor, ny*factor, a, b, order=order, origin=origin,
)
ref.setup_physics(rho, mu)
ref.setup_boundary_conditions([bottom, top, left, right],
                              pref_corner_id=corner_id, pref_value=p_analytical(x_domain[0], y_domain[0]))
ref.solve_steadystate(u0=1, p0=pref)

ref_vx, ref_vy, ref_p = ref.get_solution()



# ── collect nodes at x = 0 and x = a ─────────────────────────────────────────
sol_x_stations = {k:v for k,v in sol.group_by_x().items() if k in [-0.125, 0.25, 0.625]}
ref_x_stations = {k:v for k,v in ref.group_by_x().items() if k in [-0.125, 0.25, 0.625]}

markers    = ['o', 's', '^', 'd']
linestyles = ['--', '-', '-.']


# ── (a) velocity profiles ─────────────────────────────────────────────────────
fig, ax = plt.subplots(1, len(sol_x_stations),sharey=True, 
                    #    figsize=(14, 10)
                       )
y_fine = np.linspace(*y_domain, 100)
for i, (xs, con) in enumerate(sol_x_stations.items()):
    ax[i].set_title(f'$x={xs:.2f}$')
    ax[i].plot(vx_analytical(xs, y_fine), y_fine, 'r', lw=2, label='Analytical')
    ax[i].set_xlabel('$v_x(y)$')
    ax[i].grid()
    
for i, (xs, con) in enumerate(sol_x_stations.items()):
    ax[i].plot(sol_vx[con], sol.p2_nodes[con, 1], 'k', marker=markers[0], linestyle=linestyles[0],
            ms=8, markerfacecolor='none',
            label = 'FEM: Q9 4 x 4')
    
for i, (xs, con) in enumerate(ref_x_stations.items()):
    ax[i].plot(ref_vx[con], ref.p2_nodes[con, 1], 'k', marker=markers[1], linestyle=linestyles[1],
            ms=8, markerfacecolor='none',
            label = 'FEM: Q9 16 x 16')
    
    
ax[0].set_ylabel('$y$', rotation=0, labelpad=10)
ax[0].legend(fontsize=11)
fig.suptitle('Velocity profile')
fig.tight_layout()


# ── (b) pressure profile (along x at y = 0) ─────────────────────────────────
fig, ax = plt.subplots()
# all pressure nodes, plotted vs x
x_fine = np.linspace(*x_domain, 200)
y_target = 0.5
ax.plot(x_fine, p_analytical(x_fine, y_target), 'r', lw=2, label='Analytical')

centre_idx = np.where(np.abs(sol.p1_nodes[:, 1] - y_target) < 1e-6)[0]
ax.scatter(sol.p1_nodes[centre_idx, 0], sol_p[centre_idx], 
           c='k', s=30, zorder=3, 
           label='FEM nodes')

centre_idx = np.where(np.abs(ref.p1_nodes[:, 1] - y_target) < 1e-6)[0]
ax.scatter(ref.p1_nodes[centre_idx, 0], ref_p[centre_idx], 
           c='k', s=30, zorder=3, 
           label='FEM nodes')

ax.set_xlabel('$x$')
ax.set_ylabel('$p$', rotation=0, labelpad=10)
ax.set_title('(b) Pressure distribution')
ax.legend(fontsize=11)
ax.grid()

# # ── (c) centreline velocity vs analytical ─────────────────────────────────────
# fig, ax = plt.subplots()
# centre_idx = np.where(np.abs(sol.p2_nodes[:, 1]) < 1e-6)[0]
# xc = sol.p2_nodes[centre_idx, 0]
# ax.scatter(xc, sol_vx[centre_idx], c='k', s=40, zorder=3, label='FEM centreline')
# ax.set_xlabel('$x$')
# ax.set_ylabel('$v_x$', rotation=0, labelpad=10)
# ax.set_title('(c) Centreline velocity')
# ax.legend(fontsize=11)
# ax.grid()

# fig.suptitle('Plane Poiseuille Flow — FEM vs Analytical', fontsize=14, y=1.02)
# fig.tight_layout()


#########################################################################
# Plot streamlines

# Node data
x = sol.p2_nodes[:, 0]
y = sol.p2_nodes[:, 1]
u = sol_vx
v = sol_vy

mag = np.sqrt(u**2 + v**2)
u = [i if np.isclose(i,0.0) else i/j for i,j in zip(u,mag)]
v = [i if np.isclose(i,0.0) else i/j for i,j in zip(v,mag)]

# Create regular grid
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate velocities
Ui = griddata((x, y), u, (Xi, Yi), method='linear')
Vi = griddata((x, y), v, (Xi, Yi), method='linear')

fig3, ax3 = plt.subplots(1,2)
sol.plot_mesh(ax=ax3[0], plot_nodes=False)
ax3[0].streamplot(Xi, Yi, Ui, Vi, 
                color = 'b',
                linewidth = 0.75,
                broken_streamlines=False, 
                density = 1,
                )

# Node data
x = ref.p2_nodes[:, 0]
y = ref.p2_nodes[:, 1]
u = ref_vx
v = ref_vy

mag = np.sqrt(u**2 + v**2)
u = [i if np.isclose(i,0.0) else i/j for i,j in zip(u,mag)]
v = [i if np.isclose(i,0.0) else i/j for i,j in zip(v,mag)]

# Create regular grid
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate velocities
Ui = griddata((x, y), u, (Xi, Yi), method='linear')
Vi = griddata((x, y), v, (Xi, Yi), method='linear')

ref.plot_mesh(ax=ax3[1], plot_nodes=False)
ax3[1].streamplot(Xi, Yi, Ui, Vi, 
                color = 'b',
                linewidth = 0.75,
                broken_streamlines=False, 
                density = 1,
                )





ax3[0].axis('equal')
ax3[1].axis('equal')

plt.show()