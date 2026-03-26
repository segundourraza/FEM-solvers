from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar

plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                     'mathtext.fontset': 'cm'})


# ── Domain ────────────────────────────────────────────────────────────────────
a = b  = np.pi


order  = 2             # Q9 elements
nx = ny = 10

# ── Physics ───────────────────────────────────────────────────────────────────
Re = rho = 10.0 # 100.0
mu = 1.0


def vx_analytical(x, y):
    return np.sin(x)*np.cos(y)

def vy_analytical(x, y):
    return -np.cos(x)*np.sin(y)

def gradv_analytical(x,y):
    return np.array([[np.cos(x)*np.cos(y), -np.sin(x)*np.sin(y)],
                        [np.sin(x)*np.sin(y), -np.cos(x)*np.cos(y)]])

def p_analytical(x, y):
    return 1/4*(np.cos(2*x) + np.cos(2*y))*rho


def f1(x, y):
    return 2*mu*np.sin(x)*np.cos(y)

def f2(x, y):
    return -2*mu*np.cos(x)*np.sin(y)

def forcing_function(x,y):
    return np.array([f1(x,y), f2(x,y)])


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


sol = NavierStokesSolver.uniform_rectangular_domain_rect(nx, ny, a, b, order=order)
sol.setup_physics(rho, mu)
sol.setup_boundary_conditions([bottom, top, left, right],
                            pref_corner_id=0, pref_value=p_analytical,
                            forcing_function=forcing_function)
sol.solve_steadystate(u0=1, p0=0.0,
                        solver=0)

sol_vx, sol_vy, sol_p = sol.get_solution()

##############################################################################
# STREAMLINES
fig5, ax5 = plt.subplots()
# sol.plot_mesh(ax=ax5, plot_nodes=False)

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


ax5.streamplot(Xi, Yi, Ui, Vi, 
                color = 'b',
                linewidth = 1.25,
                broken_streamlines=False, 
                density = 0.9,
                )


u = vx_analytical(*sol.p2_nodes.T)
v = vy_analytical(*sol.p2_nodes.T)

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
ax5.streamplot(Xi, Yi, Ui, Vi, 
                color = 'r',
                linewidth = 1.25,
                broken_streamlines=False, 
                density = 0.9,
                )

L2v_norm, H1sm, H1_norm, L2_p_norm = sol.error_analysis(vx_analytical, vy_analytical, gradv_analytical, p_analytical)
print("||u-uh||_L2: {}, |u-uh|_H1: {}, ||u-uh||_H1: {}, ||p-ph||_L2: {}".format(L2v_norm, H1sm, H1_norm, L2_p_norm))

print(np.linalg.norm(sol_vx - vx_analytical(*sol.p2_nodes.T)))
print(np.linalg.norm(sol_vy - vy_analytical(*sol.p2_nodes.T)))

print(np.linalg.norm(sol_p - p_analytical(*sol.p1_nodes.T)))

fig, ax = plt.subplots()
cf = ax.tricontourf(sol.p1_nodes[:,0], sol.p1_nodes[:,1], sol_p- p_analytical(*sol.p1_nodes.T), levels=100)
fig.colorbar(cf, ax=ax)

plt.show()