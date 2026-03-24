from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar

plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                     'mathtext.fontset': 'cm'})


# ── Domain ────────────────────────────────────────────────────────────────────
a, b = 2, 6          # width, height
order  = 2             # Q9 elements
nx = ny = 10

# ── Physics ───────────────────────────────────────────────────────────────────
Re = rho = 1.0 # 100.0
mu = 1.0


pref = 10
corner_id = 0

def vx_analytical(x, y):
    return np.sin(np.pi/a*x)*np.cos(np.pi/b*y)

def vy_analytical(x, y):
    return -np.cos(np.pi/a*x)*np.sin(np.pi/b*y)

def p_analytical(x, y):
    return np.sin(np.pi/a*x)*np.cos(np.pi/b*y)
    

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
                            pref_corner_id=corner_id, pref_value=pref)
sol.solve_steadystate(u0=1, p0=pref,
                        solver=2)

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






res = np.linalg.norm(vx_analytical(*sol.p2_nodes.T) - sol_vx)
print(res)
res = np.linalg.norm(vy_analytical(*sol.p2_nodes.T) - sol_vy)
print(res)

plt.show()