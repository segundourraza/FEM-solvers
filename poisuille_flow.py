import matplotlib.pyplot as plt
import numpy as np


from fem import NavierStokesSolver, BoundaryCondition, BCVar, BCType


nx = 6
ny = nx

# ── Domain ────────────────────────────────────────────────────────────────────
a, b   = 6, 2          # width, height
order  = 2             # Q9 elements

# ── Physics ───────────────────────────────────────────────────────────────────
rho, mu = 1.0, 1.0

# ── Pressure values ───────────────────────────────────────────────────────────
p_in  = 10.0
p_out = 4.0
dPdx  = (p_out - p_in) / a    # pressure gradient (< 0 → flow in +x)

# ── Pressure values ───────────────────────────────────────────────────────────
def vx_analytical(x, y):
    """Parabolic Poiseuille profile."""
    return (-1.0 / (2.0 * mu)) * dPdx * y * (b - y)

def vy_analytical(x, y):
    """Zero vertical velocity."""
    return np.zeros_like(np.asarray(y, dtype=float))
def gradv_analytical(x,y):
    return np.array([[0, (-1.0 / (2.0 * mu)) * dPdx*(b - 2*y)],
                        [0, 0]])
def p_analytical(x, y):
    """Linear pressure from inlet to outlet."""
    return p_in + dPdx * np.asarray(x, dtype=float)


# ── Boundary conditions ───────────────────────────────────────────────────────
top = BoundaryCondition(
    name="no-slip-top",
    boundary_key="top",
    type=BCType.DIRICHLET,
    variable=BCVar.VELOCITY,
    value=(0.0, 0.0),
    apply_strong=True,
    metadata={"note": "no-slip top wall"},
)
bottom = BoundaryCondition(
    name="no-slip-bottom",
    boundary_key="bottom",
    type=BCType.DIRICHLET,
    variable=BCVar.VELOCITY,
    value=(0.0, 0.0),
    apply_strong=True,
    metadata={"note": "no-slip bottom wall"},
)
left = BoundaryCondition(
    name="pressure-inlet",
    boundary_key="left",
    type=BCType.NEUMANN,
    variable=BCVar.PRESSURE,
    value=p_in,
    metadata={"p": p_in},
)
right = BoundaryCondition(
    name="pressure-outlet",
    boundary_key="right",
    type=BCType.NEUMANN,
    variable=BCVar.PRESSURE,
    value=p_out,
    metadata={"p": p_out},
)

# ── Solve ─────────────────────────────────────────────────────────────────────
sol = NavierStokesSolver.uniform_rectangular_domain_rect(nx, ny, a, b, order=order)
sol.setup_physics(rho, mu)
sol.setup_boundary_conditions([bottom, top, left, right])

nonlinear_options = {'tol': 1e-10}
sol.solve_steadystate(u0=1, p0=p_in, nonlinear_solver_options=nonlinear_options)
sol_vx, sol_vy, sol_p = sol.get_solution()

L2v_norm, H1sm, H1_norm, L2_p_norm = sol.error_analysis(vx_analytical, vy_analytical, gradv_analytical, p_analytical)
print("||u-uh||_L2: {}, |u-uh|_H1: {}, ||u-uh||_H1: {}, ||p-ph||_L2: {}".format(L2v_norm, H1sm, H1_norm, L2_p_norm))

# ── collect nodes at x = 0 and x = a ─────────────────────────────────────────
uni_x   = sol.group_by_x()
x_stations = {k: v for k, v in uni_x.items() if k in [0.0, a]}

markers    = ['o', 's', '^', 'd']
linestyles = ['--', '-', '-.']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# ── (a) velocity profiles ─────────────────────────────────────────────────────
ax = axes[0]
y_fine = np.linspace(0, b, 200)
ax.plot(vx_analytical(a, y_fine), y_fine, 'r', lw=2, label='Analytical')
for i, (xs, con) in enumerate(x_stations.items()):
    ys = sol.p2_nodes[con, 1]
    ax.plot(sol_vx[con], ys, 'k', marker=markers[i], linestyle=linestyles[i],
            ms=8, markerfacecolor='none', label=f'FEM  $x={xs:.0f}$')
ax.set_xlabel('$v_x(y)$')
ax.set_ylabel('$y$', rotation=0, labelpad=10)
ax.set_title('(a) Velocity profile')
ax.legend(fontsize=11)
ax.grid()

# ── (b) pressure profile (along y at x = a/2) ─────────────────────────────────
ax = axes[1]
# all pressure nodes, plotted vs x
px = sol.p1_nodes[:, 0]
x_fine = np.linspace(0, a, 200)
ax.plot(x_fine, p_analytical(x_fine, 0), 'r', lw=2, label='Analytical')
ax.scatter(px, sol_p, c='k', s=30, zorder=3, label='FEM nodes')
ax.set_xlabel('$x$')
ax.set_ylabel('$p$', rotation=0, labelpad=10)
ax.set_title('(b) Pressure distribution')
ax.legend(fontsize=11)
ax.grid()

# ── (c) centreline velocity vs analytical ─────────────────────────────────────
ax = axes[2]
centre_idx = np.where(np.abs(sol.p2_nodes[:, 1] - b / 2.0) < 1e-6)[0]
xc = sol.p2_nodes[centre_idx, 0]
ax.scatter(xc, sol_vx[centre_idx], c='k', s=40, zorder=3, label='FEM centreline')
ax.set_xlabel('$x$')
ax.set_ylabel('$v_x$', rotation=0, labelpad=10)
ax.set_title('(c) Centreline velocity')
ax.legend(fontsize=11)
ax.grid()

fig.suptitle('Plane Poiseuille Flow — FEM vs Analytical', fontsize=14, y=1.02)
fig.tight_layout()
plt.show()