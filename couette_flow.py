import matplotlib.pyplot as plt
import numpy as np
from itertools import product as iproduct

plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                     'mathtext.fontset': 'cm'})

markers    = ['o', 's', '^', 'd']
linestyles = ['--', '-', '-.']
styles     = list(iproduct(linestyles, markers))


from fem import NavierStokesSolver, BoundaryCondition, BCVar, BCType

# ── Domain ────────────────────────────────────────────────────────────────────
a, b   = 6, 2          # width, height
nx = ny = 3            # elements per direction
order  = 2             # Q9 elements

# ── Physics ───────────────────────────────────────────────────────────────────
rho, mu = 1.0, 1.0
Vw      = 1.0          # top-wall speed

# ── Boundary conditions ───────────────────────────────────────────────────────
top = BoundaryCondition(
    name="moving-top-wall",
    boundary_key="top",
    type=BCType.DIRICHLET,
    variable=BCVar.VELOCITY,
    value= (Vw, 0),
    apply_strong=True,
    metadata={"Vx": Vw, "Vy": 0},
)
bottom = BoundaryCondition(
    name="no-slip",
    boundary_key="bottom",
    type=BCType.DIRICHLET,
    variable=BCVar.VELOCITY,
    value=(0.0, 0.0),
    apply_strong=True,
    metadata={"note": "no-slip"},
)
outlet = BoundaryCondition(
    name="outlet-stressfree",
    boundary_key="right",
    type=BCType.NEUMANN,
    variable=BCVar.TRACTION,
    value = (0.0, 0.0),
    apply_strong=False,
    metadata={"description": "do-nothing / traction-free outlet"},
)
inlet = BoundaryCondition(
    name="inlet-stressfree",
    boundary_key="left",
    type=BCType.NEUMANN,
    variable=BCVar.TRACTION,
    value = (0.0, 0.0),
    apply_strong=False,
    metadata={"description": "do-nothing / traction-free inlet"},
)
# ── Solve ─────────────────────────────────────────────────────────────────────
sol = NavierStokesSolver.uniform_rectangular_domain_rect(
    nx, ny, a, b, order=order, 
    alpha=0.5,
)
sol.setup_physics(rho, mu)
sol.setup_boundary_conditions([bottom, outlet, top, inlet])

sol.solve_steadystate(u0=10, p0=100, solver = 0)
sol_vx, sol_vy, sol_p = sol.get_solution()



# ── Solve ─────────────────────────────────────────────────────────────────────

def vx_analytical(x, y):
    """Linear Couette profile."""
    return Vw * y / b

def vy_analytical(x, y):
    """Zero vertical velocity."""
    return np.zeros_like(np.asarray(y, dtype=float))

def gradv_analytical(x,y):
    """velocity gradient"""
    return np.array([[0, Vw/b],
                     [0, 0]])

def p_analytical(x, y):
    """Uniform reference pressure."""
    ref = sol.p_ref_node.value
    if isinstance(x, (float, int)):
        return np.ones_like(np.asarray(y, dtype=float)) * ref
    elif isinstance(y, (float, int)):
        return np.ones_like(np.asarray(x, dtype=float)) * ref
    return ref

uni_x_clusters = sol.group_by_x()
filtered = {k: v for k, v in uni_x_clusters.items() if k in [0.0, a]}

L2v_norm, H1sm, H1_norm, L2_p_norm = sol.error_analysis(vx_analytical, vy_analytical, gradv_analytical, p_analytical)
print("||u-uh||_L2: {}, |u-uh|_H1: {}, ||u-uh||_H1: {}, ||p-ph||_L2: {}".format(L2v_norm, H1sm, H1_norm, L2_p_norm))




fig, ax = plt.subplots()
mag = np.sqrt(sol_vx**2 + sol_vy**2)
mag_safe = np.where(mag > 0, mag, 1.0)
vx_n = sol_vx / mag_safe
vy_n = sol_vy / mag_safe
sol.plot_mesh()
ax.quiver(sol.p2_nodes[:,0], sol.p2_nodes[:,1], vx_n, vy_n)

# # ── velocity profiles ─────────────────────────────────────────────────────────
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# axes = [ax1,ax2]


# ax = axes[0]
# ref_con = filtered[a]
# ax.plot(vx_analytical(a, sol.p2_nodes[ref_con, 1]),
#         sol.p2_nodes[ref_con, 1], 'r', lw=2, label='Analytical')
# for i, (xs, con) in enumerate(filtered.items()):
#     ls, m = styles[i]
#     ax.plot(sol_vx[con], sol.p2_nodes[con, 1],
#             'k', marker=m, linestyle=ls, ms=8, markerfacecolor='none',
#             label=f'FEM  $x={xs:.1f}$')
# ax.set_xlabel('$v_x$')
# ax.set_ylabel('$y$', rotation=0, labelpad=10)
# ax.set_title('Velocity profile')
# ax.legend()
# ax.grid()
# fig1.tight_layout()

# # ── pressure profile ──────────────────────────────────────────────────────────
# ax = axes[1]
# p_con = [sol.vel_2_pres_mapping[n] for n in filtered[a]
#          if n in sol.vel_2_pres_mapping]
# ax.plot(p_analytical(a, sol.p1_nodes[p_con, 1]),
#         sol.p1_nodes[p_con, 1], 'r', lw=2, label='Analytical')
# for i, (xs, con) in enumerate(filtered.items()):
#     pc = [sol.vel_2_pres_mapping[n] for n in con if n in sol.vel_2_pres_mapping]
#     ls, m = linestyles[i], markers[i]
#     ax.plot(sol_p[pc], sol.p1_nodes[pc, 1],
#             'k', marker=m, linestyle=ls, ms=8, markerfacecolor='none',
#             label=f'FEM  $x={xs:.1f}$')
# ax.set_xlabel('$p$')
# ax.set_ylabel('$y$', rotation=0, labelpad=20)
# ax.set_title('Pressure profile')
# ax.legend()
# ax.grid()
# ax.set_box_aspect()
# fig2.tight_layout()

plt.show()