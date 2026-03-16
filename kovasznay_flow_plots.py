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
def vx_analytical(x, y):
    return 1 - np.exp(lam*x)*np.cos(2*np.pi*y)

def vy_analytical(x, y):
    return (lam/(2*np.pi))*np.exp(lam*x)*np.sin(2*np.pi*y)

def p_analytical(x, y):
    return 0.5*(pref - np.exp(2*lam*x))*rho
    

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

def fig_generator():
    fig11, ax11 = plt.subplots(); fig12, ax12 = plt.subplots(); fig13, ax13 = plt.subplots()
    return [fig11, fig12, fig13], [ax11, ax12, ax13], [1e8, -1e8], [1e8, -1e8]

figs1, ax1, xl1, yl1 = fig_generator()
figs2, ax2, xl2, yl2 = fig_generator()
figs3, ax3, xl3, yl3 = fig_generator()


target_x_station =  [-0.125, 0.25, 0.625]
target_y_station =  [0.0,    0.5,  1.0]



y_fine = np.linspace(*y_domain, 100)
x_fine = np.linspace(*x_domain, 100)
for j, xs in enumerate(target_x_station):
    ax1[j].set_title(f'$x={xs:.2f}$')
    ax1[j].plot(vx_analytical(xs, y_fine), y_fine, 'r', lw=2, label='Analytical')
    ax1[j].set_xlabel('$v_x(y)$')
    ax1[j].set_ylabel('$y$', rotation=0, labelpad=10)
    ax1[j].grid()

    xl1[0] = min(xl1[0], ax1[j].get_xlim()[0])
    xl1[1] = max(xl1[1], ax1[j].get_xlim()[1])
    
    yl1[0] = min(yl1[0], ax1[j].get_ylim()[0])
    yl1[1] = max(yl1[1], ax1[j].get_ylim()[1])

for j, ys in enumerate(target_y_station):
    ax2[j].plot(x_fine,vx_analytical(x_fine, ys), 'r', lw=2, label='Analytical')
    ax2[j].set_title(f'$y={ys:.2f}$')
    ax2[j].set_xlabel('$x$')
    ax2[j].set_ylabel('$v_x$', rotation=0, labelpad=10)
    ax2[j].grid()
    figs2[j].tight_layout()

    xl2[0] = min(xl2[0], ax2[j].get_xlim()[0])
    xl2[1] = max(xl2[1], ax2[j].get_xlim()[1])
    yl2[0] = min(yl2[0], ax2[j].get_ylim()[0])
    yl2[1] = max(yl2[1], ax2[j].get_ylim()[1])

    ax3[j].plot(x_fine,vy_analytical(x_fine, ys), 'r', lw=2, label='Analytical')
    ax3[j].set_title(f'$y={ys:.2f}$')
    ax3[j].set_xlabel('$x$')
    ax3[j].set_ylabel('$v_y$', rotation=0, labelpad=10)
    ax3[j].grid()
    figs3[j].tight_layout()

    xl3[0] = min(xl3[0], ax3[j].get_xlim()[0])
    xl3[1] = max(xl3[1], ax3[j].get_xlim()[1])
    yl3[0] = min(yl3[0], ax3[j].get_ylim()[0])
    yl3[1] = max(yl3[1], ax3[j].get_ylim()[1])


markers    = ['o', 's', '^', 'd']
linestyles = ['--', '-', '-.']


factor_list = [1, 4, 8]
factor_list = [4]
fig4, ax4 = plt.subplots(1,len(factor_list))
if len(factor_list) == 1:
    ax4 = [ax4]

for i,factor in enumerate(factor_list):
    sol = NavierStokesSolver.uniform_rectangular_domain_rect(
        nx*factor, ny*factor, 
        a, b, order=order, origin=origin,
        )
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions([bottom, top, left, right],
                                pref_corner_id=corner_id, pref_value=p_analytical(x_domain[0], y_domain[0]))
    sol.solve_steadystate(u0=1, p0=pref)

    sol_vx, sol_vy, sol_p = sol.get_solution()
    
    # test = np.allclose(sol_vx, vx_analytical(*sol.p2_nodes.T), atol=1e-4)
    # if not test:
    #     mask = ~np.isclose(sol_vx, vx_analytical(*sol.p2_nodes.T), atol=1e-4)
    #     plt.close('all')

    #     plt.figure()
    #     sol.plot_mesh()
    #     plt.plot(*sol.p2_nodes[mask].T, 'sr', ms = 10)

    #     plt.figure()
    #     plt.semilogy(abs(sol_vx - vx_analytical(*sol.p2_nodes.T)))
        
    #     break
    sol_x_stations = {k:v for k,v in sol.group_by_x().items() if k in target_x_station}
    sol_y_stations = {k:v for k,v in sol.group_by_y().items() if k in target_y_station}
    #################################################
    # y vs vx
    for j, (xs, con) in enumerate(sol_x_stations.items()):
        ax1[j].plot(sol_vx[con], sol.p2_nodes[con, 1], 
                'k', marker=markers[i], linestyle=linestyles[i],
                ms=8, markerfacecolor='none',
                label = f'FEM: Q9 {nx*factor} x {ny*factor}')

    ############################################################
    # vx vs x
    for j, (ys, con) in enumerate(sol_y_stations.items()):
        ax2[j].plot(sol.p2_nodes[con, 0], sol_vx[con], 
                'k', marker=markers[i], linestyle=linestyles[i],
                ms=8, markerfacecolor='none',
                label = f'FEM: Q9 {nx*factor} x {ny*factor}')

    ############################################################
    # vy vs x
    for j, (ys, con) in enumerate(sol_y_stations.items()):
        ax3[j].plot(sol.p2_nodes[con, 0], sol_vy[con], 
                'k', marker=markers[i], linestyle=linestyles[i],
                ms=8, markerfacecolor='none',
                label = f'FEM: Q9 {nx} x {ny}')




    ##############################################################################
    # STREAMLINES
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

    sol.plot_mesh(ax=ax4[i], plot_nodes=False)
    ax4[i].streamplot(Xi, Yi, Ui, Vi, 
                    color = 'b',
                    linewidth = 0.75,
                    broken_streamlines=False, 
                    density = 1,
                    )


ax1[-1].legend(fontsize=11)
for a in ax1:
    a.set_xlim(xl1); a.set_ylim(yl1)
    a.set_aspect('equal', adjustable='box')

ax2[-1].legend(fontsize=11)
ax3[-1].legend(fontsize=11)


plt.close('all')

fig, ax = plt.subplots()

y_target = 0.0
nodes   = sol.p1_nodes
top_idx = np.where(np.abs(nodes[:, 1] - y_target) < 1e-10)[0]
ax.plot(sol.p1_nodes[top_idx, 0], sol_p[top_idx], 
        'k', marker=markers[i], linestyle=linestyles[i],
        ms=8, markerfacecolor='none',
        label = f'FEM: Q9 {nx} x {ny}')
ax.plot(x_fine,p_analytical(x_fine, y_target), 'r', lw=2, label='Analytical')
ax.set_yscale('log')
print(p_analytical(*sol.p1_nodes[top_idx].T)/sol_p[top_idx])
plt.show()