from itertools import product
from fem import IncompNavierStokesSolver2D, BoundaryCondition, BCType, BCVar


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np

np.set_printoptions(linewidth=320)
import sys
np.set_printoptions(threshold=sys.maxsize, precision=4)

if __name__ == '__main__':
    # THE FINITE ELEMENT METHOD IN HEAT TRANSFER AND FLUID DYNAMICS
    # Example 10.8.1
    a = 6
    b = 2

    nx = 6
    ny = 3
    
    
    order = 2


    rho = mu = 1


    # Analytical result
    def vx(x,y): return  3*Vw*x/(2*b)*(1 - (y/b)**2)
    def vy(x,y): return  -Vw*y/(2*b)*(3 - (y/b)**2)
    def p(x,y):  return  3*mu*Vw/(2*b**3)*(a**2 + y**2 - x**2)
    
    ##############################################################################
    # BCS
    Vw = 3.0
    bc_top = BoundaryCondition(
            name="moving-top-wall",
            boundary_key="top",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (0, -Vw),
            apply_strong=True,
            metadata={"Vx": 0, "Vy": Vw}
        )
    
    bc_outlet = BoundaryCondition(
            name="outlet-stressfree",
            boundary_key="right",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            traction=lambda x, y, t: (0.0, 0.0),
            apply_strong=False,
            metadata={"description": "do-nothing / traction-free outlet"}
        )

    bc_bot_wall = BoundaryCondition(
            name="bottom-wall-symmetry",
            boundary_key="bottom",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value=(None, 0.0),
            traction=(0.0, None),
            apply_strong=True,
            metadata={"note": "symmetry wall"}
        )

    bc_left_wall = BoundaryCondition(
            name="left-wall-symmetry",
            boundary_key="left",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value=(0.0, None),
            traction=(None, 0.0),
            apply_strong=True,
            metadata={"note": "symmetry wall"}
        )
    
    
    
    boundary_conditions = [bc_bot_wall, bc_left_wall, bc_top, bc_outlet]
    
    
    ######################################################################
    # START SETTING UP SOLVER
    
    uni = IncompNavierStokesSolver2D.uniform_rectangular_domain_rect(nx, ny, a, b, order = order)
    uni.setup_physics(rho, mu)
    uni.setup_boundary_conditions(boundary_conditions)

    dx = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
    dy = [0.25, 0.25, 0.5, 0.5, 0.25, 0.25]
    nx = len(dx)
    ny = len(dy)

    non = IncompNavierStokesSolver2D.rectangular_domain_rect(nx, ny, a, b, order = order, dx = dx, dy=dy)
    non.setup_physics(rho, mu)
    non.setup_boundary_conditions(boundary_conditions)
    
    ####################
    # EXECUTE
    u0 = 4
    pref = 0

    uni_vx, uni_vy, uni_p = uni.solve_steadystate(u0, pref)
    
    non_vx, non_vy, non_p = non.solve_steadystate(u0, pref)
    print(uni_p)
    print(non_p)

    # ####################
    # # PLOTTING
    fig, ax = plt.subplots(1,2)
    uni.plot_mesh(ax=ax[0])
    non.plot_mesh(ax=ax[1])

    
    uni_x_clusters = uni.group_by_x()
    non_x_clusters = non.group_by_x()


    markers = ['o', 's', '^', 'd']
    linestyles = ['-.', '--']
    styles = list(product(linestyles, markers))

    ###########################################################
    # VELOCITY PORFILES
    fig1, ax1 = plt.subplots(1, 2,sharey=True)
    # fig1.suptitle("{} x {} Q9".format(nx, ny))
    filtered = {k: v for k, v in uni_x_clusters.items() if k in [2.0, 4.0, 6.0]}
    # filtered = {k: v for k, v in uni_x_clusters.items() if k in non_x_clusters.keys()}
    for i,(xs,con) in enumerate(filtered.items()):
        ys = uni.nodes[con,1]
        ax1[0].plot(vx(xs,ys), ys, label = "Analytical solution at $x_s$ = {:.2f}".format(xs))
        ax1[1].plot(vy(xs,ys), ys, label = "Analytical solution at $x_s$ = {:.2f}".format(xs))
        ls, m = styles[i]
        ax1[0].plot(uni_vx[con], uni.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        ax1[1].plot(uni_vy[con], uni.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        
        con = non_x_clusters[xs]
        # ys = non.nodes[con,1]

        ax1[0].plot(non_vx[con], non.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, label = 'Non-uniform')
        ax1[1].plot(non_vy[con], non.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, label = 'Non-uniform')
        
    ax1[0].set_xlabel('$v_x(x,y)$')
    ax1[1].set_xlabel('$v_y(x,y)$')
    ax1[0].legend()
    ax1[0].set_ylabel('$y$', rotation = 0, labelpad=10)
    for _a in ax1:
        _a.grid()
        # a.set_xlim(0)
        # a.set_ylim(0)
    


    ###########################################################
    # PRESSURE PORFILE
    

    fig2, ax2 = plt.subplots(1,2, sharey=True)
    filtered = {k: v for k, v in uni.group_by_y().items() if k in [0.0, 2.0]}
    for i,(ys,con) in enumerate(filtered.items()):
        ax2[i].set_title("$y_s = {:.2f}$".format(ys))
        
        xs = uni.nodes[con,0]
        ax2[i].plot(xs, p(xs, ys), label = "Analytical solution at $y_s$ = {:.2f}".format(ys))
        
        p_con = [uni.nodes_2_p_dof_map[_] for _ in con if _ in uni.nodes_2_p_dof_map]
        mod_con = [_ for _ in con if _ in uni.nodes_2_p_dof_map]
        ax[0].plot(*uni.nodes[mod_con].T, 'xr')

        ls, m = styles[i]
        ax2[i].plot(uni.nodes[mod_con,0], uni_p[p_con], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        
    filtered = {k: v for k, v in non.group_by_y().items() if k in [0.0, 2.0]}
    for i,(ys,con) in enumerate(filtered.items()):
        
        xs = non.nodes[con,0]

        print(xs)
        p_con = [non.nodes_2_p_dof_map[_] for _ in con if _ in non.nodes_2_p_dof_map]
        mod_con = [_ for _ in con if _ in non.nodes_2_p_dof_map]
        print(non.nodes[mod_con,0])
        ax[1].plot(*non.nodes[mod_con].T, 'xr')
        
        ls, m = styles[i]        
        ax2[i].plot(non.nodes[mod_con,0], non_p[p_con], 'k', marker = m, linestyle = ls, ms = 8, label = 'Non-uniform')

    ax2[0].set_ylabel('$p(y)$', rotation = 0, labelpad=10)
    
    for _a in ax2:
        _a.grid()
        _a.set_xlabel('$x$')
        # a.set_xlim(0)
        # a.set_ylim(0)




    plt.show()