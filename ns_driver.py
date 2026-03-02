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

    nx = 3
    ny = 5
    
    nx = 6
    ny = 10
    
    # nx = 2
    # ny = 2
    
    # nx = 1
    # ny = 1
    
    
    order = 2


    rho = mu = 1
    
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
    
    uniform = IncompNavierStokesSolver2D.uniform_rectangular_domain_rect(nx, ny, a, b, order = order)
    uniform.setup_physics(rho, mu)
    uniform.setup_boundary_conditions(boundary_conditions)

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

    uni_vx, uni_vy, uni_p = uniform.solve_steadystate(u0, pref)
    
    non_vx, non_vy, non_p = non.solve_steadystate(u0, pref)


    # ####################
    # # PLOTTING

    # Analytical result
    vx = lambda x,y: 3*Vw*x/(2*b)*(1 - (y/b)**2)
    vy = lambda x,y: -Vw*y/(2*b)*(3 - (y/b)**2)
    p = lambda x,y: 3*mu*Vw/(2*b**3)*(a**2 + y**2 - x**2)

    uni_x_clusters = uniform.group_by_x()
    non_x_clusters = non.group_by_x()


    markers = ['o', 's', '^', 'd']
    linestyles = ['-.', '-']
    styles = list(product(linestyles, markers))

    fig1, ax1 = plt.subplots(1,3)
    fig1.suptitle("{} x {} Q9".format(nx, ny))
    i = -1
    for xs,con in uni_x_clusters.items():
        if xs not in [2.0, 4.0, 6.0]:
            continue
        i += 1
        ys = uniform.nodes[con,1]
        ax1[0].plot(vx(xs,ys), ys, label = "Analytical solution at $x_s$ = {:.2f}".format(xs))
        ax1[1].plot(vy(xs,ys), ys, label = "Analytical solution at $x_s$ = {:.2f}".format(xs))
        ax1[2].plot(p(xs,ys), ys, label = "Analytical solution at $x_s$ = {:.2f}".format(xs))
        ls, m = styles[i]
        ax1[0].plot(uni_vx[con], uniform.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        ax1[1].plot(uni_vy[con], uniform.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        
        con = [uniform.nodes_2_p_dof_map[_] for _ in con if _ in uniform.nodes_2_p_dof_map]
        ax1[2].plot(uni_p[con], uniform.pressure_nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')

        con = non_x_clusters[xs]
        ys = non.nodes[con,1]
        ls, m = styles[i]
        ax1[0].plot(non_vx[con], non.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, label = 'Non-uniform')
        ax1[1].plot(non_vy[con], non.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, label = 'Non-uniform')
        
        con = [non.nodes_2_p_dof_map[_] for _ in con if _ in non.nodes_2_p_dof_map]
        ax1[2].plot(non_p[con], non.pressure_nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Non-uniform')
    

    ax1[0].set_xlabel('$v_x(x,y)$')
    ax1[1].set_xlabel('$v_y(x,y)$')
    ax1[0].legend()
    for a in ax1:
        a.set_ylabel('$y$', rotation = 0, labelpad=10)
        a.grid()
        # a.set_xlim(0)
        # a.set_ylim(0)
    





    plt.show()