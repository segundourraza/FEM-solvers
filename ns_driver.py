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
    
    nx = 4
    ny = 6
    
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
    
    sol = IncompNavierStokesSolver2D.rectangular_domain_rect(b, a,
                                                      nx, ny,
                                                      order = order)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions(boundary_conditions)

    ####################
    # EXECUTE
    u0 = 4
    pref = 10


    sol.plot_mesh()
    sol_vx, sol_vy, sol_p = sol.solve_steadystate(u0, pref)
    print(sol_vy)
    
    # ####################
    # # PLOTTING

    # Analytical result
    vx = lambda x,y: 3*Vw*x/(2*b)*(1 - (y/b)**2)
    vy = lambda x,y: -Vw*y/(2*b)*(3 - (y/b)**2)
    p = lambda x,y: 3*mu*Vw/(2*b**3)*(a**2 + y**2 - x**2)

    x_clusters = sol.group_by_x()


    markers = ['o', 's', '^']
    linestyles = ['-.', '-']
    styles = list(product(linestyles, markers))

    fig1, ax1 = plt.subplots(1,2)
    i = -1
    for xs,con in list(x_clusters.items())[1::2]:
        i += 1
        ys = sol.nodes[con,1]
        ax1[0].plot(vx(xs,ys), ys, label = "Analytical solution at $x_s$ = {:.2f}".format(xs))
        ax1[1].plot(vy(xs,ys), ys, label = "Analytical solution at $x_s$ = {:.2f}".format(xs))
        ls, m = styles[i]
        ax1[0].plot(sol_vx[con], sol.nodes[con,1], marker = m, linestyle = ls)
        ax1[1].plot(sol_vy[con], sol.nodes[con,1], marker = m, linestyle = ls)
    
    ax1[0].set_xlabel('$v_x(x,y)$')
    ax1[1].set_xlabel('$v_y(x,y)$')
    ax1[0].legend()
    for a in ax1:
        a.set_ylabel('$y$', rotation = 0, labelpad=10)
        a.grid()
        # a.set_xlim(0)
        # a.set_ylim(0)
    fig1.tight_layout()
    





    plt.show()