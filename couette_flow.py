from scipy.interpolate import griddata
from itertools import product
from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np

np.set_printoptions(linewidth=320)
import sys
np.set_printoptions(threshold=sys.maxsize, precision=7)

if __name__ == '__main__':

    a = 6
    b = 2

    nx = 6
    ny = 3
    
    nx = ny = 4
    
    order = 2


    rho = 1
    mu = 1
    Vw = 1.0

    ##############################################################################
    # BCS
    top = BoundaryCondition(
            name="moving-top-wall",
            boundary_key="top",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (Vw, 0),
            apply_strong=True,
            metadata={"Vx": Vw, "Vy": 0}
        )
    
    outlet = BoundaryCondition(
            name="outlet-stressfree",
            boundary_key="right",
            bc_type=BCType.NEUMANN,
            traction=lambda x, y, t: (0.0, 0.0),
            apply_strong=False,
            metadata={"description": "do-nothing / traction-free outlet"}
        )

    inlet = BoundaryCondition(
            name="inlet-stressfree",
            boundary_key="left",
            bc_type=BCType.NEUMANN,
            traction=lambda x, y, t: (0.0, 0.0),
            apply_strong=False,
            metadata={"description": "do-nothing / traction-free inlet"}
        )

    bottom = BoundaryCondition(
            name="no-slip",
            boundary_key="bottom",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=(0.0, 0.0),
            apply_strong=True,
            metadata={"note": "no-slip"}
        )
        
    boundary_conditions = [bottom, outlet, top, inlet]
    
    
    ######################################################################
    # START SETTING UP SOLVER
    
    sol = NavierStokesSolver.uniform_rectangular_domain_rect(nx, ny, a, b, order = order)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions(boundary_conditions)
    
    ####################
    # EXECUTE
    p0 = 100
            
    u0 = 10
    
    sol.solve_steadystate(u0=u0, p0=p0)
    sol_vx, sol_vy, sol_p = sol.get_solution()
    
    
    
    def vx_analytical(x,y):return Vw*y/b
    def vy_analytical(x,y):return 0
    def p_analytical(x,y):
        if isinstance(x, (float, int)):
            return np.ones_like(y)*sol.p_ref_node.value
        elif isinstance(y, (float, int)):
            return np.ones_like(x)*sol.p_ref_node.value
        else:
            return sol.p_ref_node.value
    




    
    print(np.allclose(sol_vx, vx_analytical(sol.p2_nodes[:,0], sol.p2_nodes[:,1])))
    print(np.allclose(sol_vy, vy_analytical(sol.p2_nodes[:,0], sol.p2_nodes[:,1])))
    print(np.allclose(sol_p,  p_analytical(sol.p1_nodes[:,0], sol.p1_nodes[:,1])))
    ####################
    # PLOTTING
    
    uni_x_clusters = sol.group_by_x()
    markers = ['o', 's', '^', 'd']
    linestyles = ['--','-', '-.']
    styles = list(product(linestyles, markers))
    styles = list(product(linestyles, markers))

    ###########################################################
    # VELOCITY PORFILES
    fig1, ax1 = plt.subplots()
    # fig1.suptitle("{} x {} Q9".format(nx, ny))
    filtered = {k: v for k, v in uni_x_clusters.items() if k in [0.0, a]}
    # filtered = {k: v for k, v in uni_x_clusters.items() if k in non_x_clusters.keys()}
    con = filtered[a]
    ax1.plot(vx_analytical(a,sol.p2_nodes[con,1]), sol.p2_nodes[con,1], 'r', label = "Analytical solution")
    for i,(xs,con) in enumerate(filtered.items()):
        ys = sol.p2_nodes[con,1]
        ls, m = styles[i]
        ax1.plot(sol_vx[con], sol.p2_nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'FEM Result at $x$ = {:.2f}'.format(xs))
        
    ax1.set_xlabel('$v_x(x,y)$')
    ax1.legend()
    ax1.set_ylabel('$y$', rotation = 0, labelpad=10)
    ax1.grid()
    fig1.tight_layout()

    
    ###########################################################
    # PRESSURE PORFILE
    

    fig2, ax2 = plt.subplots()
    
    con = [sol.vel_2_pres_mapping[_] for _ in filtered[a] if _ in sol.vel_2_pres_mapping]
    ax2.plot(p_analytical(a, sol.p1_nodes[con,1]), sol.p1_nodes[con,1], 'r', label = "Analytical solution")
    for i,(xs,con) in enumerate(filtered.items()):
        
        p_con = [sol.vel_2_pres_mapping[_] for _ in con if _ in sol.vel_2_pres_mapping]
        
        ls, m = styles[i]
        ls, m = linestyles[i], markers[i]
        ax2.plot(sol_p[p_con], sol.p1_nodes[p_con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'FEM Result at $x$ = {:.2f}'.format(xs))
        
    ax2.set_xlabel('$p(y)$')
    ax2.set_ylabel('$y$', rotation = 0, labelpad=20)
    ax2.legend()
    ax2.grid()
    fig2.tight_layout()
    



    plt.show()