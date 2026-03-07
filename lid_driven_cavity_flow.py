from scipy.interpolate import griddata
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
    # AN INTRODUCTION TO NONLINEAR FINITE ELEMENT ANALYSIS: WITH APPLICATIONS TO HEAT TRANSFER, FLUID MECHANICS, AND SOLID MECHANICS (2ND EDN) -  J. N. REDDY
    # Example 10.8.2
    a = 1
    b = 1 

    
    nx = ny = 14

    order = 2

    rho = 250
    # rho = 100

    mu = 1
    V0 = 1.0
    Pref = 0


    ##############################################################################
    # BCS
    bc_top = BoundaryCondition(
            name="moving-fluid",
            boundary_key="top",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (V0, 0.0),
            apply_strong=True,
            metadata={"Vx": V0, "Vy": 0.0}
        )
    
    bc_right = BoundaryCondition(
            name="no-slip",
            boundary_key="right",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )


    bc_bot = BoundaryCondition(
            name="no-slip",
            boundary_key="bottom",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )

    bc_left = BoundaryCondition(
            name="no-slip",
            boundary_key="left",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )

    
    
    boundary_conditions = [bc_bot, bc_left, bc_top, bc_right]
    
    ######################################################################
    # START SETTING UP SOLVER
    
    sol = IncompNavierStokesSolver2D.uniform_rectangular_domain_rect(nx, ny, a, b, order = order)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions(boundary_conditions, pref_corner_id=3)

    ####################
    # EXECUTE

    tol = 1e-3
    nonlinear_option = {'tol': tol}

    uSol = sol.solve_steadystate(nonlinear_solver_options=nonlinear_option,
                                #  solver = 'newton',
                                 )
    vx, vy, p = uSol[:sol.vdof], uSol[sol.vdof:-sol.pdof], uSol[-sol.pdof:]
    
    ####################
    # PLOTTING
    x_clusters = sol.group_by_x()

    markers = ['o', 's', '^', 'd']
    linestyles = ['-.', '--']
    styles = list(product(linestyles, markers))

    ###########################################################
    # VELOCITY PORFILES
    fig1, ax1 = plt.subplots(1, 2,sharey=True)
    # fig1.suptitle("{} x {} Q9".format(nx, ny))
    filtered = {k: v for k, v in x_clusters.items() if np.isclose(k, 0.5)}
    # filtered = {k: v for k, v in uni_x_clusters.items() if k in non_x_clusters.keys()}
    for i,(xs,con) in enumerate(filtered.items()):
        ys = sol.nodes[con,1]
        ls, m = styles[i]
        ax1[0].plot(vx[con], sol.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        ax1[1].plot(vy[con], sol.nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        
    ax1[0].set_xlabel('$v_x(x,y)$')
    ax1[1].set_xlabel('$v_y(x,y)$')
    # ax1[0].legend()
    ax1[0].set_ylabel('$y$', rotation = 0, labelpad=10)
    for _a in ax1:
        _a.grid()
        # a.set_xlim(0)
        # a.set_ylim(0)

    #########################################################################
    # Plot streamlines

    # Node data
    x = sol.nodes[:, 0]
    y = sol.nodes[:, 1]
    u = vx
    v = vy

    mag = np.sqrt(u**2 + v**2)
    u = [0 if np.isclose(i,j) else i/j for i,j in zip(u,mag)]
    v = [0 if np.isclose(i,j) else i/j for i,j in zip(v,mag)]

    # Create regular grid
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate velocities
    Ui = griddata((x, y), u, (Xi, Yi), method='cubic')
    Vi = griddata((x, y), v, (Xi, Yi), method='cubic')

    fig3, ax3 = plt.subplots()
    sol.plot_mesh(ax=ax3, plot_nodes=False)
    ax3.streamplot(Xi, Yi, Ui, Vi, 
                   color = 'b',
                   linewidth = 0.75,
                   broken_streamlines=False, 
                   density = 0.5,
                   )
    
    
    ax3.quiver(Xi, Yi, Ui, Vi)
    ax3.axis('equal')
    ax3.set_xlim(-a*0.05)

    fig4, ax4 = plt.subplots()
    sol.plot_mesh(ax=ax4, plot_nodes=False)    
    ax4.tricontour(sol.nodes[:,0], sol.nodes[:,1], np.sqrt(vx**2 + vy**2))
    ###########################################################
    # PRESSURE PORFILE
    

    fig2, ax2 = plt.subplots(1)
    filtered = {k: v for k, v in sol.group_by_y().items() if k in [0.0, 1.0]}
    for i,(ys,con) in enumerate(filtered.items()):
        xs = sol.nodes[con,0]
        # ax2[i].plot(xs, p(xs, ys), label = "Analytical solution at $y_s$ = {:.2f}".format(ys))
        
        p_con = [sol.vel_2_pres_mapping[_] for _ in con if _ in sol.vel_2_pres_mapping]
        mod_con = [_ for _ in con if _ in sol.vel_2_pres_mapping]

        ls, m = styles[i]
        ax2.plot(sol.nodes[mod_con,0], p[p_con], 
                 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none',
                 label = "$y_s = {:.2f}$".format(ys))
        
    ax2.set_ylabel('$p(y)$', rotation = 0, labelpad=10)
    ax2.grid()
    ax2.set_xlabel('$x$')



    plt.show()