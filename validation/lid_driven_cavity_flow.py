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
np.set_printoptions(threshold=sys.maxsize, precision=4)

if __name__ == '__main__':

    validation_data = np.genfromtxt(r"validation\cavity_flow_u.csv", delimiter=',')
    y_coords = validation_data[1:,0]
    validation_data_u = {k:v for k,v in zip(validation_data[0,1:],validation_data[1:,1:].T)}
    
    validation_data = np.genfromtxt(r"validation\cavity_flow_v.csv", delimiter=',')
    x_coords = validation_data[1:,0]
    validation_data_v = {k:v for k,v in zip(validation_data[0,1:],validation_data[1:,1:].T)}
    

    # AN INTRODUCTION TO NONLINEAR FINITE ELEMENT ANALYSIS: WITH APPLICATIONS TO HEAT TRANSFER, FLUID MECHANICS, AND SOLID MECHANICS (2ND EDN) -  J. N. REDDY
    # Example 10.8.2
    a = 1
    b = 1 

    
    nx = ny = 14
    # nx = ny = 1
    
    
    order = 2

    rho_list = sorted(validation_data_u.keys())
    rho = rho_list[0]
    
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
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )


    bc_bot = BoundaryCondition(
            name="no-slip",
            boundary_key="bottom",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )

    bc_left = BoundaryCondition(
            name="no-slip",
            boundary_key="left",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )

    
    
    boundary_conditions = [bc_bot, bc_left, bc_top, bc_right]
    
    ######################################################################
    # START SETTING UP SOLVER
    
    sol = NavierStokesSolver.uniform_rectangular_domain_rect(nx, ny, a, b, order = order)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions(boundary_conditions, pref_corner_id=3)

    ####################
    # EXECUTE

    tol = 1e-3
    nonlinear_option = {'tol': tol,
                        # "max_iter": 1
                        }
    v0 = -0.1
    uSol = sol.solve_steadystate(v0=v0,
                                 nonlinear_solver_options=nonlinear_option,
                                #  solver = 'newton',
                                 )
    vx, vy, p = uSol[:sol.vdof], uSol[sol.vdof:-sol.pdof], uSol[-sol.pdof:]
    
    ####################
    # PLOTTING

    markers = ['s', '^', 'd', 'o']
    linestyles = ['-.', '--']
    styles = list(product(linestyles, markers))

    ###########################################################
    # VELOCITY PORFILES
    fig1, ax1 = plt.subplots(1, 2)

    filtered = {k: v for k, v in sol.group_by_x().items() if np.isclose(k, 0.5)}
    for i,(xs,con) in enumerate(filtered.items()):
        ys = sol.p2_nodes[con,1]
        ls, m = styles[i]
        ax1[0].plot(vx[con], sol.p2_nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        ax1[0].plot(validation_data_u[rho], y_coords, 'or')

    filtered = {k: v for k, v in sol.group_by_y().items() if np.isclose(k, 0.5)}
    for i,(ys,con) in enumerate(filtered.items()):
        ax1[1].plot(vy[con], sol.p2_nodes[con,0], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        ax1[1].plot(validation_data_v[rho], x_coords, 'or')
    
    ax1[0].set_ylim(0, 1)
    ax1[0].set_xlabel('$v_x(x,y)$')
    ax1[0].set_ylabel('$y$', rotation = 0, labelpad=10)

    ax1[1].set_ylim(0, 1)
    ax1[1].set_xlabel('$v_y(x,y)$')
    ax1[1].set_ylabel('$x$', rotation = 0, labelpad=10)
    for _a in ax1:
        _a.grid()
    fig1.tight_layout()


    ###########################################################
    # PRESSURE PORFILE
    

    fig2, ax2 = plt.subplots(1)
    filtered = {k: v for k, v in sol.group_by_y().items() if k in [1.0]}
    for i,(ys,con) in enumerate(filtered.items()):
        # ax2[i].plot(xs, p(xs, ys), label = "Analytical solution at $y_s$ = {:.2f}".format(ys))
        
        p_con = [sol.vel_2_pres_mapping[_] for _ in con if _ in sol.vel_2_pres_mapping]
        mod_con = [_ for _ in con if _ in sol.vel_2_pres_mapping]

        ls, m = styles[i]
        ax2.plot(sol.p2_nodes[mod_con,0], p[p_con], 
                 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none',
                 label = "$y_s = {:.2f}$".format(ys))
    ax2.set_ylabel('$p(x)$', rotation = 0, labelpad=10)
    ax2.set_xlabel('$x$')
    ax2.grid()



    #########################################################################
    # Plot streamlines

    # Node data
    x = sol.p2_nodes[:, 0]
    y = sol.p2_nodes[:, 1]
    u = vx
    v = vy

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

    plt.show()