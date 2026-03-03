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
    h2 = 2e-3
    h1 = 2*h2

    nx = 6
    ny = 3
    
    order = 2

    rho = 1
    mu = 1
    V0 = 1.0
    Pref = 0


    # Analytical result
    # def vx(x,y): return 3*Vw*x/(2*b)*(1 - (y/b)**2)
    # def vy(x,y): return -Vw*y/(2*b)*(3 - (y/b)**2)
    # def p(x,y):  return 3*mu*Vw/(2*b**3)*(a**2 + y**2 - x**2)
    
    # print(p(a,0))
    ##############################################################################
    # BCS
    bc_top = BoundaryCondition(
            name="no-slip",
            boundary_key="top",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (0.0, 0.0),
            apply_strong=True,
            metadata={"Vx": 0.0, "Vy": 0.0}
        )
    
    bc_right = BoundaryCondition(
            name="pressure-outlet",
            boundary_key="right",
            bc_type=BCType.NEUMANN,
            variable=BCVar.PRESSURE,
            value = lambda x, y, t: pref,
            apply_strong=True,
            metadata={}
        )

    bc_bot = BoundaryCondition(
            name="moving-wall",
            boundary_key="bottom",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value=(V0, 0.0),
            apply_strong=True,
            metadata={"Vx": V0, "Vy": 0.0}
        )

    bc_left = bc_right
    
    
    boundary_conditions = [bc_bot, bc_left, bc_top, bc_right]
    
    
    ######################################################################
    # START SETTING UP SOLVER
    
    uni = IncompNavierStokesSolver2D.uniform_rectangular_domain_rect(nx, ny, a, h1=h1, h2=h2, order = order)
    uni.setup_physics(rho, mu)
    uni.setup_boundary_conditions(boundary_conditions)

    ####################
    # EXECUTE

    uni_vx, uni_vy, uni_p = uni.solve_steadystate()
    
    ####################
    # PLOTTING
    fig, ax = plt.subplots()
    uni.plot_mesh(ax=ax)
    plt.show()
    
    uni_x_clusters = uni.group_by_x()

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
        
    ax1[0].set_xlabel('$v_x(x,y)$')
    ax1[1].set_xlabel('$v_y(x,y)$')
    ax1[0].legend()
    ax1[0].set_ylabel('$y$', rotation = 0, labelpad=10)
    for _a in ax1:
        _a.grid()
        # a.set_xlim(0)
        # a.set_ylim(0)

    #########################################################################
    # Plot streamlines

    # Node data
    x = uni.nodes[:, 0]
    y = uni.nodes[:, 1]
    u = uni_vx
    v = uni_vy

    # Create regular grid
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate velocities
    Ui = griddata((x, y), u, (Xi, Yi), method='linear')
    Vi = griddata((x, y), v, (Xi, Yi), method='linear')

    fig3, ax3 = plt.subplots()
    # uni.plot_mesh(ax3)
    ax3.streamplot(Xi, Yi, Ui, Vi, broken_streamlines=False, density = 0.5)
    ax3.axis('equal')
    


    ###########################################################
    # PRESSURE PORFILE
    

    fig2, ax2 = plt.subplots(1,2, sharey=True)
    filtered = {k: v for k, v in uni.group_by_y().items() if k in [0.0, 2.0]}
    for i,(ys,con) in enumerate(filtered.items()):
        ax2[i].set_title("$y_s = {:.2f}$".format(ys))
        
        xs = uni.nodes[con,0]
        ax2[i].plot(xs, p(xs, ys), label = "Analytical solution at $y_s$ = {:.2f}".format(ys))
        
        p_con = [uni.vel_2_pres_mapping[_] for _ in con if _ in uni.vel_2_pres_mapping]
        mod_con = [_ for _ in con if _ in uni.vel_2_pres_mapping]

        ls, m = styles[i]
        ax2[i].plot(uni.nodes[mod_con,0], uni_p[p_con], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        
    ax2[0].set_ylabel('$p(y)$', rotation = 0, labelpad=10)
    
    for _a in ax2:
        _a.grid()
        _a.set_xlabel('$x$')
        # a.set_xlim(0)
        # a.set_ylim(0)




    plt.show()