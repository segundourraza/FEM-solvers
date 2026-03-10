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

np.set_printoptions(linewidth=270)
import sys
np.set_printoptions(threshold=sys.maxsize, precision=5)

if __name__ == '__main__':
    # THE FINITE ELEMENT METHOD IN HEAT TRANSFER AND FLUID DYNAMICS
    # Example 10.8.1
    a = 6
    b = 2

    nx = 6
    ny = 3
    
    nx = ny = 1
    
    order = 2


    rho = 1
    mu = 1
    Vw = 1.0


    # Analytical result
    def vx(x,y): return 3*Vw*x/(2*b)*(1 - (y/b)**2)
    def vy(x,y): return -Vw*y/(2*b)*(3 - (y/b)**2)
    def p(x,y):  return 3*mu*Vw/(2*b**3)*(a**2 + y**2 - x**2)
    
    ##############################################################################
    # BCS
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
    
    # EXECUTE
    p0 = 100        
    v0 = 10
    up = uni.solve_steadystate(v0=v0, p0=p0)
    
    reduce_dim, fixed_dict, fixed_idx, free_idx = uni.__apply_dirichlet()

    R = uni.steadystate_RnJ(0, up)[0]
    plt.semilogy(free_idx, abs(R[free_idx]))
    plt.semilogy(fixed_idx, abs(R[fixed_idx]))

    idx = np.where(abs(R[free_idx]) > 1e-8)[0]
    print(len(idx))
    
    vx_idx = [_ for _ in idx if _ < uni.vdof]
    vy_idx = [_ - uni.vdof for _ in idx if _ >= uni.vdof and _ < 2*uni.vdof]
    p_idx = [_ - 2*uni.vdof for _ in idx if _ >= 2*uni.vdof]

    plt.figure()
    uni.plot_mesh()
    plt.plot(*uni.nodes[vx_idx,:].T, 'rs', markerfacecolor = 'none')
    plt.plot(*uni.nodes[vy_idx,:].T, '^b', markerfacecolor = 'none')
    print('\n'*4)
    un = uni.solve_steadystate(v0=v0, p0=p0, solver = 'newton')

    print(up)
    print(un)
    

    plt.show()