from fem import IncompNavierStokesSolver2D, BoundaryCondition, BCType, BCVar


import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(linewidth=320)


if __name__ == '__main__':
    # THE FINITE ELEMENT METHOD IN HEAT TRANSFER AND FLUID DYNAMICS
    # Example 10.8.1
    a = 6
    b = 2
    nx = ny = 2
    order = 2


    rho = mu = 1
    
    ##############################################################################
    # BCS
    Vv = -1.0
    bc_top = BoundaryCondition(
            name="moving-top-wall",
            boundary_key="top",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (0, Vv),
            metadata={"Vx": 0, "Vy": Vv}
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

    bc_left_wall = BoundaryCondition(
            name="bottom-wall-symmetry",
            boundary_key="bottom",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value=(None, 0.0),
            traction=(0.0, None),
            apply_strong=True,
            metadata={"note": "symmetry wall"}
        )

    bc_right_wall = BoundaryCondition(
            name="left-wall-symmetry",
            boundary_key="left",
            bc_type=BCType.NEUMANN,
            variable=BCVar.VELOCITY,
            value=(0.0, None),
            traction=(None, 0.0),
            apply_strong=True,
            metadata={"note": "symmetry wall"}
        )
            
    
    boundary_conditions = [bc_left_wall, bc_right_wall, bc_top, bc_outlet]
    
    
    ######################################################################
    # START SETTING UP SOLVER
    
    sol = IncompNavierStokesSolver2D.rectangular_domain_rect(a, b,
                                                      nx, ny,
                                                      order = order)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions(boundary_conditions)

    ####################
    # EXECUTE
    sol.solve_steadystate(0)

    
    
    ####################
    # PLOTTING
    sol.plot_mesh()
    
    
    
    
    plt.show()