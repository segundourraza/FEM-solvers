from fem import IncompNavierStokesSolver2D
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(linewidth=320)

from dataclasses import dataclass

@dataclass
class BoundaryCondition:

    type

if __name__ == '__main__':
    # THE FINITE ELEMENT METHOD IN HEAT TRANSFER AND FLUID DYNAMICS
    # Example 10.8.1
    a = 6
    b = 2
    nx = ny = 2
    order = 2


    rho = mu = 1

    sol = IncompNavierStokesSolver2D.rectangular_domain_rect(a, b,
                                                      nx, ny,
                                                      order = order)
    
    
    boundary_conditions = {'left':}

    sol.setup_physics(rho, mu)
    sol.solve_steadystate(0)

    sol.plot_mesh()
    plt.show()


