from fem import IncompNavierStokesSolver2D
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    height = 1
    width = 4
    nx = ny = 10
    order = 2

    sol = IncompNavierStokesSolver2D.rectangular_domain_rect(height, width, 
                                                             nx, ny,
                                                             order = order)
    sol.plot_mesh()
    plt.show()


