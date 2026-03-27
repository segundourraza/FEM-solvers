import h5py, json
from itertools import product
from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np

def execute(rho, directory, filename, ne = 20):
    
    # AN INTRODUCTION TO NONLINEAR FINITE ELEMENT ANALYSIS: WITH APPLICATIONS TO HEAT TRANSFER, FLUID MECHANICS, AND SOLID MECHANICS (2ND EDN) -  J. N. REDDY
    # Example 10.8.2
    a = 1
    b = 1 

    
    nx = ny = ne
    
    
    order = 2

    if rho not in validation_data_u:
        raise ValueError("'rho' must be in {}".format(validation_data_u.keys()))
    
    mu = 1
    V0 = 1.0
    Pref = 0


    ##############################################################################
    # BCS
    bc_top = BoundaryCondition(
            name="moving-fluid",
            boundary_key="top",
            type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (V0, 0.0),
            apply_strong=True,
            metadata={"Vx": V0, "Vy": 0.0}
        )
    
    bc_right = BoundaryCondition(
            name="no-slip",
            boundary_key="right",
            type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )


    bc_bot = BoundaryCondition(
            name="no-slip",
            boundary_key="bottom",
            type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value = lambda x, y, t: (0.0, 0.0),
            metadata={}
        )

    bc_left = BoundaryCondition(
            name="no-slip",
            boundary_key="left",
            type=BCType.DIRICHLET,
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

    tol = 1e-5
    SOLVER = 2
    NONLINEAR_OPTION = {'tol_newton': tol,
                        }
    sol.solve_steadystate(v0=-1.0,nonlinear_solver_options=NONLINEAR_OPTION,solver = SOLVER)
    sol.save(directory=directory, filename=filename, append="_Ne{}".format(sol.Ne))


def load_solution_hdf5(filepath):
    arrays = {}
    scalars = {}

    with h5py.File(filepath, "r") as f:
        sol_grp = f["solution"]

        # Load arrays
        for name, ds in sol_grp["arrays"].items():
            arrays[name] = ds[:]

        # Load scalars
        for name, ds in sol_grp["scalars"].items():
            scalars[name] = ds[()]

        be = json.loads(f.attrs['boundary_edges'])
    return arrays, scalars, be

if __name__ == '__main__':

    validation_data = np.genfromtxt(r"C:\GIT\FEM-solvers\experiments\lid-driven-cavity\cavity_flow_u.csv", delimiter=',')
    y_coords = validation_data[1:,0]
    validation_data_u = {k:v for k,v in zip(validation_data[0,1:],validation_data[1:,1:].T)}
    outliers_data_u = {100:   [9],
                       400:   [10],
                       1000:  [11], 
                       3200:  [12], 
                       5000:  [13], 
                       7500:  [14], 
                       10000: [15], 
                       }
    validation_data = np.genfromtxt(r"C:\GIT\FEM-solvers\experiments\lid-driven-cavity\cavity_flow_v.csv", delimiter=',')
    x_coords = validation_data[1:,0]
    validation_data_v = {k:v for k,v in zip(validation_data[0,1:],validation_data[1:,1:].T)}
    outliers_data_v = {100:   [7,9],
                       400:   [6, 10],
                       1000:  [5, 11], 
                       3200:  [4, 12], 
                       5000:  [3, 13], 
                       7500:  [2, 14], 
                       10000: [1, 15], 
                       }


    
    from pathlib import Path
    from fem.incompressibleNS.incNS_solver import _plot_contourf, _plot_mesh, _plot_velocity_stations, _plot_streamlines
    from fem._utils._mesh import group_by_y, group_by_x
    

    rho = 3200
    ne = 50
    
    directory=Path("C:\GIT\FEM-solvers\experiments\lid-driven-cavity\sims")
    filename='lid_sim_rho{}.h5'.format(rho)
    
    
    execute(rho, directory=directory, filename=filename, ne = ne)
    
    arrays, scalars, be = load_solution_hdf5(Path(directory)/ filename)
    vx, vy, p = arrays['vx'], arrays["vy"], arrays["p"]
    vmag = np.linalg.norm(np.column_stack([vx, vy]), axis = 1)
    p2_nodes, p1_nodes, connectivity = [arrays[_] for _ in ['p2_nodes', 'p1_nodes', 'connectivity']]

    
    fig1, ax1 = plt.subplots()
    _plot_contourf(p2_nodes, connectivity, vmag, ax=ax1)
    _plot_mesh(p2_nodes, connectivity, be, ax=ax1)
    _plot_streamlines(p2_nodes, connectivity, vx, vy, ax=ax1, density = 0.5, broken_streamlines = False)
    # _plot_streamlines(p2_nodes, connectivity, vx, vy, 
    #                   x_seed=0.9, x_n_seeds=10,
    #                   y_seed=0.5, y_n_seeds=10,
    #                   ax=ax1)
    
    fig1.tight_layout()

    
    ####################
    # PLOTTING

    markers = ['s', '^', 'd', 'o']
    linestyles = ['-.', '--']
    styles = list(product(linestyles, markers))

    ###########################################################
    # VELOCITY PORFILES
    fig11, ax11 = plt.subplots()
    fig21, ax12 = plt.subplots()
    ax1 = [ax11, ax12]
    figs = [fig11, fig21]

    filtered = {k: v for k, v in group_by_x(p2_nodes).items() if np.isclose(k, 0.5)}
    for i,(xs,con) in enumerate(filtered.items()):
        ys = p2_nodes[con,1]
        ls, m = styles[i]
        ax1[0].plot(vx[con], p2_nodes[con,1], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')
        
        mask = np.ones_like(validation_data_u[rho], bool)
        mask[outliers_data_u[rho]] = 0
        ax1[0].plot(validation_data_u[rho][mask],  y_coords[mask], 'ok')
        ax1[0].plot(validation_data_u[rho][~mask],  y_coords[~mask], 'xr', markeredgewidth = 2)
    
    filtered = {k: v for k, v in group_by_y(p2_nodes).items() if np.isclose(k, 0.5)}
    for i,(ys,con) in enumerate(filtered.items()):
        ax1[1].plot(p2_nodes[con,0], vy[con], 'k', marker = m, linestyle = ls, ms = 8, markerfacecolor = 'none', label = 'Uniform')

        mask = np.ones_like(validation_data_u[rho], bool)
        mask[outliers_data_u[rho]] = 0
        ax1[1].plot(x_coords[mask],  validation_data_v[rho][mask],  'ok')
        ax1[1].plot(x_coords[~mask], validation_data_v[rho][~mask],  'xr', markeredgewidth = 2)
    
    ax1[0].set_ylim(0, 1)
    ax1[0].set_xlabel('$v_x(0.0,y)$')
    ax1[0].set_ylabel('$y$', rotation = 0, labelpad=20)

    ax1[1].set_xlim(0, 1)
    ax1[1].set_ylabel('$v_y(x,0.0)$', rotation = 0, labelpad=20)
    ax1[1].set_xlabel('$x$')
    for _a in ax1:
        _a.grid()
    for _f in figs:
        _f.tight_layout()


    
    plt.show()