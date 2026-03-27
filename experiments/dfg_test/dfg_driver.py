import h5py, json
import matplotlib.pyplot as plt
import numpy as np
from generate_dfg_mesh import load_mesh

from fem import BoundaryCondition, NavierStokesSolver, BCType, BCVar

def execute(mesh, directory, filename):
    # ── DOMAIN ───────────────────────────────────────────────────────────────────
    nodes, connectivity, bn, _ = load_mesh(mesh)
    H = np.max(nodes[:,1]) - np.min(nodes[:,1])

    # ── Physics ───────────────────────────────────────────────────────────────────
    rho = 10.0 
    mu = 1.0
    Um = 0.3


    # ── Boundary conditions ───────────────────────────────────────────────────────
    top = BoundaryCondition(
        name="no-slip-top",
        boundary_key="top",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value=(0.0, 0.0),
        apply_strong=True,
        metadata={"note": "no-slip top wall"},
    )
    bottom = BoundaryCondition(
        name="no-slip-bottom",
        boundary_key="bottom",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value=(0.0, 0.0),
        apply_strong=True,
        metadata={"note": "no-slip bottom wall"},
    )
    cylinder = BoundaryCondition(
        name="no-slip-bottom",
        boundary_key="cylinder",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value=(0.0, 0.0),
        apply_strong=True,
        metadata={"note": "no-slip cylinder surface"},
    )

    outlet = BoundaryCondition(
        name="outlet-stressfree",
        boundary_key="right",
        type=BCType.NEUMANN,
        variable=BCVar.TRACTION,
        value = (0.0, 0.0),
        apply_strong=False,
        metadata={"description": "do-nothing / traction-free outlet"},
    )
    inlet = BoundaryCondition(
        name="inlet-stressfree",
        boundary_key="left",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value = lambda x, y, t: (4*Um*y*(H-y), 0.0),
        apply_strong=True,
        metadata={"description": "inflow"},
    )

    # ── Solve ─────────────────────────────────────────────────────────────────────
    sol = NavierStokesSolver(nodes, connectivity, bn)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions([bottom, top, inlet, outlet, cylinder])

    SOLVER = 2
    NONLINEAR_OPTIONS = {'tol_newton': 1e-6,
                        'verbose' : True}
    # SOLVER = 0
    # NONLINEAR_OPTIONS = {'tol': 1e-2,
    #                     'verbose' : True}

    sol.solve_steadystate(u0=1, p0=0, nonlinear_solver_options=NONLINEAR_OPTIONS, solver = SOLVER)
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
    from pathlib import Path
    from fem.incompressibleNS.incNS_solver import _plot_contourf, _plot_mesh, _plot_velocity_stations, _plot_streamlines
    
    directory=Path("C:\GIT\FEM-solvers\experiments/dfg_test/dfg_sims")
    
    mesh = "dfg_test/meshes/dfg_benchmark_Ne1336.npz"
    filename='dfg_sim_Ne1336.h5'
    
    mesh = "dfg_test/meshes/dfg_benchmark_Ne1554.npz"
    filename='dfg_sim_Ne1554.h5'
    
    
    execute(mesh = mesh, directory=directory, filename=filename)
    
    arrays, scalars, be = load_solution_hdf5(Path(directory)/ filename)
    vx, vy, p = arrays['vx'], arrays["vy"], arrays["p"]
    vmag = np.linalg.norm(np.column_stack([vx, vy]), axis = 1)
    p2_nodes, p1_nodes, connectivity = [arrays[_] for _ in ['p2_nodes', 'p1_nodes', 'connectivity']]

    
    fig1, ax1 = plt.subplots(figsize=(14, 3))
    _plot_contourf(p2_nodes, connectivity, vmag, ax=ax1)
    _plot_mesh(p2_nodes, connectivity, be, ax=ax1)
    _plot_streamlines(p2_nodes, connectivity, vx, vy,
                    x_seed=0.05, x_n_seeds=29, ax=ax1)
    ax1.set_aspect('equal')
    fig1.tight_layout()
    pos1 = ax1.get_position()

    fig2, ax2 = plt.subplots(figsize=(14, 3))
    # ax2 = ax1
    x_station = np.linspace(0, 0.9, 10)
    _plot_velocity_stations(p2_nodes, connectivity, vx, ax=ax2, x_stations=x_station, 
                            color = 'k',
                            )
    ax2.set_aspect('equal')
    ax2.set_position(pos1)  # force identical axes box

    # fig1.savefig("contour.pdf", bbox_inches='tight')
    # fig2.savefig("stations.pdf", bbox_inches='tight')
    
    plt.show()