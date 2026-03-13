import h5py
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar

plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                     'mathtext.fontset': 'cm'})


x_domain = [-0.5, 1.0]
y_domain = [-0.5, 1.5]

origin = (x_domain[0], y_domain[0])

a = x_domain[1] - x_domain[0]
b = y_domain[1] - y_domain[0]


order  = 2             # Q9 elements

# ── Physics ───────────────────────────────────────────────────────────────────
Re = rho = 40.0
mu = 1.0


pref = 10
corner_id = 0

    
lam = Re/2 - np.sqrt((Re/2)**2 + 4*np.pi**2)
def vx_analytical(x, y):
    return 1 - np.exp(lam*x)*np.cos(2*np.pi*y)

def vy_analytical(x, y):
    return (lam/(2*np.pi))*np.exp(lam*x)*np.sin(2*np.pi*y)

def p_analytical(x, y):
    return 0.5*(pref - np.exp(2*lam*x))

def execute(nx):
    ny = nx

    
    # ── Boundary conditions ───────────────────────────────────────────────────────
    top = BoundaryCondition(
        name="dirichlet",
        boundary_key="top",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value= lambda x, y, t: (vx_analytical(x, y), vy_analytical(x,y)),
        apply_strong=True,
        metadata={"note": "no-slip top wall"},
    )


    bottom = top.copy(); bottom.boundary_key = 'bottom'
    right = top.copy(); right.boundary_key = 'right'
    left = top.copy(); left.boundary_key = 'left'

    sol = NavierStokesSolver.uniform_rectangular_domain_rect(nx, ny, a, b, order=order, origin=origin)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions([bottom, top, left, right],
                                pref_corner_id=corner_id, pref_value=p_analytical(x_domain[0], y_domain[0]))
    
    nonlinear_options = {'tol': 1e-10}
    sol.solve_steadystate(u0=1, p0=pref, nonlinear_solver_options=nonlinear_options)
    sol.save(append=f"_nx{nx}")



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
    return arrays, scalars

def complexity_plot(prefix, fp = Path.cwd() / 'solution'):

    pattern = prefix + f"_*"
    file_list = []
    for f in list(fp.rglob(pattern)):
        file_list.append(f)
    if len(file_list) == 0:
        raise RuntimeError(f"No file found with pattern: '{pattern}'")

    target_pos1 = [-0.125, 0.0]
    convergence_data1 = np.zeros((len(file_list),2))
    
    target_pos2 = [0.625, 1.0]
    convergence_data2 = np.zeros((len(file_list),2))
    
    target_pos = [(-0.125, 0.0),
                  (0.25, 0.5),
                  (0.625, 1.0)]
    convergence_data = {_:np.zeros((len(file_list),2)) for _ in target_pos}
    
    for i,name in enumerate(file_list):
        arrays, scalars = load_solution_hdf5(fp / name)

        tolerance = 1e-6
        for pos in target_pos:
            idx = next( (i for i, node in enumerate(arrays['p2_nodes'])
                        if np.isclose(node[0], pos[0], atol=tolerance)
                        and np.isclose(node[1], pos[1], atol=tolerance)))
            convergence_data[pos][i] = scalars['Ne'], arrays['vx'][idx]

    idx = convergence_data[target_pos[0]][:, 0].argsort()
    for pos in target_pos:
        convergence_data[pos] = convergence_data[pos][idx]
    
    marker = ['o', 's', '^', 'd']
    fig1, ax1 = plt.subplots()
    for i,(target,data) in enumerate(convergence_data.items()):
        x_data = data[:-1,0]
        error = abs(data[:-1,1] - data[-1,1])
        
        l, = ax1.loglog(x_data, error, '-',
                        marker = marker[i],
                        markerfacecolor = 'none',
                         label = "(x,y) = ({},{})".format(*target) )
        m,c = np.polyfit(np.log(x_data), np.log(error), 1)
        def f(x): return x**(m)*np.exp(c)
        ax1.plot(x_data, f(x_data), '--', color = l.get_color(), 
                 label = "$\\log(e) = {:.2f}\\log(ne) + {:.2f}$".format(m,c))
        
    ax1.grid(which='major', linestyle='-', linewidth=0.8)
    ax1.grid(which='minor', linestyle='-', linewidth=0.25)
    
    ax1.legend()

    ax1.set_xlabel("Number of Elements")
    ax1.set_ylabel("$||v_x(x, T; N_e) - v_x(x, T; {:.0f})||_2$".format(convergence_data1[-1,0]))
    ax1.set_title("Spatial Computational Complexity")
    fig1.tight_layout()



if __name__ == '__main__':
    # nx_list = [4, 8, 16, 20, 24, 30]
    # nx_list = [40]
    # for nx in nx_list:
    #     execute(nx)

    pattern = "NavierStokes_steady_state"
    complexity_plot(pattern)

    plt.show()
