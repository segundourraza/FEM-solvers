import h5py
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar

plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                     'mathtext.fontset': 'cm'})



def execute_kovazney(nx):
    x_domain = [-0.5, 1.5]
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

    def gradv_analytical(x, y):
        return np.array([[-lam*np.exp(lam*x)*np.cos(2*np.pi*y),                 2*np.pi*np.exp(lam*x)*np.sin(2*np.pi*y)],
                        [lam**2/(2*np.pi)*np.exp(lam*x)*np.sin(2*np.pi*y) ,    -lam*np.exp(lam*x)*np.cos(2*np.pi*y)]])


    def p_analytical(x, y):
        return 0.5*(pref - np.exp(2*lam*x))*rho



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
    
    H1_norm, L2_norm, L2_p_norm = sol.error_analysis(vx_analytical, vy_analytical, gradv_analytical, p_analytical)
    print("H1 norm: {}, L2 norm: {}, L2 pressure norm: {}".format(H1_norm, L2_norm, L2_p_norm))
    sol.save(append=f"_kovazney_nx{nx}")


def execute_couette(nx):
    # ── Domain ────────────────────────────────────────────────────────────────────
    a, b   = 6, 2          # width, height
    order  = 2             # Q9 elements

    # ── Physics ───────────────────────────────────────────────────────────────────
    rho, mu = 1.0, 1.0
    Vw      = 1.0          # top-wall speed
    pref = 0

    def vx_analytical(x, y):
        """Linear Couette profile."""
        return Vw * y / b

    def vy_analytical(x, y):
        """Zero vertical velocity."""
        return np.zeros_like(np.asarray(y, dtype=float))

    def gradv_analytical(x,y):
        """velocity gradient"""
        return np.array([[0, Vw/b],
                         [0, 0]])
    
    def p_analytical(x, y):
        """Uniform reference pressure."""
        return np.ones_like(x)*pref
    ny = nx

    
    # ── Boundary conditions ───────────────────────────────────────────────────────
    top = BoundaryCondition(
        name="moving-top-wall",
        boundary_key="top",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value= (Vw, 0),
        apply_strong=True,
        metadata={"Vx": Vw, "Vy": 0},
    )
    bottom = BoundaryCondition(
        name="no-slip",
        boundary_key="bottom",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value=(0.0, 0.0),
        apply_strong=True,
        metadata={"note": "no-slip"},
    )
    right = BoundaryCondition(
        name="outlet-stressfree",
        boundary_key="right",
        type=BCType.NEUMANN,
        variable=BCVar.TRACTION,
        value = (0.0, 0.0),
        apply_strong=False,
        metadata={"description": "do-nothing / traction-free outlet"},
    )
    left = BoundaryCondition(
        name="inlet-stressfree",
        boundary_key="left",
        type=BCType.NEUMANN,
        variable=BCVar.TRACTION,
        value = (0.0, 0.0),
        apply_strong=False,
        metadata={"description": "do-nothing / traction-free inlet"},
    )
    # ── Solve ─────────────────────────────────────────────────────────────────────
    sol = NavierStokesSolver.uniform_rectangular_domain_rect(nx, ny, a, b, order=order)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions([bottom, top, left, right],
                                  pref_corner_id=0, pref_value=pref)
    
    nonlinear_options = {'tol': 1e-10}
    sol.solve_steadystate(u0=1, p0=pref, nonlinear_solver_options=nonlinear_options)
    
    H1_norm, L2_norm, L2_p_norm = sol.error_analysis(vx_analytical, vy_analytical, gradv_analytical, p_analytical)
    print("H1 norm: {}, L2 norm: {}, L2 pressure norm: {}".format(H1_norm, L2_norm, L2_p_norm))
    sol.save(append=f"_couette_nx{nx}")


def execute_poiseuille(nx):
    ny = nx
    
    # ── Domain ────────────────────────────────────────────────────────────────────
    a, b   = 6, 2          # width, height
    order  = 2             # Q9 elements

    # ── Physics ───────────────────────────────────────────────────────────────────
    rho, mu = 1.0, 1.0
    
    # ── Pressure values ───────────────────────────────────────────────────────────
    p_in  = 10.0
    p_out = 4.0
    dPdx  = (p_out - p_in) / a    # pressure gradient (< 0 → flow in +x)
    
    # ── Pressure values ───────────────────────────────────────────────────────────
    def vx_analytical(x, y):
        """Parabolic Poiseuille profile."""
        return (-1.0 / (2.0 * mu)) * dPdx * y * (b - y)

    def vy_analytical(x, y):
        """Zero vertical velocity."""
        return np.zeros_like(np.asarray(y, dtype=float))
    def gradv_analytical(x,y):
        return np.array([[0, (-1.0 / (2.0 * mu)) * dPdx*(b - 2*y)],
                         [0, 0]])
    def p_analytical(x, y):
        """Linear pressure from inlet to outlet."""
        return p_in + dPdx * np.asarray(x, dtype=float)

    
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
    left = BoundaryCondition(
        name="pressure-inlet",
        boundary_key="left",
        type=BCType.NEUMANN,
        variable=BCVar.PRESSURE,
        value=p_in,
        apply_strong=False,
        metadata={"p": p_in},
    )
    right = BoundaryCondition(
        name="pressure-outlet",
        boundary_key="right",
        type=BCType.NEUMANN,
        variable=BCVar.PRESSURE,
        value=p_out,
        apply_strong=False,
        metadata={"p": p_out},
    )

    # ── Solve ─────────────────────────────────────────────────────────────────────
    sol = NavierStokesSolver.uniform_rectangular_domain_rect(nx, ny, a, b, order=order)
    sol.setup_physics(rho, mu)
    sol.setup_boundary_conditions([bottom, top, left, right])
    
    nonlinear_options = {'tol': 1e-10}
    sol.solve_steadystate(u0=1, p0=p_in, nonlinear_solver_options=nonlinear_options)
    
    H1_norm, L2_norm, L2_p_norm = sol.error_analysis(vx_analytical, vy_analytical, gradv_analytical, p_analytical)
    print("H1 norm: {}, L2 norm: {}, L2 pressure norm: {}".format(H1_norm, L2_norm, L2_p_norm))
    sol.save(append=f"_poiseuille_nx{nx}")





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

    convergence_data = np.zeros((len(file_list),4))
    
    for i,name in enumerate(file_list):
        arrays, scalars = load_solution_hdf5(fp / name)

        convergence_data[i] = scalars['Ne'], scalars['L2_velocity_norm'], scalars['H1_norm'], scalars['L2_pressure_norm']

    
    convergence_data = convergence_data[convergence_data[:, 0].argsort()]
    convergence_data = convergence_data[convergence_data[:, 0].argsort()][::-1]
    
    fig1, ax1 = plt.subplots()
    
    x_data, L2_velocity_norm, H1_norm, L2_pressure_norm = convergence_data.T
    # x_data = convergence_data[:-1,0]
    # H1_norm = abs(convergence_data[:-1,1] - convergence_data[-1,1])
    # L2_velocity_norm = abs(convergence_data[:-1,2] - convergence_data[-1,2])
    # L2_pressure_norm = abs(convergence_data[:-1,3] - convergence_data[-1,3])
        
    l, = ax1.loglog(x_data, H1_norm, '-o',
                    markerfacecolor = 'none', label = "$H_1$ norm")
    m,c = np.polyfit(np.log(x_data), np.log(H1_norm), 1)
    ax1.plot(x_data, x_data**(m)*np.exp(c), '--', color = l.get_color(), 
                label = "$\\log(e) = {:.2f}\\log(ne) + {:.2f}$".format(m,c))
    
    l, = ax1.loglog(x_data, L2_velocity_norm, '-o',
                    markerfacecolor = 'none', label = "$L_2$ velocity norm")
    m,c = np.polyfit(np.log(x_data), np.log(L2_velocity_norm), 1)
    ax1.plot(x_data, x_data**(m)*np.exp(c), '--', color = l.get_color(), 
                label = "$\\log(e) = {:.2f}\\log(ne) + {:.2f}$".format(m,c))
    
    l, = ax1.loglog(x_data, L2_pressure_norm, '-o',
                    markerfacecolor = 'none', label = "$L_2$ pressure norm")
    m,c = np.polyfit(np.log(x_data), np.log(L2_pressure_norm), 1)
    ax1.plot(x_data, x_data**(m)*np.exp(c), '--', color = l.get_color(), 
                label = "$\\log(e) = {:.2f}\\log(ne) + {:.2f}$".format(m,c))
    

    ax1.grid(which='major', linestyle='-', linewidth=0.8)
    ax1.grid(which='minor', linestyle='-', linewidth=0.25)
    
    ax1.legend()

    ax1.set_xlabel("Number of elements")
    ax1.set_ylabel("$||\\mathbf{v} - \\mathbf{v}_*||_{H^1(\\Omega)}$")
    ax1.set_title("Spatial Computational Complexity")
    fig1.tight_layout()

    print_convergence_table(convergence_data)


def convergence_rate(err_coarse, err_fine, h_coarse, h_fine):
    return np.log(err_coarse / err_fine) / np.log(h_coarse / h_fine)

def print_convergence_table(results):
    """
    results: list of (h, err_u_L2, err_u_H1, err_p_L2)
    """
    print(f"{'h':>10}  {'|u-uh|_L2':>12}  {'rate':>6}  "
          f"{'|u-uh|_H1':>12}  {'rate':>6}  "
          f"{'|p-ph|_L2':>12}  {'rate':>6}")
    print("-" * 75)
    for i, (h, eL2, eH1, epL2) in enumerate(results):
        if i == 0:
            print(f"{h:>10.4e}  {eL2:>12.4e}  {'—':>6}  "
                  f"{eH1:>12.4e}  {'—':>6}  "
                  f"{epL2:>12.4e}  {'—':>6}")
        else:
            h_prev, eL2_prev, eH1_prev, epL2_prev = results[i-1]
            rL2  = convergence_rate(eL2_prev,  eL2,  h_prev, h)
            rH1  = convergence_rate(eH1_prev,  eH1,  h_prev, h)
            rpL2 = convergence_rate(epL2_prev, epL2, h_prev, h)
            print(f"{h:>10.4e}  {eL2:>12.4e}  {rL2:>6.2f}  "
                  f"{eH1:>12.4e}  {rH1:>6.2f}  "
                  f"{epL2:>12.4e}  {rpL2:>6.2f}")



if __name__ == '__main__':
    # # nx_list = [4, 8, 16, 20, 24, 30, 40]
    # nx_list = [4, 8, 16, 20, 24]
    # # nx_list = [4]
    # for nx in nx_list:
    #     # execute_kovazney(nx)
    #     # execute_couette(nx)
    #     execute_poiseuille(nx)

    # pattern = "NavierStokes_steady_state*_kovazney"
    # pattern = "NavierStokes_steady_state*_couette"
    pattern = "NavierStokes_steady_state*_poiseuille"
    complexity_plot(pattern)

    plt.show()
