import h5py
from datetime import timezone, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.linalg as sp
from scipy.sparse import csc_matrix, bmat
import scipy.sparse.linalg as linalg

from ._bcs import BoundaryCondition, BCVar, BCType

from .._utils import LinearRectElement, QuadraticRectElement
from .._utils import generate_rect_mesh, generate_rectangular_domain, boundary_edges_connectivity
from .._utils import _progress_range, tqdm, NonConstantJacobian
from .._utils._mesh import EdgesDict


class IncompNavierStokesSolver2D():
    
    __setup_physics = False
    __setup_bcs = False

    def __init__(self, nodes: np.ndarray, connectivity: np.ndarray, boundary_edges:EdgesDict):
        
        # Nodes and connectivity
        self.__nodes = nodes
        self.__velocity_connectivity = connectivity

        # Number of nodes and elements
        self.__Nv = len(self.__nodes)
        self.__Nele = len(self.__velocity_connectivity)
        
        # Number of nodes per element
        self.__nv = len(self.__velocity_connectivity[0])
        if self.__nv == 3:
            raise NotImplementedError("Triangular Elements are not yet supported for {}".format(self.__class__.__name__))
        elif self.__nv == 4:
            raise NotImplementedError("MINI Elements are not yet supported for {}".format(self.__class__.__name__))
        elif self.__nv == 9:
            self.velocity_element = QuadraticRectElement()
            self.pressure_element = LinearRectElement()
            
            self.__np = 4
            self.__build_pressure_mesh()
        else:
            raise ValueError(f"No compatible element for a {self.__nv} point element.")
        
        self.__edges: EdgesDict = boundary_edges



    #####################################################################
    # CONSTRUCTORS
    @classmethod
    def rectangular_domain_tri(cls, height, width, mesh_size = 0.08):
        """
        Generate a 2D triangular mesh of a rectangle height x width.
        """
        nodes, connectivity = generate_rectangular_domain(height=height, width=width, mesh_size=mesh_size)
        return cls(nodes=nodes, connectivity=connectivity)
    
    @classmethod
    def rectangular_domain_rect(cls, height, width, nx, ny, order, element = 'complete'):
        """
        Generate a 2D rectangular mesh of a rectangle height x width.
        """
        nodes, connectivity = generate_rect_mesh(nx, ny, width, height, order=order)
        boundary_edges = boundary_edges_connectivity(connectivity, nx, ny, order=order, element=element)
        return cls(nodes=nodes, connectivity=connectivity, boundary_edges=boundary_edges)
    
    @classmethod
    def duct_domain_rect(cls, h1, h2, width, nx, ny, order,element = 'complete'):
        """
        Generate a 2D duct mesh of a rectangle height x width.
        """
        nodes, connectivity = generate_rect_mesh(nx, ny, width, h1=h1, h2=h2, order=order,element=element)
        boundary_edges = boundary_edges_connectivity(connectivity, nx, ny, order=order, element=element)
        return cls(nodes=nodes, connectivity=connectivity, boundary_edges=boundary_edges)
    


    ###############################################################
    # SETTING UP METHODS
    def setup_physics(self, rho, viscosity):
        self.rho = rho
        self.mu = viscosity
        
        self.__setup_physics = True

    def setup_boundary_conditions(self, bc_list:Iterable[BoundaryCondition]):
        for bc in bc_list:
            if isinstance(bc, BoundaryCondition):
                bc.attach_segments_from_edges(edges_dict=self.__edges)
            else:
                raise TypeError("All elements of 'bc_list' must be of type {}".format(BoundaryCondition))
        self.__boundary_conditions = bc_list
        self.__setup_bcs = True




    ###############################################################
    # SIMULATION EXECUTION
    def solve_steadystate(self, u0, nonlinear_solver_options:dict = {}):
        if (not self.__setup_bcs) or (not self.__setup_physics):
            raise RuntimeError("Physics have not been set. Please use 'setup_physics' method.")

        self.__ss_preprocessing()
        
        pass

    
    def solve_transient(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', nonlinear_solver_options:dict = {}, 
              terminate_solver = True):
        
        self.__T = T
        self.__dt = dt
        self.__t = np.arange(0, T+dt, dt)
        self.__nt = len(self.__t)

        if isinstance(time_integrator, str):
            time_integrator = TIME_INTEGRATOR_STR2INT_MAP[time_integrator.lower()]
        self.__solver_name = TIME_INTEGRATOR_INT2STR_MAP[time_integrator]
        match time_integrator:
            case 3:
                _time_stepper = self._semi_implicit_B
            case 5:
                _time_stepper = self._1SSI_scheme
            case 6:
                _time_stepper = self._2SSI_scheme
            case _:
                raise ValueError
            

        ###########
        self.__nonlinear_solver_parameters = {k:v for k,v in nonlinear_solver_options.items() if v is not None}
        


        # CONSTRUCT SOLUTION VECTOR U = [C , W]
        self.__u = np.zeros((self.__nt, 2*self.__Nv), dtype=float)
        if callable(u0):
            self.__u[0][:self.__Nv] = u0(self.__nodes[:,0], self.__nodes[:,1])
        elif len(u0) == self.__Nv:
            self.__u[0][:self.__Nv] = u0

        # Computation of W[0] via the variational problem
        N0 = self.__assemble_N(self.__u[0,:self.__Nv])
        b = self.epsilon*(self.K@self.__u[0,:self.__Nv]) + 1/self.epsilon * N0
        self.__u[0,self.__Nv:] = linalg.spsolve(self.M, b)
        
        # CONSERVED QUANTITIES
        self.__mass = np.zeros((self.__nt,), dtype=float)
        self.__energy = np.zeros((self.__nt,), dtype=float)
        self.__mass[0] = self.__compute_mass(self.__u[0])
        self.__energy[0] = self.__compute_energy(self.__u[0])

        
        ################################################
        # TIME STEPPING
        # return
        print()
        self.__termination_flag = _time_stepper(terminate_solver)
        tqdm.write("Simulation ended.\n")
        
    

    ####################################################################
    # ASSEMBLE GLOBAL LINEAR SYSTEMS
    
    def _assemble_S_mat(self,):
        S11 = np.zeros((self.__Nv, self.__Nv))
        S22 = np.zeros((self.__Nv, self.__Nv))
        S12 = np.zeros((self.__Nv, self.__Nv))
        for e, con in enumerate(self.__velocity_connectivity):
            self.velocity_element.Se(self.__nodes, con, S11, S22, S12)
        return (_*self.mu for _ in [S11, S22, S12])
    
    def __assemble_R_mat(self,):
        R10 = np.zeros((self.__Nv, self.__Np))
        R20 = np.zeros((self.__Nv, self.__Np))
        for con_v, con_p in zip(self.__velocity_connectivity, self.__pressure_connectivity):
            for (xi,eta), wi in zip(*self.velocity_element.quadrature_points(9)):
                # VELOCITY FINITE ELEMENT
                grad_psi_hat = self.velocity_element.grad_basis_functions(xi,eta)
                jac = self.velocity_element.jacobian(self.__nodes[con_v], xi, eta)
                detJ = jac[0,0]*jac[1,1] - jac[1,0]*jac[0,1]
                invJ = np.array([[jac[1,1], -jac[1,0]],
                                [-jac[0,1], jac[0,0]]])*(1/detJ)
                grad_psi = grad_psi_hat@invJ # Map grad of shape function back to physical coordinates

                # PRESSURE FINITE ELEMENT
                phi = self.pressure_element.basis_functions(xi, eta)

                R10[np.ix_(con_v,con_p)] += np.outer(grad_psi[:,0], phi)*detJ*wi
                R20[np.ix_(con_v,con_p)] += np.outer(grad_psi[:,1], phi)*detJ*wi
        return R10, R20

    def evaluate_C(self, evaluation_u):
        C = np.zeros((self.__Nv, self.__Nv))
        for con in self.__velocity_connectivity:
            self.velocity_element._C(self.__nodes, con, C, evaluation_u[:self.__Nv], evaluation_u[self.__Nv:2*self.__Nv])
        return C
    

    #####################################################################
    # AUXILIARY FUNCTIONS
    def plot_mesh(self, ax = None, linewidth = 0.6, color = 'k', plot_nodes = True, node_color = 'k', node_size = 6, **kwargs):
        if ax is None:
            ax = plt.gca()  
        
        if plot_nodes:
            ax.plot(self.__nodes[:,0], self.__nodes[:,1], '.', color = node_color, ms = node_size)
            idx = []
            for con in self.__velocity_connectivity:
                idx.extend(con[:4])
            ax.plot(self.__nodes[idx,0], self.__nodes[idx,1], 'o', markerfacecolor = 'none', color = node_color, ms = node_size*1.5)
        
        if self.velocity_element.n == 9:
            for e, con in enumerate(self.__velocity_connectivity):
                temp = np.vstack([self.__nodes[con[:4]],self.__nodes[con[0]]]).T
                ax.plot(*temp, '-', color = color, linewidth= linewidth)
        else:
            for e, con in enumerate(self.__velocity_connectivity):
                temp = np.vstack([self.__nodes[con],self.__nodes[con[0]]]).T
                ax.plot(*temp, '-', color = color, linewidth= linewidth)

    def plot_solution(self, z, ax = None, cmap = 'jet', levels = 100, plot_mesh = False, **kwargs):
        if ax is None:
            ax = plt.gca()  
            
        vmin = kwargs.get('vmin', min(np.nanmin(z), -1.0))
        vmax = kwargs.get('vmax', max(np.nanmax(z), 1.0))

        levels = np.linspace(vmin, vmax, levels)
        if self.__nv == 3:
            tcf = ax.tricontourf(self.__tri, z, levels, cmap = cmap)
        else:
            tcf = ax.tricontourf(self.__nodes[:,0], self.__nodes[:,1], z, levels, cmap = cmap)
        
        if plot_mesh:
            self.plot_mesh(ax=ax, **kwargs)
        
        return tcf, levels

    def animate_solution(self, vector = 'c', fps = 10, cmap = 'jet', levels = 100, prepend = None, directory = None, show_mesh = False):

        if vector == 'c':
            v = self.sol_c
        elif vector == 'w':
            v = self.sol_w
        else:
            RuntimeError()
        
        if prepend is None:
            filename = self.simulation_name + "_gif_" + vector
        else:
            filename = prepend + "_" + self.simulation_name + "_gif_" + vector 

        # Ensure extension
        if not filename.endswith(".gif"):
            filename += ".gif"
        
        # Determine directory
        if directory is None:
            directory = Path.cwd() / "gifs"
        else:
            directory = Path(directory)

        # Create directory if it does not exist
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("tricontourf GIF example")

        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        tcf,_ = self.plot_solution(v[0], ax = ax, levels = levels, cmap = cmap, vmin= vmin, vmax = vmax, show_mesh=show_mesh)
        ax.set_title(f"Tme step: 0")
        cbar = fig.colorbar(tcf, ax=ax)
        
        def update(i):
            ax.clear()
            tcf,_ = self.plot_solution(v[i], ax = ax, levels = levels, cmap = cmap, vmin= vmin, vmax = vmax, show_mesh=show_mesh)
            # cbar.update_normal(tcf)
            ax.set_title(f"Time step: {i}")
            return tcf
        
        anim = FuncAnimation(fig, update, frames=self.__nt, interval=100, blit=False)

        # Save as GIF
        writer = PillowWriter(fps=fps)   # frames per second
        # pbar = tqdm(total= self.__nt, leave= LEAVE_TQDM_BAR, desc= "Animating solution:")
        pbar = _progress_range(range(self.__nt), f"Animating '{vector}' solution")
        
        def progress(i, n):
            pbar.update(1)
        
        anim.save(filepath,writer=writer,dpi=150,progress_callback=progress)
        pbar.close()
        tqdm.write(f"File saved successfully: {filepath}\n")
        plt.close(fig)
            
    def save(self, prepend = None, directory = None, append_time = False):
        if prepend is None:
            filename = self.simulation_name
        else:
            filename = prepend + "_" + self.simulation_name

        if append_time:
            filename += "_" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")

        # Ensure extension
        if not filename.endswith(".h5"):
            filename += ".h5"
        
        # Determine directory
        if directory is None:
            directory = Path.cwd() / "solution2D"
        else:
            directory = Path(directory)

        # Create directory if it does not exist
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename

        with h5py.File(filepath, "w") as f:
            sol_grp = f.create_group("solution")
            
            # -----------------
            # Save arrays
            # -----------------
            arr_grp = sol_grp.create_group("arrays")
            for name, array in zip(['sol_c', 'sol_w', 't', 'mass', 'energy', 'nodes', 'connectivity'], [self.sol_c, self.sol_w, self.t, self.__mass, self.__energy, self.__nodes, self.__velocity_connectivity]):
                arr_grp.create_dataset(
                    name,
                    data=array,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True
            )


            # -----------------
            # Save scalars
            # -----------------
            scal_grp = sol_grp.create_group("scalars")
            for name, value in zip(['Ne', 'N', 'dt', 'T'], [self.__Nele, self.__Nv, self.__dt, self.__T]):
                scal_grp.create_dataset(name, data=value)

            # -----------------
            # Save metadata
            # -----------------
            sol_grp.attrs["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
        
        print("Simulation successfully saved as : {}\n".format(filepath))

    #####################################################################
    # HELPER FUNCTIONS

    def __build_pressure_mesh(self):
        corners = self.__velocity_connectivity[:, :4]                    # (n_elems, 4)
        unique_ids = np.unique(corners)           # sorted unique global node indices
        # fast mapping old_idx -> new contiguous idx
        maxid = unique_ids.max()
        map_array = -np.ones(maxid + 1, dtype=int) # -1 for unused indices
        map_array[unique_ids] = np.arange(unique_ids.size, dtype=int)

        # produce pressure connectivity (mapped)
        self.__pressure_connectivity = map_array[corners]        # same shape as corners (n_elems, 4)
        self.__Np = int(unique_ids.shape[0])
    
    def __ss_preprocessing(self,):

        self.S11, self.S22, self.S12 = self._assemble_S_mat()
        self.R10, self.R20 = self.__assemble_R_mat()

        pass

    def __ts_preprocessing(self,):

        # Compute Jacobian
        self.__detJ = [0]*self.__Nele
        self.__InvJ = [0]*self.__Nele
        try: 
            for e,con in enumerate(self.__velocity_connectivity):
                self.__detJ[e], self.__InvJ[e] = self.velocity_element.compute_ele_properties(e,self.__nodes[con])
        
            # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
            self.M = self.__assemble_M_constant_jac()
            self.K = self.__assemble_K_constant_jac()

        except NonConstantJacobian:
            for e,con in enumerate(self.__velocity_connectivity):
                pass
        except Exception as e:
            raise e


    def Residual(self,evaluation_u):
        pass

    


    ######################################################
    # PROPERTIES

    @property
    def dt(self):
        """time step"""
        return self.__dt

    @property
    def nt(self):
        """Number of time steps"""
        return self.__nt
    
    @property
    def Nv(self):
        """Number of velocity nodes"""
        return self.__Nv
    
    @property
    def Np(self):
        """Number of pressure nodes"""
        return self.__Np
    
    @property
    def Ne(self):
        """Number of elements"""
        return self.__Nele
        
    @property
    def t(self):
        """time array"""
        return self.__t
    
    @property
    def sol_u(self):
        """solution of w"""
        return self.__u
    
    @property
    def nodes(self):
        """Nodes"""
        return self.__nodes
        
    @property
    def connectivity(self):
        """connectivity"""
        return self.__velocity_connectivity