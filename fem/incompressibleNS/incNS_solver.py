import h5py
from datetime import timezone, datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import csc_matrix, bmat
import scipy.sparse.linalg as linalg


from .._utils import LinearRectElement, QuadraticRectElement
from .._utils import generate_rect_mesh, generate_rectangular_domain
from .._utils import _progress_range, tqdm



class IncompNavierStokesSolver2D():

    def __init__(self, nodes: np.ndarray, connectivity: np.ndarray):
        
        # Nodes and connectivity
        self.__nodes = nodes
        self.__connectivity = connectivity

        self.__n = len(self.__connectivity[0])
        
        if self.__n == 3:
            raise NotImplementedError("Triangular Elements are not yet supported for {}".format(self.__class__.__name__))
        elif self.__n == 4:
            raise NotImplementedError("MINI Elements are not yet supported for {}".format(self.__class__.__name__))
        elif self.__n == 9:
            self.velocity_element = QuadraticRectElement()
            self.pressure_element = LinearRectElement()
            self.__pressure_connectivity = [con[:4] for con in self.__connectivity]
        else:
            raise ValueError(f"No compatible element for a {self.__n} point element.")

        # Number of nodes and elements
        self.__N = len(self.__nodes)
        self.__Ne = len(self.__connectivity)

        
        # Preprocessing
        self.__preprocessing()

    def __preprocessing(self,):

        # Compute Jacobian
        self.__detJ = np.zeros((self.__Ne,))
        self.__InvJ = [0]*self.__Ne
        for e,con in enumerate(self.__connectivity):
            self.__detJ[e], self.__InvJ[e] = self.velocity_element.compute_ele_properties(self.__nodes[con])
            

        # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
        self.M = self.__assemble_M()
        self.K = self.__assemble_K()

    
    def solve(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', nonlinear_solver_options:dict = {}, 
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
        self.__u = np.zeros((self.__nt, 2*self.__N), dtype=float)
        if callable(u0):
            self.__u[0][:self.__N] = u0(self.__nodes[:,0], self.__nodes[:,1])
        elif len(u0) == self.__N:
            self.__u[0][:self.__N] = u0

        # Computation of W[0] via the variational problem
        N0 = self.__assemble_N(self.__u[0,:self.__N])
        b = self.epsilon*(self.K@self.__u[0,:self.__N]) + 1/self.epsilon * N0
        self.__u[0,self.__N:] = linalg.spsolve(self.M, b)
        
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
    def __assemble_M(self)->csc_matrix:
        """Assemble Mass Matrix"""
        M = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e,con in enumerate(self.__connectivity):
            self.velocity_element.Me(M, con, self.__detJ[e])
        return csc_matrix(M)
    
    def __assemble_K(self)->csc_matrix:
        """Assemble Stiffness Matrix"""
        K = np.zeros((self.__N, self.__N), dtype= float)
        for e,con in enumerate(self.__connectivity):
            self.velocity_element.Ke(K, con, self.__detJ[e], self.__InvJ[e])
        return csc_matrix(K)    
    
    def __assemble_N(self, evaluation_C):
        """Assemble non-linear mass matrix"""
        N = np.zeros((self.__N,), dtype=float)
        for e,con in enumerate(self.__connectivity):
            # Classic overlapping block assembly
            self.velocity_element.Ne(N, con, self.__detJ[e], evaluation_C[con])
        return N
    
    
    
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
    def rectangular_domain_rect(cls, height, width, nx, ny, order):
        """
        Generate a 2D triangular mesh of a rectangle height x width.
        """
        nodes, connectivity = generate_rect_mesh(nx, ny, width, height, order=order)
        return cls(nodes=nodes, connectivity=connectivity)
    
    @classmethod
    def 

    #####################################################################
    # AUXILIARY FUNCTIONS
    def plot_mesh(self, ax = None, linewidth = 0.6, color = 'k', plot_nodes = True, node_color = 'k', node_size = 6, **kwargs):
        if ax is None:
            ax = plt.gca()  
        
        if plot_nodes:
            ax.plot(self.__nodes[:,0], self.__nodes[:,1], '.', color = node_color, ms = node_size)
            idx = []
            for con in self.__pressure_connectivity:
                idx.extend(con)
            ax.plot(self.__nodes[idx,0], self.__nodes[idx,1], 'o', markerfacecolor = 'none', color = node_color, ms = node_size*1.5)
        
        if self.__tri:
            ax.triplot(self.__tri, linewidth = linewidth, color = color)
        else:
            if self.velocity_element.n == 9:
                for e, con in enumerate(self.__connectivity):
                    temp = np.vstack([self.__nodes[con[:4]],self.__nodes[con[0]]]).T
                    ax.plot(*temp, '-', color = color, linewidth= linewidth)
            else:
                for e, con in enumerate(self.__connectivity):
                    temp = np.vstack([self.__nodes[con],self.__nodes[con[0]]]).T
                    ax.plot(*temp, '-', color = color, linewidth= linewidth)


    def plot_solution(self, z, ax = None, cmap = 'jet', levels = 100, plot_mesh = False, **kwargs):
        if ax is None:
            ax = plt.gca()  
            
        vmin = kwargs.get('vmin', min(np.nanmin(z), -1.0))
        vmax = kwargs.get('vmax', max(np.nanmax(z), 1.0))

        levels = np.linspace(vmin, vmax, levels)
        if self.__n == 3:
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
            for name, array in zip(['sol_c', 'sol_w', 't', 'mass', 'energy', 'nodes', 'connectivity'], [self.sol_c, self.sol_w, self.t, self.__mass, self.__energy, self.__nodes, self.__connectivity]):
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
            for name, value in zip(['Ne', 'N', 'dt', 'T'], [self.__Ne, self.__N, self.__dt, self.__T]):
                scal_grp.create_dataset(name, data=value)

            # -----------------
            # Save metadata
            # -----------------
            sol_grp.attrs["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
        
        print("Simulation successfully saved as : {}\n".format(filepath))

    #####################################################################
    # HELPER FUNCTIONS
    


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
    def N(self):
        """Number of nodes"""
        return self.__N
    
    @property
    def Ne(self):
        """Number of elements"""
        return self.__Ne
        
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
        return self.__connectivity