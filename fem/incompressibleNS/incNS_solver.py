import h5py, warnings
from copy import deepcopy
from datetime import timezone, datetime
from pathlib import Path
from typing import Iterable, Tuple, Callable, List, Dict, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix, bmat, csr_matrix, block_diag

from ._bcs import BoundaryCondition, BCVar, BCType, SegmentsList, PressureReferenceNode, SegmentWithElem

from .._utils import LinearRectElement, QuadraticRectElement
from .._utils import (generate_uniform_rect_mesh, boundary_edges_connectivity, generate_nonuniform_rect_mesh)
from .._utils import _progress_range, tqdm, NonConstantJacobian
from .._utils._mesh import EdgesDict, group_array

np.random.seed(0)
class IncompNavierStokesSolver2D():
    
    def __init__(self, nodes: np.ndarray, connectivity: np.ndarray, boundary_edges:EdgesDict):
        
        self.__setup_physics = False
        self.__setup_bcs = False
        self.p_ref_node = None
    
        # Nodes and connectivity
        self.__nodes = nodes
        self.__velocity_connectivity = connectivity

        # Number of nodes and elements
        self.__N_vel_nodes = len(self.__nodes)
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



    def __build_pressure_mesh(self):
        corners = self.__velocity_connectivity[:, :4]                    # (n_elems, 4)
        unique_ids = np.unique(corners)           # sorted unique global node indices
        # fast mapping old_idx -> new contiguous idx
        maxid = unique_ids.max()
        map_array = -np.ones(maxid + 1, dtype=int) # -1 for unused indices
        map_array[unique_ids] = np.arange(unique_ids.size, dtype=int)
        self.vel_2_pres_mapping = {int(i):int(map_array[i]) for i in unique_ids}
        # produce pressure connectivity (mapped)
        self.__pressure_connectivity = map_array[corners]        # same shape as corners (n_elems, 4)
        
        self.__N_pres_nodes = int(unique_ids.shape[0])
    
    def __ss_preprocessing(self,v0,p0, u0 = None):

        self.S11, self.S22, self.S12 = self._assemble_S_mat()        
        self.Q1, self.Q2 = self._assemble_Q_mat()
        if u0 is None:
            u0 = np.zeros((self.ndof,))
            try:
                u0[:-self.__N_pres_nodes] = float(v0)
            except:

                raise ValueError("'v0' must be a scalar, currently of type '{}'".format(type(v0)))
            try:
                u0[-self.__N_pres_nodes:] = float(p0)
            except:
                raise ValueError("'p0' must be a scalar, currently of type '{}'".format(type(p0)))
            return u0
        elif isinstance(u0, Iterable) and len(u0) == self.ndof:
            return u0
        else:
            raise RuntimeError()
            


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




    #####################################################################
    # CONSTRUCTORS
    @classmethod
    def uniform_rectangular_domain_rect(cls,nx: int,
                                        ny: int,
                                        width: float,
                                        h1: float,
                                        h2: float = None,
                                        order: int = 2,
                                        element: str = 'complete'):
        """
        Generate a 2D rectangular mesh of a rectangle height x width.
        """    
        nodes, connectivity = generate_uniform_rect_mesh(nx=nx, ny=ny, width=width, h1=h1, h2=h2, order=order, element=element)
        boundary_edges = boundary_edges_connectivity(connectivity, nx, ny, order=order, element=element)
        return cls(nodes=nodes, connectivity=connectivity, boundary_edges=boundary_edges)
    
    @classmethod
    def rectangular_domain_rect(cls, 
                                nx: int,
                                ny: int,
                                width: float,
                                h1: float,
                                h2: float = None,
                                order: int = 2,
                                element: str = 'complete',
                                dx: list = None,
                                dy: list = None,
                                origin: Tuple[float,float] = (0.0, 0.0)):
        """
        Generate a 2D rectangular mesh of a rectangle height x width.
        """    
        nodes, connectivity = generate_nonuniform_rect_mesh(nx=nx, ny=ny, width=width, h1=h1, h2=h2, order=order, element=element,dx=dx,dy=dy,origin=origin)
        boundary_edges = boundary_edges_connectivity(connectivity, nx, ny, order=order, element=element)
        return cls(nodes=nodes, connectivity=connectivity, boundary_edges=boundary_edges)
        
    @classmethod
    def duct_domain_rect(cls, h1, h2, width, nx, ny, order,element = 'complete'):
        """
        Generate a 2D duct mesh of a rectangle height x width.
        """
        nodes, connectivity = generate_uniform_rect_mesh(nx, ny, width, h1=h1, h2=h2, order=order,element=element)
        boundary_edges = boundary_edges_connectivity(connectivity, nx, ny, order=order, element=element)
        return cls(nodes=nodes, connectivity=connectivity, boundary_edges=boundary_edges)
    


    ###############################################################
    # SETTING UP METHODS
    def setup_physics(self, rho, viscosity):
        self.rho = rho
        self.mu = viscosity
        
        self.__setup_physics = True

    def setup_boundary_conditions(self, bc_list:Iterable[BoundaryCondition],pref_corner_id = 1, pref_node:PressureReferenceNode = None, pref_value:float = 0.0):
        self.__bc_dict: Dict[str, BoundaryCondition] = {bc.boundary_key: deepcopy(bc) for bc in bc_list}
        for bc in self.__bc_dict.values():
            if isinstance(bc, BoundaryCondition):
                bc.attach_segments_from_edges(edges_dict=self.__edges)
            else:
                raise TypeError("All elements of 'bc_list' must be of type {}".format(BoundaryCondition))

        corners = [bc.segments[-1][0][-1] for bc in self.__bc_dict.values()]

        if pref_node is None:
            self.p_ref_node = PressureReferenceNode(float(pref_value), self.vel_2_pres_mapping[corners[pref_corner_id]])
        
        
        self.__setup_bcs = True
        




    ###############################################################
    # SIMULATION EXECUTION
    def solve_steadystate(self, v0 = 0.0, p0 = 0.0, u0 = None, solver = 'picard', nonlinear_solver_options:dict = {}):
        if (not self.__setup_bcs) or (not self.__setup_physics):
            raise RuntimeError("Physics have not been set. Please use 'setup_physics' method.")
        
        self.__nonlinear_solver_parameters = {k:v for k,v in nonlinear_solver_options.items() if v is not None}

        u0 = self.__ss_preprocessing(v0,p0, u0 = None)
        if solver == 'picard':
            uSol = self._picards_iteration(0.0, u0, **self.__nonlinear_solver_parameters)            
        elif solver == 'newton':
            uSol = self._NewtonRaphson(0.0, u0, self.steadystate_RnJ, **self.__nonlinear_solver_parameters)        
        else:
            raise ValueError()
        return uSol
        return uSol[:self.__N_vel_nodes], uSol[self.__N_vel_nodes:-self.__N_pres_nodes], uSol[-self.__N_pres_nodes:]
    

    ####################################################################
    # ASSEMBLE GLOBAL LINEAR SYSTEMS
    
    def _assemble_S_mat(self,)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S11 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        S22 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        S12 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        for con in self.__velocity_connectivity:
            self.velocity_element.Se(self.__nodes, con, S11, S22, S12)
        return (_*self.mu for _ in [S11, S22, S12])
    
    def _assemble_Q_mat(self,r=2):
        Q1 = np.zeros((self.__N_vel_nodes, self.__N_pres_nodes))
        Q2 = np.zeros((self.__N_vel_nodes, self.__N_pres_nodes))
        for con_v, con_p in zip(self.__velocity_connectivity, self.__pressure_connectivity):
            for (xi,eta), wi in zip(*self.velocity_element.quadrature_points(r)):
                # VELOCITY FINITE ELEMENT
                grad_psi_hat = self.velocity_element.grad_basis_functions(xi,eta)
                jac = self.velocity_element.jacobian(self.__nodes[con_v], xi, eta)
                detJ = jac[0,0]*jac[1,1] - jac[1,0]*jac[0,1]
                invJ = np.array([[jac[1,1], -jac[1,0]],
                                [-jac[0,1], jac[0,0]]])*(1/detJ)
                grad_psi = grad_psi_hat@invJ # Map grad of shape function back to physical coordinates

                # PRESSURE FINITE ELEMENT
                phi = self.pressure_element.basis_functions(xi, eta)

                Q1[np.ix_(con_v,con_p)] += np.outer(grad_psi[:,0], phi)*detJ*wi
                Q2[np.ix_(con_v,con_p)] += np.outer(grad_psi[:,1], phi)*detJ*wi
        return Q1, Q2

    def _evaluate_C(self, u_eval)->np.ndarray:
        C = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        for con in self.__velocity_connectivity:
            self.velocity_element._C(self.__nodes, con, C, u_eval[self.vx_dof(con)], u_eval[self.vy_dof(con)])
        C *= self.rho
        return C
    
    
    
    ####################################################################
    # BOUNDARY CONDITION ENFORCEMENT
    
    def __collect_fixed_velocity_dofs(self, t: float = 0.0 ) -> Dict[int, float]:
        """
        Build list of fixed DOFs + prescribed values from BCs
        
        Returns dict fixed_dofs: dof_index -> prescribed_value
        Only collects velocity Dirichlet components. Pressure not here.
        """
        fixed = {}
        seen_dofs = {}
        for key,bc in self.__bc_dict.items():
            if not getattr(bc, "active", True):
                continue
            if bc.variable not in (BCVar.VELOCITY, BCVar.BOTH):
                continue

            # Get list of nodes to where apply BC to
            # node_list = _extract_nodes_from_segments(bc.segments)
            node_list = np.unique([i for _ in bc.segments for i in _[0]])
            
            if len(node_list) == 0:
                # No segments attached to the BC
                continue

            for node in node_list:
                # evaluate prescribed value at (x,y,t)
                if callable(bc.value):
                    vx_val, vy_val = bc.value(*self.__nodes[node,:], t)
                else:
                    # expected tuple or None
                    if bc.value is None:
                        vx_val, vy_val = (None, None)
                    else:
                        if isinstance(bc.value, tuple):
                            vx_val, vy_val = bc.value
                        else:
                            raise ValueError("Velocity Dirichlet value must be tuple (v_x,v_y) or callable")
                if vx_val is not None:
                    if self.vx_dof(node) in seen_dofs:
                        if bc.apply_strong:
                            fixed[self.vx_dof(node)] = float(vx_val)
                            seen_dofs[self.vx_dof(node)] = key
                    else:
                        fixed[self.vx_dof(node)] = float(vx_val)
                        seen_dofs[self.vx_dof(node)] = key

                if vy_val is not None:
                    if self.vy_dof(node) in seen_dofs:
                        if bc.apply_strong:
                            fixed[self.vy_dof(node)] = float(vy_val)
                            seen_dofs[self.vy_dof(node)] = key
                    else:
                        fixed[self.vy_dof(node)] = float(vy_val)
                        seen_dofs[self.vx_dof(node)] = key

        return fixed

    def __apply_neumann(self, t = 0.0):
        
        for bc in self.__bc_dict.values():
            # print(bc.segments)
            pass




        
    def __enforce_bcs(self, t: float = 0.0):

        # Collect fixed velocity DOFs from BCs
        fixed_dict = self.__collect_fixed_velocity_dofs(t)

        # Add Neumann traction contributions into R here as needed
        self.__apply_neumann(t)
        
        # Combine fixed dict with pressure reference if given
        if self.p_ref_node is not None:
            fixed_dict[int(self.p_dof(self.p_ref_node.index))] = self.p_ref_node.value

        if len(fixed_dict) == 0:
            return False, None, None, None
        else:
            # Build boolean mask arrays for fixed / free DOFs
            fixed_idx = np.array(sorted(fixed_dict.keys()), dtype=int)
            mask = np.ones(self.ndof, dtype=bool)
            mask[fixed_idx] = False
            free_idx = np.nonzero(mask)[0].astype(int)
        
            return True, fixed_dict, fixed_idx, free_idx

    #####################################################################
    # PICARDS ITERATION NON-LINEAR SOLVER

    def _picards_iteration(self, t_eval, u_prev,
                           tol = 1e-8, max_iter = 100,
                           relaxation_parameter = 0,
                           verbose = True)-> np.ndarray:
        """
        Picards iteration
        """
        # start from previous solution as initial guess
        
        if relaxation_parameter < 0 or relaxation_parameter >=1.0:
            raise ValueError(f"'relaxation_parameter' must be on range (0, 1), currently equal to {relaxation_parameter}.")
        
        update_rule = lambda u, ustar: u*relaxation_parameter + (1-relaxation_parameter)*ustar
        
        Z = csr_matrix((self.__N_pres_nodes, self.__N_pres_nodes))
        A = bmat([[2*self.S11 + self.S22,   self.S12,               -self.Q1],
                  [self.S12.T,              self.S11 + 2*self.S22,  -self.Q2],
                  [-self.Q1.T,              -self.Q2.T,             Z]], format='csc')
        
        b = np.zeros((self.ndof,), dtype= float)
    
        # ENFORCE BCs
        reduce_dim, fixed_dict, fixed_idx, free_idx = self.__enforce_bcs(t_eval)
        # enforce fixed DOFs exactly on previous solution
        for dof, pres in fixed_dict.items():
            u_prev[dof] = pres    
            
        for k in range(max_iter):

            # SOLVE
            if reduce_dim:
                # Build reduced system
                C = self._evaluate_C(u_prev)
                A_full = A + block_diag([C, C, Z], format='csc')
                A_ff = A_full[free_idx][:,free_idx].tocsc()   # use CSC for solve if needed
                A_fc = A_full[free_idx][:,fixed_idx]          # sparse shape (#free, #fixed)
                
                # compute RHS_f = b - A @ u[fixed dofs]
                rhs_f = b[free_idx] - A_fc.dot(u_prev[fixed_idx])

                # SOLVE REDUCED SYSTEM
                lu = spla.splu(A_ff)
                ustar = np.empty((self.ndof))
                ustar[fixed_idx] = u_prev[fixed_idx]
                ustar[free_idx] = lu.solve(rhs_f)
            else:
                # No nodes to eliminate, solve full system
                ustar = spla.spsolve(A, b)
            # update rule
            u_next = update_rule(u_prev, ustar)
            
            if not all(np.isclose(u_next[_], fixed_dict[_]) for _ in fixed_idx):
                raise RuntimeError()
            
            # convergence check (you can check norm of R or norm of delta)
            du_norm = np.linalg.norm(u_next - u_prev)
            if verbose:
                print(f"Iteration {k: >{len(str(max_iter))}}: ||du||={du_norm:.3e}, fixed_dofs={len(fixed_dict)}")
            if  du_norm < tol:
                if verbose:
                    print()
                break
            u_prev = u_next
        else:
            # if we exit loop not converged:
            warnings.warn("Picard's Iteration failed to converge after {}".format(max_iter))
        
        return u_next



    #####################################################################
    # NEWTON-RAPHSON NON-LINEAR SOLVER

    def steadystate_RnJ(self, u_prev, u_current)->Tuple[np.ndarray, csr_matrix]:
        v1, v2 =  u_current[:self.__N_vel_nodes], u_current[self.__N_vel_nodes:-self.__N_pres_nodes]
        p = u_current[-self.__N_pres_nodes:]

        C = self._evaluate_C(u_current)

        # Compute Matrices
        C1_1 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        C2_1 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        for con in self.__velocity_connectivity:
            self.velocity_element._C1n2(self.__nodes, con, C1_1, C2_1)
        C1_1 *= self.rho
        C2_1 *= self.rho

        # Compute Forcing vector
        F1 = F2 = np.zeros((self.__N_vel_nodes,))
        
        # COMPUTING RESIDUAL VECTOR
        R1 = C.dot(v1) + 2*self.S11.dot(v1) + self.S22.dot(v1) + self.S12.dot(v2) - self.Q1.dot(p) - F1
        R2 = C.dot(v2) + (self.S12.T).dot(v1) + self.S11.dot(v2) + 2*self.S22.dot(v2) - self.Q2.dot(p) - F2
        R3 = -(self.Q1.T).dot(v1) - (self.Q2.T).dot(v2)
        

        # COMPUTING JACOBIAN
        dR1dv1 = C + C1_1@v1 + 2*self.S11 + self.S22
        dR1dv2 = C2_1@v1 + self.S12
        dR1dp  = -self.Q1

        dR2dv1 = C1_1@v2 + self.S12.T
        dR2dv2 = C + C2_1@v2 + self.S11 + 2*self.S22
        dR2dp  = -self.Q2

        dR3dv1 = -self.Q1.T
        dR3dv2 = -self.Q2.T
        
        Tangent = bmat([[dR1dv1, dR1dv2, dR1dp],
                        [dR2dv1, dR2dv2, dR2dp],
                        [dR3dv1, dR3dv2, np.zeros((self.__N_pres_nodes, self.__N_pres_nodes))]], format='csr')
        return np.concatenate([R1, R2, R3]), Tangent


    def _NewtonRaphson(self, t_eval, u_prev, RnJ,
                       tol = 1e-8, max_iter = 10,
                       line_search = None, relaxation_parameter = 0,
                       verbose = True, run_checks = True):
        """
        Newton solver for one implicit time-step.
        u_prev : vector at previous time step (size 2*N)
        Returns u_new (size 2*N)
        """
        # start from previous solution as initial guess
        
        if relaxation_parameter < 0 or relaxation_parameter >=1.0:
            raise ValueError(f"'relaxation_parameter' must be on range (0, 1), currently equal to {relaxation_parameter}.")
        
        if isinstance(line_search, str):
            match line_search.lower():
                case 'armijo':
                    Residual = lambda u_prev, u: self.steadystate_RnJ(u_prev, u)[0]
                    update_rule = lambda u_prev, u, du, res_norm: self.apply_backtracking(u_prev, u, du, res_norm,Residual, relaxation_parameter=relaxation_parameter)[0]
                case _: raise ValueError("'line_search' must be on eof {'armijo'}, currently {}".format(line_search))
        elif line_search is None:
            update_rule = lambda u_prev, u, du, res_norm: u + (1-relaxation_parameter)*du 
        else:
            raise ValueError("Unrecognized 'line_search' algorithim")
        
        reduce_dim, fixed_dict, fixed_idx, free_idx = self.__enforce_bcs(t_eval)
        # enforce fixed DOFs exactly on previous solution
        for dof, pres in fixed_dict.items():
            u_prev[dof] = pres
        
        u_next = np.copy(u_prev)
        
        for k in range(max_iter):
            # build residual and jacobian at current iterate
            Res, Jac = RnJ(u_prev, u_next)
            res_norm = np.linalg.norm(Res)
            if verbose:
                print('Iteration: {}'.format(k))
            if run_checks:
                j_res= self.fd_jacobian_check(RnJ, u_prev, u_next)
                print(f"\t Jacobian relative error = {j_res:.4e}")

            if reduce_dim:
                # Partition vectors/matrices
                # Δ_c (fixed) = prescribed - current
                delta_c = np.empty(len(fixed_idx), dtype=float)
                for i, dof in enumerate(fixed_idx):
                    delta_c[i] = fixed_dict[dof] - u_next[dof]

                # Build J_ff (free x free), J_fc (free x fixed), R_f
                # ensure CSR for slicing
                J_ff = Jac[free_idx][:,free_idx].tocsc()   # use CSC for solve if needed
                J_fc = Jac[free_idx][:,fixed_idx]          # sparse shape (#free, #fixed)
                R_f  = Res[free_idx]
                
                # compute RHS_f = -R_f - J_fc * delta_c
                rhs_f = -R_f - (J_fc.dot(delta_c))

                # SOLVE REDUCED SYSTEM
                # use sparse direct solve (CSC works for splu), or use spsolve with CSR works too
                # convert to CSR for spsolve
                lu = spla.splu(J_ff)
                delta_f = lu.solve(rhs_f)

                # Build full delta
                delta = np.empty(self.ndof, dtype=float)
                delta[free_idx] = delta_f
                delta[fixed_idx] = delta_c
            else:
                raise ValueError()
            
            
            u_next = update_rule(u_prev, u_next, delta, res_norm)
            
            # if not all(np.isclose(u_next[_], fixed_dict[_]) for _ in fixed_idx):
            #     raise RuntimeError()
            # # enforce fixed DOFs exactly (remove roundoff)
            for dof, pres in fixed_dict.items():
                u_next[dof] = pres

            # convergence check (you can check norm of R or norm of delta)
            res_norm = np.linalg.norm(Res)
            du_norm = np.linalg.norm(delta)
            if verbose:
                print(f"Newton iter {k}: ||R||={res_norm:.3e}, ||Δ||={du_norm:.3e}, fixed_dofs={len(fixed_dict)}")
            if res_norm < tol and du_norm < tol:
                if verbose:
                    print()
                break
        else:
            # if we exit loop not converged:
            warnings.warn("Newton-Raphson failed to converge after max_iter")
        
        return u_next

    def apply_backtracking(self, u_prev, u, du, res_norm, Residual, relaxation_parameter = 0.0, max_iters=10, c=1e-4, rho=0.5):
        """Backtracking Armijo line-search. Returns new u, alpha used."""
        alpha = 1.0 - relaxation_parameter
        Fu = lambda v: np.linalg.norm(Residual(u_prev, v))  # adjust if your Residual needs different args
        f0 = res_norm
        for k in range(max_iters):
            u_trial = u + alpha * du
            f_trial = Fu(u_trial)
            if f_trial <= f0 + c * alpha * (-np.dot(Residual(u_prev, u), du)):  # Armijo condition
                return u_trial, alpha
            alpha *= rho
        # if line search fails, return the damped update
        return u + alpha * du, alpha
    
    def fd_jacobian_check(self, RnJ, u_prev, u, eps=1e-6):
        Fu,J = RnJ(u_prev, u)
        v = np.random.randn(u.size)
        v /= np.linalg.norm(v)
        Jv = J.dot(v)
        FD = (RnJ(u_prev, u + eps*v)[0] - Fu) / eps
        print(FD)
        print(Jv)
        print(FD - Jv)
        res = np.linalg.norm(Jv - FD) / (np.linalg.norm(Jv) + 1e-16)
        if res < eps or np.isclose(res, eps):
            return res
        else:
            raise RuntimeError(f"Finite Difference Jacobian Check Failed. eps ({eps}) != residual ({res})")


    
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
            ax.plot(self.__nodes[idx,0], self.__nodes[idx,1], 'o', markerfacecolor = 'none', color = node_color, ms = node_size*1.25)
        


        if self.velocity_element.n == 9:
            for e, con in enumerate(self.__velocity_connectivity):
                temp = np.vstack([self.__nodes[con[:4]],self.__nodes[con[0]]]).T
                ax.plot(*temp, '-', color = color, linewidth= linewidth)
        else:
            for e, con in enumerate(self.__velocity_connectivity):
                temp = np.vstack([self.__nodes[con],self.__nodes[con[0]]]).T
                ax.plot(*temp, '-', color = color, linewidth= linewidth)


        for i,(k,bc) in enumerate(self.__bc_dict.items()):
            line_nodes = []
            for edge in bc.segments:
                line_nodes.extend(edge[0])
            lc = LineCollection([self.__nodes[line_nodes,:]], colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
            lc.set_linewidth(2.5)
            lc.set_zorder(1e9)
            ax.add_collection(lc)

        


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
            for name, value in zip(['Ne', 'N', 'dt', 'T'], [self.__Nele, self.__N_vel_nodes, self.__dt, self.__T]):
                scal_grp.create_dataset(name, data=value)

            # -----------------
            # Save metadata
            # -----------------
            sol_grp.attrs["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
        
        print("Simulation successfully saved as : {}\n".format(filepath))

    #####################################################################
    # HELPER FUNCTIONS

    def group_by_x(self):
        return {k:sorted(v,key=lambda p: self.__nodes[p,1]) for k,v in group_array(self.__nodes[:,0]).items()}
    
    def group_by_y(self):
        return {k:sorted(v,key=lambda p: self.__nodes[p,0]) for k,v in group_array(self.__nodes[:,1]).items()}
        


    ######################################################
    # PROPERTIES    
    @property
    def vdof(self):
        """Number of velocity nodes"""
        return self.__N_vel_nodes
    
    @property
    def pdof(self):
        """Number of pressure nodes"""
        return self.__N_pres_nodes
    
    @property
    def ndof(self):
        """Number of Degrees of freedom"""
        return self.__N_pres_nodes + 2*self.__N_vel_nodes
    
    @property
    def Ne(self):
        """Number of elements"""
        return self.__Nele
        
    @property
    def u(self):
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
    

    def vx_dof(self, node_idx: int) -> int:
        if isinstance(node_idx, Iterable):
            return [int(_) for _ in node_idx]
        else:
            return int(node_idx)
    
    def vy_dof(self, node_idx: int) -> int:
        if isinstance(node_idx, Iterable):
            return [int(self.__N_vel_nodes + _) for _ in node_idx]
        else:
            return int(self.__N_vel_nodes + node_idx)
    
    def p_dof(self, node_idx: int) -> int:
        if isinstance(node_idx, Iterable):
            return [int(2*self.__N_vel_nodes + _) for _ in node_idx]
        else:
            return int(2 * self.__N_vel_nodes + node_idx)


    @property
    def t(self):
        """time array"""
        return self.__t
    
    @property
    def dt(self):
        """time step"""
        return self.__dt

    @property
    def nt(self):
        """Number of time steps"""
        return self.__nt




def _extract_nodes_from_segments(segments:SegmentsList)->Set[int]:
    if segments is None:
            return []
    nodes = []
    for s in segments:
        if isinstance(s, SegmentWithElem):
            seg = s[0]
            nodes.extend([seg[0], seg[1]])
    return sorted(set(nodes))
