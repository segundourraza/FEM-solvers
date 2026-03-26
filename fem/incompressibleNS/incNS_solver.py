import h5py, warnings, inspect
from enum import Enum, auto
from copy import deepcopy
from datetime import timezone, datetime
from pathlib import Path
from typing import Iterable, Tuple, Callable, List, Dict, Set, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import bmat, csr_matrix, block_diag

from ._bcs import BoundaryCondition, BCVar, BCType, PressureReferenceNode

from .._utils import LinearRectElement, QuadraticRectElement
from .._utils import (generate_uniform_rect_mesh, boundary_edges_connectivity, generate_nonuniform_rect_mesh)
from .._utils import _progress_range, tqdm, NonConstantJacobian
from .._utils._mesh import EdgesDict, group_array, perturb_interior_nodes


class BoundaryConditionSingularityWarning(Warning):
    def __init__(self, *args):
        super().__init__(*args)


NONLINEAR_SOLVER_INT2STR = {0: 'picard',
                            1: 'newton',
                            2: 'continuation'}
np.random.seed(0)
class NavierStokesSolver():
    
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
            
    #####################################################################
    # CONSTRUCTORS
    @classmethod
    def uniform_rectangular_domain_rect(cls,nx: int,
                                        ny: int,
                                        width: float,
                                        h1: float,
                                        h2: float = None,
                                        order: int = 2,
                                        origin: Tuple[float,float] = (0.0, 0.0),
                                        alpha = None,
                                        element: str = 'complete'):
        """
        Generate a 2D rectangular mesh of a rectangle height x width.
        """    
        nodes, connectivity = generate_uniform_rect_mesh(nx=nx, ny=ny, width=width, h1=h1, h2=h2, order=order, element=element, origin=origin)
        boundary_edges = boundary_edges_connectivity(connectivity, nx, ny, order=order, element=element)
        if alpha is not None:
            perturb_interior_nodes(nodes, alpha, [_ for edge in boundary_edges.values() for _e in edge for _ in _e[0]])
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

    def setup_boundary_conditions(self, bc_list:Iterable[BoundaryCondition], 
                                  pref_corner_id = 1, pref_value:float = 0.0, pref_node:PressureReferenceNode = None,
                                  forcing_function:Callable = None):
        self.__forcing_func = forcing_function

        self.__bc_dict: Dict[str, BoundaryCondition] = {bc.boundary_key: deepcopy(bc) for bc in bc_list}
        require_pref = True
        for bc in self.__bc_dict.values():
            if isinstance(bc, BoundaryCondition):
                bc.attach_segments_from_edges(edges_dict=self.__edges)
            else:
                raise TypeError("All elements of 'bc_list' must be of type {}".format(BoundaryCondition))

            # Pressure specificied boundary is actually a Neumann BC
            if bc.variable == BCVar.PRESSURE:
                require_pref = False
                if bc.type == BCType.DIRICHLET:
                    bc.type = BCType.NEUMANN

            

        
        self.corner_nodes = [self.__bc_dict[keys].segments[0][0][0] for keys in ['bottom', 'right', 'top', 'left']]

        if require_pref and pref_node is None:
            if isinstance(pref_value, (float, int)):
                self.p_ref_node = PressureReferenceNode(float(pref_value), self.vel_2_pres_mapping[self.corner_nodes[pref_corner_id]])
            elif isinstance(pref_value, Callable):
                self.p_ref_node = PressureReferenceNode(pref_value(*self.__nodes[self.corner_nodes[pref_corner_id]]), self.vel_2_pres_mapping[self.corner_nodes[pref_corner_id]])
        self.__setup_bcs = True
        




    ###############################################################
    # SIMULATION EXECUTION
    def solve_steadystate(self, v0 = 0.0, p0 = 0.0, u0 = None, solver = 'newton',
                          nonlinear_solver_options:dict = {}):
        
        _solver = solver
        if isinstance(_solver,str):
            _solver = _solver.lower()
            
        if (not self.__setup_bcs) or (not self.__setup_physics):
            raise RuntimeError("Physics have not been set. Please use 'setup_physics' method.")
        
        self.nonlinear_solver_parameters = {k:v for k,v in nonlinear_solver_options.items() if v is not None}

        u0 = self.__ss_preprocessing(v0,p0, u0 = None)
        if _solver in [0,'picard']:
            _process_solver_parameter_dict(self._picards_iteration, self.nonlinear_solver_parameters)
            uSol = self._picards_iteration(0.0, u0, **self.nonlinear_solver_parameters)
        elif _solver in [1, 'newton']:
            _process_solver_parameter_dict(self._NewtonRaphson, self.nonlinear_solver_parameters)
            uSol = self._NewtonRaphson(0.0, u0, self.residual, self.Jacobian, **self.nonlinear_solver_parameters)        
        elif _solver in [2, 'continutation']:
            _process_solver_parameter_dict(self._continuation_method, self.nonlinear_solver_parameters)
            uSol = self._continuation_method(0.0, u0, **self.nonlinear_solver_parameters)

        else:
            raise ValueError()
        if isinstance(_solver, int):
            _solver = NONLINEAR_SOLVER_INT2STR[_solver]
        self.simulation_name = f"NavierStokes_steady_state_{_solver}"
        self.solution = uSol
        return uSol
    

    ####################################################################
    # ASSEMBLE GLOBAL LINEAR SYSTEMS
    
    def _assemble_S_mat(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S11 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        S22 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        S12 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        for con in self.__velocity_connectivity:
            self.velocity_element.Se(self.__nodes, con, S11, S22, S12)
        return S11*self.mu, S22*self.mu, S12*self.mu
    
    def _assemble_Q_mat(self,r=4):
        Q1 = np.zeros((self.__N_vel_nodes, self.__N_pres_nodes))
        Q2 = np.zeros((self.__N_vel_nodes, self.__N_pres_nodes))
        for con_v, con_p in zip(self.__velocity_connectivity, self.__pressure_connectivity):
            for (xi,eta), wi in zip(*self.velocity_element.quadrature_points(r)):
                # VELOCITY FINITE ELEMENT
                *_, detJ, grad_psi = self.velocity_element.properties(self.__nodes[con_v], xi, eta)                
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
    
    def _assemble_F(self):
        F = np.zeros((self.ndof,), dtype= float)
        if self.__forcing_func is not None:
            for con in self.__velocity_connectivity:
                for (xi,eta), wi in zip(*self.velocity_element.quadrature_points(self.velocity_element.r_viscous)):
                    psi_hat, _, detJ, _ = self.velocity_element.properties(self.__nodes[con], xi, eta)
                    
                    coords = psi_hat.T @ self.__nodes[con,:]
                    df = self.__forcing_func(*coords)*detJ*wi
                    F[self.vx_dof(con)] += psi_hat*df[0]
                    F[self.vy_dof(con)] += psi_hat*df[1]
        return F
    
    def _evaluate_traction(self, F:np.ndarray, con:Iterable[int], u:np.ndarray, bc:BoundaryCondition = None, t: float = None):
        v_nodes = np.column_stack([u[self.vx_dof(con)], u[self.vy_dof(con)]])
        p_nodes = u[[self.p_dof(self.vel_2_pres_mapping[_])  for _ in con if _ in self.vel_2_pres_mapping]]

        flag = True if bc is None else False
        
        for xi, wi in zip(*self.velocity_element.edge_quadrature_points(self.velocity_element.r_convective)):
            # VELOCITY FINITE ELEMENT
            psi_hat = self.velocity_element.edge_basis_function(xi)
            grad_psi_hat = self.velocity_element.edge_grad_basis_function(xi)

            # EDGE PARAMETERS
            t = grad_psi_hat @ self.__nodes[con,:] # tangent vector
            detJ = np.linalg.norm(t)
            t_unit = t / detJ
            n = np.array([t_unit[1], -t_unit[0]])

            if flag:
                
                # PRESSURE FINITE ELEMENT
                phi = self.pressure_element.edge_basis_function(xi)

                grad_v = grad_psi_hat@v_nodes
                p = phi@p_nodes
                d = 0.5*(grad_v + grad_v.T)
                tau = 2*self.mu*d

                sigma = tau - p*np.eye(2)
                t = sigma @ n
            else:
                if callable(bc.value):
                    coords = psi_hat @ self.__nodes[con,:]
                    t = bc.value(*coords, t)
                elif isinstance(bc.value, tuple):
                    t = list(bc.value)
                elif bc.variable == BCVar.PRESSURE and isinstance(bc.value,(float, int)):
                    t = - bc.value*n
                else:
                    raise ValueError            
            test = np.outer(psi_hat, t)*detJ*wi
            F[self.vx_dof(con)] += test[:,0]
            F[self.vy_dof(con)] += test[:,1]


    
    ####################################################################
    # BOUNDARY CONDITION ENFORCEMENT
    
    def __find_dirichlet_dofs(self, t: float = 0.0) -> Dict[int, float]:
        """
        Build list of dirichlet DOFs + prescribed values from BCs
        
        Returns dict fixed_dofs: dof_index -> prescribed_value
        Only collects velocity Dirichlet components. Pressure not here.
        """
        fixed = {}
        seen_dofs = {}
        for key,bc in self.__bc_dict.items():
            if not bc.active:
                continue
            if bc.variable != BCVar.VELOCITY:
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
                        if np.isclose(vx_val,fixed[self.vx_dof(node)]):
                            pass
                        elif bc.apply_strong:
                            warnings.warn("Boundary condition singularity detected for v_x: bc at '{}' and '{}' have same node index '{}'. Value for bc '{}' is applied due to 'strong_apply' flag".format(key, seen_dofs[self.vx_dof(node)], self.vx_dof(node), key), BoundaryConditionSingularityWarning)
                            fixed[self.vx_dof(node)] = float(vx_val)
                            seen_dofs[self.vx_dof(node)] = key
                    else:
                        fixed[self.vx_dof(node)] = float(vx_val)
                        seen_dofs[self.vx_dof(node)] = key

                if vy_val is not None:
                    if self.vy_dof(node) in seen_dofs:
                        if np.isclose(vy_val,fixed[self.vy_dof(node)]):
                            pass
                        elif bc.apply_strong:
                            warnings.warn("Boundary condition singularity detected for v_y: bc at '{}' and '{}' have same node index '{}'. Value for bc '{}' is applied due to 'strong_apply' flag".format(key, seen_dofs[self.vy_dof(node)], self.vy_dof(node), key), BoundaryConditionSingularityWarning)
                            fixed[self.vy_dof(node)] = float(vy_val)
                            seen_dofs[self.vy_dof(node)] = key
                    else:
                        fixed[self.vy_dof(node)] = float(vy_val)
                        seen_dofs[self.vy_dof(node)] = key

        return fixed

    def __assemble_rhs(self, u:np.ndarray, t:float = 0.0):
        
        F = self._assemble_F()
        for bc in self.__bc_dict.values():
            if bc.type == BCType.NEUMANN and bc.active:
                for segment in bc.segments:
                    self._evaluate_traction(F, segment[0], u, bc=bc, t=t)
        return F
    

        
    def _apply_dirichlet(self, t: float = 0.0):

        # Collect fixed velocity DOFs from BCs
        fixed_dict = self.__find_dirichlet_dofs(t)

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

        # ENFORCE BCs
        reduce_dim, fixed_dict, fixed_idx, free_idx = self._apply_dirichlet(t_eval)

        # enforce fixed DOFs exactly on previous solution
        for dof, pres in fixed_dict.items():
            u_prev[dof] = pres    
            
        for k in range(max_iter):

            # Add Neumann traction contributions into rhs here as needed
            b = self.__assemble_rhs(u=u_prev, t=t_eval)

            # SOLVE
            if reduce_dim:
                # Build reduced system
                C = self._evaluate_C(u_prev)
                C *= 0

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
            new_res = self.residual(u_prev, u_next)[free_idx]
            new_res_norm = np.linalg.norm(new_res)
            if verbose:    
                print(f"Iteration {k: <{len(str(max_iter))}}: ||R||={new_res_norm:.3e}, ||du||={du_norm:.3e}, fixed_dofs={len(fixed_idx)}, free_dofs = {len(free_idx)}")
            if new_res_norm < tol:
                break
            else:
                if k > 0 and abs(res_norm-new_res_norm)< tol:
                    break
            u_prev = u_next
            res_norm = new_res_norm
        else:
            # if we exit loop not converged:
            warnings.warn("Picard's Iteration failed to converge after {}".format(max_iter))
        if verbose:
            print()        
        return u_next



    #####################################################################
    # NEWTON-RAPHSON NON-LINEAR SOLVER

    def residual(self, u_prev, u_current)->np.ndarray:
        v1, v2 =  u_current[:self.vdof], u_current[self.vdof:-self.pdof]
        p = u_current[-self.pdof:]

        C = self._evaluate_C(u_current)

        # Add Neumann traction contributions into rhs here as needed
        F = self.__assemble_rhs(u=u_current)

        # COMPUTING RESIDUAL VECTOR
        R1 = C@v1 + 2*self.S11.dot(v1) + self.S22.dot(v1) + self.S12.dot(v2) - self.Q1.dot(p)
        R2 = C@v2 + (self.S12.T).dot(v1) + self.S11.dot(v2) + 2*self.S22.dot(v2) - self.Q2.dot(p)
        R3 = -(self.Q1.T).dot(v1) - (self.Q2.T).dot(v2)
    
        return np.concatenate([R1, R2, R3]) - F

    def Jacobian(self, u_prev, u_current):
        v1 = u_current[:self.__N_vel_nodes]
        v2 = u_current[self.__N_vel_nodes:-self.__N_pres_nodes]

        C    = self._evaluate_C(u_current)
        K1v1 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        K2v1 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        K1v2 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))
        K2v2 = np.zeros((self.__N_vel_nodes, self.__N_vel_nodes))

        for con in self.__velocity_connectivity:
            self.velocity_element._C1n2(self.__nodes, con, v1, v1, K1v1, K2v1)
            self.velocity_element._C1n2(self.__nodes, con, v2, v2, K1v2, K2v2)

        K1v1 *= self.rho;  K2v1 *= self.rho
        K1v2 *= self.rho;  K2v2 *= self.rho

        dR1dv1 = C + K1v1 + 2*self.S11 + self.S22
        dR1dv2 =     K2v1 + self.S12        
        dR2dv1 =     K1v2 + self.S12.T        
        dR2dv2 = C + K2v2 + self.S11 + 2*self.S22
        
        
        return bmat([[dR1dv1,       dR1dv2,     -self.Q1],
                     [dR2dv1,       dR2dv2,     -self.Q2],
                     [-self.Q1.T,   -self.Q2.T, np.zeros((self.__N_pres_nodes, self.__N_pres_nodes))]], 
                     format='csr')


    def steadystate_RnJ(self, u_prev, u_current)->Tuple[np.ndarray, csr_matrix]:
        return self.residual(u_prev, u_current), self.Jacobian(u_prev, u_current)
    
    def _NewtonRaphson(self, t_eval, u_prev, 
                       residual:Callable, jacobian:Callable,
                       tol = 1e-8, max_iter = 25,
                       line_search = None, relaxation_parameter = 0,
                       verbose = True, run_checks = False):
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
                    update_rule = lambda u_prev, u, du, res_norm: self.apply_backtracking(u_prev, u, du, res_norm, self.residual, relaxation_parameter=relaxation_parameter)[0]
                case _: raise ValueError("'line_search' must be on eof {'armijo'}, currently {}".format(line_search))
        elif line_search is None:
            update_rule = lambda u_prev, u, du, res_norm: u + (1-relaxation_parameter)*du 
        else:
            raise ValueError("Unrecognized 'line_search' algorithim")
        
        reduce_dim, fixed_dict, fixed_idx, free_idx = self._apply_dirichlet(t_eval)
        # enforce fixed DOFs exactly on previous solution
        for dof, pres in fixed_dict.items():
            u_prev[dof] = pres
        
        u_next = np.copy(u_prev)
        
        Res = residual(u_prev, u_next)
        Jac = jacobian(u_prev, u_next)
        for k in range(max_iter):
            if reduce_dim:
                # Partition vectors/matrices
                # delta_c (fixed) = prescribed - current
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
            
            
            if run_checks:
                j_res= self.fd_jacobian_check(residual, jacobian, u_prev, u_next)
                # j_res= self.fd_jacobian_check(residual, jacobian, u_prev, u_next, free_idx=free_idx)
                print(f"\t Jacobian relative error = {j_res:.4e}")

            res_norm = np.linalg.norm(R_f)
            u_next = update_rule(u_prev, u_next, delta, res_norm)

            # build residual and jacobian at updated iterate
            Res = residual(u_prev, u_next)
            Jac = jacobian(u_prev, u_next)
            res_norm = np.linalg.norm(Res[free_idx])
                
            # enforce fixed DOFs exactly (remove roundoff)
            for dof, pres in fixed_dict.items():
                u_next[dof] = pres

            # convergence check
            du_norm = np.linalg.norm(delta[free_idx])
            if verbose:
                print(f"Iteration {k: <{len(str(max_iter))}}: ||R||={res_norm:.3e}, ||du||={du_norm:.3e}, fixed_dofs={len(fixed_idx)}, free_dofs = {len(free_idx)}")
            if res_norm < tol and du_norm < tol:
                if verbose:
                    print()
                
                break
        else:
            # if we exit loop not converged:
            warnings.warn("Newton-Raphson failed to converge after max_iter")
        
            
        # plt.scatter(free_idx, abs(R_f), marker ='.')
        # plt.gca().set_yscale('log')
        # plt.axvline(self.vdof, color = 'r')
        # plt.axvline(2*self.vdof, color = 'r')
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
    
    def fd_jacobian_check(self, residual, jacobian, u_prev, u, free_idx = None, eps=1e-6):
        if free_idx is None:
            free_idx = np.ones_like(u, dtype=bool)
        Fu = residual(u_prev, u)[free_idx]
        J = jacobian(u_prev, u)[free_idx][:,free_idx]
        v = np.random.randn(u[free_idx].size)
        v /= np.linalg.norm(v)
        Jv = J.dot(v)
        _u = np.copy(u)
        _u[free_idx] += eps*v
        FD = (residual(u_prev, _u)[free_idx] - Fu) / eps
        res = np.linalg.norm(Jv - FD) / (np.linalg.norm(Jv) + 1e-16)
        if res < eps or np.isclose(res, eps):
            return res
        else:
            raise RuntimeError(f"Finite Difference Jacobian Check Failed. eps ({eps}) != residual ({res})")

    

    #####################################################################
    # CONTINUATION METHOD NON-LINEAR SOLVER
    def _continuation_method(self, t_eval, u_prev,
                             tol_picard = 1e-1, max_iter_picard = 25,
                             tol_newton = 1e-8, max_iter_newton = 10,
                             line_search = None, relaxation_parameter = 0,
                             verbose = True, run_checks = False):
        uSol = self._picards_iteration(0.0, u_prev, tol = tol_picard, max_iter=max_iter_picard, relaxation_parameter=relaxation_parameter, verbose=verbose)
        uSol = self._NewtonRaphson(0.0, uSol, self.residual, self.Jacobian, tol=tol_newton, max_iter=max_iter_newton, line_search=line_search, relaxation_parameter=relaxation_parameter, verbose=verbose,run_checks=run_checks)
        return uSol
    
    #####################################################################
    # AUXILIARY FUNCTIONS
    def plot_mesh(self, ax = None, linewidth = 0.6, color = 'k', plot_nodes = True, node_color = 'k', node_size = 6, **kwargs):
        if ax is None:
            ax = plt.gca()  
        
        if plot_nodes:
            ax.plot(*self.p2_nodes.T, '.', color = node_color, ms = node_size)
            ax.plot(*self.p1_nodes.T, 'o', markerfacecolor = 'none', color = node_color, ms = node_size*1.25)
        


        if self.velocity_element.n == 9:
            idx = [0, 4, 1, 5, 2, 6, 3, 7, 0]
            for e, con in enumerate(self.__velocity_connectivity):                
                temp = self.__nodes[con[idx]].T
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
            
    def save(self, prepend = None, append=None, directory = None, append_time = False):
        if prepend is None:
            filename = self.simulation_name
        else:
            filename = prepend + "_" + self.simulation_name

        if append:
            filename += str(append)
            
        if append_time:
            filename += "_" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")

        # Ensure extension
        if not filename.endswith(".h5"):
            filename += ".h5"
        
        # Determine directory
        if directory is None:
            directory = Path.cwd() / "solution"
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
            for name, array in zip(['vx', 'vy', 'p', 'p2_nodes', 'p1_nodes', 'connectivity'], [*self.get_solution(),  self.p2_nodes, self.p1_nodes, self.__velocity_connectivity]):
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
            for name, value in zip(['Ne', 'vdof', 'pdof'], [self.__Nele, self.vdof, self.pdof]):
                scal_grp.create_dataset(name, data=value)

            if hasattr(self, 'H1_norm'): scal_grp.create_dataset('H1_norm', data=self.H1_norm)
            if hasattr(self, 'H1_seminorm'): scal_grp.create_dataset('H1_seminorm', data=self.H1_seminorm)
            if hasattr(self, 'L2_velocity_norm'): scal_grp.create_dataset('L2_velocity_norm', data=self.L2_velocity_norm)
            if hasattr(self, 'L2_pressure_norm'):scal_grp.create_dataset('L2_pressure_norm', data=self.L2_pressure_norm)
            
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
    
    def get_solution(self):
        return self._get_components(self.solution)

    def _get_components(self, u):
        return u[:self.vdof], u[self.vdof:-self.pdof], u[-self.pdof:]

    def error_analysis(self, vx_analytical:Callable, vy_analytical:Callable, gradv_analytical:Callable, p_analytical:Callable):
        vx, vy, p = self.get_solution()

        L2_pressure_norm = 0
        L2_velocity_norm = 0
        H1_seminorm = 0
        r = 4

        
        for con in self.__velocity_connectivity:
            # plt.figure()
            # self.plot_mesh()
            # for i,e in enumerate(con):
            #     plt.text(*self.__nodes[e,:].T, "{} ({})".format(i,e))
            # plt.plot(*self.__nodes[con[:4],:].T, 'sr')
            # plt.plot(*self.__nodes[con[4:-1],:].T, 'ob')
            # plt.plot(*self.__nodes[con[-1],:].T, '^g')
            # plt.show()
            # Gauss-Legendre Quadrature
            for (xi, eta), wi in zip(*self.velocity_element.quadrature_points(r)):
                psi_hat         = self.velocity_element.basis_functions(xi, eta)
                grad_psi_hat    = self.velocity_element.grad_basis_functions(xi, eta)
                jac             = self.velocity_element.jacobian(self.__nodes[con], xi, eta)
                detJ = np.linalg.det(jac)
                invJ_T = np.array([[jac[1,1], -jac[1,0]],
                                   [-jac[0,1], jac[0,0]]])*(1/detJ)
                grad_psi = grad_psi_hat@invJ_T # Map grad of shape function back to physical coordinates
                
                coord = psi_hat @ self.__nodes[con,:] # Physical coordinate sof quadrature point
                
                # L2 terms
                vx_q = np.dot(vx[con], psi_hat)
                vy_q = np.dot(vy[con], psi_hat)
                L2_velocity_norm += ((vx_q - vx_analytical(*coord))**2 + (vy_q - vy_analytical(*coord))**2)*detJ*wi

                # H1 seminorm
                gradv = [vx[con], vy[con]]@grad_psi
                H1_seminorm += np.linalg.norm(gradv - gradv_analytical(*coord), ord = 'fro')**2*detJ*wi

        for con in self.__pressure_connectivity:
            # Gauss-Legendre Quadrature
            for (xi, eta), wi in zip(*self.pressure_element.quadrature_points(r)):
                psi_hat      = self.pressure_element.basis_functions(xi, eta)
                jac          = self.pressure_element.jacobian(self.p1_nodes[con,:], xi, eta)
                detJ = np.linalg.det(jac)
                
                # L2 terms
                coord = psi_hat @ self.p1_nodes[con,:] # Physical coordinate sof quadrature point
                pq = np.dot(psi_hat, p[con])
                L2_pressure_norm += (pq - p_analytical(*coord))**2*detJ*wi

        self.L2_velocity_norm = np.sqrt(L2_velocity_norm)
        self.H1_seminorm      = np.sqrt(H1_seminorm)
        self.H1_norm          = np.sqrt(L2_velocity_norm + H1_seminorm)
        self.L2_pressure_norm = np.sqrt(L2_pressure_norm)
        return self.L2_velocity_norm, self.H1_seminorm, self.H1_norm, self.L2_pressure_norm



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
    def p2_nodes(self):
        """Nodes"""
        return self.__nodes

    @property
    def p1_nodes(self):
        """Nodes"""
        return self.__nodes[list(self.vel_2_pres_mapping.keys()),:]
        
        
    @property
    def connectivity(self):
        """connectivity"""
        return self.__velocity_connectivity

    
    def vx_dof(self, node_idx: Union[int, Iterable]) -> int:
        if isinstance(node_idx, Iterable):
            return [int(_) for _ in node_idx]
        else:
            return int(node_idx)
    
    def vy_dof(self, node_idx: Union[int, Iterable]) -> int:
        if isinstance(node_idx, Iterable):
            return [int(self.__N_vel_nodes + _) for _ in node_idx]
        else:
            return int(self.__N_vel_nodes + node_idx)
    
    def p_dof(self, node_idx: Union[int, Iterable]) -> int:
        if isinstance(node_idx, Iterable):
            return [int(2*self.__N_vel_nodes + _) for _ in node_idx]
        else:
            return int(2 * self.__N_vel_nodes + node_idx)


def _process_solver_parameter_dict(func, options:dict):
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty and name not in options:
            options[name] = param.default
        




