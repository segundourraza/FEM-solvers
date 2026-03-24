from abc import ABC, abstractmethod
import numpy as np

from ._quadrature import triangle_quadrature
from ._utils import NonConstantJacobian

############################################################################
# BASIS FUNCTIONS
def linear_basis_functions(xi):
    return np.array([0.5*(1-xi),
                     0.5*(1+xi)])

def linear_grad_basis_functions(xi):
    return np.array([-0.5,
                     0.5])

def quad_basis_function(xi):
    return np.array([-0.5*xi*(1-xi), 
                     1 - xi*xi, 
                     0.5*xi*(1+xi)])

def quad_grad_basis_function(xi):
    return np.array([-0.5*(1-2*xi), 
                     -2*xi, 
                     0.5*(1+2*xi)])



class _LegendreElement(ABC):

    def __init__(self):
        super().__init__()
        self.r_mass = int(np.ceil(0.5*(self.degree+1)))

    @property
    @abstractmethod
    def degree(self)->int:
        """Degree of polynomial"""
        pass

    @property
    def n(self)->int:
        """Number of node in element"""
        return self.degree + 1
    
    @property
    @abstractmethod
    def r_Ne(self)->int:
        """Quadrature points used for Ne"""
        pass

    @property
    @abstractmethod
    def r_J(self)->int:
        """Quadrature points used for Jacobian Nonlinear block"""
        pass

    
    @staticmethod
    @abstractmethod
    def basis_functions(xi):
        """Basis functions"""
        pass
    
    @staticmethod
    @abstractmethod
    def grad_basis_functions(xi):
        """Basis functions"""
        pass
    
    @staticmethod
    @abstractmethod
    def Me(he):
        """Mass Matrix"""
        pass
    
    @staticmethod
    @abstractmethod
    def Ke(he):
        """Stiffness Matrix"""
        pass
    

    def Ne(self, N, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            for i in range(self.n):
                N[i] += (ch3 - ch)*phi[i]*(he/2)*wi
    
    def He(self, H, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch2 = np.dot(Ce, phi)**2
            H[:] += np.outer(phi, phi)*(3*ch2)*(he/2)*wi

    #################################
    # TIME STEPPING
    def b2_b(self, b2, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            for i in range(self.n):
                b2[i] += (ch3 - 3*ch)*phi[i]*(he/2)*wi


    def _c1(self, b, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            b += ch*phi*(he/2)*wi

    def _c3(self, b, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch3 = np.dot(Ce, phi)**3
            b += ch3*phi*(he/2)*wi
    
    
    def Awc_c(self, H, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch2 = np.dot(Ce, phi)**2
            H[:] += 3*ch2*np.outer(phi, phi)*(he/2)*wi

    
    #################################
    # AUXILIARY 
    
    def compute_J_e(self, eps, he, Ce):
        J = 0
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_J)):
            phi = self.basis_functions(xi)
            grad_phi = self.grad_basis_functions(xi)
            ch = np.dot(Ce,phi)
            grad_ch = np.dot(Ce, grad_phi)
            J += wi*((1 - ch**2)**2/(4*eps)  + eps/2 * (grad_ch)**2)
        return J*he/2

    def compute_mass_e(self, he, Ce):
        M = 0
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_mass)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            M += wi*(he/2)*ch
        return M
    
class LinearLegendreElement(_LegendreElement):

    degree: int = 1
    
    # Quadrature points
    r_Ne:int = 3
    r_J: int = 3


    @staticmethod
    def basis_functions(xi):
        return linear_basis_functions(xi)
    
    @staticmethod
    def grad_basis_functions(xi):
        return linear_grad_basis_functions(xi)
    

    ##################################################
    # FINITE ELEMENT DISCRETIZATION
    @staticmethod
    def Me(M, he):
        M[:] += he/6*np.array([[2,1],[1,2]], dtype=float)
    
    @staticmethod
    def Ke(K, he):
        K[:] += 1/he * np.array([[1, -1], [-1, 1]], dtype=float)

    ############################################
    # AUXILIARY
    def compute_mass_e(self, he, Ce):
        return he/2*(Ce[0] + Ce[1])

class QuadraticLegendreElement(_LegendreElement):

    degree: int = 2
    
    # Quadrature points
    r_Ne:int = 5
    r_J: int = 5


    @staticmethod
    def basis_functions(xi):
        return quad_basis_function(xi)
    
    @staticmethod
    def grad_basis_functions(xi):
        return quad_grad_basis_function(xi)

    ##################################################
    # FINITE ELEMENT DISCRETIZATION
    @staticmethod
    def Me(M, he):
        M[:] += he/30*np.array([[ 4, 2, -1],
                                [ 2, 16, 2],
                                [-1, 2,  4]], dtype=float)
    
    @staticmethod
    def Ke(K, he):
        K[:] += 1/(3*he)*np.array([[7,  -8,  1],
                                   [-8, 16, -8],
                                   [1,  -8,  7]], dtype = float)



class _LegendreElement2D(ABC):

    n = None
    degree = None

    r_viscous = None
    r_convective = None

    @staticmethod
    @abstractmethod
    def edge_basis_function(xi)->np.ndarray: ...
    
    @staticmethod
    @abstractmethod
    def edge_grad_basis_function(xi)->np.ndarray: ...

    @abstractmethod
    def edge_properties(self,xi:float,edge_nodes)->np.ndarray: ...


    @staticmethod
    @abstractmethod
    def basis_functions(xi, eta)->np.ndarray: ...
        
    @staticmethod
    @abstractmethod
    def grad_basis_functions(xi, eta)->np.ndarray: ...

    @staticmethod
    @abstractmethod
    def quadrature_points(n_points): ...
    
    @abstractmethod
    def compute_ele_properties(self,nodes):...
    
    def jacobian(self, nodes, xi,eta)->np.ndarray:
        return nodes.T@self.grad_basis_functions(xi, eta)
    
    def detJ(self, nodes, xi, eta):
        return np.linalg.det(self.jacobian(nodes, xi, eta))
   
    def Se(self, nodes, con, S11, S22, S12):
        for (xi,eta), wi in zip(*self.quadrature_points(self.r_viscous)):
            grad_psi_hat = self.grad_basis_functions(xi,eta)
            jac = self.jacobian(nodes[con], xi, eta)
            detJ = np.linalg.det(jac)
            invJ_T = np.array([[jac[1,1], -jac[1,0]],
                             [-jac[0,1], jac[0,0]]])*(1/detJ)
            grad_psi = grad_psi_hat@invJ_T # Map grad of shape function back to physical coordinates
            S11[np.ix_(con,con)] += np.outer(grad_psi[:,0], grad_psi[:,0])*detJ*wi
            S22[np.ix_(con,con)] += np.outer(grad_psi[:,1], grad_psi[:,1])*detJ*wi
            S12[np.ix_(con,con)] += np.outer(grad_psi[:,0], grad_psi[:,1])*detJ*wi

    def _C(self, nodes, con, C_global, ve_x, ve_y):
        Ve = np.vstack([ve_x,ve_y])
        for (xi, eta), wi in zip(*self.quadrature_points(self.r_convective)):
            psi_hat = self.basis_functions(xi, eta)
            grad_psi_hat = self.grad_basis_functions(xi, eta)
            jac = self.jacobian(nodes[con], xi, eta)
            detJ = np.linalg.det(jac)
            invJ_T = np.array([[jac[1,1], -jac[1,0]],
                             [-jac[0,1], jac[0,0]]])*(1/detJ)
            grad_psi = grad_psi_hat@invJ_T # Map grad of shape function back to physical coordinates
            
            Vh = np.dot(Ve, psi_hat)
            C_global[np.ix_(con,con)] += np.outer(psi_hat, np.dot(grad_psi, Vh))*detJ*wi
    
    def _C1n2(self, nodes:np.ndarray, con:np.ndarray, 
               ve_x, ve_y,
               C1:np.ndarray, C2:np.ndarray):
        for (xi, eta), wi in zip(*self.quadrature_points(self.r_convective)):
            psi_hat = self.basis_functions(xi, eta)
            grad_psi_hat = self.grad_basis_functions(xi, eta)
            jac = self.jacobian(nodes[con], xi, eta)
            detJ = np.linalg.det(jac)
            invJ_T = np.array([[jac[1,1], -jac[1,0]],
                             [-jac[0,1], jac[0,0]]])*(1/detJ)
            grad_psi = grad_psi_hat@invJ_T # Map grad of shape function back to physical coordinates
            
            dvdx = grad_psi[:, 0] @ ve_x[con]
            dvdy = grad_psi[:, 1] @ ve_y[con]

            C1[np.ix_(con, con)] += np.outer(psi_hat, psi_hat) * dvdx * detJ * wi
            C2[np.ix_(con, con)] += np.outer(psi_hat, psi_hat) * dvdy * detJ * wi

class LinearTriangularElement(_LegendreElement2D):
    
    n = 3 # Number of nodes in element
    degree = 1

    # Quadrature points
    r_Ne:int = 6
    r_mass: int = 1

    #############################################
    # BASIC ELEMENT MATRICES
    
    _M = (1/24)*np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]], dtype=float)
    
    _A = 0.5*np.array([[1, -1, 0],
                        [-1, 1, 0],
                        [0, 0, 0]], dtype=float)
    
    _BpC = 0.5*np.array([[2, -1, -1],
                          [-1, 0, 1],
                          [-1, 1, 0]], dtype=float)
    
    _D = 0.5*np.array([[1, 0, -1],
                        [0, 0, 0],
                        [-1, 0, 1]], dtype=float)
    

    @staticmethod
    def quadrature_points(n_points):
        return triangle_quadrature(n_points)
    

    @staticmethod
    def basis_functions(xi, eta):
        return np.array([1 - xi - eta, xi, eta], dtype =float)
    
    @staticmethod
    def grad_basis_functions(xi, eta):
        return np.array([[-1, -1],
                         [1, 0],
                         [0, 1]], dtype=float)
    
    def compute_ele_properties(self, nodes):
        x1, x2, x3 = nodes[:,0]
        y1, y2, y3 = nodes[:,1]

        dx31 = x3 - x1
        dx21 = x2 - x1
        dy21 = y2 - y1
        dy31 = y3 - y1
        
        detJ = dy31*dx21 - dx31*dy21
        invJ_T = 1/detJ*np.array([[dy31, -dx31],
                                [-dy21, dx21]])
        return detJ, invJ_T
    
    
class LinearRectElement(_LegendreElement2D):

    n:int = 4
    degree:int = 1
    
    # Quadrature points
    r_viscous: int = 2
    r_convective: int = 2

    @staticmethod
    def edge_basis_function(xi):
        return linear_basis_functions(xi)
    
    @staticmethod
    def edge_grad_basis_function(xi):
        return linear_grad_basis_functions(xi)
    
    def edge_properties(self, xi, nodes):
        raise NotImplementedError()
    
    @staticmethod
    def basis_functions(xi, eta):
        return 0.25*np.array([(1 - xi)*(1-eta),
                              (1 + xi)*(1-eta),
                              (1 + xi)*(1+eta),
                              (1 - xi)*(1+eta)], dtype = float)
    
    @staticmethod
    def grad_basis_functions(xi, eta):
        return 0.25*np.array([[-1 + eta, -1 + xi],
                              [ 1 - eta, -1 - xi],
                              [ 1 + eta,  1 + xi],
                              [-1 - eta,  1 - xi]
                              ])

    @staticmethod
    def quadrature_points(n_points):
        x, w = np.polynomial.legendre.leggauss(n_points)
        X, Y = np.meshgrid(x, x)
        return np.vstack([X.ravel(), Y.ravel()]).T, np.outer(w,w).ravel()
    
    def compute_ele_properties(self, nodes):
        J = nodes[:,:2].T@self.grad_basis_functions(-1,-1)
        a = nodes[1,0] - nodes[0,0] # width
        b = nodes[2,1] - nodes[1,1] # height
        if J[0,0] == a/2 and J[1,1] == b/2:
            return a*b/4, np.diag([2/a, 2/b])
        else:
            raise ValueError("Rectangular element is not aligned with axis")
            
class QuadraticRectElement(_LegendreElement2D):

    n:int = 9
    degree:int = 2
    
    # Quadrature points
    r_viscous: int    = 3
    r_convective: int = 4

    @staticmethod
    def edge_basis_function(xi):
        return quad_basis_function(xi)
    
    @staticmethod
    def edge_grad_basis_function(xi):
        return quad_grad_basis_function(xi)

    def edge_properties(self, xi, nodes):
        
        psi      = self.edge_basis_function(xi)
        psi_grad = self.edge_grad_basis_function(xi)
        
        coords = psi @ nodes
        t = psi_grad @ nodes # tangent vector

        detJ = np.linalg.norm(t)
        t_unit = t / detJ

        n = np.array([t_unit[1], -t_unit[0]])

        return coords, n, detJ




    @staticmethod
    def basis_functions(xi, eta):
        return np.array([0.25*(xi**2 - xi)*(eta**2 - eta),
                         0.25*(xi**2 + xi)*(eta**2 - eta),
                         0.25*(xi**2 + xi)*(eta**2 + eta),
                         0.25*(xi**2 - xi)*(eta**2 + eta),
                         0.5*(1 - xi**2)*(eta**2 - eta),
                         0.5*(xi**2 + xi)*(1 - eta**2),
                         0.5*(1 - xi**2)*(eta**2 + eta),
                         0.5*(xi**2 - xi)*(1 - eta**2),
                         (1 - xi**2)*(1-eta**2)], dtype = float)
    
    
    @staticmethod
    def grad_basis_functions(x, y):
        return np.array([[1/4*(-1 + 2*x)*(-1 + y)*y, 1/4*x*(-1 + x)*(-1 + 2*y)],
                         [1/4*( 1 + 2*x)*(-1 + y)*y, 1/4*x*(1 + x)*(-1 + 2*y)],
                         [1/4*( 1 + 2*x)*(1 + y)*y,  1/4*x*(1 + x)*(1 + 2*y)],
                         [1/4*(-1 + 2*x)*(1 + y)*y,  1/4*x*(-1 + x)*(1 + 2*y)],
                         
                         [-x*(-1 + y)*y, -(1/2)*(-1 + x**2)*(-1 + 2*y)],
                         [-(1/2)*(1 + 2*x)*(-1 + y**2), -x*(1 + x)*y],
                         [-x*y*(1 + y), -(1/2)*(-1 + x**2)*(1 + 2*y)],
                         [-(1/2)*(-1 + 2*x)*(-1 + y**2), -(-1 + x)*x*y],
                         
                         [2*x*(-1 + y**2), 2*(-1 + x**2)*y]])

    @staticmethod
    def quadrature_points(n_points):
        x, w = np.polynomial.legendre.leggauss(n_points)
        X, Y = np.meshgrid(x, x)
        return np.vstack([X.ravel(), Y.ravel()]).T, np.outer(w,w).ravel()
    

    @staticmethod
    def edge_quadrature_points(n_points):
        return np.polynomial.legendre.leggauss(n_points)
    

    def compute_ele_properties(self,e, nodes):
        J = nodes[:,:2].T@self.grad_basis_functions(-1,-1)
        a = nodes[1,0] - nodes[0,0] # width
        b = nodes[2,1] - nodes[1,1] # height
        if np.isclose(2*J[0,0], a) and np.isclose(2*J[1,1],b):
            return a*b/4, np.diag([2/a, 2/b])
        else:
            raise NonConstantJacobian(e)
    