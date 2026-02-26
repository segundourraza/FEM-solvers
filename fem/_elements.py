from abc import ABC, abstractmethod
from ._quadrature import triangle_quadrature
import numpy as np

############################################################################
# BASIS FUNCTIONS
linear_basis_functions = [lambda xi: 0.5*(1-xi),
                          lambda xi: 0.5*(1+xi)]

linear_grad_basis_functions = [lambda xi: -0.5,
                               lambda xi: 0.5]

quad_basis_function = [lambda x: -0.5*x*(1-x), 
                       lambda x: 1 - x*x, 
                       lambda x: 0.5*x*(1+x)]

quad_grad_basis_function = [lambda x: -0.5*(1-2*x), 
                            lambda x: -2*x, 
                            lambda x: 0.5*(1+2*x)]



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
        return np.array([_(xi) for _ in linear_basis_functions])
    
    @staticmethod
    def grad_basis_functions(xi):
        return np.array([_(xi) for _ in linear_grad_basis_functions])
    

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
        return np.array([_(xi) for _ in quad_basis_function])
    
    @staticmethod
    def grad_basis_functions(xi):
        return np.array([_(xi) for _ in quad_grad_basis_function])
    

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

    r_Ne = None
    r_mass = None

    _A = None
    _BpC = None
    _D = None
    _M = None
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.n is None:
            raise TypeError("Subclasses must define 'n'")
        
        if cls.degree is None:
            raise TypeError("Subclasses must define 'degree'")
        
        if cls._A is None:
            raise TypeError("Subclasses must define '__A'")
        
        if cls._BpC is None:
            raise TypeError("Subclasses must define '__BpC'")
        
        if cls._D is None:
            raise TypeError("Subclasses must define '__D'")
        
        if cls._M is None:
            raise TypeError("Subclasses must define '__M'")
    
        if cls.r_Ne is None:
            raise TypeError("Subclasses must define 'r_Ne'")
            
        if cls.r_mass is None:
            raise TypeError("Subclasses must define 'r_mass'")
    
    @staticmethod
    @abstractmethod
    def basis_functions(xi, eta): ...
        
    @staticmethod
    @abstractmethod
    def grad_basis_function(xi, eta): ...

    @staticmethod
    @abstractmethod
    def quadrature_points(n_points): ...
    
    @abstractmethod
    def compute_ele_properties(self,nodes):...

    

    def Me(self, M_global, con, detJ):
        M_global[np.ix_(con,con)] += detJ*self._M
    
    def Ke(self, K_global, con,detJ, invJ):
        Se = (invJ[0,0]**2)*self._A +(invJ[0,0]*invJ[0,1])*self._BpC + (invJ[0,1]**2)*self._D
        Sn = (invJ[1,0]**2)*self._A +(invJ[1,0]*invJ[1,1])*self._BpC + (invJ[1,1]**2)*self._D
        K_global[np.ix_(con,con)] += detJ*(Se + Sn)

    def Ne(self, N, con, detJ, Ce):
        for (xi,eta), wi in zip(*self.quadrature_points(self.r_Ne)):
            phi = self.basis_functions(xi,eta)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            N[con] += (ch3 - ch)*phi*detJ*wi
    
    ################################################################
    # TIME STEPPING SPECIFIC MATRICES

    def b2_b(self, b_global, con, detJ, Ce):
        for (xi, eta), wi in zip(*self.quadrature_points(self.r_Ne)):
            phi = self.basis_functions(xi, eta)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            b_global[con] += (ch3 - 3*ch)*phi*(detJ)*wi

    def _c3(self, b_global, con, detJ, Ce):
        for (xi, eta), wi in zip(*self.quadrature_points(self.r_Ne)):
            phi = self.basis_functions(xi, eta)
            ch3 = np.dot(Ce, phi)**3
            b_global[con] += ch3*phi*(detJ)*wi

    def compute_energy(self, detJ, invJ, Ce, eps):
        E = 0
        for (xi, eta), wi in zip(*self.quadrature_points(self.r_Ne)):
            phi = self.basis_functions(xi, eta)
            ch2 = np.dot(Ce, phi)**2

            grad_phi = self.grad_basis_function(xi, eta)
        
            grad_c = ((invJ@grad_phi.T)@Ce)
            grad_c2 = np.dot(grad_c, grad_c)
            
            E += (1/(4*eps)*(1- ch2**2)**2 + eps/2*grad_c2)*(detJ)*wi
        return E
    
    def compute_mass(self, detJ, Ce):
        dM = 0
        for (xi, eta), wi in zip(*self.quadrature_points(self.r_mass)):
            phi = self.basis_functions(xi, eta)
            ch = np.dot(Ce, phi)
            dM += ch*(detJ)*wi
        return dM
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
    def grad_basis_function(xi, eta):
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
        invJ = 1/detJ*np.array([[dy31, -dx31],
                                [-dy21, dx21]])
        return detJ, invJ
    
    
class LinearRectElement(_LegendreElement2D):

    n:int = 4
    degree:int = 1
    
    # Quadrature points
    r_Ne:int = 3
    r_mass: int = 1

    _M = (1/9)*np.array([[4, 2, 2, 1],
                         [2, 4, 1, 2],
                         [2, 1, 4, 2],
                         [1, 2, 2, 4]], dtype=float)
    
    _A   = 1/6*np.array([[ 2, -2, -1,  1],
                         [-2,  2,  1, -1],
                         [-1,  1,  2, -2],
                         [ 1, -1, -2,  2]], dtype=float)

    _D = 1/6*np.array([[ 2,  1,  -1, -2],
                       [ 1,  2,  -2, -1],
                       [-1, -2,  2,  1],
                       [-2, -1,  1,  2]], dtype=float)

    _BpC = 1/2*np.array([[ 1,  0, -1,  0],
                         [ 0, -1,  0,  1],
                         [-1,  0,  1,  0],
                         [ 0,  1,  0, -1]], dtype=float)
   
    @staticmethod
    def basis_functions(xi, eta):
        return 0.25*np.array([(1 - xi)*(1-eta),
                              (1 + xi)*(1-eta),
                              (1 + xi)*(1+eta),
                              (1 - xi)*(1+eta)], dtype = float)
    
    @staticmethod
    def grad_basis_function(xi, eta):
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
        J = nodes[:,:2].T@self.grad_basis_function(-1,-1)
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
    r_Ne:int = 5
    r_mass: int = 3

    
    _M = (1/225)*np.array([[16, -4, 1, -4, 8, -2, -2, 8, 4],
                           [-4, 16, -4, 1, 8, 8, -2, -2, 4],
                           [1, -4, 16, -4, -2, 8, 8, -2, 4],
                           [-4, 1, -4, 16, -2, -2, 8, 8, 4],
                           [8, 8, -2, -2, 64, 4, -16, 4, 32],
                           [-2, 8, 8, -2, 4, 64, 4, -16, 32],
                           [-2, -2, 8, 8, -16, 4, 64, 4, 32],
                           [8, -2, -2, 8, 4, -16, 4, 64, 32],
                           [4, 4, 4, 4, 32, 32, 32, 32, 256]], dtype = float)

    _A = 1/90 * np.array([[28, 4, -1, -7, -32, 2, 8, 14, -16],
                          [4, 28, -7, -1, -32, 14, 8, 2, -16],
                          [-1, -7, 28, 4, 8, 14, -32, 2, -16],
                          [-7, -1, 4, 28, 8, 2, -32, 14, -16],
                          [-32, -32, 8, 8, 64, -16, -16, -16, 32],
                          [2, 14, 14, 2, -16, 112, -16, 16, -128],
                          [8, 8, -32, -32, -16, -16, 64, -16, 32],
                          [14, 2, 2, 14, -16, 16, -16, 112, -128],
                          [-16, -16, -16, -16, 32, -128, 32, -128, 256]], dtype=float)

    _D   = 1/90*np.array([[28, -7, -1, 4, 14, 8, 2, -32, -16],
                          [-7, 28, 4, -1, 14, -32, 2, 8, -16],
                          [-1, 4, 28, -7, 2, -32, 14, 8, -16],
                          [4, -1, -7, 28, 2, 8, 14, -32, -16],
                          [14, 14, 2, 2, 112, -16, 16, -16, -128],
                          [8, -32, -32, 8, -16, 64, -16, -16, 32],
                          [2, 2, 14, 14, 16, -16, 112, -16, -128],
                          [-32, 8, 8, -32, -16, -16, -16, 64, 32],
                          [-16, -16, -16, -16, -128, 32, -128, 32, 256]], dtype=float)
    
    _BpC = 1/18 * np.array([[9, 0, -1, 0, 0, 4, 4, 0, -16],
                            [0, -9, 0, 1, 0, 0, -4, -4, 16],
                            [-1, 0, 9, 0, 4, 0, 0, 4, -16],
                            [0, 1, 0, -9, -4, -4, 0, 0, 16],
                            [0, 0, 4, -4, 0, -16, 0, 16, 0],
                            [4, 0, 0, -4, -16, 0, 16, 0, 0],
                            [4, -4, 0, 0, 0, 16, 0, -16, 0],
                            [0, -4, 4, 0, 16, 0, -16, 0, 0],
                            [-16, 16, -16, 16, 0, 0, 0, 0, 0]], dtype= float)
    
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
    def grad_basis_function(x, y):
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
    
    def compute_ele_properties(self, nodes):
        J = nodes[:,:2].T@self.grad_basis_function(-1,-1)
        a = nodes[1,0] - nodes[0,0] # width
        b = nodes[2,1] - nodes[1,1] # height
        if np.isclose(2*J[0,0], a) and np.isclose(2*J[1,1],b):
            return a*b/4, np.diag([2/a, 2/b])
        else:
            raise ValueError("Rectangular element is not aligned with axis")
    