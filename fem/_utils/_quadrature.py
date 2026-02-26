import numpy as np

def quadrature(f, n):
    I = 0
    for xi, wi in zip(*np.polynomial.legendre.leggauss(n)):
        I += wi*f(xi)
    return I

def triangle_quadrature(npts):
    """
    Return quadrature points and weights on the reference triangle
    with vertices (0,0), (1,0), (0,1).
    
    Exact Polynomial Integral map
    1-point → degree 1
    3-point → degree 2
    4-point → degree 3
    6-point → degree 4
    7-point → degree 5
    
    Parameters
    ----------
    npts : int
        Number of quadrature points. Supported: 1, 3, 4, 6, 7.
    
    Returns
    -------
    points : ndarray, shape (npts, 2)
        Quadrature points in reference coordinates (xi, eta).
    weights : ndarray, shape (npts,)
        Quadrature weights (they sum to area = 1/2).
    
    Raises
    ------
    ValueError
        If npts is not one of the supported values.
    """
    if npts == 1:
        points = np.array([[1/3, 1/3]])
        weights = np.array([0.5])
    elif npts == 3:
        points = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3],
        ])
        weights = np.array([1/6, 1/6, 1/6])
    elif npts == 4:
        points = np.array([
            [1/3, 1/3],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2],
        ])
        weights = np.array([
            -27.0/96.0,
             25.0/96.0,
             25.0/96.0,
             25.0/96.0,
        ])
    elif npts == 6:
        points = np.array([
            [0.445948490915965, 0.445948490915965],
            [0.445948490915965, 0.108103018168070],
            [0.108103018168070, 0.445948490915965],
            [0.091576213509771, 0.091576213509771],
            [0.091576213509771, 0.816847572980459],
            [0.816847572980459, 0.091576213509771],
        ])
        weights = np.array([
            0.111690794839005,
            0.111690794839005,
            0.111690794839005,
            0.054975871827661,
            0.054975871827661,
            0.054975871827661,
        ])
    elif npts == 7:
        points = np.array([
            [1/3, 1/3],
            [0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770],
            [0.059715871789770, 0.470142064105115],
            [0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087],
            [0.797426985353087, 0.101286507323456],
        ])
        weights = np.array([
            0.1125,
            0.066197076394253,
            0.066197076394253,
            0.066197076394253,
            0.062969590272414,
            0.062969590272414,
            0.062969590272414,
        ])
    else:
        raise ValueError("Unsupported npts. Use one of: 1, 3, 4, 6, 7.")
    return points, weights