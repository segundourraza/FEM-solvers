import numpy as np


# ---------------------------------------------------------------------------
# Gauss quadrature on the reference segment [-1, 1]
# ---------------------------------------------------------------------------

def gauss_quadrature_1d(n_points: int):
    """
    Return Gauss-Legendre quadrature points and weights on [-1, 1].

    Parameters
    ----------
    n_points : int
        Number of quadrature points (1–5 sufficient for P2 face integrals).

    Returns
    -------
    xi : (n_points,) array   — reference coordinates in [-1, 1]
    w  : (n_points,) array   — weights (sum = 2)
    """
    xi, w = np.polynomial.legendre.leggauss(n_points)
    return xi, w


# ---------------------------------------------------------------------------
# 1-D P2 shape functions on the reference segment [-1, 1]
# Node layout:  0 -------- 2 -------- 1
#              xi=-1      xi=0      xi=+1
# ---------------------------------------------------------------------------

def p2_shape_1d(xi: float):
    """
    Quadratic (P2) Lagrange shape functions on [-1, 1].

    Nodes:  N0 at xi=-1,  N1 at xi=+1,  N2 at xi=0  (mid-node)

    Returns
    -------
    N  : (3,) array of shape function values
    """
    N = np.array([
        0.5 * xi * (xi - 1.0),   # N0, node at xi = -1
        0.5 * xi * (xi + 1.0),   # N1, node at xi = +1
        1.0 - xi**2               # N2, node at xi =  0  (mid-node)
    ])
    return N


def p1_shape_1d(xi: float):
    """
    Linear (P1) Lagrange shape functions on [-1, 1].

    Nodes:  M0 at xi=-1,  M1 at xi=+1

    Returns
    -------
    M  : (2,) array of shape function values
    """
    M = np.array([
        0.5 * (1.0 - xi),   # M0, node at xi = -1
        0.5 * (1.0 + xi),   # M1, node at xi = +1
    ])
    return M


# ---------------------------------------------------------------------------
# Geometric mapping: reference segment → physical edge
# ---------------------------------------------------------------------------

def edge_geometry(xi: float, x_nodes: np.ndarray):
    """
    Map reference coordinate xi ∈ [-1,1] to a physical 2D edge using
    a P2 (quadratic) mapping defined by 3 nodes.

    Parameters
    ----------
    xi      : float          — reference coordinate
    x_nodes : (3, 2) array  — physical coordinates of the 3 edge nodes
                              [endpoint0, endpoint1, midpoint]

    Returns
    -------
    x      : (2,) array  — physical point
    n      : (2,) array  — outward unit normal
    jac    : float       — Jacobian |dx/dxi|
    """
    N = p2_shape_1d(xi)

    # Physical position
    x = N @ x_nodes                          # (2,)

    # Tangent vector  t = dx/dxi
    dN_dxi = np.array([
        xi - 0.5,    # dN0/dxi
        xi + 0.5,    # dN1/dxi
        -2.0 * xi    # dN2/dxi
    ])
    t = dN_dxi @ x_nodes                     # (2,)

    jac = np.linalg.norm(t)
    t_unit = t / jac

    # Outward normal (rotate tangent 90° — convention: n points outward)
    # For a boundary traversed left-to-right when the domain is to the left:
    #   n = (t_y, -t_x)  [rotated clockwise]
    n = np.array([t_unit[1], -t_unit[0]])

    return x, n, jac


# ---------------------------------------------------------------------------
# Velocity gradient at a point — given solution DOFs
# ---------------------------------------------------------------------------

def velocity_gradient_at_point(
    xi: float,
    x_nodes: np.ndarray,
    u_nodes: np.ndarray,
    dN2d_dx: callable
):
    """
    Compute the velocity gradient tensor ∇u at a boundary quadrature point.

    NOTE: This requires the 2-D shape-function gradients evaluated at the
    physical point.  In a real solver you would pass in the precomputed
    gradients from the parent element.  Here we accept a callable
    `dN2d_dx(x)` that returns the (n_nodes, 2) gradient array for the
    parent element at physical point x.

    Parameters
    ----------
    xi       : float         — reference coord on the boundary edge
    x_nodes  : (3, 2) array — edge node coordinates (P2)
    u_nodes  : (n_vel, 2)   — velocity DOF values at ALL velocity nodes
                              of the parent element (P2, so n_vel = 6 in 2D)
    dN2d_dx  : callable      — dN2d_dx(x) → (n_vel, 2) shape-fn gradients

    Returns
    -------
    grad_u : (2, 2) array   — velocity gradient  [∂u_i/∂x_j]
    """
    N = p2_shape_1d(xi)
    x, _, _ = edge_geometry(xi, x_nodes)

    dN_dx = dN2d_dx(x)          # (n_vel, 2)  from the 2-D parent element

    # grad_u[i, j] = sum_a  u_a[i] * dN_a/dx_j
    grad_u = u_nodes.T @ dN_dx  # (2, n_vel) @ (n_vel, 2) → (2, 2)
    return grad_u


# ---------------------------------------------------------------------------
# Core function: traction boundary integral
# ---------------------------------------------------------------------------

def compute_traction_boundary_integral(
    x_edge_nodes: np.ndarray,
    u_edge_nodes: np.ndarray,
    p_edge_nodes: np.ndarray,
    grad_u_fn: callable,
    mu: float,
    n_gauss: int = 3,
    traction_fn=None
):
    """
    Compute the traction boundary integral for one edge of a 2D
    Taylor-Hood (P2/P1) Navier-Stokes element.

    The traction vector at each boundary point is:

        t = σ · n = (-p I + 2μ ε(u)) · n

    where  ε(u) = ½(∇u + ∇uᵀ).

    The contribution to the residual / RHS for velocity test functions is:

        f_i^bc = ∫_Γ  t · φ_i  dΓ       (i = 0,1,2  — P2 velocity nodes)

    Parameters
    ----------
    x_edge_nodes : (3, 2) ndarray
        Physical coordinates of the 3 edge nodes (P2 layout):
          [0] → endpoint 0,  [1] → endpoint 1,  [2] → midpoint.

    u_edge_nodes : (3, 2) ndarray
        Velocity values at the 3 edge nodes  (u_x, u_y) per row.
        Used only if `traction_fn` is None (computed from solution).

    p_edge_nodes : (2,) ndarray
        Pressure values at the 2 *endpoint* nodes (P1 nodes on the edge).
        Used only if `traction_fn` is None.

    grad_u_fn : callable  xi → (2,2) ndarray
        Function that returns the velocity gradient ∇u at reference
        coordinate xi on this edge.  Signature:
            grad_u = grad_u_fn(xi)
        Pass `None` if `traction_fn` is provided instead.

    mu : float
        Dynamic viscosity.

    n_gauss : int, optional
        Number of Gauss points (default 3, exact for P2×P2 integrands).

    traction_fn : callable or None, optional
        If provided, overrides the computed traction with a prescribed one.
        Signature:  t = traction_fn(x, n)  → (2,) ndarray.
        Use this for Neumann BCs (e.g. pressure outlet, known stress).

    Returns
    -------
    F_vel : (3, 2) ndarray
        Traction load vector contributions for the 3 P2 velocity nodes
        on this edge.  Add these into the global RHS at the corresponding
        velocity DOFs.

    traction_history : list of dict
        Quadrature-point data for debugging / post-processing:
        keys: 'x', 'n', 'p', 'grad_u', 'eps', 'sigma', 't', 'weight'.
    """

    xi_pts, w_pts = gauss_quadrature_1d(n_gauss)

    F_vel = np.zeros((3, 2))   # (3 P2 nodes) × (2 components)
    history = []

    for xi, w in zip(xi_pts, w_pts):

        # --- geometry ---
        x, n, jac = edge_geometry(xi, x_edge_nodes)

        # --- traction ---
        if traction_fn is not None:
            # Prescribed Neumann traction (e.g. do-nothing, pressure outlet)
            t = traction_fn(x, n)
            record = {'x': x, 'n': n, 't': t,
                      'p': None, 'grad_u': None, 'eps': None, 'sigma': None}
        else:
            # Compute traction from the current solution (u, p)

            # Pressure at quadrature point  (P1 interpolation, 2 endpoints)
            M = p1_shape_1d(xi)
            p_qp = M @ p_edge_nodes          # scalar

            # Velocity gradient at quadrature point
            grad_u = grad_u_fn(xi)           # (2, 2)

            # Strain-rate tensor  ε(u) = ½(∇u + ∇uᵀ)
            eps = 0.5 * (grad_u + grad_u.T)  # (2, 2)

            # Cauchy stress  σ = -p I + 2μ ε
            sigma = -p_qp * np.eye(2) + 2.0 * mu * eps   # (2, 2)

            # Traction vector  t = σ · n
            t = sigma @ n                    # (2,)

            record = {
                'x': x, 'n': n, 'p': p_qp,
                'grad_u': grad_u, 'eps': eps,
                'sigma': sigma, 't': t
            }

        record['weight'] = w * jac
        history.append(record)

        # --- shape functions at this quadrature point ---
        N = p2_shape_1d(xi)      # (3,) — one per P2 velocity node

        # --- accumulate:  F_i += w * jac * (t · e_k) * N_i ---
        # outer product N[:, None] * t[None, :] → (3, 2)
        F_vel += (w * jac) * np.outer(N, t)

    return F_vel, history


# ---------------------------------------------------------------------------
# Convenience wrappers for common BC types
# ---------------------------------------------------------------------------

def traction_do_nothing(x, n):
    """Zero traction — natural outflow 'do-nothing' BC."""
    return np.zeros(2)


def traction_pressure_outlet(p_out: float):
    """
    Prescribed pressure outlet:  t = -p_out * n
    (viscous part is neglected — suitable for fully-developed outflow).
    """
    def _fn(x, n):
        return -p_out * n
    return _fn


def traction_full_stress(p_out: float, mu: float, grad_u_fn: callable):
    """
    Full traction with prescribed pressure and viscous stress:
        t = -p_out * n + 2μ ε(u) · n
    Requires a grad_u_fn(x) callable that returns (2,2) ∇u at point x.
    """
    def _fn(x, n):
        grad_u = grad_u_fn(x)
        eps = 0.5 * (grad_u + grad_u.T)
        return -p_out * n + 2.0 * mu * eps @ n
    return _fn


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Straight horizontal edge from (0,0) to (1,0), outward normal = (0,-1)
    x_edge = np.array([
        [0.0, 0.0],   # endpoint 0
        [1.0, 0.0],   # endpoint 1
        [0.5, 0.0],   # midpoint
    ])

    # Uniform pressure p=2, zero velocity gradient
    p_edge = np.array([2.0, 2.0])
    u_edge = np.zeros((3, 2))

    mu = 0.01

    def grad_u_zero(xi):
        return np.zeros((2, 2))

    F, history = compute_traction_boundary_integral(
        x_edge_nodes=x_edge,
        u_edge_nodes=u_edge,
        p_edge_nodes=p_edge,
        grad_u_fn=grad_u_zero,
        mu=mu,
        n_gauss=3
    )

    print("=== Smoke test: uniform p=2, zero viscous stress ===")
    print(f"Expected traction:  t = -p*n = -2*(0,-1) = (0, 2)")
    print(f"Expected F_y total: 2 * edge_length = 2 * 1.0 = 2.0")
    print(f"\nF_vel (3 nodes × 2 components):\n{F}")
    print(f"Sum F_y = {F[:, 1].sum():.6f}  (should be 2.0)")

    # Do-nothing BC test
    F_dn, _ = compute_traction_boundary_integral(
        x_edge_nodes=x_edge,
        u_edge_nodes=u_edge,
        p_edge_nodes=p_edge,
        grad_u_fn=None,
        mu=mu,
        n_gauss=3,
        traction_fn=traction_do_nothing
    )
    print(f"\n=== Do-nothing BC: F_vel should be zero ===")
    print(f"F_vel:\n{F_dn}")