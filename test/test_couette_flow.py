import unittest
import numpy as np
from itertools import product

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar


class TestCouetteFlow(unittest.TestCase):
    """
    Unit tests for the steady-state Couette flow problem solved via FEM.

    Domain   : rectangular, width a=6, height b=2
    Physics  : incompressible Navier-Stokes, rho=1, mu=1
    BCs      : moving top wall (Vw=1), no-slip bottom,
               traction-free left & right boundaries
    Element  : Q9 (order=2), 4x4 mesh

    Analytical solution
    -------------------
    vx(x, y) = Vw * y / b
    vy(x, y) = 0
    p(x, y)  = p_ref  (uniform)
    """

    # ------------------------------------------------------------------ #
    #  Class-level setup – solver is built once and shared across tests   #
    # ------------------------------------------------------------------ #
    @classmethod
    def setUpClass(cls):
        cls.a   = 6       # domain width
        cls.b   = 2       # domain height
        cls.nx  = cls.ny = 4
        cls.order = 2

        cls.rho = 1.0
        cls.mu  = 1.0
        cls.Vw  = 1.0

        # --- boundary conditions ---
        top = BoundaryCondition(
            name="moving-top-wall",
            boundary_key="top",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (cls.Vw, 0),
            apply_strong=True,
            metadata={"Vx": cls.Vw, "Vy": 0},
        )
        bottom = BoundaryCondition(
            name="no-slip",
            boundary_key="bottom",
            bc_type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=(0.0, 0.0),
            apply_strong=True,
            metadata={"note": "no-slip"},
        )
        outlet = BoundaryCondition(
            name="outlet-stressfree",
            boundary_key="right",
            bc_type=BCType.NEUMANN,
            traction=lambda x, y, t: (0.0, 0.0),
            apply_strong=False,
            metadata={"description": "do-nothing / traction-free outlet"},
        )
        inlet = BoundaryCondition(
            name="inlet-stressfree",
            boundary_key="left",
            bc_type=BCType.NEUMANN,
            traction=lambda x, y, t: (0.0, 0.0),
            apply_strong=False,
            metadata={"description": "do-nothing / traction-free inlet"},
        )

        # build & run solver
        sol = NavierStokesSolver.uniform_rectangular_domain_rect(
            cls.nx, cls.ny, cls.a, cls.b, order=cls.order)
        sol.setup_physics(cls.rho, cls.mu)
        sol.setup_boundary_conditions([bottom, outlet, top, inlet])
        sol.solve_steadystate(u0=10, p0=100)

        cls.sol = sol
        cls.sol_vx, cls.sol_vy, cls.sol_p = sol.get_solution()

    ######################################################################
    # ANALYTICAL SOLUTION HELPERS
    
    def vx_analytical(self, x, y):
        return self.Vw * y / self.b

    def vy_analytical(self, x, y):
        return np.zeros_like(np.asarray(y, dtype=float))

    def p_analytical(self, x, y):
        ref = self.sol.p_ref_node.value
        if isinstance(x, (float, int)):
            return np.ones_like(np.asarray(y, dtype=float)) * ref
        elif isinstance(y, (float, int)):
            return np.ones_like(np.asarray(x, dtype=float)) * ref
        else:
            return ref

    ######################################################################
    # TESTS

    def test_vx_matches_analytical(self):
        """Horizontal velocity must match the linear Couette profile."""
        nodes = self.sol.p2_nodes
        expected = self.vx_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_vx, expected,
            rtol=1e-8, atol=1e-10,
            err_msg="vx FEM solution deviates from analytical Couette profile.",
        )

    def test_vy_is_zero(self):
        """Vertical velocity must be zero everywhere (pure shear flow)."""
        nodes = self.sol.p2_nodes
        expected = self.vy_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_vy, expected,
            rtol=1e-8, atol=1e-10,
            err_msg="vy FEM solution is not zero (expected pure shear).",
        )

    def test_pressure_is_uniform(self):
        """Pressure must be uniform and equal to the reference value."""
        nodes = self.sol.p1_nodes
        expected = self.p_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_p, expected,
            rtol=1e-8, atol=1e-10,
            err_msg="Pressure FEM solution is not uniform.",
        )

    def test_top_wall_velocity(self):
        """All nodes on the top wall must have vx == Vw and vy == 0."""
        nodes  = self.sol.p2_nodes
        tol    = 1e-10
        top_idx = np.where(np.abs(nodes[:, 1] - self.b) < tol)[0]
        self.assertTrue(len(top_idx) > 0, "No nodes found on top wall.")
        np.testing.assert_allclose(
            self.sol_vx[top_idx], self.Vw,
            atol=1e-10,
            err_msg="Top-wall vx != Vw.",
        )
        np.testing.assert_allclose(
            self.sol_vy[top_idx], 0.0,
            atol=1e-10,
            err_msg="Top-wall vy != 0.",
        )

    def test_bottom_wall_no_slip(self):
        """All nodes on the bottom wall must have vx == 0 and vy == 0."""
        nodes   = self.sol.p2_nodes
        tol     = 1e-10
        bot_idx = np.where(np.abs(nodes[:, 1]) < tol)[0]
        self.assertTrue(len(bot_idx) > 0, "No nodes found on bottom wall.")
        np.testing.assert_allclose(
            self.sol_vx[bot_idx], 0.0,
            atol=1e-10,
            err_msg="Bottom-wall vx != 0 (no-slip violated).",
        )
        np.testing.assert_allclose(
            self.sol_vy[bot_idx], 0.0,
            atol=1e-10,
            err_msg="Bottom-wall vy != 0 (no-slip violated).",
        )

    def test_velocity_profile_linearity(self):
        """
        At a fixed x station, vx should vary linearly with y.
        """
        nodes = self.sol.p2_nodes
        x_target = self.a / 2
        tol      = 1e-6
        idx      = np.where(np.abs(nodes[:, 0] - x_target) < tol)[0]
        if len(idx) < 3:
            self.skipTest("Not enough nodes at x = a/2 to test linearity.")
        y_vals  = nodes[idx, 1]
        vx_vals = self.sol_vx[idx]
        
        np.testing.assert_allclose(
            vx_vals, self.vx_analytical(x_target, y_vals),
            atol=1e-10,
            err_msg="Bottom-wall vy != 0 (no-slip violated).",
        )

    def test_solution_shape(self):
        """Solution arrays must have consistent sizes with node arrays."""
        self.assertEqual(self.sol_vx.shape[0], self.sol.p2_nodes.shape[0])
        self.assertEqual(self.sol_vy.shape[0], self.sol.p2_nodes.shape[0])
        self.assertEqual(self.sol_p.shape[0],  self.sol.p1_nodes.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)