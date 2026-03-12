import unittest
import numpy as np

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar


class TestPoiseuilleFlow(unittest.TestCase):
    """
    Unit tests for the steady-state plane Poiseuille flow problem solved via FEM.

    A pressure difference dP drives viscous flow between two stationary
    parallel plates separated by height b.

    Domain      : rectangular, width a=6, height b=2  (channel: y in [0, b])
    Physics     : incompressible Navier-Stokes, rho=1, mu=1
    BCs         : no-slip top & bottom walls (Dirichlet)
                  pressure inlet  p_in  at x=0  (Neumann traction)
                  pressure outlet p_out at x=a  (Neumann traction)
    Element     : Q9 (order=2), 4x4 mesh

    Analytical solution (channel centreline at y = b/2)
    ----------------------------------------------------
    Let  dPdx = (p_out - p_in) / a   (negative for left-to-right flow)

         vx(x, y) = -1/(2*mu) * dPdx * y * (b - y)
         vy(x, y) = 0
         p(x,  y) = p_in + dPdx * x          (linear in x)

    Peak velocity at centreline (y = b/2):
         vx_max = -dPdx * b**2 / (8*mu)
    """

    # ------------------------------------------------------------------ #
    #  Class-level setup – solver is built once and shared across tests   #
    # ------------------------------------------------------------------ #
    @classmethod
    def setUpClass(cls):
        cls.a     = 6       # channel length
        cls.b     = 2       # channel height
        cls.nx    = cls.ny = 4
        cls.order = 2

        cls.rho   = 1.0
        cls.mu    = 1.0

        # Pressure boundary values
        cls.p_in  = 10.0
        cls.p_out = 4.0
        cls.dPdx  = (cls.p_out - cls.p_in) / cls.a   # < 0  →  flow in +x

        # ── boundary conditions ──────────────────────────────────────────
        # No-slip on top and bottom walls
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

        # Pressure inlet: traction = (p_in, 0) on the inward normal
        # Left boundary outward normal is (-1,0), so traction = (-p_in, 0)
        # in the Neumann form  t = sigma·n  →  use (p_in, 0) with sign
        # convention of the solver (stress-free value = prescribed pressure)
        inlet = BoundaryCondition(
            name="pressure-inlet",
            boundary_key="left",
            type=BCType.NEUMANN,
            variable=BCVar.PRESSURE,
            value= cls.p_in,
            apply_strong=False,
            metadata={"p": cls.p_in},
        )
        outlet = BoundaryCondition(
            name="pressure-outlet",
            boundary_key="right",
            type=BCType.NEUMANN,
            variable=BCVar.PRESSURE,
            value= cls.p_out,
            apply_strong=False,
            metadata={"p": cls.p_out},
        )

        # ── build & run solver ───────────────────────────────────────────
        sol = NavierStokesSolver.uniform_rectangular_domain_rect(
            cls.nx, cls.ny, cls.a, cls.b, order=cls.order
        )
        sol.setup_physics(cls.rho, cls.mu)
        sol.setup_boundary_conditions([bottom, top, inlet, outlet])
        sol.solve_steadystate(u0=1, p0=cls.p_in)

        cls.sol     = sol
        cls.sol_vx, cls.sol_vy, cls.sol_p = sol.get_solution()

    # ------------------------------------------------------------------ #
    #  Analytical solution helpers                                         #
    # ------------------------------------------------------------------ #
    def vx_analytical(self, x, y):
        """Parabolic Poiseuille profile."""
        return (-1.0 / (2.0 * self.mu)) * self.dPdx * y * (self.b - y)

    def vy_analytical(self, x, y):
        return np.zeros_like(np.asarray(y, dtype=float))

    def p_analytical(self, x, y):
        """Linear pressure drop from inlet to outlet."""
        return self.p_in + self.dPdx * np.asarray(x, dtype=float)

    @property
    def vx_max(self):
        """Peak centreline velocity."""
        return -self.dPdx * self.b**2 / (8.0 * self.mu)

    # ------------------------------------------------------------------ #
    #  Tests                                                               #
    # ------------------------------------------------------------------ #
    def test_vx_matches_analytical(self):
        """vx must match the parabolic Poiseuille profile at every node."""
        nodes    = self.sol.p2_nodes
        expected = self.vx_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_vx, expected,
            rtol=1e-8, atol=1e-10,
            err_msg="vx deviates from the analytical parabolic Poiseuille profile.",
        )

    def test_vy_is_zero(self):
        """vy must be zero everywhere (fully-developed channel flow)."""
        nodes    = self.sol.p2_nodes
        expected = self.vy_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_vy, expected,
            rtol=1e-8, atol=1e-10,
            err_msg="vy is not zero — unexpected cross-flow detected.",
        )

    def test_pressure_linear_in_x(self):
        """Pressure must vary linearly in x and match p_in + dPdx·x."""
        nodes    = self.sol.p1_nodes
        expected = self.p_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_p, expected,
            rtol=1e-8, atol=1e-10,
            err_msg="Pressure does not match the linear analytical profile.",
        )

    def test_pressure_inlet_value(self):
        """Pressure at x=0 must equal p_in."""
        nodes   = self.sol.p1_nodes
        tol     = 1e-10
        in_idx  = np.where(np.abs(nodes[:, 0]) < tol)[0]
        self.assertGreater(len(in_idx), 0, "No pressure nodes found at x=0.")
        np.testing.assert_allclose(
            self.sol_p[in_idx], self.p_in,
            atol=1e-8,
            err_msg=f"Inlet pressure != {self.p_in}.",
        )

    def test_pressure_outlet_value(self):
        """Pressure at x=a must equal p_out."""
        nodes    = self.sol.p1_nodes
        tol      = 1e-10
        out_idx  = np.where(np.abs(nodes[:, 0] - self.a) < tol)[0]
        self.assertGreater(len(out_idx), 0, "No pressure nodes found at x=a.")
        np.testing.assert_allclose(
            self.sol_p[out_idx], self.p_out,
            atol=1e-8,
            err_msg=f"Outlet pressure != {self.p_out}.",
        )

    def test_no_slip_top_wall(self):
        """Top wall must be stationary: vx = vy = 0."""
        nodes   = self.sol.p2_nodes
        top_idx = np.where(np.abs(nodes[:, 1] - self.b) < 1e-10)[0]
        self.assertGreater(len(top_idx), 0, "No nodes found on top wall.")
        np.testing.assert_allclose(self.sol_vx[top_idx], 0.0, atol=1e-10,
                                   err_msg="Top-wall vx != 0.")
        np.testing.assert_allclose(self.sol_vy[top_idx], 0.0, atol=1e-10,
                                   err_msg="Top-wall vy != 0.")

    def test_no_slip_bottom_wall(self):
        """Bottom wall must be stationary: vx = vy = 0."""
        nodes   = self.sol.p2_nodes
        bot_idx = np.where(np.abs(nodes[:, 1]) < 1e-10)[0]
        self.assertGreater(len(bot_idx), 0, "No nodes found on bottom wall.")
        np.testing.assert_allclose(self.sol_vx[bot_idx], 0.0, atol=1e-10,
                                   err_msg="Bottom-wall vx != 0.")
        np.testing.assert_allclose(self.sol_vy[bot_idx], 0.0, atol=1e-10,
                                   err_msg="Bottom-wall vy != 0.")

    def test_peak_velocity_at_centreline(self):
        """
        The maximum vx must occur at the centreline y = b/2 and equal
        vx_max = -dPdx * b² / (8*mu).
        """
        nodes      = self.sol.p2_nodes
        y_centre   = self.b / 2.0
        centre_idx = np.where(np.abs(nodes[:, 1] - y_centre) < 1e-6)[0]
        if len(centre_idx) == 0:
            self.skipTest("No nodes found at centreline y = b/2.")
        np.testing.assert_allclose(
            self.sol_vx[centre_idx], self.vx_max,
            rtol=1e-8, atol=1e-10,
            err_msg=f"Centreline vx != vx_max ({self.vx_max:.6f}).",
        )

    def test_velocity_profile_parabolic(self):
        """
        At mid-domain (x = a/2), vx(y) must be well-fitted by a degree-2
        polynomial with R² ≥ 0.9999.
        """
        nodes    = self.sol.p2_nodes
        x_target = self.a / 2.0
        idx      = np.where(np.abs(nodes[:, 0] - x_target) < 1e-6)[0]
        if len(idx) < 4:
            self.skipTest("Not enough nodes at x = a/2 to test parabolicity.")
        y_vals  = nodes[idx, 1]
        vx_vals = self.sol_vx[idx]
        residuals = vx_vals - np.polyval(np.polyfit(y_vals, vx_vals, 2), y_vals)
        ss_res  = np.sum(residuals**2)
        ss_tot  = np.sum((vx_vals - vx_vals.mean())**2)
        r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        self.assertGreaterEqual(r2, 0.9999,
                                f"vx profile is not parabolic (R²={r2:.6f}).")

    def test_velocity_profile_symmetric(self):
        """
        vx must be symmetric about the centreline:
        vx(x, y) == vx(x, b - y)  for all nodes.
        """
        nodes   = self.sol.p2_nodes
        tol_pos = 1e-8
        for i in range(len(nodes)):
            xi, yi = nodes[i, 0], nodes[i, 1]
            y_mirror = self.b - yi
            # find the mirror node
            dists = np.sqrt((nodes[:, 0] - xi)**2 + (nodes[:, 1] - y_mirror)**2)
            j = np.argmin(dists)
            if dists[j] < tol_pos:
                self.assertAlmostEqual(
                    self.sol_vx[i], self.sol_vx[j], places=8,
                    msg=f"Symmetry broken at node ({xi:.3f}, {yi:.3f}).",
                )

    def test_flow_rate_matches_analytical(self):
        """
        The volumetric flow rate Q = ∫₀ᵇ vx dy must equal Q_exact = vx_max * 2b/3
        at every x-station (here checked at x = a/2).
        """
        nodes    = self.sol.p2_nodes
        x_target = self.a / 2.0
        idx      = np.where(np.abs(nodes[:, 0] - x_target) < 1e-6)[0]
        if len(idx) < 3:
            self.skipTest("Not enough nodes at x = a/2 to integrate flow rate.")
        y_vals  = nodes[idx, 1]
        vx_vals = self.sol_vx[idx]
        sort_order = np.argsort(y_vals)
        Q_fem   = np.trapezoid(vx_vals[sort_order], y_vals[sort_order])
        Q_exact = self.vx_max * 2.0 * self.b / 3.0
        self.assertAlmostEqual(Q_fem, Q_exact, places=6,
                               msg=f"Flow rate Q={Q_fem:.8f} != Q_exact={Q_exact:.8f}.")

    def test_solution_shape(self):
        """Solution arrays must be consistent with node array sizes."""
        self.assertEqual(self.sol_vx.shape[0], self.sol.p2_nodes.shape[0])
        self.assertEqual(self.sol_vy.shape[0], self.sol.p2_nodes.shape[0])
        self.assertEqual(self.sol_p.shape[0],  self.sol.p1_nodes.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)