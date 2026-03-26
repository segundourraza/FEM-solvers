import unittest
import numpy as np

from fem import NavierStokesSolver, BoundaryCondition, BCType, BCVar


class TestKovasznayFlow(unittest.TestCase):
    """
    Unit tests for the steady-state Kovasznay flow problem solved via FEM.

    The Kovasznay flow is an exact solution of the incompressible Navier-Stokes
    equations that describes a periodic shear-layer wake behind a row of
    cylinders.  Its closed-form velocity and pressure fields make it an ideal
    benchmark for nonlinear (convective) FEM solvers.

    Domain      : x in [-0.5, 1.0],  y in [-0.5, 1.5]
    Physics     : incompressible Navier-Stokes,  Re = rho = 40,  mu = 1
    BCs         : exact Dirichlet velocity on all four walls
    Element     : Q9 (order = 2),  16 x 16 mesh  (refined)
    Pressure ref: p = 0 at corner node 0

    Analytical solution
    -------------------
    lam = Re/2 - sqrt((Re/2)^2 + 4*pi^2)

    vx(x, y) = 1 - exp(lam*x) * cos(2*pi*y)
    vy(x, y) = (lam / (2*pi)) * exp(lam*x) * sin(2*pi*y)
    p(x,  y) = p_ref - 0.5 * exp(2*lam*x)
    """

    # ------------------------------------------------------------------ #
    #  Class-level setup – solver is built once and shared across tests   #
    # ------------------------------------------------------------------ #
    @classmethod
    def setUpClass(cls):
        # ── Domain ───────────────────────────────────────────────────────
        cls.x_domain = (-0.5, 1.0)
        cls.y_domain = (-0.5, 1.5)
        cls.nx = cls.ny = 32          # refined mesh
        cls.order = 2                 # Q9 elements

        cls.origin = (cls.x_domain[0], cls.y_domain[0])
        cls.a = cls.x_domain[1] - cls.x_domain[0]   # 1.5
        cls.b = cls.y_domain[1] - cls.y_domain[0]   # 2.0

        # ── Physics ──────────────────────────────────────────────────────
        cls.Re  = cls.rho = 100.0
        cls.mu  = 1.0

        cls.pref      = 10.0
        cls.corner_id = 0

        # Pre-compute the Kovasznay decay parameter
        cls.lam = cls.Re / 2.0 - np.sqrt((cls.Re / 2.0)**2 + 4.0 * np.pi**2)

        # ── Boundary conditions (exact Dirichlet on all walls) ────────────
        # Store lam as a local so the lambdas capture it cleanly
        lam = cls.lam

        dirichlet = BoundaryCondition(
            name="dirichlet",
            boundary_key="top",
            type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (
                1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y),
                (lam / (2.0 * np.pi)) * np.exp(lam * x) * np.sin(2.0 * np.pi * y),
            ),
            apply_strong=True,
            metadata={"note": "exact Kovasznay Dirichlet"},
        )

        top    = dirichlet.copy(); top.boundary_key    = "top"
        bottom = dirichlet.copy(); bottom.boundary_key = "bottom"
        left   = dirichlet.copy(); left.boundary_key   = "left"
        right  = dirichlet.copy(); right.boundary_key  = "right"

        # ── Build & run solver ───────────────────────────────────────────
        sol = NavierStokesSolver.uniform_rectangular_domain_rect(
            cls.nx, cls.ny, cls.a, cls.b,
            order=cls.order, origin=cls.origin,
        )
        sol.setup_physics(cls.rho, cls.mu)
        sol.setup_boundary_conditions(
            [bottom, top, left, right],
            pref_corner_id=cls.corner_id,
            pref_value=0.5*(cls.pref - np.exp(2.0 * cls.lam * cls.x_domain[0]))*cls.rho,
        )
        sol.solve_steadystate(u0=1, p0=cls.pref, solver=2, nonlinear_solver_options={'verbose': False})

        cls.sol     = sol
        cls.sol_vx, cls.sol_vy, cls.sol_p = sol.get_solution()

    # ------------------------------------------------------------------ #
    #  Analytical solution helpers                                         #
    # ------------------------------------------------------------------ #
    def vx_analytical(self, x, y):
        return 1.0 - np.exp(self.lam * x) * np.cos(2.0 * np.pi * y)

    def vy_analytical(self, x, y):
        return (self.lam / (2.0 * np.pi)) * np.exp(self.lam * x) * np.sin(2.0 * np.pi * y)

    def p_analytical(self, x, y):
        return 0.5*(self.pref - np.exp(2.0 * self.lam * x))*self.rho

    # ------------------------------------------------------------------ #
    #  Tests                                                               #
    # ------------------------------------------------------------------ #
    def test_lam_is_negative(self):
        """
        The decay parameter lam must be negative so that the solution
        decays in the +x direction.
        """
        self.assertLess(self.lam, 0.0,
                        f"lam={self.lam:.6f} is not negative.")

    def test_vx_matches_analytical(self):
        """
        vx must match the Kovasznay profile 1 - exp(lam*x)*cos(2pi*y)
        at every velocity node to within a tight tolerance.
        """
        nodes    = self.sol.p2_nodes
        expected = self.vx_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_vx, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="vx deviates from the analytical Kovasznay profile.",
        )

    def test_vy_matches_analytical(self):
        """
        vy must match (lam/2pi)*exp(lam*x)*sin(2pi*y) at every velocity node.
        """
        nodes    = self.sol.p2_nodes
        expected = self.vy_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_vy, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="vy deviates from the analytical Kovasznay profile.",
        )
        
    def test_p_matches_analytical(self):
        """
        p must match  0.5*(pref - np.exp(2*lam*x))*rho at every pressure node.
        """
        nodes    = self.sol.p1_nodes
        expected = self.p_analytical(nodes[:, 0], nodes[:, 1])
        np.testing.assert_allclose(
            self.sol_p, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="p deviates from the analytical Kovasznay profile.",
        )

    def test_vx_l2_error(self):
        """
        The L2 relative error in vx must be below 1 % on the 32x32 mesh.
        """
        nodes    = self.sol.p2_nodes
        expected = self.vx_analytical(nodes[:, 0], nodes[:, 1])
        l2_err   = np.linalg.norm(self.sol_vx - expected) / np.linalg.norm(expected)
        self.assertLess(l2_err, 0.01,
                        f"L2 relative error in vx = {l2_err:.4%} exceeds 1 %.")

    def test_vy_l2_error(self):
        """
        The L2 relative error in vy must be below 1 % on the 32x32 mesh.
        """
        nodes    = self.sol.p2_nodes
        expected = self.vy_analytical(nodes[:, 0], nodes[:, 1])
        # vy can be near zero so use absolute norm as denominator guard
        denom    = max(np.linalg.norm(expected), 1e-12)
        l2_err   = np.linalg.norm(self.sol_vy - expected) / denom
        self.assertLess(l2_err, 0.01,
                        f"L2 relative error in vy = {l2_err:.4%} exceeds 1 %.")

    def test_dirichlet_bcs_top(self):
        """Velocity nodes on the top wall must satisfy the exact Dirichlet BC."""
        nodes   = self.sol.p2_nodes
        top_idx = np.where(np.abs(nodes[:, 1] - self.y_domain[1]) < 1e-10)[0]
        self.assertGreater(len(top_idx), 0, "No nodes found on top wall.")
        expected_vx = self.vx_analytical(nodes[top_idx, 0], nodes[top_idx, 1])
        expected_vy = self.vy_analytical(nodes[top_idx, 0], nodes[top_idx, 1])
        np.testing.assert_allclose(self.sol_vx[top_idx], expected_vx, atol=1e-10,
                                   err_msg="Top-wall vx BC not satisfied.")
        np.testing.assert_allclose(self.sol_vy[top_idx], expected_vy, atol=1e-10,
                                   err_msg="Top-wall vy BC not satisfied.")

    def test_dirichlet_bcs_bottom(self):
        """Velocity nodes on the bottom wall must satisfy the exact Dirichlet BC."""
        nodes   = self.sol.p2_nodes
        bot_idx = np.where(np.abs(nodes[:, 1] - self.y_domain[0]) < 1e-10)[0]
        self.assertGreater(len(bot_idx), 0, "No nodes found on bottom wall.")
        expected_vx = self.vx_analytical(nodes[bot_idx, 0], nodes[bot_idx, 1])
        expected_vy = self.vy_analytical(nodes[bot_idx, 0], nodes[bot_idx, 1])
        np.testing.assert_allclose(self.sol_vx[bot_idx], expected_vx, atol=1e-10,
                                   err_msg="Bottom-wall vx BC not satisfied.")
        np.testing.assert_allclose(self.sol_vy[bot_idx], expected_vy, atol=1e-10,
                                   err_msg="Bottom-wall vy BC not satisfied.")

    def test_dirichlet_bcs_left(self):
        """Velocity nodes on the left wall must satisfy the exact Dirichlet BC."""
        nodes    = self.sol.p2_nodes
        left_idx = np.where(np.abs(nodes[:, 0] - self.x_domain[0]) < 1e-10)[0]
        self.assertGreater(len(left_idx), 0, "No nodes found on left wall.")
        expected_vx = self.vx_analytical(nodes[left_idx, 0], nodes[left_idx, 1])
        expected_vy = self.vy_analytical(nodes[left_idx, 0], nodes[left_idx, 1])
        np.testing.assert_allclose(self.sol_vx[left_idx], expected_vx, atol=1e-10,
                                   err_msg="Left-wall vx BC not satisfied.")
        np.testing.assert_allclose(self.sol_vy[left_idx], expected_vy, atol=1e-10,
                                   err_msg="Left-wall vy BC not satisfied.")

    def test_dirichlet_bcs_right(self):
        """Velocity nodes on the right wall must satisfy the exact Dirichlet BC."""
        nodes     = self.sol.p2_nodes
        right_idx = np.where(np.abs(nodes[:, 0] - self.x_domain[1]) < 1e-10)[0]
        self.assertGreater(len(right_idx), 0, "No nodes found on right wall.")
        expected_vx = self.vx_analytical(nodes[right_idx, 0], nodes[right_idx, 1])
        expected_vy = self.vy_analytical(nodes[right_idx, 0], nodes[right_idx, 1])
        np.testing.assert_allclose(self.sol_vx[right_idx], expected_vx, atol=1e-10,
                                   err_msg="Right-wall vx BC not satisfied.")
        np.testing.assert_allclose(self.sol_vy[right_idx], expected_vy, atol=1e-10,
                                   err_msg="Right-wall vy BC not satisfied.")

    def test_pressure_reference_value(self):
        """
        The pressure at the reference corner node must equal pref = 0.
        Note: the solver pins the pressure at corner_id; we verify the
        FEM value matches the analytical pressure at that node location.
        """
        ref_node = self.sol.p_ref_node
        x_ref, y_ref = self.sol.p1_nodes[ref_node.index]
        p_exact = self.p_analytical(x_ref, y_ref)
        self.assertAlmostEqual(
            ref_node.value, p_exact, places=6,
            msg=f"Reference pressure {ref_node.value:.8f} != analytical {p_exact:.8f}.",
        )
        
    def test_solution_shape(self):
        """Solution arrays must be consistent with node array sizes."""
        self.assertEqual(self.sol_vx.shape[0], self.sol.p2_nodes.shape[0])
        self.assertEqual(self.sol_vy.shape[0], self.sol.p2_nodes.shape[0])
        self.assertEqual(self.sol_p.shape[0],  self.sol.p1_nodes.shape[0])

    def test_mesh_size(self):
        """Confirm the solver was built with the required 32x32 refined mesh."""
        # Number of Q9 velocity nodes on a (nx x ny) mesh with order=2:
        # (2*nx + 1) * (2*ny + 1)
        expected_v_nodes = (2 * self.nx + 1) * (2 * self.ny + 1)
        expected_p_nodes = (self.nx + 1) * (self.ny + 1)
        self.assertEqual(
            self.sol.p2_nodes.shape[0], expected_v_nodes,
            f"Expected {expected_v_nodes} velocity nodes for a "
            f"{self.nx}x{self.ny} Q9 mesh.",
        )
        self.assertEqual(
            self.sol.p1_nodes.shape[0], expected_p_nodes,
            f"Expected {expected_p_nodes} pressure nodes for a "
            f"{self.nx}x{self.ny} Q9 mesh.",
        )

class TestKovasznayConvergence(unittest.TestCase):
    """
    Mesh-convergence test for the Kovasznay flow benchmark.

    Expected optimal rates for Q2/Q1 Taylor-Hood on quads:
        ||u - u_h||_L2  ~ O(h^3)
        ||p - p_h||_L2  ~ O(h^2)

    We run on meshes {4, 8, 16, 32}, fit the slope in log-log space,
    and assert the measured rate is within a tolerance of the theoretical one.
    """

    # ── Parameters ────────────────────────────────────────────────────
    x_domain = (-0.5, 1.0)
    y_domain = (-0.5, 1.5)
    order    = 2
    Re = rho = 100.0
    mu       = 1.0
    pref     = 10.0

    mesh_sizes = [4, 8, 16, 32]

    # Theoretical rates and acceptable tolerance (allow ~0.3 degradation)
    expected_velocity_rate = 3.0
    expected_pressure_rate = 2.0
    rate_tolerance         = 0.4

    @classmethod
    def setUpClass(cls):
        cls.origin = (cls.x_domain[0], cls.y_domain[0])
        cls.a = cls.x_domain[1] - cls.x_domain[0]
        cls.b = cls.y_domain[1] - cls.y_domain[0]
        cls.lam = cls.Re / 2.0 - np.sqrt((cls.Re / 2.0)**2 + 4.0 * np.pi**2)

        lam = cls.lam

        dirichlet = BoundaryCondition(
            name="dirichlet",
            boundary_key="top",
            type=BCType.DIRICHLET,
            variable=BCVar.VELOCITY,
            value=lambda x, y, t: (
                1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y),
                (lam / (2.0 * np.pi)) * np.exp(lam * x) * np.sin(2.0 * np.pi * y),
            ),
            apply_strong=True,
            metadata={"note": "exact Kovasznay Dirichlet"},
        )

        cls.h_vals     = []
        cls.err_vx_l2  = []
        cls.err_vy_l2  = []
        cls.err_p_l2   = []

        for n in cls.mesh_sizes:
            sol = NavierStokesSolver.uniform_rectangular_domain_rect(
                n, n, cls.a, cls.b,
                order=cls.order, origin=cls.origin,
            )
            sol.setup_physics(cls.rho, cls.mu)

            bcs = []
            for key in ("top", "bottom", "left", "right"):
                bc = dirichlet.copy()
                bc.boundary_key = key
                bcs.append(bc)

            sol.setup_boundary_conditions(
                bcs,
                pref_corner_id=0,
                pref_value=0.5 * (cls.pref - np.exp(2.0 * cls.lam * cls.x_domain[0])) * cls.rho,
            )
            sol.solve_steadystate(u0=1, p0=cls.pref, solver=2, nonlinear_solver_options={'verbose': False})
            vx, vy, p = sol.get_solution()

            # ── Velocity errors (Q2 nodes) ──
            v_nodes = sol.p2_nodes
            vx_ex = 1.0 - np.exp(lam * v_nodes[:, 0]) * np.cos(2.0 * np.pi * v_nodes[:, 1])
            vy_ex = (lam / (2.0 * np.pi)) * np.exp(lam * v_nodes[:, 0]) * np.sin(2.0 * np.pi * v_nodes[:, 1])

            # ── Pressure errors (Q1 nodes) ──
            p_nodes = sol.p1_nodes
            p_ex = 0.5 * (cls.pref - np.exp(2.0 * cls.lam * p_nodes[:, 0])) * cls.rho

            h = max(cls.a / n, cls.b / n)
            cls.h_vals.append(h)
            cls.err_vx_l2.append(np.linalg.norm(vx - vx_ex) / np.linalg.norm(vx_ex))
            cls.err_vy_l2.append(np.linalg.norm(vy - vy_ex) / max(np.linalg.norm(vy_ex), 1e-12))
            cls.err_p_l2.append(np.linalg.norm(p - p_ex) / np.linalg.norm(p_ex))

        cls.h_vals    = np.array(cls.h_vals)
        cls.err_vx_l2 = np.array(cls.err_vx_l2)
        cls.err_vy_l2 = np.array(cls.err_vy_l2)
        cls.err_p_l2  = np.array(cls.err_p_l2)

    @staticmethod
    def _fit_rate(h, err):
        """Least-squares fit of log(err) = rate * log(h) + const."""
        log_h   = np.log(h)
        log_err = np.log(err)
        A = np.vstack([log_h, np.ones_like(log_h)]).T
        rate, _ = np.linalg.lstsq(A, log_err, rcond=None)[0]
        return rate

    # ── Tests ─────────────────────────────────────────────────────────

    def test_vx_convergence_rate(self):
        """vx L2 error must converge at ~O(h^3) for Q2 elements."""
        rate = self._fit_rate(self.h_vals, self.err_vx_l2)
        self.assertGreater(
            rate, self.expected_velocity_rate - self.rate_tolerance,
            f"vx convergence rate {rate:.2f} below threshold "
            f"{self.expected_velocity_rate - self.rate_tolerance:.1f}.",
        )

    def test_vy_convergence_rate(self):
        """vy L2 error must converge at ~O(h^3) for Q2 elements."""
        rate = self._fit_rate(self.h_vals, self.err_vy_l2)
        self.assertGreater(
            rate, self.expected_velocity_rate - self.rate_tolerance,
            f"vy convergence rate {rate:.2f} below threshold "
            f"{self.expected_velocity_rate - self.rate_tolerance:.1f}.",
        )

    def test_pressure_convergence_rate(self):
        """Pressure L2 error must converge at ~O(h^2) for Q1 elements."""
        rate = self._fit_rate(self.h_vals, self.err_p_l2)
        self.assertGreater(
            rate, self.expected_pressure_rate - self.rate_tolerance,
            f"Pressure convergence rate {rate:.2f} below threshold "
            f"{self.expected_pressure_rate - self.rate_tolerance:.1f}.",
        )

    def test_errors_monotonically_decrease(self):
        """Errors must decrease on every refinement step."""
        for name, errs in [("vx", self.err_vx_l2), ("vy", self.err_vy_l2), ("p", self.err_p_l2)]:
            for i in range(1, len(errs)):
                self.assertLess(
                    errs[i], errs[i - 1],
                    f"{name} error did not decrease from mesh {self.mesh_sizes[i-1]} "
                    f"to {self.mesh_sizes[i]}: {errs[i-1]:.3e} -> {errs[i]:.3e}",
                )

if __name__ == "__main__":
    unittest.main(verbosity=2)
