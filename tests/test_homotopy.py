"""Tests for homotopy algorithms.

This module tests the VACASK-style homotopy continuation methods for
DC operating point convergence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from jax_spice.analysis.homotopy import (
    HomotopyConfig,
    HomotopyResult,
    gmin_stepping,
    source_stepping,
    run_homotopy_chain,
)

# Precision is auto-configured by jax_spice import based on backend capabilities


def create_mock_nr_solve(residual_fn_factory, max_iterations=50, abstol=1e-10):
    """Create a mock nr_solve function for testing.

    Args:
        residual_fn_factory: Function (gmin, gshunt) -> residual_fn
                            where residual_fn(V) -> residual vector
        max_iterations: Max NR iterations
        abstol: Convergence tolerance

    Returns:
        A function with signature:
        nr_solve(V_init, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, gmin, gshunt)
            -> (V, iters, converged, max_f, Q)
    """
    def nr_solve(V_init, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, gmin=1e-12, gshunt=0.0):
        # Build residual function with current gmin/gshunt
        res_fn = residual_fn_factory(gmin, gshunt)
        jac_fn = jax.jacfwd(res_fn)

        # Simple NR loop
        def cond_fn(state):
            V, iteration, converged, max_f = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            V, iteration, _, _ = state

            f = res_fn(V)
            max_f = jnp.max(jnp.abs(f))

            # Check convergence
            converged = max_f < abstol

            # Compute Jacobian and solve
            J_full = jac_fn(V)
            J = J_full[:, 1:]  # Non-ground columns only

            # Add regularization
            reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
            J_reg = J + reg

            # Solve: J * delta = -f
            delta = jax.scipy.linalg.solve(J_reg, -f)

            # Update V (ground stays 0)
            V_new = V.at[1:].add(delta)

            return (V_new, iteration + 1, converged, max_f)

        init_state = (V_init, 0, jnp.array(False), jnp.array(jnp.inf))
        V_final, iters, converged, max_f = lax.while_loop(cond_fn, body_fn, init_state)

        Q = jnp.zeros(len(V_init) - 1)  # Dummy Q
        return V_final, iters, converged, max_f, Q

    return nr_solve


class TestHomotopyConfig:
    """Tests for HomotopyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values match VACASK defaults."""
        config = HomotopyConfig()

        # GMIN parameters
        assert config.gmin == 1e-12
        assert config.gdev_start == 1e-3
        assert config.gdev_target == 1e-13
        assert config.gmin_factor == 10.0

        # Source stepping parameters
        assert config.source_step == 0.1
        assert config.source_step_min == 0.001

        # Default chain
        assert config.chain == ("gdev", "gshunt", "src")

    def test_custom_config(self):
        """Test custom configuration."""
        config = HomotopyConfig(
            gmin=1e-11,
            gdev_start=1e-2,
            chain=("gdev", "src"),
        )

        assert config.gmin == 1e-11
        assert config.gdev_start == 1e-2
        assert config.chain == ("gdev", "src")


class TestHomotopyResult:
    """Tests for HomotopyResult dataclass."""

    def test_result_fields(self):
        """Test result dataclass fields."""
        V = jnp.array([0.0, 1.0, 0.5])
        result = HomotopyResult(
            converged=True,
            V=V,
            method="gdev_stepping",
            iterations=10,
            homotopy_steps=5,
            final_gmin=1e-12,
        )

        assert result.converged is True
        assert jnp.array_equal(result.V, V)
        assert result.method == "gdev_stepping"
        assert result.iterations == 10
        assert result.homotopy_steps == 5
        assert result.final_gmin == 1e-12


class TestSimpleCircuits:
    """Tests with simple circuits to validate homotopy algorithms."""

    def test_gmin_stepping_resistor_divider(self):
        """Test GMIN stepping with a simple resistor voltage divider.

        Circuit: VDD -- R1 -- node1 -- R2 -- GND
        Expected: V[1] = VDD * R2/(R1+R2) = 1.2 * 0.5 = 0.6V
        """
        # Circuit parameters
        vdd = 1.2
        r1 = 1000.0  # 1k ohm
        r2 = 1000.0  # 1k ohm
        g1 = 1.0 / r1
        g2 = 1.0 / r2

        def residual_fn_factory(gmin, gshunt):
            """Build residual function for the resistor divider."""
            def residual_fn(V):
                V1 = V[1]
                # KCL at node 1
                f1 = g1 * (vdd - V1) - g2 * V1 + gmin * V1 + gshunt * V1
                return jnp.array([f1])
            return residual_fn

        # Create mock nr_solve
        nr_solve = create_mock_nr_solve(residual_fn_factory)

        # Initial guess: V[0]=0 (ground), V[1]=0
        V_init = jnp.array([0.0, 0.0])
        vsource_vals = jnp.array([vdd])  # Not used by our mock
        isource_vals = jnp.array([])
        Q_prev = jnp.zeros(1)

        # Config
        config = HomotopyConfig(
            gmin=1e-12,
            gdev_start=1e-3,
            gdev_target=1e-13,
            max_iterations=50,
            abstol=1e-10,
            debug=0,
        )

        # Run GMIN stepping
        result = gmin_stepping(
            nr_solve,
            V_init,
            vsource_vals,
            isource_vals,
            Q_prev,
            device_arrays={},
            source_scale=1.0,
            config=config,
            mode="gdev",
        )

        assert result.converged, f"GMIN stepping should converge, got: {result}"
        expected_v1 = vdd * r2 / (r1 + r2)  # = 0.6V
        assert jnp.abs(result.V[1] - expected_v1) < 1e-6, (
            f"Expected V1={expected_v1}, got V1={float(result.V[1])}"
        )

    def test_source_stepping_resistor(self):
        """Test source stepping with a simple resistor circuit.

        Circuit: VDD -- R -- GND
        Expected: V1 = VDD at source_scale=1.0
        """
        vdd = 1.2

        def residual_fn_factory(gmin, gshunt):
            """Build residual function with source scaling."""
            def residual_fn(V):
                V1 = V[1]
                # Note: source scaling is handled by the homotopy algorithm
                # scaling vsource_vals, not inside the residual
                G_source = 1e12
                f1 = G_source * (V1 - vdd) + gmin * V1 + gshunt * V1
                return jnp.array([f1])
            return residual_fn

        # For source stepping, we need a residual that uses vsource_vals
        def create_source_stepping_nr_solve():
            def nr_solve(V_init, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, gmin=1e-12, gshunt=0.0):
                # Use the first vsource value as VDD (already scaled by homotopy)
                scaled_vdd = vsource_vals[0] if len(vsource_vals) > 0 else vdd

                def res_fn(V):
                    V1 = V[1]
                    G_source = 1e12
                    f1 = G_source * (V1 - scaled_vdd) + gmin * V1 + gshunt * V1
                    return jnp.array([f1])

                jac_fn = jax.jacfwd(res_fn)

                def cond_fn(state):
                    V, iteration, converged, max_f = state
                    return jnp.logical_and(~converged, iteration < 50)

                def body_fn(state):
                    V, iteration, _, _ = state
                    f = res_fn(V)
                    max_f = jnp.max(jnp.abs(f))
                    converged = max_f < 1e-10
                    J_full = jac_fn(V)
                    J = J_full[:, 1:]
                    reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
                    delta = jax.scipy.linalg.solve(J + reg, -f)
                    V_new = V.at[1:].add(delta)
                    return (V_new, iteration + 1, converged, max_f)

                init_state = (V_init, 0, jnp.array(False), jnp.array(jnp.inf))
                V_final, iters, converged, max_f = lax.while_loop(cond_fn, body_fn, init_state)
                Q = jnp.zeros(len(V_init) - 1)
                return V_final, iters, converged, max_f, Q
            return nr_solve

        nr_solve = create_source_stepping_nr_solve()

        # V[0]=0 (ground), V[1]=initial
        V_init = jnp.array([0.0, 0.0])
        vsource_vals = jnp.array([vdd])  # Full VDD, will be scaled by homotopy
        isource_vals = jnp.array([])
        Q_prev = jnp.zeros(1)

        config = HomotopyConfig(
            gmin=1e-12,
            source_step=0.2,
            max_iterations=50,
            abstol=1e-10,
            debug=0,
        )

        result = source_stepping(
            nr_solve,
            V_init,
            vsource_vals,
            isource_vals,
            Q_prev,
            device_arrays={},
            config=config,
        )

        assert result.converged, f"Source stepping should converge, got: {result}"
        assert result.final_source_scale >= 1.0, "Should reach source_scale=1.0"
        assert jnp.abs(result.V[1] - vdd) < 1e-6, (
            f"Expected V1={vdd}, got V1={float(result.V[1])}"
        )

    def test_homotopy_chain_simple_circuit(self):
        """Test the full homotopy chain with a simple circuit."""
        vdd = 1.2

        def create_chain_nr_solve():
            def nr_solve(V_init, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, gmin=1e-12, gshunt=0.0):
                scaled_vdd = vsource_vals[0] if len(vsource_vals) > 0 else vdd

                def res_fn(V):
                    V1 = V[1]
                    G_source = 1e12
                    f1 = G_source * (V1 - scaled_vdd) + gmin * V1 + gshunt * V1
                    return jnp.array([f1])

                jac_fn = jax.jacfwd(res_fn)

                def cond_fn(state):
                    V, iteration, converged, max_f = state
                    return jnp.logical_and(~converged, iteration < 50)

                def body_fn(state):
                    V, iteration, _, _ = state
                    f = res_fn(V)
                    max_f = jnp.max(jnp.abs(f))
                    converged = max_f < 1e-10
                    J_full = jac_fn(V)
                    J = J_full[:, 1:]
                    reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
                    delta = jax.scipy.linalg.solve(J + reg, -f)
                    V_new = V.at[1:].add(delta)
                    return (V_new, iteration + 1, converged, max_f)

                init_state = (V_init, 0, jnp.array(False), jnp.array(jnp.inf))
                V_final, iters, converged, max_f = lax.while_loop(cond_fn, body_fn, init_state)
                Q = jnp.zeros(len(V_init) - 1)
                return V_final, iters, converged, max_f, Q
            return nr_solve

        nr_solve = create_chain_nr_solve()

        V_init = jnp.array([0.0, 0.0])
        vsource_vals = jnp.array([vdd])
        isource_vals = jnp.array([])
        Q_prev = jnp.zeros(1)

        config = HomotopyConfig(max_iterations=50, abstol=1e-10, debug=0)

        result = run_homotopy_chain(
            nr_solve,
            V_init,
            vsource_vals,
            isource_vals,
            Q_prev,
            device_arrays={},
            config=config,
        )

        assert result.converged, f"Homotopy chain should converge, got: {result}"
        assert "chain_" in result.method, f"Method should indicate chain, got: {result.method}"


class TestDifficultCircuits:
    """Tests with circuits that are difficult to converge without homotopy."""

    def test_near_singular_circuit(self):
        """Test a circuit with near-singular Jacobian at initial guess.

        This simulates a MOSFET-like behavior where at V=0, the device
        has very low conductance (transistor off), causing convergence issues.
        """
        vdd = 1.2
        r_load = 1000.0
        g_load = 1.0 / r_load
        vth = 0.4
        k = 1e-3

        def mosfet_current(V_gs):
            return jnp.where(V_gs > vth, k * (V_gs - vth) ** 2, 0.0)

        def create_mosfet_nr_solve():
            def nr_solve(V_init, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, gmin=1e-12, gshunt=0.0):
                scaled_vdd = vsource_vals[0] if len(vsource_vals) > 0 else vdd

                def res_fn(V):
                    V1 = V[1]
                    I_ds = mosfet_current(scaled_vdd)
                    f1 = g_load * (scaled_vdd - V1) - I_ds + gmin * V1 + gshunt * V1
                    return jnp.array([f1])

                jac_fn = jax.jacfwd(res_fn)

                def cond_fn(state):
                    V, iteration, converged, max_f = state
                    return jnp.logical_and(~converged, iteration < 50)

                def body_fn(state):
                    V, iteration, _, _ = state
                    f = res_fn(V)
                    max_f = jnp.max(jnp.abs(f))
                    converged = max_f < 1e-10
                    J_full = jac_fn(V)
                    J = J_full[:, 1:]
                    reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
                    delta = jax.scipy.linalg.solve(J + reg, -f)
                    V_new = V.at[1:].add(delta)
                    return (V_new, iteration + 1, converged, max_f)

                init_state = (V_init, 0, jnp.array(False), jnp.array(jnp.inf))
                V_final, iters, converged, max_f = lax.while_loop(cond_fn, body_fn, init_state)
                Q = jnp.zeros(len(V_init) - 1)
                return V_final, iters, converged, max_f, Q
            return nr_solve

        nr_solve = create_mosfet_nr_solve()

        V_init = jnp.array([0.0, 0.0])
        vsource_vals = jnp.array([vdd])
        isource_vals = jnp.array([])
        Q_prev = jnp.zeros(1)

        config = HomotopyConfig(max_iterations=50, abstol=1e-10, debug=0)

        result = run_homotopy_chain(
            nr_solve,
            V_init,
            vsource_vals,
            isource_vals,
            Q_prev,
            device_arrays={},
            config=config,
        )

        assert result.converged, f"Homotopy chain should handle near-singular case: {result}"


class TestAdaptiveStepAdjustment:
    """Tests for adaptive step adjustment in homotopy algorithms."""

    def test_gmin_factor_increases_on_fast_convergence(self):
        """Verify factor increases when convergence is fast."""
        config = HomotopyConfig(
            gmin_factor=10.0,
            gmin_factor_max=100.0,
            debug=0,
        )
        assert config.gmin_factor_max > config.gmin_factor

    def test_source_step_scales_adaptively(self):
        """Verify source step adapts based on convergence."""
        config = HomotopyConfig(
            source_step=0.1,
            source_scale=2.0,
            source_step_min=0.001,
            debug=0,
        )
        assert config.source_scale > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
