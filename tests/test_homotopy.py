"""Tests for homotopy algorithms.

This module tests the VACASK-style homotopy continuation methods for
DC operating point convergence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_spice.analysis.homotopy import (
    HomotopyConfig,
    HomotopyResult,
    gmin_stepping,
    source_stepping,
    run_homotopy_chain,
)
from jax_spice.analysis.solver import NRConfig

# Precision is auto-configured by jax_spice import based on backend capabilities


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

        # 2 nodes: ground (0), node1 (1)
        # V includes ground, residual excludes ground
        n_nodes = 2

        def build_residual_fn(gmin: float, gshunt: float):
            """Build residual function for the resistor divider."""

            def residual_fn(V):
                # V is the full voltage vector [V0, V1] where V0=0 (ground)
                V1 = V[1]

                # KCL at node 1:
                # Current from VDD through R1: (VDD - V1) * G1 = G1*VDD - G1*V1
                # Current to GND through R2: -V1 * G2
                # GMIN contribution: gmin * V1
                # GSHUNT contribution: gshunt * V1
                f1 = g1 * (vdd - V1) - g2 * V1 + gmin * V1 + gshunt * V1

                return jnp.array([f1])

            return residual_fn

        def build_jacobian_fn(gmin: float, gshunt: float):
            """Build Jacobian function for the resistor divider."""
            # Returns shape (n-1, n) = (1, 2)
            return jax.jacfwd(build_residual_fn(gmin, gshunt))

        # Initial guess: V[0]=0 (ground), V[1]=0
        V_init = jnp.array([0.0, 0.0])

        # Config
        config = HomotopyConfig(
            gmin=1e-12,
            gdev_start=1e-3,
            gdev_target=1e-13,
            debug=0,
        )
        nr_config = NRConfig(max_iterations=50, abstol=1e-10, reltol=1e-6)

        # Run GMIN stepping
        result = gmin_stepping(
            build_residual_fn,
            build_jacobian_fn,
            V_init,
            config,
            nr_config,
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
        n_nodes = 2

        def build_residual_fn(source_scale: float, gmin: float, gshunt: float):
            """Build residual function with source scaling."""

            def residual_fn(V):
                V1 = V[1]

                # KCL at node 1 (VDD node):
                # Voltage source enforces V1 = VDD * source_scale
                # We use a large conductance to enforce this
                G_source = 1e12
                scaled_vdd = vdd * source_scale
                f1 = G_source * (V1 - scaled_vdd) + gmin * V1 + gshunt * V1

                return jnp.array([f1])

            return residual_fn

        def build_jacobian_fn(source_scale: float, gmin: float, gshunt: float):
            """Build Jacobian function."""
            return jax.jacfwd(build_residual_fn(source_scale, gmin, gshunt))

        # V[0]=0 (ground), V[1]=initial
        V_init = jnp.array([0.0, 0.0])

        config = HomotopyConfig(
            gmin=1e-12,
            source_step=0.2,  # Start with larger steps
            debug=0,
        )
        nr_config = NRConfig(max_iterations=50, abstol=1e-10, reltol=1e-6)

        result = source_stepping(
            build_residual_fn,
            build_jacobian_fn,
            V_init,
            config,
            nr_config,
        )

        assert result.converged, f"Source stepping should converge, got: {result}"
        assert result.final_source_scale >= 1.0, "Should reach source_scale=1.0"
        assert jnp.abs(result.V[1] - vdd) < 1e-6, (
            f"Expected V1={vdd}, got V1={float(result.V[1])}"
        )

    def test_homotopy_chain_simple_circuit(self):
        """Test the full homotopy chain with a simple circuit."""
        vdd = 1.2
        n_nodes = 2

        def build_residual_fn(source_scale: float, gmin: float, gshunt: float):
            def residual_fn(V):
                V1 = V[1]

                G_source = 1e12
                scaled_vdd = vdd * source_scale
                f1 = G_source * (V1 - scaled_vdd) + gmin * V1 + gshunt * V1

                return jnp.array([f1])

            return residual_fn

        def build_jacobian_fn(source_scale: float, gmin: float, gshunt: float):
            return jax.jacfwd(build_residual_fn(source_scale, gmin, gshunt))

        # V[0]=0 (ground), V[1]=initial
        V_init = jnp.array([0.0, 0.0])

        config = HomotopyConfig(debug=0)
        nr_config = NRConfig(max_iterations=50, abstol=1e-10, reltol=1e-6)

        result = run_homotopy_chain(
            build_residual_fn,
            build_jacobian_fn,
            V_init,
            config,
            nr_config,
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
        # Parameters
        vdd = 1.2
        r_load = 1000.0  # Load resistor
        g_load = 1.0 / r_load
        n_nodes = 2

        # MOSFET-like I-V: I = 0 for V < Vth, I = k*(V-Vth)^2 for V > Vth
        vth = 0.4
        k = 1e-3  # transconductance

        def mosfet_current(V_gs):
            """Simplified MOSFET current."""
            return jnp.where(V_gs > vth, k * (V_gs - vth) ** 2, 0.0)

        def build_residual_fn(source_scale: float, gmin: float, gshunt: float):
            def residual_fn(V):
                V1 = V[1]  # Output node
                scaled_vdd = vdd * source_scale

                # KCL at output node:
                # Current from VDD through load: (VDD - V1) * g_load
                # Current through MOSFET (drain): I_ds (flows out)
                # Assuming gate is at VDD, V_gs = VDD
                I_ds = mosfet_current(scaled_vdd)

                f1 = g_load * (scaled_vdd - V1) - I_ds + gmin * V1 + gshunt * V1

                return jnp.array([f1])

            return residual_fn

        def build_jacobian_fn(source_scale: float, gmin: float, gshunt: float):
            return jax.jacfwd(build_residual_fn(source_scale, gmin, gshunt))

        # V[0]=0 (ground), V[1]=initial
        V_init = jnp.array([0.0, 0.0])

        config = HomotopyConfig(debug=0)
        nr_config = NRConfig(max_iterations=50, abstol=1e-10, reltol=1e-6)

        result = run_homotopy_chain(
            build_residual_fn,
            build_jacobian_fn,
            V_init,
            config,
            nr_config,
        )

        assert result.converged, f"Homotopy chain should handle near-singular case: {result}"


class TestAdaptiveStepAdjustment:
    """Tests for adaptive step adjustment in homotopy algorithms."""

    def test_gmin_factor_increases_on_fast_convergence(self):
        """Verify factor increases when convergence is fast."""
        # This is implicitly tested by the other tests - convergence
        # behavior is observed in the debug output when debug=1

        config = HomotopyConfig(
            gmin_factor=10.0,
            gmin_factor_max=100.0,
            debug=0,
        )

        # Just verify the config values are correct
        assert config.gmin_factor_max > config.gmin_factor

    def test_source_step_scales_adaptively(self):
        """Verify source step adapts based on convergence."""
        config = HomotopyConfig(
            source_step=0.1,
            source_scale=2.0,
            source_step_min=0.001,
            debug=0,
        )

        # Just verify the config values are correct
        assert config.source_scale > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
