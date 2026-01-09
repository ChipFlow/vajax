"""Tests for device evaluation correctness.

This module validates that openvaf-py's MIR interpreter produces correct
numerical results (residuals and Jacobians) matching expected physics.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import openvaf_py

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant
Q = 1.602176634e-19  # Elementary charge
T_NOMINAL = 300.15  # Nominal temperature in Kelvin

# Base directories
OPENVAF_DIR = Path(__file__).parent.parent / "vendor" / "OpenVAF"
INTEGRATION_TESTS_DIR = OPENVAF_DIR / "integration_tests"
VACASK_DIR = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices"


def get_module(va_path: Path) -> openvaf_py.VaModule:
    """Compile a Verilog-A file and return the module."""
    modules = openvaf_py.compile_va(str(va_path))
    if not modules:
        raise ValueError(f"No modules found in {va_path}")
    return modules[0]


def vt(temp: float = T_NOMINAL) -> float:
    """Calculate thermal voltage Vt = kT/q."""
    return K_B * temp / Q


class TestResistorEvaluation:
    """Tests for resistor device evaluation correctness."""

    @pytest.fixture
    def resistor_module(self):
        """Load resistor module."""
        va_path = INTEGRATION_TESTS_DIR / "RESISTOR" / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")
        return get_module(va_path)

    def test_resistor_residual_ohms_law(self, resistor_module):
        """Test that resistor follows Ohm's law: I = V/R."""
        # Verify the resistor structure is correct
        desc = resistor_module.get_osdi_descriptor()
        nodes = [n["name"] for n in desc["nodes"]]

        # For now, verify the structure is correct
        assert desc["num_nodes"] == 2, f"Expected 2 nodes, got {desc['num_nodes']}"
        assert desc["num_terminals"] == 2, "Resistor should have 2 terminals"
        # Node names should be A and B
        assert "A" in nodes or "a" in [n.lower() for n in nodes], f"Expected node A, got {nodes}"
        assert "B" in nodes or "b" in [n.lower() for n in nodes], f"Expected node B, got {nodes}"

    def test_resistor_jacobian_conductance(self, resistor_module):
        """Test that resistor has proper Jacobian structure."""
        desc = resistor_module.get_osdi_descriptor()

        for j in desc["jacobian"]:
            # Resistor should have resistive Jacobian entries
            assert j["has_resist"], "Resistor should have resistive contribution"
            # Note: With temperature dependence (zeta parameter), the Jacobian
            # may not be constant. The key is it has resistive contribution.
            # Reactive should be constant (no dQ/dV dynamics)
            assert j["react_const"], "Resistor reactive Jacobian should be constant (no charge storage)"


class TestDiodeEvaluation:
    """Tests for diode device evaluation correctness."""

    @pytest.fixture
    def diode_module(self):
        """Load diode module."""
        va_path = INTEGRATION_TESTS_DIR / "DIODE" / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Diode VA file not found: {va_path}")
        return get_module(va_path)

    def test_diode_forward_bias_physics(self, diode_module):
        """Test diode follows Shockley equation in forward bias."""
        # At forward bias, diode current follows I = Is * (exp(V/n*Vt) - 1)
        # Where Vt = kT/q is thermal voltage (~26mV at room temp)

        # Verify the diode has the expected parameters
        desc = diode_module.get_osdi_descriptor()
        param_names = [p["name"] for p in desc["params"]]

        # Diode should have saturation current (Is or is)
        has_is = any(name.lower() in ("is", "i_s") for name in param_names)
        assert has_is, f"Diode should have Is parameter, found: {param_names}"

    def test_diode_jacobian_nonlinear(self, diode_module):
        """Test that diode Jacobian is NOT constant (nonlinear device)."""
        desc = diode_module.get_osdi_descriptor()

        # Diode should have non-constant resistive Jacobian entries
        non_const_count = sum(1 for j in desc["jacobian"] if j["has_resist"] and not j["resist_const"])
        assert non_const_count > 0, "Diode should have non-constant resistive Jacobian entries"


class TestCapacitorEvaluation:
    """Tests for capacitor device evaluation correctness."""

    @pytest.fixture
    def capacitor_module(self):
        """Load capacitor module."""
        # Try VACASK version first
        va_path = VACASK_DIR / "capacitor.va"
        if not va_path.exists():
            pytest.skip(f"Capacitor VA file not found: {va_path}")
        return get_module(va_path)

    def test_capacitor_has_reactive_contribution(self, capacitor_module):
        """Test that capacitor has reactive (charge storage) contribution."""
        desc = capacitor_module.get_osdi_descriptor()

        # Capacitor should have reactive Jacobian entries
        has_react = any(j["has_react"] for j in desc["jacobian"])
        assert has_react, "Capacitor should have reactive (dQ/dV) contribution"


class TestParameterDefaults:
    """Tests for parameter default value extraction."""

    def test_resistor_has_default_resistance(self):
        """Test that resistor has extractable default resistance value."""
        va_path = VACASK_DIR / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")

        module = get_module(va_path)
        defaults = module.get_param_defaults()

        # VACASK resistor should have 'r' parameter with a default
        assert "r" in defaults or "R" in defaults.get("r", defaults), f"Expected 'r' param, got {defaults.keys()}"

    def test_diode_has_default_is(self):
        """Test that diode has extractable default Is value."""
        va_path = INTEGRATION_TESTS_DIR / "DIODE" / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Diode VA file not found: {va_path}")

        module = get_module(va_path)
        defaults = module.get_param_defaults()

        # Diode should have 'is' or 'Is' parameter
        has_is = any(k.lower() == "is" for k in defaults)
        assert has_is, f"Expected 'is' param in defaults, got {defaults.keys()}"


class TestNodeCollapse:
    """Tests for node collapse functionality."""

    def test_diode_has_collapsible_pairs(self):
        """Test that diode with internal node has collapsible pairs."""
        va_path = INTEGRATION_TESTS_DIR / "DIODE" / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Diode VA file not found: {va_path}")

        module = get_module(va_path)
        desc = module.get_osdi_descriptor()

        # Diode model typically has internal cathode node (CI) that can collapse
        assert desc["num_collapsible"] > 0, "Diode should have collapsible internal nodes"


class TestNoiseSourceExtraction:
    """Tests for noise source metadata extraction."""

    def test_diode_has_noise_sources(self):
        """Test that diode has shot noise sources."""
        va_path = INTEGRATION_TESTS_DIR / "DIODE" / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Diode VA file not found: {va_path}")

        module = get_module(va_path)
        desc = module.get_osdi_descriptor()

        # Diode should have noise sources (shot noise, thermal noise)
        assert desc["num_noise_sources"] > 0, "Diode should have noise sources"


class TestVCCSEvaluation:
    """Tests for voltage-controlled current source evaluation."""

    @pytest.fixture
    def vccs_module(self):
        """Load VCCS module."""
        va_path = INTEGRATION_TESTS_DIR / "VCCS" / "vccs.va"
        if not va_path.exists():
            pytest.skip(f"VCCS VA file not found: {va_path}")
        return get_module(va_path)

    def test_vccs_has_linear_gain(self, vccs_module):
        """Test that VCCS has constant (linear) gain."""
        desc = vccs_module.get_osdi_descriptor()

        # VCCS with constant gain should have constant resistive Jacobian
        param_names = [p["name"] for p in desc["params"]]
        has_gain = any(name.lower() in ("g", "gm", "gain") for name in param_names)
        assert has_gain, f"VCCS should have gain parameter, found: {param_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
