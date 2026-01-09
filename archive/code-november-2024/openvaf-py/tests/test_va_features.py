"""Tests for specific Verilog-A features that may be incorrectly translated.

These tests target VA features identified during the Ring benchmark investigation:
1. Noise correlation nodes with high conductance (G > 1e20)
2. Conditional contributions (CollapsableR pattern: if R>0: I<+... else V<+0)
3. Branch current access I(branch) as input
4. Internal node allocation and voltage handling

See: PSP103_module.include lines 1834-1836 for the noise branch issue.
"""

import pytest
import numpy as np
from pathlib import Path
from conftest import INTEGRATION_PATH


class TestCollapsableResistor:
    """Test conditional contributions: if (R > minr) I<+V/R else V<+0.

    This pattern is used in PSP103's CollapsableR macro and diode.va.
    When R=0, nodes should be collapsed (V=0 between them).
    """

    @pytest.fixture
    def diode_model(self, compile_model):
        """Compile diode model which has CollapsableR pattern."""
        model_path = INTEGRATION_PATH / "DIODE" / "diode.va"
        return compile_model(str(model_path))

    def test_resistive_mode(self, diode_model):
        """Test diode with rs > 0 (resistor mode, I<+V/R contribution)."""
        # With rs > minr, the CI-C branch should act as a resistor
        rs = 100.0  # 100 ohms
        voltage_ci_c = 1.0  # 1V across the resistor

        # Build inputs with rs > 0
        # We need to find the param order from the model
        param_names = diode_model.param_names
        param_kinds = diode_model.param_kinds

        # Build default inputs
        inputs = diode_model.build_default_inputs()

        # Find and set rs parameter
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if name == 'rs' and kind == 'param':
                inputs[i] = rs
            elif 'V(CI,C)' in name or 'V(br_ci_c)' in name:
                inputs[i] = voltage_ci_c

        # Evaluate
        residuals, jacobian = diode_model.jax_fn(inputs)

        # The CI node residual should include the resistor contribution
        # I = V/R = 1.0/100 = 0.01 A
        # We just check it's finite and reasonable (not ~1e40)
        for node, res in residuals.items():
            resist_val = float(res.get('resist', 0.0))
            assert np.isfinite(resist_val), f"Non-finite residual at {node}"
            assert abs(resist_val) < 1e10, f"Residual too large at {node}: {resist_val}"

    def test_collapsed_mode(self, diode_model):
        """Test diode with rs = 0 (collapsed mode, V<+0 contribution)."""
        # With rs = 0, the CI-C branch should collapse (V=0)
        rs = 0.0  # Zero resistance -> collapse

        # Build default inputs
        inputs = diode_model.build_default_inputs()
        param_names = diode_model.param_names
        param_kinds = diode_model.param_kinds

        # Set rs = 0
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if name == 'rs' and kind == 'param':
                inputs[i] = rs

        # Evaluate
        residuals, jacobian = diode_model.jax_fn(inputs)

        # All residuals should be finite and reasonable
        for node, res in residuals.items():
            resist_val = float(res.get('resist', 0.0))
            assert np.isfinite(resist_val), f"Non-finite residual at {node}"
            # In collapsed mode, residuals should NOT have 1e40-scale values
            assert abs(resist_val) < 1e20, f"Residual too large in collapsed mode at {node}: {resist_val}"


class TestPSP103NoiseCorrelationNode:
    """Test PSP103 NOI (noise correlation) internal node handling.

    PSP103 has: I(NOIR) <+ V(NOI) / mig where mig = 1e-40
    This creates a conductance G = 1e40 to ground.
    If V(NOI) is not 0, residual = V(NOI) * 1e40 which corrupts NR.
    """

    @pytest.fixture
    def psp103_model(self, compile_model):
        """Compile PSP103 model."""
        model_path = INTEGRATION_PATH / "PSP103" / "psp103.va"
        return compile_model(str(model_path))

    def test_has_noi_node(self, psp103_model):
        """PSP103 should have NOI as an internal node."""
        nodes = psp103_model.nodes
        # NOI should be node4 based on PSP103_module.include
        assert len(nodes) >= 5, "PSP103 should have at least 5 nodes"
        # The nodes list includes internal nodes
        assert 'node4' in nodes or any('noi' in n.lower() for n in nodes), \
            f"NOI node not found in {nodes}"

    def test_has_vnoi_voltage_input(self, psp103_model):
        """PSP103 should have V(NOI) as a voltage input parameter."""
        param_names = psp103_model.param_names
        param_kinds = psp103_model.param_kinds

        vnoi_found = False
        for name, kind in zip(param_names, param_kinds):
            if 'V(NOI)' in name and kind == 'voltage':
                vnoi_found = True
                break

        assert vnoi_found, f"V(NOI) voltage param not found. Voltage params: " + \
            str([n for n, k in zip(param_names, param_kinds) if k == 'voltage'])

    def test_has_branch_current_input(self, psp103_model):
        """PSP103 should have I(NOII) as a current input parameter."""
        param_names = psp103_model.param_names
        param_kinds = psp103_model.param_kinds

        inoii_found = False
        for name, kind in zip(param_names, param_kinds):
            if 'I(NOII)' in name and kind == 'current':
                inoii_found = True
                break

        assert inoii_found, f"I(NOII) current param not found. Current params: " + \
            str([n for n, k in zip(param_names, param_kinds) if k == 'current'])

    def test_noi_zero_voltage_residuals(self, psp103_model):
        """With V(NOI)=0, all residuals should be finite and reasonable."""
        # Build default inputs
        inputs = psp103_model.build_default_inputs()
        param_names = psp103_model.param_names
        param_kinds = psp103_model.param_kinds

        # Ensure V(NOI) = 0
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if 'V(NOI)' in name and kind == 'voltage':
                inputs[i] = 0.0
            # Also ensure I(NOII) = 0 for DC
            if 'I(NOII)' in name and kind == 'current':
                inputs[i] = 0.0

        # Evaluate
        residuals, jacobian = psp103_model.jax_fn(inputs)

        # All residuals should be finite and reasonable (NOT 1e40 scale)
        max_residual = 0.0
        for node, res in residuals.items():
            resist_val = float(res.get('resist', 0.0))
            max_residual = max(max_residual, abs(resist_val))
            assert np.isfinite(resist_val), f"Non-finite residual at {node}"

        # The key test: residuals should NOT be ~1e40
        assert max_residual < 1e30, \
            f"Residuals too large (max={max_residual:.2e}), likely NOI node issue"

    @pytest.mark.xfail(reason="Known issue: V(NOI) non-zero causes 1e40 residuals")
    def test_noi_nonzero_voltage_stability(self, psp103_model):
        """With V(NOI)=0.6V, residuals should still be bounded.

        This test is expected to fail until the issue is fixed.
        The I(NOIR) <+ V(NOI)/mig contribution with mig=1e-40 creates
        residual = 0.6 * 1e40 = 6e39 which corrupts NR.
        """
        inputs = psp103_model.build_default_inputs()
        param_names = psp103_model.param_names
        param_kinds = psp103_model.param_kinds

        # Set V(NOI) = 0.6V (the problematic value from Ring benchmark)
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if 'V(NOI)' in name and kind == 'voltage':
                inputs[i] = 0.6

        # Evaluate
        residuals, jacobian = psp103_model.jax_fn(inputs)

        # Check residuals - this will fail with current implementation
        max_residual = 0.0
        for node, res in residuals.items():
            resist_val = float(res.get('resist', 0.0))
            max_residual = max(max_residual, abs(resist_val))

        # This assertion will fail: we expect ~6e39, want < 1e30
        assert max_residual < 1e30, \
            f"V(NOI)=0.6 causes residual {max_residual:.2e} (expected ~6e39 bug)"


class TestCollapsiblePairs:
    """Test node collapse information from OpenVAF."""

    @pytest.fixture
    def psp103_module(self):
        """Get raw PSP103 module."""
        import openvaf_py
        model_path = INTEGRATION_PATH / "PSP103" / "psp103.va"
        modules = openvaf_py.compile_va(str(model_path))
        return modules[0]

    def test_has_collapsible_pairs(self, psp103_module):
        """PSP103 should report collapsible node pairs."""
        pairs = psp103_module.collapsible_pairs
        n_collapsible = psp103_module.num_collapsible

        assert n_collapsible > 0, "PSP103 should have collapsible pairs"
        assert len(pairs) == n_collapsible

        # Expected pairs from CollapsableR macro:
        # G-GP, S-SI, D-DI, BP-BI, BS-BI, BD-BI, B-BI
        assert n_collapsible >= 7, f"Expected >= 7 collapsible pairs, got {n_collapsible}"

    def test_noi_not_collapsible(self, psp103_module):
        """NOI node should NOT be in collapsible pairs."""
        pairs = psp103_module.collapsible_pairs
        nodes = list(psp103_module.nodes)

        # Find NOI node index (should be node4)
        noi_idx = None
        for i, node in enumerate(nodes):
            if 'node4' in node or 'noi' in node.lower():
                noi_idx = i
                break

        if noi_idx is not None:
            # NOI should not appear in any collapsible pair
            for pair in pairs:
                assert noi_idx not in pair, \
                    f"NOI (node {noi_idx}) should not be collapsible, found in {pair}"


class TestLargeConductance:
    """Test numerical stability with large conductance values (> 1e20)."""

    @pytest.fixture
    def resistor_model(self, compile_model):
        """Compile resistor model."""
        model_path = INTEGRATION_PATH / "RESISTOR" / "resistor.va"
        return compile_model(str(model_path))

    def test_small_resistance_large_conductance(self, resistor_model):
        """Test resistor with very small R (large G = 1/R)."""
        # R = 1e-10 -> G = 1e10
        small_r = 1e-10
        voltage = 1.0

        inputs = resistor_model.build_default_inputs()
        param_names = resistor_model.param_names
        param_kinds = resistor_model.param_kinds

        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if name == 'R' and kind == 'param':
                inputs[i] = small_r
            elif 'V(A,B)' in name:
                inputs[i] = voltage

        residuals, jacobian = resistor_model.jax_fn(inputs)

        # Check residual is I = V/R = 1e10
        for node, res in residuals.items():
            resist_val = float(res.get('resist', 0.0))
            # Should be ±1e10
            if abs(resist_val) > 1e5:  # Skip small values
                expected = voltage / small_r
                assert abs(resist_val) < 2 * expected, \
                    f"Large conductance residual wrong: {resist_val} vs expected {expected}"


class TestInternalNodeAllocation:
    """Test that internal nodes are properly allocated in models."""

    @pytest.fixture
    def diode_module(self):
        """Get raw diode module."""
        import openvaf_py
        model_path = INTEGRATION_PATH / "DIODE" / "diode.va"
        modules = openvaf_py.compile_va(str(model_path))
        return modules[0]

    def test_diode_has_internal_node(self, diode_module):
        """Diode should have CI as internal node."""
        nodes = list(diode_module.nodes)
        # Diode has A (anode), C (cathode), CI (internal cathode), dT (thermal)
        assert len(nodes) >= 3, f"Diode should have >= 3 nodes, got {nodes}"

    def test_diode_residual_count_matches_nodes(self, diode_module):
        """Number of residuals should match number of nodes."""
        n_nodes = len(diode_module.nodes)
        n_residuals = diode_module.num_residuals

        # Residuals = one per node (KCL for each)
        assert n_residuals == n_nodes, \
            f"Residual count ({n_residuals}) != node count ({n_nodes})"
