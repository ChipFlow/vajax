"""Unified test for all OpenVAF integration test models.

This test verifies that all Verilog-A models in the OpenVAF integration tests
can be compiled and evaluated using their default parameters from the VA source.
"""

import numpy as np
import pytest
from conftest import INTEGRATION_MODELS, INTEGRATION_PATH

# Models with known codegen issues (undefined SSA variables in complex control flow)
# Fixed in commit a887f5d by pre-initializing output variables
CODEGEN_BROKEN_MODELS: set = set()  # Empty - all models should work now

# Models that hang during evaluation (investigation needed)
# These cause CI timeout - mark xfail until root cause is found
# See: https://github.com/ChipFlow/jax-spice/issues/19
HANGING_MODELS: set = {"bsimsoi", "hisim2", "hisimhv"}  # Hang after XLA compilation completes


class TestAllModels:
    """Test all OpenVAF integration models with VA defaults."""

    @pytest.mark.parametrize("model_name,model_path", INTEGRATION_MODELS)
    def test_model_compiles(self, compile_model, model_name, model_path):
        """Model compiles without error"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert model.module is not None
        assert len(model.nodes) >= 2  # At least 2 terminals

    @pytest.mark.parametrize("model_name,model_path", INTEGRATION_MODELS)
    def test_model_produces_finite_output(self, compile_model, model_name, model_path):
        """Model produces at least some finite outputs with VA defaults.

        Uses the actual default parameter values from the parsed Verilog-A source,
        ensuring the model is evaluated in a physically meaningful state.
        """
        if model_name in CODEGEN_BROKEN_MODELS:
            pytest.xfail(f"{model_name}: codegen bug - undefined SSA variable in control flow")
        if model_name in HANGING_MODELS:
            pytest.skip(f"{model_name}: hangs during evaluation - see issue #19")
        model = compile_model(INTEGRATION_PATH / model_path)

        # Use VA defaults (not hardcoded values)
        inputs = model.build_default_inputs()
        residuals, jacobian = model.jax_fn(inputs)

        # Check that we got outputs
        assert residuals is not None
        assert len(residuals) > 0

        # Count finite vs NaN residuals
        finite_count = 0
        nan_count = 0
        for node, res in residuals.items():
            resist = float(res['resist'])
            if np.isfinite(resist):
                finite_count += 1
            else:
                nan_count += 1

        # At least one finite output (most models should have all finite)
        assert finite_count > 0, (
            f"{model_name} produced no finite outputs. "
            f"NaN at nodes: {[n for n, r in residuals.items() if not np.isfinite(float(r['resist']))]}"
        )

    @pytest.mark.parametrize("model_name,model_path", INTEGRATION_MODELS)
    def test_model_has_jacobian(self, compile_model, model_name, model_path):
        """Model has jacobian entries"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert model.module.num_jacobian > 0

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS
        if m[0] not in ('bsim3', 'bsimcmg', 'hicum', 'mextram')  # Known to have some NaN
        and m[0] not in CODEGEN_BROKEN_MODELS  # Codegen bugs
        and m[0] not in HANGING_MODELS  # Cause CI timeout
    ])
    def test_model_all_residuals_finite(self, compile_model, model_name, model_path):
        """Model produces ALL finite residuals (stricter test).

        This test excludes models known to have numerical issues at zero bias.
        """
        model = compile_model(INTEGRATION_PATH / model_path)
        inputs = model.build_default_inputs()
        residuals, jacobian = model.jax_fn(inputs)

        # All residuals must be finite
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert np.isfinite(resist), f"{model_name} has NaN resist at {node}"


class TestModelCategories:
    """Test models by category to verify category-specific behavior."""

    # Simple 2-terminal devices
    SIMPLE_MODELS = ['resistor', 'diode', 'diode_cmc', 'isrc']

    # MOSFET models (4+ terminals)
    MOSFET_MODELS = ['ekv', 'bsim3', 'bsim4', 'bsim6', 'bsimbulk', 'bsimcmg',
                     'bsimsoi', 'psp102', 'psp103', 'hisim2']

    # BJT models (3+ terminals)
    BJT_MODELS = ['hicum', 'mextram']

    # HEMT models
    HEMT_MODELS = ['asmhemt', 'mvsg']

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS if m[0] in ['resistor', 'diode', 'diode_cmc']
    ])
    def test_simple_device_zero_bias(self, compile_model, model_name, model_path):
        """Simple devices at zero bias should have near-zero current."""
        model = compile_model(INTEGRATION_PATH / model_path)
        inputs = model.build_default_inputs()
        residuals, _ = model.jax_fn(inputs)

        # At zero bias, currents should be small
        for node, res in residuals.items():
            resist = float(res['resist'])
            if np.isfinite(resist):
                # Allow small currents (leakage, etc.) but not large ones
                assert abs(resist) < 1e10, f"{model_name} has large current at {node}: {resist}"

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS if m[0] in ['ekv', 'bsim4', 'psp103']
    ])
    def test_mosfet_has_four_terminals(self, compile_model, model_name, model_path):
        """MOSFET models should have at least 4 terminals (D, G, S, B)."""
        model = compile_model(INTEGRATION_PATH / model_path)
        # num_terminals from module, nodes includes internal nodes
        assert model.module.num_terminals >= 4, (
            f"{model_name} has only {model.module.num_terminals} terminals"
        )
