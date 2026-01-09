"""Tests for BJT models: HICUM, MEXTRAM"""

import pytest
import numpy as np
from conftest import INTEGRATION_PATH, CompiledModel


@pytest.fixture(scope="module")
def hicum_model(compile_model) -> CompiledModel:
    """Compiled HICUM L2 model"""
    return compile_model(INTEGRATION_PATH / "HICUML2/hicuml2.va")


@pytest.fixture(scope="module")
def mextram_model(compile_model) -> CompiledModel:
    """Compiled MEXTRAM model"""
    return compile_model(INTEGRATION_PATH / "MEXTRAM/mextram.va")


class TestHICUM:
    """Test HICUM Level 2 BJT model"""

    def test_compilation(self, hicum_model: CompiledModel):
        """HICUM model compiles without error"""
        assert hicum_model.module is not None
        assert 'hicum' in hicum_model.name.lower()
        assert len(hicum_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, hicum_model: CompiledModel):
        """HICUM produces valid outputs"""
        inputs = hicum_model.build_default_inputs()
        residuals, jacobian = hicum_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_has_jacobian(self, hicum_model: CompiledModel):
        """HICUM has jacobian entries"""
        assert hicum_model.module.num_jacobian > 0

    def test_complexity(self, hicum_model: CompiledModel):
        """HICUM is a complex model"""
        hidden_count = sum(
            1 for k in hicum_model.param_kinds if k == 'hidden_state'
        )
        assert hidden_count > 50, f"HICUM should be complex, has {hidden_count} hidden states"


class TestMEXTRAM:
    """Test MEXTRAM BJT model"""

    def test_compilation(self, mextram_model: CompiledModel):
        """MEXTRAM model compiles without error"""
        assert mextram_model.module is not None
        # MEXTRAM is also known as bjt505t
        assert len(mextram_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, mextram_model: CompiledModel):
        """MEXTRAM produces valid outputs"""
        inputs = mextram_model.build_default_inputs()
        residuals, jacobian = mextram_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_has_jacobian(self, mextram_model: CompiledModel):
        """MEXTRAM has jacobian entries"""
        assert mextram_model.module.num_jacobian > 0

    def test_complexity(self, mextram_model: CompiledModel):
        """MEXTRAM is a complex model"""
        hidden_count = sum(
            1 for k in mextram_model.param_kinds if k == 'hidden_state'
        )
        assert hidden_count > 50, f"MEXTRAM should be complex, has {hidden_count} hidden states"


class TestBJTBehavior:
    """Test physical behavior of BJT models"""

    @pytest.mark.parametrize("fixture_name", ["hicum_model", "mextram_model"])
    def test_has_multiple_nodes(self, request, fixture_name):
        """BJT model has multiple terminal nodes"""
        model = request.getfixturevalue(fixture_name)
        # BJTs should have at least 4 nodes (c, b, e, substrate or thermal)
        assert len(model.nodes) >= 4, f"{model.name} should have at least 4 nodes"
