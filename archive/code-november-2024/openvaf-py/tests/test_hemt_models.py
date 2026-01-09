"""Tests for HEMT models: ASMHEMT, MVSG"""

import pytest
import numpy as np
from conftest import INTEGRATION_PATH, CompiledModel


@pytest.fixture(scope="module")
def asmhemt_model(compile_model) -> CompiledModel:
    """Compiled ASMHEMT model"""
    return compile_model(INTEGRATION_PATH / "ASMHEMT/asmhemt.va")


@pytest.fixture(scope="module")
def mvsg_model(compile_model) -> CompiledModel:
    """Compiled MVSG model"""
    return compile_model(INTEGRATION_PATH / "MVSG_CMC/mvsg_cmc.va")


class TestASMHEMT:
    """Test ASMHEMT GaN HEMT model"""

    def test_compilation(self, asmhemt_model: CompiledModel):
        """ASMHEMT model compiles without error"""
        assert asmhemt_model.module is not None
        assert 'asmhemt' in asmhemt_model.name.lower()
        # ASMHEMT may have more internal nodes
        assert len(asmhemt_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, asmhemt_model: CompiledModel):
        """ASMHEMT produces valid outputs"""
        inputs = asmhemt_model.build_default_inputs()
        residuals, jacobian = asmhemt_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_has_jacobian(self, asmhemt_model: CompiledModel):
        """ASMHEMT has jacobian entries"""
        assert asmhemt_model.module.num_jacobian > 0


class TestMVSG:
    """Test MVSG GaN HEMT model"""

    def test_compilation(self, mvsg_model: CompiledModel):
        """MVSG model compiles without error"""
        assert mvsg_model.module is not None
        assert 'mvsg' in mvsg_model.name.lower()
        # MVSG may have more internal nodes
        assert len(mvsg_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, mvsg_model: CompiledModel):
        """MVSG produces valid outputs"""
        inputs = mvsg_model.build_default_inputs()
        residuals, jacobian = mvsg_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_has_jacobian(self, mvsg_model: CompiledModel):
        """MVSG has jacobian entries"""
        assert mvsg_model.module.num_jacobian > 0


class TestHEMTBehavior:
    """Test physical behavior of HEMT models"""

    @pytest.mark.parametrize("fixture_name", ["asmhemt_model", "mvsg_model"])
    def test_has_multiple_nodes(self, request, fixture_name):
        """HEMT model has multiple terminal nodes"""
        model = request.getfixturevalue(fixture_name)
        # HEMTs should have at least 4 nodes (d, g, s, substrate)
        assert len(model.nodes) >= 4, f"{model.name} should have at least 4 nodes"
