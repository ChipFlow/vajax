"""Tests for HiSIM models: HiSIM2, HiSIMHV"""

import numpy as np
import pytest
from conftest import INTEGRATION_PATH, CompiledModel


@pytest.fixture(scope="module")
def hisim2_model(compile_model) -> CompiledModel:
    """Compiled HiSIM2 model"""
    return compile_model(INTEGRATION_PATH / "HiSIM2/hisim2.va")


@pytest.fixture(scope="module")
def hisimhv_model(compile_model) -> CompiledModel:
    """Compiled HiSIMHV model"""
    return compile_model(INTEGRATION_PATH / "HiSIMHV/hisimhv.va")


class TestHiSIM2:
    """Test HiSIM2 MOSFET model"""

    def test_compilation(self, hisim2_model: CompiledModel):
        """HiSIM2 model compiles without error"""
        assert hisim2_model.module is not None
        assert 'hisim2' in hisim2_model.name.lower()
        assert len(hisim2_model.nodes) >= 4

    def test_valid_output(self, hisim2_model: CompiledModel):
        """HiSIM2 produces valid outputs"""
        inputs = hisim2_model.build_default_inputs()
        residuals, jacobian = hisim2_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_has_jacobian(self, hisim2_model: CompiledModel):
        """HiSIM2 has jacobian entries"""
        assert hisim2_model.module.num_jacobian > 0


class TestHiSIMHV:
    """Test HiSIMHV high-voltage MOSFET model"""

    def test_compilation(self, hisimhv_model: CompiledModel):
        """HiSIMHV model compiles without error"""
        assert hisimhv_model.module is not None
        assert 'hisimhv' in hisimhv_model.name.lower()
        assert len(hisimhv_model.nodes) >= 4

    def test_valid_output(self, hisimhv_model: CompiledModel):
        """HiSIMHV produces valid outputs"""
        inputs = hisimhv_model.build_default_inputs()
        residuals, jacobian = hisimhv_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_complexity(self, hisimhv_model: CompiledModel):
        """HiSIMHV is a complex model"""
        # HiSIMHV should have many parameters
        param_count = sum(
            1 for k in hisimhv_model.param_kinds if k == 'param'
        )
        assert param_count > 100, f"HiSIMHV should have many params, has {param_count}"
