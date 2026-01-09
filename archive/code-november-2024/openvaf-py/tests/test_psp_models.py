"""Tests for PSP models: PSP102, PSP103, JUNCAP200"""

import pytest
import numpy as np
from conftest import INTEGRATION_PATH, CompiledModel


@pytest.fixture(scope="module")
def psp102_model(compile_model) -> CompiledModel:
    """Compiled PSP102 model"""
    return compile_model(INTEGRATION_PATH / "PSP102/psp102.va")


@pytest.fixture(scope="module")
def psp103_model(compile_model) -> CompiledModel:
    """Compiled PSP103 model"""
    return compile_model(INTEGRATION_PATH / "PSP103/psp103.va")


@pytest.fixture(scope="module")
def juncap_model(compile_model) -> CompiledModel:
    """Compiled JUNCAP200 model"""
    return compile_model(INTEGRATION_PATH / "PSP103/juncap200.va")


class TestPSP102:
    """Test PSP102 MOSFET model"""

    def test_compilation(self, psp102_model: CompiledModel):
        """PSP102 model compiles without error"""
        assert psp102_model.module is not None
        assert 'psp102' in psp102_model.name.lower()
        assert len(psp102_model.nodes) >= 4

    def test_valid_output(self, psp102_model: CompiledModel):
        """PSP102 produces valid outputs"""
        inputs = psp102_model.build_default_inputs()
        residuals, jacobian = psp102_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_has_jacobian(self, psp102_model: CompiledModel):
        """PSP102 has jacobian entries"""
        assert psp102_model.module.num_jacobian > 0


class TestPSP103:
    """Test PSP103 MOSFET model (latest PSP version)"""

    def test_compilation(self, psp103_model: CompiledModel):
        """PSP103 model compiles without error"""
        assert psp103_model.module is not None
        assert 'psp103' in psp103_model.name.lower()
        assert len(psp103_model.nodes) >= 4

    def test_valid_output(self, psp103_model: CompiledModel):
        """PSP103 produces valid outputs"""
        inputs = psp103_model.build_default_inputs()
        residuals, jacobian = psp103_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_complexity(self, psp103_model: CompiledModel):
        """PSP103 is a complex model"""
        # PSP103 should have many hidden states
        hidden_count = sum(
            1 for k in psp103_model.param_kinds if k == 'hidden_state'
        )
        assert hidden_count > 100, f"PSP103 should be complex, has {hidden_count} hidden states"


class TestJUNCAP:
    """Test JUNCAP200 junction capacitance model"""

    def test_compilation(self, juncap_model: CompiledModel):
        """JUNCAP200 model compiles without error"""
        assert juncap_model.module is not None
        assert 'juncap' in juncap_model.name.lower()
        assert len(juncap_model.nodes) >= 2

    def test_valid_output(self, juncap_model: CompiledModel):
        """JUNCAP200 produces valid outputs"""
        inputs = juncap_model.build_default_inputs()
        residuals, jacobian = juncap_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"NaN at {node}"

    def test_is_two_terminal(self, juncap_model: CompiledModel):
        """JUNCAP is a two-terminal device"""
        assert len(juncap_model.nodes) == 2
