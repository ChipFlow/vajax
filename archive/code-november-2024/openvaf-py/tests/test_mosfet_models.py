"""Tests for MOSFET models: EKV, BSIM3, BSIM4, BSIM6, BSIMBULK, BSIMCMG, BSIMSOI"""

import pytest
import numpy as np
from conftest import INTEGRATION_PATH, CompiledModel


@pytest.fixture(scope="module")
def ekv_model(compile_model) -> CompiledModel:
    """Compiled EKV model"""
    return compile_model(INTEGRATION_PATH / "EKV/ekv.va")


@pytest.fixture(scope="module")
def bsim3_model(compile_model) -> CompiledModel:
    """Compiled BSIM3 model"""
    return compile_model(INTEGRATION_PATH / "BSIM3/bsim3.va")


@pytest.fixture(scope="module")
def bsim4_model(compile_model) -> CompiledModel:
    """Compiled BSIM4 model"""
    return compile_model(INTEGRATION_PATH / "BSIM4/bsim4.va")


@pytest.fixture(scope="module")
def bsim6_model(compile_model) -> CompiledModel:
    """Compiled BSIM6 model"""
    return compile_model(INTEGRATION_PATH / "BSIM6/bsim6.va")


@pytest.fixture(scope="module")
def bsimbulk_model(compile_model) -> CompiledModel:
    """Compiled BSIMBULK model"""
    return compile_model(INTEGRATION_PATH / "BSIMBULK/bsimbulk.va")


@pytest.fixture(scope="module")
def bsimcmg_model(compile_model) -> CompiledModel:
    """Compiled BSIMCMG model"""
    return compile_model(INTEGRATION_PATH / "BSIMCMG/bsimcmg.va")


@pytest.fixture(scope="module")
def bsimsoi_model(compile_model) -> CompiledModel:
    """Compiled BSIMSOI model"""
    return compile_model(INTEGRATION_PATH / "BSIMSOI/bsimsoi.va")


class TestEKV:
    """Test EKV MOSFET model"""

    def test_compilation(self, ekv_model: CompiledModel):
        """EKV model compiles without error"""
        assert ekv_model.module is not None
        assert 'ekv' in ekv_model.name.lower()
        assert len(ekv_model.nodes) >= 4

    def test_valid_output(self, ekv_model: CompiledModel):
        """EKV produces valid outputs"""
        inputs = ekv_model.build_default_inputs()
        residuals, jacobian = ekv_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            assert not np.isnan(float(res['resist'])), f"NaN at {node}"


class TestBSIM3:
    """Test BSIM3 MOSFET model"""

    def test_compilation(self, bsim3_model: CompiledModel):
        """BSIM3 model compiles without error"""
        assert bsim3_model.module is not None
        assert 'bsim3' in bsim3_model.name.lower()
        assert len(bsim3_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, bsim3_model: CompiledModel):
        """BSIM3 produces valid outputs"""
        inputs = bsim3_model.build_default_inputs()
        residuals, jacobian = bsim3_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            assert not np.isnan(float(res['resist'])), f"NaN at {node}"

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_jacobian_valid(self, bsim3_model: CompiledModel):
        """BSIM3 jacobian is valid"""
        inputs = bsim3_model.build_default_inputs()
        residuals, jacobian = bsim3_model.jax_fn(inputs)

        assert jacobian is not None
        assert len(jacobian) > 0


class TestBSIM4:
    """Test BSIM4 MOSFET model"""

    def test_compilation(self, bsim4_model: CompiledModel):
        """BSIM4 model compiles without error"""
        assert bsim4_model.module is not None
        assert 'bsim4' in bsim4_model.name.lower()
        assert len(bsim4_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, bsim4_model: CompiledModel):
        """BSIM4 produces valid outputs"""
        inputs = bsim4_model.build_default_inputs()
        residuals, jacobian = bsim4_model.jax_fn(inputs)

        assert residuals is not None
        # Check at least some outputs are finite
        finite_count = sum(
            1 for res in residuals.values()
            if np.isfinite(float(res['resist']))
        )
        assert finite_count > 0, "BSIM4 produced no finite outputs"

    def test_has_many_jacobian_entries(self, bsim4_model: CompiledModel):
        """BSIM4 has substantial jacobian (complex model)"""
        assert bsim4_model.module.num_jacobian > 50


class TestBSIM6:
    """Test BSIM6 MOSFET model"""

    def test_compilation(self, bsim6_model: CompiledModel):
        """BSIM6 model compiles without error"""
        assert bsim6_model.module is not None
        assert 'bsim6' in bsim6_model.name.lower()
        assert len(bsim6_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, bsim6_model: CompiledModel):
        """BSIM6 produces valid outputs"""
        inputs = bsim6_model.build_default_inputs()
        residuals, jacobian = bsim6_model.jax_fn(inputs)

        assert residuals is not None


class TestBSIMBULK:
    """Test BSIMBULK MOSFET model"""

    def test_compilation(self, bsimbulk_model: CompiledModel):
        """BSIMBULK model compiles without error"""
        assert bsimbulk_model.module is not None
        assert 'bsimbulk' in bsimbulk_model.name.lower()
        assert len(bsimbulk_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, bsimbulk_model: CompiledModel):
        """BSIMBULK produces valid outputs"""
        inputs = bsimbulk_model.build_default_inputs()
        residuals, jacobian = bsimbulk_model.jax_fn(inputs)

        assert residuals is not None


class TestBSIMCMG:
    """Test BSIMCMG FinFET model"""

    def test_compilation(self, bsimcmg_model: CompiledModel):
        """BSIMCMG model compiles without error"""
        assert bsimcmg_model.module is not None
        assert 'bsimcmg' in bsimcmg_model.name.lower()
        assert len(bsimcmg_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, bsimcmg_model: CompiledModel):
        """BSIMCMG produces valid outputs"""
        inputs = bsimcmg_model.build_default_inputs()
        residuals, jacobian = bsimcmg_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            assert not np.isnan(float(res['resist'])), f"NaN at {node}"


class TestBSIMSOI:
    """Test BSIMSOI SOI MOSFET model"""

    def test_compilation(self, bsimsoi_model: CompiledModel):
        """BSIMSOI model compiles without error"""
        assert bsimsoi_model.module is not None
        assert 'bsimsoi' in bsimsoi_model.name.lower()
        assert len(bsimsoi_model.nodes) >= 4

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_valid_output(self, bsimsoi_model: CompiledModel):
        """BSIMSOI produces valid outputs"""
        inputs = bsimsoi_model.build_default_inputs()
        residuals, jacobian = bsimsoi_model.jax_fn(inputs)

        assert residuals is not None


class TestMOSFETBehavior:
    """Test physical behavior of MOSFET models"""

    @pytest.mark.parametrize("fixture_name", [
        "ekv_model",
        "bsim3_model",
    ])
    def test_has_multiple_nodes(self, request, fixture_name):
        """MOSFET model has multiple terminal nodes"""
        model = request.getfixturevalue(fixture_name)
        # MOSFETs should have at least 4 nodes (d, g, s, b)
        assert len(model.nodes) >= 4, f"{model.name} should have at least 4 nodes"
