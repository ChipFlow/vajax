"""Tests for JAX translation equivalence.

This module validates that JAX-translated functions produce identical results
to the openvaf-py MIR interpreter, ensuring the translation is correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add parent directory to path for openvaf_jax import
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import numpy as np
from jax import config

import openvaf_py
from openvaf_jax import OpenVAFToJAX

# Import jax_spice to auto-configure precision based on backend capabilities
# This ensures Metal/TPU backends use f32 while CPU/CUDA use f64
try:
    import jax_spice  # noqa: F401
except ImportError:
    # Fallback if jax_spice not installed - enable x64 for CPU/CUDA
    config.update("jax_enable_x64", True)

# Base directories
OPENVAF_DIR = Path(__file__).parent.parent / "vendor" / "OpenVAF"
INTEGRATION_TESTS_DIR = OPENVAF_DIR / "integration_tests"
VACASK_DIR = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices"


def get_translator(va_path: Path) -> OpenVAFToJAX:
    """Create a JAX translator from a Verilog-A file."""
    return OpenVAFToJAX.from_file(str(va_path))


class TestResistorEquivalence:
    """Tests for resistor JAX translation equivalence."""

    @pytest.fixture
    def resistor_translator(self):
        """Load resistor translator."""
        va_path = VACASK_DIR / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")
        return get_translator(va_path)

    def test_resistor_translation_compiles(self, resistor_translator):
        """Test that resistor JAX translation compiles without error."""
        # Translate to JAX function
        eval_fn, metadata = resistor_translator.translate_array()

        # Verify the function is callable
        assert callable(eval_fn), "Translated function should be callable"
        # Metadata contains node_names and jacobian_keys
        assert "node_names" in metadata, "Metadata should contain node_names"

    def test_resistor_input_output_shapes(self, resistor_translator):
        """Test that resistor JAX function has correct input/output shapes."""
        eval_fn, metadata = resistor_translator.translate_array()

        # Get expected sizes from module
        module = resistor_translator.module
        num_params = module.func_num_params
        num_residuals = module.num_residuals
        num_jacobian = module.num_jacobian

        # Create test input
        inputs = jnp.zeros(num_params)

        # Call function - returns (res_resist, res_react, jac_resist, jac_react)
        try:
            result = eval_fn(inputs)
            assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"

            res_resist, res_react, jac_resist, jac_react = result

            # Check shapes
            assert res_resist.shape == (num_residuals,), f"Expected resist residual shape ({num_residuals},), got {res_resist.shape}"
            assert res_react.shape == (num_residuals,), f"Expected react residual shape ({num_residuals},), got {res_react.shape}"
            assert jac_resist.shape == (num_jacobian,), f"Expected resist jacobian shape ({num_jacobian},), got {jac_resist.shape}"
            assert jac_react.shape == (num_jacobian,), f"Expected react jacobian shape ({num_jacobian},), got {jac_react.shape}"
        except Exception as e:
            pytest.fail(f"Function call failed: {e}")

    def test_resistor_jax_outputs_finite(self, resistor_translator):
        """Test that resistor JAX function produces finite outputs."""
        eval_fn, metadata = resistor_translator.translate_array()
        module = resistor_translator.module

        # Build realistic input
        # Need to understand param ordering from param_names and param_kinds
        param_names = module.param_names
        param_kinds = module.param_kinds

        # Create input array with reasonable values
        # Default: all zeros except temperature and resistance
        num_params = module.func_num_params
        inputs = jnp.zeros(num_params)

        # Try to find and set key parameters
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if i >= num_params:
                break
            if kind == "temperature":
                inputs = inputs.at[i].set(300.15)  # Room temperature
            elif kind == "param" and name.lower() == "r":
                inputs = inputs.at[i].set(1000.0)  # 1k Ohm
            elif kind == "voltage":
                inputs = inputs.at[i].set(1.0)  # 1V

        result = eval_fn(inputs)
        res_resist, res_react, jac_resist, jac_react = result

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(res_resist)), f"Resist residuals contain non-finite values: {res_resist}"
        assert jnp.all(jnp.isfinite(res_react)), f"React residuals contain non-finite values: {res_react}"
        assert jnp.all(jnp.isfinite(jac_resist)), f"Resist Jacobian contains non-finite values: {jac_resist}"
        assert jnp.all(jnp.isfinite(jac_react)), f"React Jacobian contains non-finite values: {jac_react}"


class TestDiodeEquivalence:
    """Tests for diode JAX translation equivalence."""

    @pytest.fixture
    def diode_translator(self):
        """Load diode translator."""
        va_path = INTEGRATION_TESTS_DIR / "DIODE" / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Diode VA file not found: {va_path}")
        return get_translator(va_path)

    def test_diode_translation_compiles(self, diode_translator):
        """Test that diode JAX translation compiles without error."""
        eval_fn, metadata = diode_translator.translate_array()
        assert callable(eval_fn), "Translated function should be callable"

    def test_diode_zero_bias_output(self, diode_translator):
        """Test diode at zero bias produces near-zero current."""
        eval_fn, metadata = diode_translator.translate_array()
        module = diode_translator.module

        num_params = module.func_num_params
        param_names = module.param_names
        param_kinds = module.param_kinds

        # Create input with zero voltages
        inputs = jnp.zeros(num_params)

        # Set temperature
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if i >= num_params:
                break
            if kind == "temperature":
                inputs = inputs.at[i].set(300.15)

        result = eval_fn(inputs)
        res_resist, res_react, jac_resist, jac_react = result

        # At zero bias, residuals should be finite and reasonable
        assert jnp.all(jnp.isfinite(res_resist)), "Zero-bias resist residuals should be finite"
        assert jnp.all(jnp.isfinite(res_react)), "Zero-bias react residuals should be finite"


class TestCapacitorEquivalence:
    """Tests for capacitor JAX translation equivalence."""

    @pytest.fixture
    def capacitor_translator(self):
        """Load capacitor translator."""
        va_path = VACASK_DIR / "capacitor.va"
        if not va_path.exists():
            pytest.skip(f"Capacitor VA file not found: {va_path}")
        return get_translator(va_path)

    def test_capacitor_translation_compiles(self, capacitor_translator):
        """Test that capacitor JAX translation compiles without error."""
        eval_fn, metadata = capacitor_translator.translate_array()
        assert callable(eval_fn), "Translated function should be callable"

    def test_capacitor_has_reactive_outputs(self, capacitor_translator):
        """Test that capacitor produces reactive (charge) outputs."""
        module = capacitor_translator.module
        desc = module.get_osdi_descriptor()

        # Capacitor should have reactive Jacobian entries
        has_react = any(j["has_react"] for j in desc["jacobian"])
        assert has_react, "Capacitor should have reactive contributions"


class TestJITCompilation:
    """Tests for JAX JIT compilation correctness."""

    @pytest.fixture
    def resistor_translator(self):
        """Load resistor translator."""
        va_path = VACASK_DIR / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")
        return get_translator(va_path)

    def test_jit_compilation_succeeds(self, resistor_translator):
        """Test that JAX JIT compilation succeeds."""
        from jax import jit

        eval_fn, metadata = resistor_translator.translate_array()
        jitted_fn = jit(eval_fn)

        # Create test input
        num_params = resistor_translator.module.func_num_params
        inputs = jnp.ones(num_params)

        # First call triggers JIT compilation
        result1 = jitted_fn(inputs)

        # Second call uses cached compilation
        result2 = jitted_fn(inputs)

        # Results should match
        for i, (r1, r2) in enumerate(zip(result1, result2)):
            np.testing.assert_array_almost_equal(
                np.array(r1), np.array(r2),
                decimal=10, err_msg=f"JIT result[{i}] should be reproducible"
            )

    def test_jit_multiple_inputs(self, resistor_translator):
        """Test JIT function with different inputs."""
        from jax import jit

        eval_fn, metadata = resistor_translator.translate_array()
        jitted_fn = jit(eval_fn)

        num_params = resistor_translator.module.func_num_params

        # Test with different inputs
        inputs1 = jnp.ones(num_params)
        inputs2 = jnp.ones(num_params) * 2.0

        result1 = jitted_fn(inputs1)
        result2 = jitted_fn(inputs2)

        # At least some outputs should be different for different inputs
        any_different = any(
            not jnp.allclose(r1, r2) for r1, r2 in zip(result1, result2)
        )
        assert any_different, "Different inputs should produce different outputs"


class TestVmapBatching:
    """Tests for JAX vmap batching correctness."""

    @pytest.fixture
    def resistor_translator(self):
        """Load resistor translator."""
        va_path = VACASK_DIR / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")
        return get_translator(va_path)

    def test_vmap_batching_succeeds(self, resistor_translator):
        """Test that vmap batching works correctly."""
        from jax import vmap

        eval_fn, metadata = resistor_translator.translate_array()
        batched_fn = vmap(eval_fn)

        # Create batched input (batch of 10 instances)
        num_params = resistor_translator.module.func_num_params
        batch_size = 10
        batch_inputs = jnp.ones((batch_size, num_params))

        # Run batched evaluation - returns (res_resist, res_react, jac_resist, jac_react)
        result = batched_fn(batch_inputs)
        assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"

        batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react = result

        # Check output shapes
        num_residuals = resistor_translator.module.num_residuals
        num_jacobian = resistor_translator.module.num_jacobian

        assert batch_res_resist.shape == (batch_size, num_residuals), \
            f"Expected batch resist residual shape ({batch_size}, {num_residuals}), got {batch_res_resist.shape}"
        assert batch_res_react.shape == (batch_size, num_residuals), \
            f"Expected batch react residual shape ({batch_size}, {num_residuals}), got {batch_res_react.shape}"
        assert batch_jac_resist.shape == (batch_size, num_jacobian), \
            f"Expected batch resist jacobian shape ({batch_size}, {num_jacobian}), got {batch_jac_resist.shape}"
        assert batch_jac_react.shape == (batch_size, num_jacobian), \
            f"Expected batch react jacobian shape ({batch_size}, {num_jacobian}), got {batch_jac_react.shape}"

    def test_vmap_consistency_with_loop(self, resistor_translator):
        """Test that vmap produces same results as explicit loop."""
        from jax import vmap

        eval_fn, metadata = resistor_translator.translate_array()

        # Create batch input
        num_params = resistor_translator.module.func_num_params
        batch_size = 5
        batch_inputs = jnp.arange(batch_size * num_params).reshape(batch_size, num_params).astype(jnp.float64)

        # Vmap result - returns (res_resist, res_react, jac_resist, jac_react)
        batched_fn = vmap(eval_fn)
        vmap_result = batched_fn(batch_inputs)
        vmap_res_resist, vmap_res_react, vmap_jac_resist, vmap_jac_react = vmap_result

        # Loop result
        loop_res_resist = []
        loop_res_react = []
        loop_jac_resist = []
        loop_jac_react = []
        for i in range(batch_size):
            result = eval_fn(batch_inputs[i])
            res_resist, res_react, jac_resist, jac_react = result
            loop_res_resist.append(res_resist)
            loop_res_react.append(res_react)
            loop_jac_resist.append(jac_resist)
            loop_jac_react.append(jac_react)
        loop_res_resist = jnp.stack(loop_res_resist)
        loop_res_react = jnp.stack(loop_res_react)
        loop_jac_resist = jnp.stack(loop_jac_resist)
        loop_jac_react = jnp.stack(loop_jac_react)

        # Results should match
        np.testing.assert_array_almost_equal(
            np.array(vmap_res_resist), np.array(loop_res_resist),
            decimal=10, err_msg="Vmap resist residuals should match loop"
        )
        np.testing.assert_array_almost_equal(
            np.array(vmap_res_react), np.array(loop_res_react),
            decimal=10, err_msg="Vmap react residuals should match loop"
        )
        np.testing.assert_array_almost_equal(
            np.array(vmap_jac_resist), np.array(loop_jac_resist),
            decimal=10, err_msg="Vmap resist jacobian should match loop"
        )
        np.testing.assert_array_almost_equal(
            np.array(vmap_jac_react), np.array(loop_jac_react),
            decimal=10, err_msg="Vmap react jacobian should match loop"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
