"""Tests for AST-based code generation equivalence.

This module validates that AST-generated functions produce identical results
to string-based generated functions for the same inputs.
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
try:
    import jax_spice  # noqa: F401
except ImportError:
    # Fallback if jax_spice not installed - enable x64 for CPU/CUDA
    config.update("jax_enable_x64", True)

# Base directories
OPENVAF_DIR = Path(__file__).parent.parent / "vendor" / "OpenVAF"
INTEGRATION_TESTS_DIR = OPENVAF_DIR / "integration_tests"


def get_translator(va_path: Path) -> OpenVAFToJAX:
    """Create a JAX translator from a Verilog-A file."""
    return OpenVAFToJAX.from_file(str(va_path))


class TestASTvsStringGeneration:
    """Tests comparing AST-based and string-based code generation."""

    @pytest.fixture
    def resistor_translator(self):
        """Load resistor translator."""
        va_path = INTEGRATION_TESTS_DIR / "RESISTOR" / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")
        return get_translator(va_path)

    @pytest.fixture
    def diode_translator(self):
        """Load diode translator."""
        va_path = INTEGRATION_TESTS_DIR / "DIODE" / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Diode VA file not found: {va_path}")
        return get_translator(va_path)

    @pytest.fixture
    def cccs_translator(self):
        """Load CCCS translator."""
        va_path = INTEGRATION_TESTS_DIR / "CCCS" / "cccs.va"
        if not va_path.exists():
            pytest.skip(f"CCCS VA file not found: {va_path}")
        return get_translator(va_path)

    @pytest.fixture
    def vccs_translator(self):
        """Load VCCS translator."""
        va_path = INTEGRATION_TESTS_DIR / "VCCS" / "vccs.va"
        if not va_path.exists():
            pytest.skip(f"VCCS VA file not found: {va_path}")
        return get_translator(va_path)

    @pytest.fixture
    def isrc_translator(self):
        """Load current source translator."""
        va_path = INTEGRATION_TESTS_DIR / "CURRENT_SOURCE" / "current_source.va"
        if not va_path.exists():
            pytest.skip(f"Current source VA file not found: {va_path}")
        return get_translator(va_path)

    def _compare_outputs(self, translator: OpenVAFToJAX, test_name: str):
        """Compare string-based and AST-based generation outputs.

        Args:
            translator: The OpenVAFToJAX translator
            test_name: Name for error messages
        """
        # Generate using string-based method
        string_fn, string_metadata = translator.translate_array()

        # Generate using AST-based method
        ast_fn, ast_metadata = translator.translate_array_ast()

        # Create test inputs
        module = translator.module
        num_params = module.func_num_params
        param_names = module.param_names
        param_kinds = module.param_kinds

        # Build a realistic input array
        inputs = jnp.zeros(num_params)
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if i >= num_params:
                break
            if kind == "temperature":
                inputs = inputs.at[i].set(300.15)
            elif kind == "param":
                if "r" in name.lower() or "resistance" in name.lower():
                    inputs = inputs.at[i].set(1000.0)
                elif "is" in name.lower():  # Saturation current
                    inputs = inputs.at[i].set(1e-14)
                elif "n" == name.lower():  # Ideality factor
                    inputs = inputs.at[i].set(1.0)
                else:
                    inputs = inputs.at[i].set(1.0)
            elif kind == "voltage":
                inputs = inputs.at[i].set(0.5)

        # Run both functions
        string_result = string_fn(inputs)
        ast_result = ast_fn(inputs)

        # Compare outputs
        assert len(string_result) == len(ast_result), (
            f"{test_name}: Output tuple length mismatch: "
            f"string={len(string_result)}, ast={len(ast_result)}"
        )

        output_names = ["res_resist", "res_react", "jac_resist", "jac_react"]
        for i, (s_out, a_out, name) in enumerate(zip(string_result, ast_result, output_names)):
            # Check shapes match
            assert s_out.shape == a_out.shape, (
                f"{test_name} {name}: Shape mismatch: "
                f"string={s_out.shape}, ast={a_out.shape}"
            )

            # Check values match
            np.testing.assert_allclose(
                np.array(s_out), np.array(a_out),
                rtol=1e-10, atol=1e-15,
                err_msg=f"{test_name} {name}: Values don't match"
            )

    def test_resistor_equivalence(self, resistor_translator):
        """Test resistor AST generation matches string generation."""
        self._compare_outputs(resistor_translator, "Resistor")

    def test_diode_equivalence(self, diode_translator):
        """Test diode AST generation matches string generation."""
        self._compare_outputs(diode_translator, "Diode")

    def test_cccs_equivalence(self, cccs_translator):
        """Test CCCS AST generation matches string generation."""
        self._compare_outputs(cccs_translator, "CCCS")

    def test_vccs_equivalence(self, vccs_translator):
        """Test VCCS AST generation matches string generation."""
        self._compare_outputs(vccs_translator, "VCCS")

    def test_isrc_equivalence(self, isrc_translator):
        """Test current source AST generation matches string generation."""
        self._compare_outputs(isrc_translator, "CurrentSource")


class TestASTGeneratedCode:
    """Tests for AST-generated code quality."""

    @pytest.fixture
    def resistor_translator(self):
        """Load resistor translator."""
        va_path = INTEGRATION_TESTS_DIR / "RESISTOR" / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")
        return get_translator(va_path)

    def test_ast_code_is_valid_python(self, resistor_translator):
        """Test that AST generates valid, parseable Python code."""
        import ast as python_ast

        # Get the AST-generated code
        code = resistor_translator.get_generated_code_ast()

        # Verify it's valid Python by parsing it
        try:
            python_ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Generated code is not valid Python: {e}\n\nCode:\n{code}")

    def test_ast_code_compiles(self, resistor_translator):
        """Test that AST-generated code compiles and executes."""
        # This implicitly tests compilation since translate_array_ast
        # compiles and executes the code
        eval_fn, metadata = resistor_translator.translate_array_ast()
        assert callable(eval_fn), "AST-generated function should be callable"

    def test_ast_code_jit_compatible(self, resistor_translator):
        """Test that AST-generated function is JIT-compatible."""
        from jax import jit

        eval_fn, metadata = resistor_translator.translate_array_ast()
        jitted_fn = jit(eval_fn)

        num_params = resistor_translator.module.func_num_params
        inputs = jnp.ones(num_params)

        # Should not raise during JIT compilation
        result = jitted_fn(inputs)
        assert len(result) == 4, "JIT function should return 4 arrays"


class TestASTVmapBatching:
    """Tests for vmap compatibility of AST-generated functions."""

    @pytest.fixture
    def resistor_translator(self):
        """Load resistor translator."""
        va_path = INTEGRATION_TESTS_DIR / "RESISTOR" / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Resistor VA file not found: {va_path}")
        return get_translator(va_path)

    def test_ast_vmap_batching(self, resistor_translator):
        """Test that AST-generated function works with vmap."""
        from jax import vmap

        eval_fn, metadata = resistor_translator.translate_array_ast()
        batched_fn = vmap(eval_fn)

        num_params = resistor_translator.module.func_num_params
        batch_size = 10
        batch_inputs = jnp.ones((batch_size, num_params))

        result = batched_fn(batch_inputs)
        assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"

        # Verify batch dimension is present
        for i, arr in enumerate(result):
            assert arr.shape[0] == batch_size, (
                f"Output {i} should have batch dimension {batch_size}, got {arr.shape[0]}"
            )


class TestASTWithLoops:
    """Tests for models that contain loops (lax.while_loop)."""

    @pytest.fixture
    def ekv_translator(self):
        """Load EKV model which may contain loops."""
        va_path = INTEGRATION_TESTS_DIR / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"EKV VA file not found: {va_path}")
        return get_translator(va_path)

    def test_ekv_ast_compiles(self, ekv_translator):
        """Test that EKV model AST generation compiles."""
        try:
            eval_fn, metadata = ekv_translator.translate_array_ast()
            assert callable(eval_fn)
        except Exception as e:
            pytest.fail(f"EKV AST generation failed: {e}")

    def test_ekv_ast_matches_string(self, ekv_translator):
        """Test that EKV AST output matches string output."""
        # Get both versions
        string_fn, _ = ekv_translator.translate_array()
        ast_fn, _ = ekv_translator.translate_array_ast()

        # Create test input
        module = ekv_translator.module
        num_params = module.func_num_params
        inputs = jnp.zeros(num_params)

        # Set reasonable defaults
        param_names = module.param_names
        param_kinds = module.param_kinds
        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if i >= num_params:
                break
            if kind == "temperature":
                inputs = inputs.at[i].set(300.15)
            elif kind == "voltage":
                inputs = inputs.at[i].set(0.0)

        # Compare
        string_result = string_fn(inputs)
        ast_result = ast_fn(inputs)

        for i, (s_out, a_out) in enumerate(zip(string_result, ast_result)):
            np.testing.assert_allclose(
                np.array(s_out), np.array(a_out),
                rtol=1e-10, atol=1e-15,
                err_msg=f"EKV output[{i}] mismatch"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
