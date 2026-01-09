"""Tests for GF130 PDK Verilog-A models

These tests require the GF130 PDK to be available via PDK_GF130_PATH environment variable.
"""

import pytest
from pathlib import Path

from pdk_utils import PDK_PATHS, sanitize_pdk_path, CompiledPDKModel, compile_pdk_model


# Models known to fail compilation (Spectre-specific syntax not supported by OpenVAF)
KNOWN_FAILING_MODELS = {
    # These use Spectre-specific syntax (_spe suffix indicates Spectre models)
    "esdnsh_16p0_stk_va_spe",
    "esdnsh_30p0_stk_va_spe",
    "esdpfet_1p5_spe",
    "esdnfet_1p5_spe",
    "esdnfet_5p0_spe",
    "esdnfet_5p0_ulr_spe",
    "esdpfet_5p0_spe",
    # Non-_spe models that also fail (use net references in arithmetic)
    "esdpfet_1p5",
}


@pytest.mark.requires_pdk("gf130")
class TestGF130Compilation:
    """Test that GF130 models compile to JAX"""

    def test_pdk_path_exists(self):
        """GF130 PDK path is valid"""
        gf130_path = PDK_PATHS["gf130"]
        assert gf130_path is not None, "PDK_GF130_PATH not set"
        assert gf130_path.exists(), "PDK path does not exist"

    def test_has_va_files(self):
        """GF130 PDK contains .va files"""
        gf130_path = PDK_PATHS["gf130"]
        va_files = list(gf130_path.glob("**/*.va"))
        assert len(va_files) > 0, "No .va files found in GF130 PDK"

    def test_model_count(self):
        """GF130 has expected number of models"""
        gf130_path = PDK_PATHS["gf130"]
        va_files = list(gf130_path.glob("**/*.va"))
        # GF130 should have 30+ models
        assert len(va_files) >= 30, f"Expected 30+ models, found {len(va_files)}"

    def test_all_models_compile(self):
        """GF130 .va files compile to JAX (excluding known failures)"""
        gf130_path = PDK_PATHS["gf130"]
        va_files = list(gf130_path.glob("**/*.va"))

        results = []
        for va_file in va_files:
            model_name = va_file.stem
            try:
                model = compile_pdk_model(va_file, allow_analog_in_cond=True)
                results.append((model_name, True, None))
            except Exception as e:
                # Sanitize error message
                error_msg = sanitize_pdk_path(str(e))
                results.append((model_name, False, error_msg))

        # Report results
        passed = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]

        # Print summary (model names only, not paths)
        print(f"\nGF130 Compilation Results: {len(passed)}/{len(results)} passed")
        if failed:
            print("Failed models:")
            for name, _, error in failed:
                # Truncate error message for readability
                short_error = error[:100] + "..." if len(error) > 100 else error
                print(f"  - {name}: {short_error}")

        # Check for unexpected failures (not in known failing list)
        unexpected_failures = [r for r in failed if r[0] not in KNOWN_FAILING_MODELS]
        if unexpected_failures:
            names = [r[0] for r in unexpected_failures]
            pytest.fail(f"Unexpected compilation failures: {names}")

        # Ensure we have a good success rate (at least 70%)
        success_rate = len(passed) / len(results)
        assert success_rate >= 0.70, f"Success rate {success_rate:.0%} below 70% threshold"


@pytest.mark.requires_pdk("gf130")
class TestGF130ModelProperties:
    """Test properties of compiled GF130 models"""

    @pytest.fixture(scope="class")
    def sample_model(self) -> CompiledPDKModel:
        """Get a sample compiled GF130 model for testing"""
        gf130_path = PDK_PATHS["gf130"]
        # Find first .va file
        va_files = list(gf130_path.glob("**/*.va"))
        if not va_files:
            pytest.skip("No .va files in GF130 PDK")
        return compile_pdk_model(va_files[0], allow_analog_in_cond=True)

    def test_model_has_nodes(self, sample_model):
        """Compiled model has nodes"""
        assert len(sample_model.nodes) >= 2, "Model should have at least 2 nodes"

    def test_model_has_params(self, sample_model):
        """Compiled model has parameters"""
        assert len(sample_model.param_names) > 0, "Model should have parameters"

    def test_model_has_jax_function(self, sample_model):
        """Compiled model has JAX function"""
        assert sample_model.jax_fn is not None, "Model should have JAX function"


@pytest.mark.requires_pdk("gf130")
class TestGF130JAXExecution:
    """Test JAX execution of GF130 models"""

    @pytest.fixture(scope="class")
    def compiled_models(self):
        """Compile a few sample GF130 models"""
        gf130_path = PDK_PATHS["gf130"]
        va_files = list(gf130_path.glob("**/*.va"))[:5]  # First 5 models

        models = []
        for va_file in va_files:
            try:
                model = compile_pdk_model(va_file, allow_analog_in_cond=True)
                models.append(model)
            except Exception:
                pass  # Skip models that fail to compile

        return models

    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_models_produce_output(self, compiled_models):
        """GF130 models produce valid JAX output"""
        import numpy as np

        for model in compiled_models:
            inputs = model.build_default_inputs()
            residuals, jacobian = model.jax_fn(inputs)

            assert residuals is not None, f"{model.name} returned None residuals"

            # Check at least one output is not NaN
            has_valid = False
            for node, res in residuals.items():
                if np.isfinite(float(res['resist'])):
                    has_valid = True
                    break

            assert has_valid, f"{model.name} produced no valid outputs"
