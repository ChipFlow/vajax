"""Tests for GPU backend selection."""

import jax
import jax.numpy as jnp
import pytest

from vajax.analysis.gpu_backend import (
    BackendConfig,
    backend_info,
    get_default_dtype,
    get_device,
    is_gpu_available,
    select_backend,
)

# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestBackendSelection:
    """Tests for automatic backend selection."""

    def test_small_circuit_uses_cpu(self):
        """Circuits below threshold should use CPU."""
        backend = select_backend(num_nodes=100)
        assert backend == "cpu"

    def test_medium_circuit_uses_cpu_without_gpu(self):
        """Circuits above threshold but without GPU available should use CPU."""
        # Force CPU-only check
        config = BackendConfig(gpu_threshold=500, force_backend=None)
        backend = select_backend(num_nodes=1000, config=config)

        # If no GPU available, should fall back to CPU
        if not is_gpu_available():
            assert backend == "cpu"
        else:
            assert backend == "gpu"

    def test_force_cpu_backend(self):
        """Force CPU should always return CPU."""
        config = BackendConfig(force_backend="cpu")
        backend = select_backend(num_nodes=10000, config=config)
        assert backend == "cpu"

    def test_force_gpu_without_gpu_falls_back(self):
        """Force GPU without GPU available should fall back to CPU."""
        if is_gpu_available():
            pytest.skip("GPU available - cannot test fallback")

        config = BackendConfig(force_backend="gpu")
        backend = select_backend(num_nodes=100, config=config)
        assert backend == "cpu"

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        config = BackendConfig(gpu_threshold=50)
        backend_small = select_backend(num_nodes=40, config=config)
        backend_large = select_backend(num_nodes=60, config=config)

        assert backend_small == "cpu"
        # Large circuit uses GPU only if available
        if is_gpu_available():
            assert backend_large == "gpu"
        else:
            assert backend_large == "cpu"


class TestDeviceSelection:
    """Tests for device selection."""

    def test_get_cpu_device(self):
        """Should be able to get CPU device."""
        device = get_device("cpu")
        assert device.platform == "cpu"

    def test_get_gpu_device_when_available(self):
        """Should get GPU device if available."""
        if not is_gpu_available():
            pytest.skip("No GPU available")

        device = get_device("gpu")
        assert device.platform != "cpu"

    def test_get_gpu_device_when_unavailable_raises(self):
        """Should raise error if GPU requested but not available."""
        if is_gpu_available():
            pytest.skip("GPU available - cannot test error case")

        with pytest.raises(RuntimeError, match="no GPU available"):
            get_device("gpu")


class TestDtype:
    """Tests for dtype selection."""

    def test_default_dtype_is_float64_on_cpu(self):
        """CPU should use float64 by default."""
        dtype = get_default_dtype("cpu")
        assert dtype == jnp.float64

    def test_metal_uses_float32(self):
        """Metal backend should use float32."""
        if jax.default_backend() != "METAL":
            pytest.skip("Not running on Metal backend")

        dtype = get_default_dtype("gpu")
        assert dtype == jnp.float32


class TestBackendInfo:
    """Tests for backend_info utility."""

    def test_backend_info_returns_dict(self):
        """Should return dict with expected keys."""
        info = backend_info()
        assert isinstance(info, dict)
        assert "default_backend" in info
        assert "gpu_available" in info
        assert "default_threshold" in info

    def test_backend_info_threshold(self):
        """Should report default threshold."""
        info = backend_info()
        assert info["default_threshold"] == 500
