"""Tests for GPU backend selection and GPU-native solvers."""

import pytest
import jax
import jax.numpy as jnp

from jax_spice.analysis.gpu_backend import (
    BackendConfig,
    select_backend,
    is_gpu_available,
    get_device,
    get_default_dtype,
    backend_info,
)
from jax_spice.devices.base import DeviceStamps


# =============================================================================
# Simple device evaluation functions for testing
# =============================================================================

def resistor_eval(voltages, params, context):
    """Resistor evaluation function."""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', params.get('R', 1000.0)))
    G = 1.0 / R
    I = G * (Vp - Vn)
    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
            ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
        }
    )


def capacitor_eval(voltages, params, context):
    """Capacitor evaluation function."""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    C = float(params.get('c', params.get('C', 1e-12)))
    V = Vp - Vn
    Q = C * V
    G_small = 1e-15  # Small leakage
    return DeviceStamps(
        currents={'p': jnp.array(0.0), 'n': jnp.array(0.0)},
        conductances={
            ('p', 'p'): jnp.array(G_small), ('p', 'n'): jnp.array(-G_small),
            ('n', 'p'): jnp.array(-G_small), ('n', 'n'): jnp.array(G_small)
        },
        charges={'p': jnp.array(Q), 'n': jnp.array(-Q)},
        capacitances={
            ('p', 'p'): jnp.array(C), ('p', 'n'): jnp.array(-C),
            ('n', 'p'): jnp.array(-C), ('n', 'n'): jnp.array(C)
        }
    )


def vsource_eval(voltages, params, context):
    """Voltage source evaluation function (DC only)."""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    V_target = float(params.get('v', params.get('dc', 0.0)))
    V_actual = Vp - Vn

    G_big = 1e12  # Large conductance to force voltage
    I = G_big * (V_actual - V_target)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
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
        from jax_spice.analysis.gpu_backend import is_metal_backend
        if not is_metal_backend():
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


# =============================================================================
# GPU DC Solver Tests
# =============================================================================

class TestGPUDCSolver:
    """Tests for GPU-native DC solver."""

    def test_dc_gpu_simple_resistor_divider(self):
        """Test GPU DC solver with simple resistor divider."""
        from jax_spice.analysis.mna import MNASystem, DeviceInfo
        from jax_spice.analysis.dc import dc_operating_point_gpu

        # Create simple voltage divider: VDD -> R1 -> out -> R2 -> GND
        # Node indices: 0=gnd, 1=vdd, 2=out
        node_names = {"0": 0, "vdd": 1, "out": 2}
        system = MNASystem(num_nodes=3, node_names=node_names)

        # Add voltage source: enforces V(vdd) = 5.0
        system.devices.append(DeviceInfo(
            name="V1",
            model_name="vsource",
            terminals=["p", "n"],
            node_indices=[1, 0],  # vdd, gnd
            params={"v": 5.0, "dc": 5.0},
            eval_fn=vsource_eval,
        ))

        # Add R1: vdd -> out
        system.devices.append(DeviceInfo(
            name="R1",
            model_name="resistor",
            terminals=["p", "n"],
            node_indices=[1, 2],  # vdd, out
            params={"r": 1000.0},
            eval_fn=resistor_eval,
        ))

        # Add R2: out -> gnd
        system.devices.append(DeviceInfo(
            name="R2",
            model_name="resistor",
            terminals=["p", "n"],
            node_indices=[2, 0],  # out, gnd
            params={"r": 1000.0},
            eval_fn=resistor_eval,
        ))

        # Build device groups for GPU functions
        system.build_device_groups()

        # Solve with GPU backend (will use CPU if GPU not available)
        V, info = dc_operating_point_gpu(system, vdd=5.0, backend="cpu")

        assert info["converged"]
        # Output should be ~2.5V (voltage divider)
        out_idx = node_names["out"]
        assert abs(V[out_idx] - 2.5) < 0.1

    def test_dc_gpu_matches_cpu(self):
        """GPU and CPU solvers should give same results."""
        from jax_spice.analysis.mna import MNASystem, DeviceInfo
        from jax_spice.analysis import dc_operating_point
        from jax_spice.analysis.dc import dc_operating_point_gpu

        # Create circuit: VDD -> R1 -> mid -> R2 -> out -> R3 -> GND
        # Node indices: 0=gnd, 1=vdd, 2=mid, 3=out
        node_names = {"0": 0, "vdd": 1, "mid": 2, "out": 3}
        system = MNASystem(num_nodes=4, node_names=node_names)

        # Add voltage source
        system.devices.append(DeviceInfo(
            name="V1",
            model_name="vsource",
            terminals=["p", "n"],
            node_indices=[1, 0],
            params={"v": 3.3, "dc": 3.3},
            eval_fn=vsource_eval,
        ))

        # Add resistors
        system.devices.append(DeviceInfo(
            name="R1",
            model_name="resistor",
            terminals=["p", "n"],
            node_indices=[1, 2],  # vdd -> mid
            params={"r": 470.0},
            eval_fn=resistor_eval,
        ))
        system.devices.append(DeviceInfo(
            name="R2",
            model_name="resistor",
            terminals=["p", "n"],
            node_indices=[2, 3],  # mid -> out
            params={"r": 330.0},
            eval_fn=resistor_eval,
        ))
        system.devices.append(DeviceInfo(
            name="R3",
            model_name="resistor",
            terminals=["p", "n"],
            node_indices=[3, 0],  # out -> gnd
            params={"r": 1000.0},
            eval_fn=resistor_eval,
        ))

        system.build_device_groups()

        # Solve with both methods
        V_cpu, info_cpu = dc_operating_point(system)
        V_gpu, info_gpu = dc_operating_point_gpu(system, vdd=3.3, backend="cpu")

        assert info_cpu["converged"]
        assert info_gpu["converged"]

        # Results should match
        for name, idx in node_names.items():
            if idx > 0:  # Skip ground
                assert abs(float(V_cpu[idx]) - float(V_gpu[idx])) < 1e-6, (
                    f"Node {name}: CPU={V_cpu[idx]}, GPU={V_gpu[idx]}"
                )


# =============================================================================
# Transient Backend Tests
# =============================================================================

class TestTransientBackend:
    """Tests for transient analysis with backend selection."""

    def test_transient_jit_with_backend_param(self):
        """Transient analysis should accept backend parameter."""
        from jax_spice.analysis.mna import MNASystem, DeviceInfo
        from jax_spice.analysis.transient import transient_analysis_jit

        # Simple RC circuit: VDD -> R -> out -> C -> GND
        # Node indices: 0=gnd, 1=vdd, 2=out
        node_names = {"0": 0, "vdd": 1, "out": 2}
        system = MNASystem(num_nodes=3, node_names=node_names)

        # Add voltage source
        system.devices.append(DeviceInfo(
            name="V1",
            model_name="vsource",
            terminals=["p", "n"],
            node_indices=[1, 0],
            params={"v": 1.0, "dc": 1.0},
            eval_fn=vsource_eval,
        ))

        # Add resistor
        system.devices.append(DeviceInfo(
            name="R1",
            model_name="resistor",
            terminals=["p", "n"],
            node_indices=[1, 2],  # vdd -> out
            params={"r": 1000.0},
            eval_fn=resistor_eval,
        ))

        # Add capacitor
        system.devices.append(DeviceInfo(
            name="C1",
            model_name="capacitor",
            terminals=["p", "n"],
            node_indices=[2, 0],  # out -> gnd
            params={"c": 1e-9},
            eval_fn=capacitor_eval,
        ))

        # Run with explicit CPU backend
        times, solutions, info = transient_analysis_jit(
            system,
            t_stop=1e-6,
            t_step=1e-8,
            initial_conditions={"vdd": 1.0, "out": 0.0},
            backend="cpu",
        )

        assert info["backend"] == "cpu"
        assert len(times) > 0
        assert solutions.shape[0] == len(times)
