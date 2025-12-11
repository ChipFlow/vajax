"""Unified system building for JAX-SPICE Newton-Raphson solvers.

This module provides a unified interface for building Jacobian matrices and
residual vectors from circuits. It supports:
- Simple devices (resistors, capacitors, voltage sources, etc.)
- OpenVAF-compiled Verilog-A devices with analytical Jacobians
- Mixed circuits with both device types

The key advantage is using analytical Jacobians from OpenVAF instead of
autodiff, which provides better numerical stability and performance.

Usage:
    from jax_spice.analysis.system import SystemBuilder

    # Create builder from circuit devices
    builder = SystemBuilder.from_devices(devices, num_nodes, ground=0)
    builder.prepare()  # JIT compile device evaluations

    # Build system during NR iteration
    J, f = builder.build_system(V, t=0.0, dt=1e-9)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, NamedTuple
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Add openvaf-py to path for optional OpenVAF support
_openvaf_path = Path(__file__).parent.parent.parent / "openvaf-py"
if str(_openvaf_path) not in sys.path:
    sys.path.insert(0, str(_openvaf_path))

try:
    from jax_spice.devices.openvaf_device import CompiledModelBatch, HAS_OPENVAF
except ImportError:
    CompiledModelBatch = None
    HAS_OPENVAF = False


class SystemResult(NamedTuple):
    """Result from system building."""
    J: Array  # Jacobian matrix (n-1, n-1) excluding ground
    f: Array  # Residual vector (n-1,) excluding ground


@dataclass
class SimpleDevice:
    """Simple device for stamping (resistor, capacitor, vsource, isource, diode)."""
    name: str
    device_type: str  # 'resistor', 'capacitor', 'vsource', 'isource', 'diode'
    nodes: List[int]  # Node indices
    params: Dict[str, Any]  # Device parameters


@dataclass
class OpenVAFDeviceInfo:
    """Info for an OpenVAF device to be added to a batch."""
    name: str
    model_type: str  # e.g., 'psp103'
    va_path: str  # Path to .va file
    ext_nodes: List[int]  # External terminal nodes [d, g, s, b]
    params: Dict[str, Any]  # Model parameters


@dataclass
class SystemBuilder:
    """Unified system builder for Newton-Raphson solvers.

    This class manages both simple devices and OpenVAF-compiled devices,
    providing a single interface for building Jacobian and residual.

    Attributes:
        num_nodes: Total number of nodes (including ground at index 0)
        ground: Ground node index (always 0)
        simple_devices: List of simple devices
        openvaf_batches: Dict of model_type -> CompiledModelBatch
        _prepared: Whether prepare() has been called
    """
    num_nodes: int
    ground: int = 0
    simple_devices: List[SimpleDevice] = field(default_factory=list)
    openvaf_batches: Dict[str, Any] = field(default_factory=dict)  # model_type -> CompiledModelBatch
    openvaf_va_paths: Dict[str, str] = field(default_factory=dict)  # model_type -> va_path
    _internal_node_map: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _total_nodes: Optional[int] = field(default=None)
    _prepared: bool = field(default=False)

    @classmethod
    def from_devices(
        cls,
        devices: List[Dict[str, Any]],
        num_nodes: int,
        ground: int = 0,
        openvaf_model_paths: Optional[Dict[str, str]] = None,
    ) -> 'SystemBuilder':
        """Create a SystemBuilder from a list of device dictionaries.

        Args:
            devices: List of device dicts with keys:
                - name: Device instance name
                - model/type: Device type ('resistor', 'psp103', etc.)
                - nodes: List of node indices
                - params: Device parameters dict
                - is_openvaf: Optional bool indicating OpenVAF device
            num_nodes: Number of nodes in circuit
            ground: Ground node index
            openvaf_model_paths: Dict mapping model type to .va file path

        Returns:
            SystemBuilder ready to accept devices and prepare
        """
        builder = cls(num_nodes=num_nodes, ground=ground)

        for dev in devices:
            name = dev['name']
            model_type = dev.get('model') or dev.get('type', 'unknown')
            nodes = dev['nodes']
            params = dev.get('params', {})
            is_openvaf = dev.get('is_openvaf', False)

            if is_openvaf and HAS_OPENVAF:
                # OpenVAF device
                va_path = None
                if openvaf_model_paths:
                    va_path = openvaf_model_paths.get(model_type)
                if va_path:
                    builder.add_openvaf_device(
                        name=name,
                        model_type=model_type,
                        va_path=va_path,
                        ext_nodes=nodes,
                        params=params,
                    )
                else:
                    # Fall back to simple device if no VA path
                    builder.add_simple_device(name, model_type, nodes, params)
            else:
                # Simple device
                builder.add_simple_device(name, model_type, nodes, params)

        return builder

    def add_simple_device(
        self,
        name: str,
        device_type: str,
        nodes: List[int],
        params: Dict[str, Any],
    ) -> None:
        """Add a simple device to the builder."""
        if self._prepared:
            raise RuntimeError("Cannot add devices after prepare()")
        self.simple_devices.append(SimpleDevice(
            name=name,
            device_type=device_type.lower(),
            nodes=nodes,
            params=params,
        ))

    def add_openvaf_device(
        self,
        name: str,
        model_type: str,
        va_path: str,
        ext_nodes: List[int],
        params: Dict[str, Any],
        temperature: float = 300.15,
    ) -> None:
        """Add an OpenVAF device to the builder.

        The device will be added to a batch for its model type, enabling
        efficient vmapped evaluation.

        Args:
            name: Device instance name
            model_type: Model type identifier (e.g., 'psp103')
            va_path: Path to .va file
            ext_nodes: External terminal node indices
            params: Device parameters
            temperature: Device temperature in Kelvin
        """
        if self._prepared:
            raise RuntimeError("Cannot add devices after prepare()")

        if not HAS_OPENVAF:
            raise ImportError("OpenVAF support not available")

        # Create batch for this model type if needed
        if model_type not in self.openvaf_batches:
            batch = CompiledModelBatch.from_va_file(va_path)
            self.openvaf_batches[model_type] = batch
            self.openvaf_va_paths[model_type] = va_path

        batch = self.openvaf_batches[model_type]

        # Allocate internal nodes for this device
        internal_nodes = self._allocate_internal_nodes(name, batch)

        # Add device to batch
        batch.add_device(
            name=name,
            ext_nodes=ext_nodes,
            internal_nodes=internal_nodes,
            params=params,
            ground=self.ground,
            temperature=temperature,
        )

    def _allocate_internal_nodes(
        self,
        device_name: str,
        batch: Any,
    ) -> Dict[str, int]:
        """Allocate internal nodes for an OpenVAF device."""
        if self._total_nodes is None:
            self._total_nodes = self.num_nodes

        model_nodes = batch.model_nodes
        n_model_nodes = len(model_nodes)

        # Allocate internal nodes (skip first 4 external and last 1 branch)
        internal_map = {}
        for i in range(4, n_model_nodes - 1):
            node_name = model_nodes[i]
            internal_map[node_name] = self._total_nodes
            self._total_nodes += 1

        self._internal_node_map[device_name] = internal_map
        return internal_map

    def prepare(self) -> None:
        """Finalize setup. Must be called before build_system()."""
        if self._prepared:
            return

        # Finalize total node count
        if self._total_nodes is None:
            self._total_nodes = self.num_nodes

        # Prepare all OpenVAF batches
        for batch in self.openvaf_batches.values():
            batch.prepare()

        self._prepared = True

    @property
    def total_nodes(self) -> int:
        """Total number of nodes including internal nodes."""
        if self._total_nodes is None:
            return self.num_nodes
        return self._total_nodes

    def build_system(
        self,
        V: np.ndarray,
        t: float = 0.0,
        dt: float = 1e-9,
        V_prev: Optional[np.ndarray] = None,
        source_values: Optional[Dict[str, float]] = None,
    ) -> SystemResult:
        """Build Jacobian and residual for current voltages.

        Args:
            V: Current voltage solution (length: total_nodes)
            t: Current simulation time
            dt: Time step (for capacitor companion model)
            V_prev: Previous voltage (for capacitor companion model in transient)
            source_values: Override values for sources {device_name: value}

        Returns:
            SystemResult with (J, f) where:
                J: Jacobian matrix (n-1, n-1)
                f: Residual vector (n-1,)
        """
        if not self._prepared:
            self.prepare()

        n_unknowns = self.total_nodes - 1  # Exclude ground

        # Initialize system
        J = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
        f = np.zeros(n_unknowns, dtype=np.float64)

        # Stamp simple devices
        self._stamp_simple_devices(V, J, f, t, dt, V_prev, source_values)

        # Stamp OpenVAF devices
        self._stamp_openvaf_devices(V, J, f)

        return SystemResult(J=jnp.array(J), f=jnp.array(f))

    def _stamp_simple_devices(
        self,
        V: np.ndarray,
        J: np.ndarray,
        f: np.ndarray,
        t: float,
        dt: float,
        V_prev: Optional[np.ndarray],
        source_values: Optional[Dict[str, float]],
    ) -> None:
        """Stamp simple devices into system matrices."""
        ground = self.ground

        for dev in self.simple_devices:
            dtype = dev.device_type
            nodes = dev.nodes
            params = dev.params

            if dtype == 'resistor':
                self._stamp_resistor(V, J, f, nodes, params, ground)
            elif dtype == 'capacitor':
                self._stamp_capacitor(V, J, f, nodes, params, dt, V_prev, ground)
            elif dtype == 'vsource':
                value = params.get('dc', params.get('value', 0.0))
                if source_values and dev.name in source_values:
                    value = source_values[dev.name]
                self._stamp_vsource(V, J, f, nodes, value, ground)
            elif dtype == 'isource':
                value = params.get('dc', params.get('value', 0.0))
                if source_values and dev.name in source_values:
                    value = source_values[dev.name]
                self._stamp_isource(f, nodes, value, ground)
            elif dtype == 'diode':
                self._stamp_diode(V, J, f, nodes, params, ground)

    def _stamp_resistor(
        self,
        V: np.ndarray,
        J: np.ndarray,
        f: np.ndarray,
        nodes: List[int],
        params: Dict[str, Any],
        ground: int,
    ) -> None:
        """Stamp resistor into system."""
        n1, n2 = nodes[0], nodes[1]
        r = float(params.get('r', params.get('value', 1000.0)))
        mfactor = float(params.get('mfactor', 1.0))
        g = mfactor / r  # conductance

        v1 = V[n1] if n1 < len(V) else 0.0
        v2 = V[n2] if n2 < len(V) else 0.0
        i = g * (v1 - v2)

        # Stamp current into residual
        if n1 != ground:
            f[n1 - 1] += i
        if n2 != ground:
            f[n2 - 1] -= i

        # Stamp conductance into Jacobian
        if n1 != ground:
            J[n1 - 1, n1 - 1] += g
        if n2 != ground:
            J[n2 - 1, n2 - 1] += g
        if n1 != ground and n2 != ground:
            J[n1 - 1, n2 - 1] -= g
            J[n2 - 1, n1 - 1] -= g

    def _stamp_capacitor(
        self,
        V: np.ndarray,
        J: np.ndarray,
        f: np.ndarray,
        nodes: List[int],
        params: Dict[str, Any],
        dt: float,
        V_prev: Optional[np.ndarray],
        ground: int,
    ) -> None:
        """Stamp capacitor companion model (backward Euler)."""
        n1, n2 = nodes[0], nodes[1]
        c = float(params.get('c', params.get('value', 1e-12)))
        mfactor = float(params.get('mfactor', 1.0))
        c_eff = c * mfactor

        # Companion model: G_eq = C/dt, I_eq = C/dt * V_prev
        g_eq = c_eff / dt

        v1 = V[n1] if n1 < len(V) else 0.0
        v2 = V[n2] if n2 < len(V) else 0.0

        if V_prev is not None:
            v1_prev = V_prev[n1] if n1 < len(V_prev) else 0.0
            v2_prev = V_prev[n2] if n2 < len(V_prev) else 0.0
        else:
            v1_prev, v2_prev = v1, v2

        # Current: i = G_eq * (v - v_prev)
        i = g_eq * ((v1 - v2) - (v1_prev - v2_prev))

        # Stamp current
        if n1 != ground:
            f[n1 - 1] += i
        if n2 != ground:
            f[n2 - 1] -= i

        # Stamp conductance
        if n1 != ground:
            J[n1 - 1, n1 - 1] += g_eq
        if n2 != ground:
            J[n2 - 1, n2 - 1] += g_eq
        if n1 != ground and n2 != ground:
            J[n1 - 1, n2 - 1] -= g_eq
            J[n2 - 1, n1 - 1] -= g_eq

    def _stamp_vsource(
        self,
        V: np.ndarray,
        J: np.ndarray,
        f: np.ndarray,
        nodes: List[int],
        value: float,
        ground: int,
    ) -> None:
        """Stamp voltage source using large conductance method."""
        n_pos, n_neg = nodes[0], nodes[1]
        G_large = 1e12  # Large conductance

        v_pos = V[n_pos] if n_pos < len(V) else 0.0
        v_neg = V[n_neg] if n_neg < len(V) else 0.0
        v_diff = v_pos - v_neg

        # Current = G * (V_diff - V_target)
        i = G_large * (v_diff - value)

        # Stamp current
        if n_pos != ground:
            f[n_pos - 1] += i
        if n_neg != ground:
            f[n_neg - 1] -= i

        # Stamp conductance
        if n_pos != ground:
            J[n_pos - 1, n_pos - 1] += G_large
        if n_neg != ground:
            J[n_neg - 1, n_neg - 1] += G_large
        if n_pos != ground and n_neg != ground:
            J[n_pos - 1, n_neg - 1] -= G_large
            J[n_neg - 1, n_pos - 1] -= G_large

    def _stamp_isource(
        self,
        f: np.ndarray,
        nodes: List[int],
        value: float,
        ground: int,
    ) -> None:
        """Stamp current source."""
        n_pos, n_neg = nodes[0], nodes[1]

        # Current flows from pos to neg
        if n_pos != ground:
            f[n_pos - 1] -= value
        if n_neg != ground:
            f[n_neg - 1] += value

    def _stamp_diode(
        self,
        V: np.ndarray,
        J: np.ndarray,
        f: np.ndarray,
        nodes: List[int],
        params: Dict[str, Any],
        ground: int,
    ) -> None:
        """Stamp diode using Shockley equation with limiting."""
        n_anode, n_cathode = nodes[0], nodes[1]

        Is = float(params.get('is', params.get('Is', 1e-14)))
        n_factor = float(params.get('n', 1.0))
        mfactor = float(params.get('mfactor', 1.0))
        vt = 0.02585  # Thermal voltage at 300K

        v_anode = V[n_anode] if n_anode < len(V) else 0.0
        v_cathode = V[n_cathode] if n_cathode < len(V) else 0.0
        vd = v_anode - v_cathode

        # Voltage limiting for convergence
        vd_crit = n_factor * vt * np.log(n_factor * vt / (np.sqrt(2) * Is))
        if vd > vd_crit:
            vd = vd_crit + n_factor * vt * np.log(1 + (vd - vd_crit) / (n_factor * vt))

        # Diode current and conductance
        if vd < -5 * n_factor * vt:
            # Reverse bias - avoid underflow
            Id = -Is * mfactor
            gd = Is * mfactor / (n_factor * vt)
        else:
            exp_term = np.exp(vd / (n_factor * vt))
            Id = Is * mfactor * (exp_term - 1)
            gd = Is * mfactor * exp_term / (n_factor * vt)

        # Companion model current
        ieq = Id - gd * vd

        # Stamp current
        if n_anode != ground:
            f[n_anode - 1] += Id
        if n_cathode != ground:
            f[n_cathode - 1] -= Id

        # Stamp conductance
        if n_anode != ground:
            J[n_anode - 1, n_anode - 1] += gd
        if n_cathode != ground:
            J[n_cathode - 1, n_cathode - 1] += gd
        if n_anode != ground and n_cathode != ground:
            J[n_anode - 1, n_cathode - 1] -= gd
            J[n_cathode - 1, n_anode - 1] -= gd

    def _stamp_openvaf_devices(
        self,
        V: np.ndarray,
        J: np.ndarray,
        f: np.ndarray,
    ) -> None:
        """Stamp all OpenVAF devices into system matrices."""
        for model_type, batch in self.openvaf_batches.items():
            # Evaluate batch - returns analytical Jacobian
            batch_residuals, batch_jacobian = batch.evaluate(V)

            # Stamp into system
            batch.stamp_into_system(
                batch_residuals=batch_residuals,
                batch_jacobian=batch_jacobian,
                f=f,
                J=J,
                ground=self.ground,
            )
