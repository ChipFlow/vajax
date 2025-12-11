"""OpenVAF-based device evaluation with analytical Jacobians

This module provides device evaluation functions compiled from Verilog-A
models using openvaf_jax. The key advantage is that these functions return
BOTH residuals AND analytical Jacobians in a single call, avoiding the
autodiff issues that cause GPU solver convergence problems.

Features:
- Single-device evaluation via VADevice.evaluate()
- Batched evaluation via CompiledModelBatch for multiple devices of same type
- JIT-compiled vmapped functions for efficient parallel evaluation
- Static/dynamic input separation for NR iteration performance

Usage:
    from jax_spice.devices.openvaf_device import CompiledModelBatch, VADevice

    # Single device evaluation
    device = VADevice.from_va_file("path/to/model.va")
    residuals, jacobian = device.evaluate(voltages, params)

    # Batched evaluation (preferred for circuits)
    batch = CompiledModelBatch.from_va_file("path/to/model.va")
    batch.add_device(device_info, node_map, params)
    batch.prepare()  # JIT compile vmapped function
    residuals, jacobian = batch.evaluate(V)  # Fast batched evaluation
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Add openvaf-py to path
_openvaf_path = Path(__file__).parent.parent.parent / "openvaf-py"
if str(_openvaf_path) not in sys.path:
    sys.path.insert(0, str(_openvaf_path))

try:
    import openvaf_py
    import openvaf_jax
    HAS_OPENVAF = True
except ImportError:
    HAS_OPENVAF = False
    openvaf_py = None
    openvaf_jax = None


class DeviceContext(NamedTuple):
    """Context for a single device in a batch.

    Stores the mapping from model nodes to circuit nodes for stamping.
    """
    name: str
    node_map: Dict[str, int]  # Model node name -> circuit node index
    ext_nodes: List[int]  # External terminal nodes [d, g, s, b]
    voltage_node_pairs: List[Tuple[int, int]]  # (n1, n2) for each voltage param


@dataclass
class VADevice:
    """Wrapper for an OpenVAF-compiled Verilog-A device model.

    Provides evaluation with analytical Jacobians suitable for Newton-Raphson.
    This class is for single-device evaluation. For batched evaluation of
    multiple devices of the same type, use CompiledModelBatch.
    """
    name: str
    module: Any  # openvaf_py.VaModule
    eval_fn: Callable
    param_names: List[str]
    param_kinds: List[str]
    nodes: List[str]

    @classmethod
    def from_va_file(cls, va_path: str, allow_analog_in_cond: bool = False) -> 'VADevice':
        """Compile a Verilog-A file and create a device wrapper.

        Args:
            va_path: Path to .va file
            allow_analog_in_cond: Allow analog statements in conditionals

        Returns:
            VADevice instance ready for evaluation
        """
        if not HAS_OPENVAF:
            raise ImportError("openvaf_py and openvaf_jax required. "
                            "Ensure openvaf-py is built and in path.")

        modules = openvaf_py.compile_va(str(va_path), allow_analog_in_cond)
        if not modules:
            raise ValueError(f"No modules found in {va_path}")

        module = modules[0]
        translator = openvaf_jax.OpenVAFToJAX(module)
        eval_fn = translator.translate()

        return cls(
            name=module.name,
            module=module,
            eval_fn=eval_fn,
            param_names=list(module.param_names),
            param_kinds=list(module.param_kinds),
            nodes=list(module.nodes),
        )

    def build_inputs(self, voltages: Dict[str, float], params: Dict[str, Any],
                     temperature: float = 300.0) -> List[float]:
        """Build input array for evaluation.

        Args:
            voltages: Terminal voltages by name (e.g., {'V(A,B)': 1.0})
            params: Device parameters (e.g., {'r': 1000.0})
            temperature: Device temperature in Kelvin

        Returns:
            Input array matching param_names order
        """
        inputs = []
        for name, kind in zip(self.param_names, self.param_kinds):
            if kind == 'voltage':
                # Try exact match first, then pattern matching
                if name in voltages:
                    inputs.append(float(voltages[name]))
                else:
                    inputs.append(0.0)
            elif kind == 'temperature' or 'temperature' in name.lower():
                inputs.append(temperature)
            elif kind in ('param', 'sysfun'):
                # Look up in params dict (sysfun includes mfactor)
                param_lower = name.lower()
                # Default mfactor to 1.0
                default = 1.0 if 'mfactor' in param_lower else 1.0
                value = params.get(name, params.get(param_lower, default))
                inputs.append(float(value))
            elif kind == 'hidden_state':
                inputs.append(0.0)
            else:
                # Unknown kind - check if it's in params anyway
                if name in params:
                    inputs.append(float(params[name]))
                elif name.lower() in params:
                    inputs.append(float(params[name.lower()]))
                else:
                    inputs.append(0.0)
        return inputs

    def evaluate(self, voltages: Dict[str, float], params: Dict[str, Any],
                 temperature: float = 300.0) -> Tuple[Dict, Dict]:
        """Evaluate device and return residuals and Jacobian.

        Args:
            voltages: Terminal voltages
            params: Device parameters
            temperature: Temperature in Kelvin

        Returns:
            Tuple of (residuals, jacobian) where:
                residuals: Dict[node, {'resist': float, 'react': float}]
                jacobian: Dict[(row, col), {'resist': float, 'react': float}]
        """
        inputs = self.build_inputs(voltages, params, temperature)
        return self.eval_fn(inputs)


@dataclass
class CompiledModelBatch:
    """Batched evaluation for multiple devices of the same OpenVAF model type.

    This class provides efficient evaluation of many devices using JAX vmap.
    The key optimization is separating static inputs (parameters) from dynamic
    inputs (voltages) so that only voltages need to be updated each NR iteration.

    Usage:
        batch = CompiledModelBatch.from_va_file("path/to/psp103.va")

        # Add devices during circuit setup
        for dev in psp103_devices:
            batch.add_device(dev, node_map, params)

        # Prepare JIT-compiled batched function
        batch.prepare()

        # During NR iteration - fast evaluation
        residuals, jacobian = batch.evaluate(V)
    """
    name: str
    module: Any
    translator: Any
    jax_fn_array: Callable
    vmapped_fn: Callable
    param_names: List[str]
    param_kinds: List[str]
    model_nodes: List[str]
    array_metadata: Dict[str, Any]
    # Batch state
    _static_inputs: Optional[np.ndarray] = field(default=None, repr=False)
    _voltage_indices: List[int] = field(default_factory=list, repr=False)
    _device_contexts: List[DeviceContext] = field(default_factory=list, repr=False)
    _prepared: bool = field(default=False, repr=False)

    @classmethod
    def from_va_file(cls, va_path: str, allow_analog_in_cond: bool = False) -> 'CompiledModelBatch':
        """Compile a Verilog-A model for batched evaluation.

        Args:
            va_path: Path to .va file
            allow_analog_in_cond: Allow analog statements in conditionals

        Returns:
            CompiledModelBatch ready to accept devices
        """
        if not HAS_OPENVAF:
            raise ImportError("openvaf_py and openvaf_jax required")

        modules = openvaf_py.compile_va(str(va_path), allow_analog_in_cond)
        if not modules:
            raise ValueError(f"No modules found in {va_path}")

        module = modules[0]
        translator = openvaf_jax.OpenVAFToJAX(module)
        jax_fn_array, array_metadata = translator.translate_array()
        vmapped_fn = jax.jit(jax.vmap(jax_fn_array))

        return cls(
            name=module.name,
            module=module,
            translator=translator,
            jax_fn_array=jax_fn_array,
            vmapped_fn=vmapped_fn,
            param_names=list(module.param_names),
            param_kinds=list(module.param_kinds),
            model_nodes=list(module.nodes),
            array_metadata=array_metadata,
        )

    @classmethod
    def from_module(cls, module: Any) -> 'CompiledModelBatch':
        """Create from an already-compiled openvaf_py module.

        Args:
            module: VaModule from openvaf_py.compile_va()

        Returns:
            CompiledModelBatch ready to accept devices
        """
        if not HAS_OPENVAF:
            raise ImportError("openvaf_py and openvaf_jax required")

        translator = openvaf_jax.OpenVAFToJAX(module)
        jax_fn_array, array_metadata = translator.translate_array()
        vmapped_fn = jax.jit(jax.vmap(jax_fn_array))

        return cls(
            name=module.name,
            module=module,
            translator=translator,
            jax_fn_array=jax_fn_array,
            vmapped_fn=vmapped_fn,
            param_names=list(module.param_names),
            param_kinds=list(module.param_kinds),
            model_nodes=list(module.nodes),
            array_metadata=array_metadata,
        )

    def add_device(
        self,
        name: str,
        ext_nodes: List[int],
        internal_nodes: Dict[str, int],
        params: Dict[str, Any],
        ground: int = 0,
        temperature: float = 300.15,
    ) -> None:
        """Add a device to the batch.

        Args:
            name: Device instance name
            ext_nodes: External terminal node indices [d, g, s, b]
            internal_nodes: Internal node name -> global node index mapping
            params: Device parameters
            ground: Ground node index
            temperature: Device temperature in Kelvin
        """
        if self._prepared:
            raise RuntimeError("Cannot add devices after prepare() called")

        # Build node map: model node name -> global circuit node index
        node_map = {}
        for i, model_node in enumerate(self.model_nodes[:4]):
            if i < len(ext_nodes):
                node_map[model_node] = ext_nodes[i]
            else:
                node_map[model_node] = ground

        # Internal nodes
        for model_node, global_idx in internal_nodes.items():
            node_map[model_node] = global_idx

        # Also map sim_node names (OpenVAF uses these internally)
        for model_node in self.model_nodes[:-1]:
            node_map[f'sim_{model_node}'] = node_map.get(model_node, ground)

        # Pre-compute voltage node pairs for fast update
        voltage_node_pairs = []
        voltage_indices = []
        for i, kind in enumerate(self.param_kinds):
            if kind == 'voltage':
                voltage_indices.append(i)
                name_param = self.param_names[i]
                node_pair = self._parse_voltage_param(name_param, node_map, ground)
                voltage_node_pairs.append(node_pair)

        # Store voltage indices if not already set
        if not self._voltage_indices:
            self._voltage_indices = voltage_indices

        # Build input array (static params only, voltages set to 0)
        inputs = []
        for param_name, kind in zip(self.param_names, self.param_kinds):
            if kind == 'voltage':
                inputs.append(0.0)  # Will be updated dynamically
            elif kind == 'param':
                param_lower = param_name.lower()
                if param_lower in params:
                    inputs.append(float(params[param_lower]))
                elif param_name in params:
                    inputs.append(float(params[param_name]))
                elif 'temperature' in param_lower or param_name == '$temperature':
                    inputs.append(temperature)
                elif param_lower in ('tnom', 'tref', 'tr'):
                    inputs.append(300.0)
                elif param_lower == 'mfactor':
                    inputs.append(params.get('mfactor', 1.0))
                else:
                    inputs.append(1.0)
            elif kind == 'hidden_state':
                inputs.append(0.0)
            elif kind == 'temperature':
                inputs.append(temperature)
            else:
                inputs.append(0.0)

        # Accumulate inputs
        if self._static_inputs is None:
            self._static_inputs = np.array([inputs])
        else:
            self._static_inputs = np.vstack([self._static_inputs, inputs])

        self._device_contexts.append(DeviceContext(
            name=name,
            node_map=node_map,
            ext_nodes=ext_nodes,
            voltage_node_pairs=voltage_node_pairs,
        ))

    def _parse_voltage_param(
        self,
        name: str,
        node_map: Dict[str, int],
        ground: int
    ) -> Tuple[int, int]:
        """Parse a voltage parameter name and return (node1_idx, node2_idx).

        Handles formats like "V(GP,SI)" or "V(DI)".
        """
        # PSP103 internal node mapping
        internal_name_map = {
            'GP': 'node4', 'SI': 'node5', 'DI': 'node6', 'BP': 'node7',
            'BS': 'node8', 'BD': 'node9', 'BI': 'node10', 'NOI': 'node11',
            'G': 'node1', 'D': 'node0', 'S': 'node2', 'B': 'node3',
        }

        match = re.match(r'V\(([^,)]+)(?:,([^)]+))?\)', name)
        if not match:
            return (ground, ground)

        node1_name = match.group(1).strip()
        node2_name = match.group(2).strip() if match.group(2) else None

        # Resolve node1
        if node1_name in internal_name_map:
            node1_name = internal_name_map[node1_name]
        node1_idx = node_map.get(node1_name, node_map.get(node1_name.lower(), ground))

        # Resolve node2
        if node2_name:
            if node2_name in internal_name_map:
                node2_name = internal_name_map[node2_name]
            node2_idx = node_map.get(node2_name, node_map.get(node2_name.lower(), ground))
        else:
            node2_idx = ground

        return (node1_idx, node2_idx)

    def prepare(self) -> None:
        """Finalize batch setup. Call after all devices are added."""
        if not self._device_contexts:
            raise RuntimeError("No devices added to batch")
        self._prepared = True

    def num_devices(self) -> int:
        """Return number of devices in batch."""
        return len(self._device_contexts)

    def update_voltage_inputs(self, V: np.ndarray) -> jnp.ndarray:
        """Update voltage parameters from current voltage solution.

        This is the fast path called each NR iteration - only updates voltages.

        Args:
            V: Current voltage solution array

        Returns:
            JAX array with updated inputs ready for vmapped evaluation
        """
        inputs = self._static_inputs.copy()

        for dev_idx, ctx in enumerate(self._device_contexts):
            for i, (n1, n2) in enumerate(ctx.voltage_node_pairs):
                v1 = V[n1] if n1 < len(V) else 0.0
                v2 = V[n2] if n2 < len(V) else 0.0
                inputs[dev_idx, self._voltage_indices[i]] = v1 - v2

        return jnp.array(inputs)

    def evaluate(self, V: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate all devices at current voltages.

        Args:
            V: Current voltage solution array

        Returns:
            (batch_residuals, batch_jacobian) where:
                batch_residuals: Shape (num_devices, num_nodes)
                batch_jacobian: Shape (num_devices, num_jac_entries)
        """
        if not self._prepared:
            self.prepare()

        inputs = self.update_voltage_inputs(V)
        return self.vmapped_fn(inputs)

    def stamp_into_system(
        self,
        batch_residuals: jnp.ndarray,
        batch_jacobian: jnp.ndarray,
        f: np.ndarray,
        J: np.ndarray,
        ground: int = 0,
    ) -> None:
        """Stamp batched evaluation results into system matrices.

        Args:
            batch_residuals: Shape (num_devices, num_nodes) residuals
            batch_jacobian: Shape (num_devices, num_jac_entries) jacobian values
            f: Residual vector to stamp into (modified in place)
            J: Jacobian matrix to stamp into (modified in place)
            ground: Ground node index
        """
        node_names = self.array_metadata['node_names']
        jacobian_keys = self.array_metadata['jacobian_keys']

        # Convert to numpy for stamping
        batch_residuals_np = np.asarray(batch_residuals)
        batch_jacobian_np = np.asarray(batch_jacobian)

        for dev_idx, ctx in enumerate(self._device_contexts):
            node_map = ctx.node_map

            # Stamp residuals
            for res_idx, node_name in enumerate(node_names):
                # Map node name to global index
                if node_name.startswith('sim_'):
                    model_node = node_name[4:]
                else:
                    model_node = node_name

                node_idx = node_map.get(model_node, node_map.get(node_name, None))
                if node_idx is None or node_idx == ground:
                    continue

                resist = batch_residuals_np[dev_idx, res_idx]
                if not np.isnan(resist) and node_idx > 0 and node_idx - 1 < len(f):
                    f[node_idx - 1] += resist

            # Stamp Jacobian
            for jac_idx, (row_name, col_name) in enumerate(jacobian_keys):
                row_model = row_name[4:] if row_name.startswith('sim_') else row_name
                col_model = col_name[4:] if col_name.startswith('sim_') else col_name

                row_idx = node_map.get(row_model, node_map.get(row_name, None))
                col_idx = node_map.get(col_model, node_map.get(col_name, None))

                if row_idx is None or col_idx is None:
                    continue
                if row_idx == ground or col_idx == ground:
                    continue

                resist = batch_jacobian_np[dev_idx, jac_idx]
                if not np.isnan(resist):
                    ri = row_idx - 1
                    ci = col_idx - 1
                    if 0 <= ri < len(f) and 0 <= ci < J.shape[1]:
                        J[ri, ci] += resist

    @property
    def node_names(self) -> List[str]:
        """Node names in residual array order."""
        return self.array_metadata['node_names']

    @property
    def jacobian_keys(self) -> List[Tuple[str, str]]:
        """(row, col) tuples in jacobian array order."""
        return self.array_metadata['jacobian_keys']


# Module-level cache for compiled models
_model_cache: Dict[str, Any] = {}


def compile_model(
    va_path: str,
    allow_analog_in_cond: bool = False,
    cache: bool = True,
) -> CompiledModelBatch:
    """Compile a Verilog-A model for batched evaluation.

    Args:
        va_path: Path to .va file
        allow_analog_in_cond: Allow analog statements in conditionals
        cache: Whether to cache the compiled model

    Returns:
        CompiledModelBatch ready to accept devices
    """
    va_path = str(va_path)
    cache_key = f"{va_path}:{allow_analog_in_cond}"

    if cache and cache_key in _model_cache:
        # Return new batch using cached module
        return CompiledModelBatch.from_module(_model_cache[cache_key])

    batch = CompiledModelBatch.from_va_file(va_path, allow_analog_in_cond)

    if cache:
        _model_cache[cache_key] = batch.module

    return batch


def clear_model_cache() -> None:
    """Clear the compiled model cache."""
    _model_cache.clear()


# Legacy functions for backwards compatibility
def get_vacask_resistor() -> VADevice:
    """Get compiled VACASK resistor model."""
    key = 'vacask_resistor'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices" / "resistor.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def get_vacask_diode() -> VADevice:
    """Get compiled VACASK diode model."""
    key = 'vacask_diode'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices" / "diode.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def get_vacask_capacitor() -> VADevice:
    """Get compiled VACASK capacitor model."""
    key = 'vacask_capacitor'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices" / "capacitor.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def get_mosfet_level1() -> VADevice:
    """Get compiled level-1 MOSFET model."""
    key = 'mosfet_level1'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "tests" / "models" / "mosfet_level1.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def stamp_device_into_system(
    residual: Array,
    jacobian_data: List[float],
    jacobian_rows: List[int],
    jacobian_cols: List[int],
    device: VADevice,
    node_map: Dict[str, int],
    voltages: Dict[str, float],
    params: Dict[str, Any],
    temperature: float = 300.0,
) -> Tuple[Array, List[float], List[int], List[int]]:
    """Stamp device residuals and Jacobian into circuit system.

    Args:
        residual: Circuit residual vector to update
        jacobian_data/rows/cols: COO format Jacobian to update
        device: VADevice to evaluate
        node_map: Maps device terminal names to circuit node indices
        voltages: Terminal voltages
        params: Device parameters
        temperature: Temperature

    Returns:
        Updated (residual, jacobian_data, jacobian_rows, jacobian_cols)
    """
    dev_residuals, dev_jacobian = device.evaluate(voltages, params, temperature)

    # Stamp residuals
    for node_name, res in dev_residuals.items():
        if node_name in node_map:
            node_idx = node_map[node_name]
            if node_idx > 0:  # Skip ground
                residual = residual.at[node_idx].add(float(res['resist']))

    # Stamp Jacobian
    for (row_name, col_name), jac in dev_jacobian.items():
        if row_name in node_map and col_name in node_map:
            row_idx = node_map[row_name]
            col_idx = node_map[col_name]
            if row_idx > 0 and col_idx > 0:  # Skip ground
                jacobian_data.append(float(jac['resist']))
                jacobian_rows.append(row_idx - 1)  # 0-indexed for reduced system
                jacobian_cols.append(col_idx - 1)

    return residual, jacobian_data, jacobian_rows, jacobian_cols
