"""Modified Nodal Analysis (MNA) matrix assembly for JAX-SPICE

The MNA formulation:
    G * V + C * dV/dt = I
    
Where:
    G = conductance matrix (from resistive elements)
    C = capacitance matrix (from reactive elements) 
    V = node voltage vector
    I = current source vector

For Newton-Raphson iteration, we solve:
    J * delta_V = -residual
    
Where J is the Jacobian (partial derivatives of residual w.r.t. voltages).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jax_spice.netlist.circuit import Circuit, Instance, Model
from jax_spice.analysis.context import AnalysisContext


# =============================================================================
# Parameter Evaluation
# =============================================================================


def eval_param_simple(value, vdd: float = 1.2, defaults: dict = None):
    """Simple parameter evaluation for common cases.

    Handles:
    - Numbers (int, float)
    - String 'vdd' -> vdd value
    - String '0' -> 0.0
    - String 'w', 'l' -> default MOSFET dimensions
    - SPICE number suffixes (1u, 100n, etc.)
    """
    if defaults is None:
        defaults = {
            'w': 1e-6,      # Default MOSFET width = 1u
            'l': 0.2e-6,    # Default MOSFET length = 0.2u
            'ld': 0.5e-6,   # Default drain extension
            'ls': 0.5e-6,   # Default source extension
        }

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value_lower = value.lower().strip()

        # Common parameter references
        if value_lower == 'vdd':
            return vdd
        if value_lower in ('0', '0.0', 'vss', 'gnd'):
            return 0.0
        if value_lower in defaults:
            return defaults[value_lower]

        # SPICE number suffixes
        suffixes = {
            't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
            'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
        }
        for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
            if value_lower.endswith(suffix):
                try:
                    return float(value_lower[:-len(suffix)]) * mult
                except ValueError:
                    pass

        # Try direct conversion
        try:
            return float(value)
        except ValueError:
            pass

    return 0.0


# =============================================================================
# Device Type Enumeration
# =============================================================================


class DeviceType(Enum):
    """Enumeration of supported device types for vectorized evaluation"""
    VSOURCE = 'vsource'
    ISOURCE = 'isource'
    RESISTOR = 'resistor'
    CAPACITOR = 'capacitor'
    MOSFET = 'mosfet'
    VERILOG_A = 'verilog_a'
    UNKNOWN = 'unknown'


# =============================================================================
# Vectorized Device Group
# =============================================================================


@dataclass
class VectorizedDeviceGroup:
    """Group of devices of the same type for vectorized evaluation

    This enables GPU-friendly batch processing of device evaluations.
    Instead of iterating over devices one-by-one with Python loops,
    we evaluate all devices of the same type in parallel using JAX.

    Attributes:
        device_type: Type of devices in this group
        n_devices: Number of devices in the group
        device_names: List of device instance names (for debugging)
        node_indices: (n_devices, n_terminals) int array of node indices
        params: Dict of parameter arrays, each shape (n_devices,)
    """
    device_type: DeviceType
    n_devices: int
    device_names: List[str]
    node_indices: Array  # (n_devices, n_terminals) int32 array
    params: Dict[str, Array]  # Parameter name -> (n_devices,) array

    @property
    def n_terminals(self) -> int:
        """Number of terminals per device"""
        return self.node_indices.shape[1] if self.n_devices > 0 else 0


@dataclass
class DeviceInfo:
    """Information about a device instance for simulation"""
    name: str
    model_name: str
    terminals: List[str]  # Terminal names
    node_indices: List[int]  # Corresponding node indices
    params: Dict[str, Any]  # Instance parameters
    eval_fn: Optional[Callable] = None  # Device evaluation function


@dataclass 
class MNASystem:
    """MNA system for circuit simulation
    
    Manages the mapping between circuit topology and matrix entries.
    Supports both dense and sparse matrix assembly.
    
    Attributes:
        num_nodes: Number of nodes (including ground = 0)
        node_names: Mapping from node name to index
        devices: List of device instances
        ground_node: Index of ground node (always 0)
    """
    num_nodes: int
    node_names: Dict[str, int]
    devices: List[DeviceInfo] = field(default_factory=list)
    ground_node: int = 0

    # Model registry: model_name -> (device_class, default_params)
    model_registry: Dict[str, Tuple[Any, Dict]] = field(default_factory=dict)

    # Vectorized device groups for GPU-friendly batch evaluation
    device_groups: List[VectorizedDeviceGroup] = field(default_factory=list)
    
    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        top_subckt: str,
        model_registry: Dict[str, Tuple[Any, Dict]]
    ) -> 'MNASystem':
        """Create MNA system from parsed circuit
        
        Args:
            circuit: Parsed circuit from netlist
            top_subckt: Name of top-level subcircuit to simulate
            model_registry: Mapping from model names to device classes
            
        Returns:
            MNASystem ready for simulation
        """
        # Flatten the circuit hierarchy
        flat_instances, nodes = circuit.flatten(top_subckt)
        
        # Create the system
        system = cls(
            num_nodes=len(nodes),
            node_names=nodes,
            model_registry=model_registry
        )
        
        # Add devices
        for inst in flat_instances:
            # Look up the model
            model_info = circuit.models.get(inst.model)
            if model_info is None:
                raise ValueError(f"Model '{inst.model}' not found for instance '{inst.name}'")
            
            # Look up the device class
            if model_info.module not in model_registry:
                raise ValueError(f"Unknown device module '{model_info.module}' for model '{inst.model}'")
            
            device_class, default_params = model_registry[model_info.module]
            
            # Merge parameters: defaults < model params < instance params
            params = {**default_params, **model_info.params, **inst.params}
            
            # Map terminal names to node indices
            node_indices = [nodes[t] for t in inst.terminals]
            
            device = DeviceInfo(
                name=inst.name,
                model_name=inst.model,
                terminals=inst.terminals,
                node_indices=node_indices,
                params=params,
                eval_fn=device_class
            )
            system.devices.append(device)
        
        return system
    
    def build_jacobian_and_residual(
        self,
        voltages: Array,
        context: AnalysisContext
    ) -> Tuple[Array, Array]:
        """Build Jacobian matrix and residual vector
        
        Args:
            voltages: Current node voltage estimates (shape: [num_nodes])
            context: Analysis context with time, etc.
            
        Returns:
            Tuple of (jacobian, residual) where:
                jacobian: Shape [num_nodes-1, num_nodes-1] (ground eliminated)
                residual: Shape [num_nodes-1]
        """
        n = self.num_nodes - 1  # Exclude ground

        # Use float32 on Metal (no float64 support), float64 elsewhere
        dtype = jnp.float32 if jax.default_backend() == 'METAL' else jnp.float64

        # Initialize Jacobian and residual
        jacobian = jnp.zeros((n, n), dtype=dtype)
        residual = jnp.zeros(n, dtype=dtype)
        
        # Evaluate each device and stamp into matrix
        for device in self.devices:
            if device.eval_fn is None:
                continue
                
            # Build voltage dictionary for this device
            dev_voltages = {}
            for i, (term, node_idx) in enumerate(zip(device.terminals, device.node_indices)):
                dev_voltages[term] = voltages[node_idx]
            
            # Evaluate device
            stamps = device.eval_fn(dev_voltages, device.params, context)
            
            # Stamp currents into residual
            for term, current in stamps.currents.items():
                term_idx = device.terminals.index(term)
                node_idx = device.node_indices[term_idx]
                if node_idx != self.ground_node:
                    residual = residual.at[node_idx - 1].add(current)
            
            # Stamp conductances into Jacobian
            for (term_i, term_j), conductance in stamps.conductances.items():
                idx_i = device.terminals.index(term_i)
                idx_j = device.terminals.index(term_j)
                node_i = device.node_indices[idx_i]
                node_j = device.node_indices[idx_j]
                
                # Skip ground entries
                if node_i != self.ground_node and node_j != self.ground_node:
                    jacobian = jacobian.at[node_i - 1, node_j - 1].add(conductance)
        
        return jacobian, residual
    
    def get_node_voltage(self, solution: Array, node_name: str) -> float:
        """Get voltage at a named node from solution vector
        
        Args:
            solution: Solution vector (length num_nodes - 1, ground excluded)
            node_name: Name of node
            
        Returns:
            Voltage at node (0.0 for ground)
        """
        node_idx = self.node_names.get(node_name)
        if node_idx is None:
            raise ValueError(f"Unknown node: {node_name}")
        if node_idx == self.ground_node:
            return 0.0
        return float(solution[node_idx - 1])
    
    def full_voltage_vector(self, solution: Array) -> Array:
        """Convert solution to full voltage vector including ground

        Args:
            solution: Solution vector (length num_nodes - 1)

        Returns:
            Full voltage vector (length num_nodes) with ground = 0
        """
        return jnp.concatenate([jnp.array([0.0]), solution])

    def build_sparse_jacobian_and_residual(
        self,
        voltages: Array,
        context: AnalysisContext
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]], np.ndarray]:
        """Build Jacobian matrix and residual vector in sparse CSR format

        This is more memory-efficient for large circuits. Uses COO accumulation
        then converts to CSR for the sparse solver.

        Args:
            voltages: Current node voltage estimates (shape: [num_nodes])
            context: Analysis context with time, etc.

        Returns:
            Tuple of ((data, indices, indptr, shape), residual) where:
                data, indices, indptr: CSR format arrays
                shape: (n, n) matrix shape
                residual: Shape [num_nodes-1] numpy array
        """
        n = self.num_nodes - 1  # Exclude ground

        # Accumulate triplets (row, col, value) for COO format
        rows = []
        cols = []
        values = []
        residual = np.zeros(n, dtype=np.float64)

        # Evaluate each device and accumulate stamps
        for device in self.devices:
            if device.eval_fn is None:
                continue

            # Build voltage dictionary for this device
            dev_voltages = {}
            for i, (term, node_idx) in enumerate(zip(device.terminals, device.node_indices)):
                dev_voltages[term] = float(voltages[node_idx])

            # Evaluate device
            stamps = device.eval_fn(dev_voltages, device.params, context)

            # Stamp currents into residual
            for term, current in stamps.currents.items():
                term_idx = device.terminals.index(term)
                node_idx = device.node_indices[term_idx]
                if node_idx != self.ground_node:
                    residual[node_idx - 1] += float(current)

            # Stamp conductances into Jacobian (triplet format)
            for (term_i, term_j), conductance in stamps.conductances.items():
                idx_i = device.terminals.index(term_i)
                idx_j = device.terminals.index(term_j)
                node_i = device.node_indices[idx_i]
                node_j = device.node_indices[idx_j]

                # Skip ground entries
                if node_i != self.ground_node and node_j != self.ground_node:
                    rows.append(node_i - 1)
                    cols.append(node_j - 1)
                    values.append(float(conductance))

        # Add GMIN to all diagonal entries for numerical stability
        # This creates a small conductance from each node to ground
        # Read gmin from context (allows GMIN stepping for convergence)
        gmin = context.gmin if hasattr(context, 'gmin') else 1e-12
        for i in range(n):
            rows.append(i)
            cols.append(i)
            values.append(gmin)

        # Convert to CSR format
        from jax_spice.analysis.sparse import build_csr_arrays

        rows_arr = np.array(rows, dtype=np.int32)
        cols_arr = np.array(cols, dtype=np.int32)
        values_arr = np.array(values, dtype=np.float64)

        data, indices, indptr = build_csr_arrays(
            rows_arr, cols_arr, values_arr, (n, n)
        )

        return (data, indices, indptr, (n, n)), residual

    def build_vectorized_jacobian_and_residual(
        self,
        voltages: Array,
        context: AnalysisContext
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]], np.ndarray]:
        """Build Jacobian matrix and residual using vectorized device evaluation

        GPU-friendly version that evaluates all devices of the same type in parallel
        using JAX operations instead of Python loops.

        Requires device_groups to be populated (via build_device_groups()).

        Args:
            voltages: Current node voltage estimates (shape: [num_nodes])
            context: Analysis context with time, etc.

        Returns:
            Tuple of ((data, indices, indptr, shape), residual) where:
                data, indices, indptr: CSR format arrays
                shape: (n, n) matrix shape
                residual: Shape [num_nodes-1] numpy array
        """
        from jax_spice.devices.vsource import vsource_batch
        from jax_spice.devices.resistor import resistor_batch

        n = self.num_nodes - 1  # Exclude ground

        # Accumulate triplets (row, col, value) for COO format
        rows = []
        cols = []
        values = []
        residual = np.zeros(n, dtype=np.float64)

        # Process each device group with vectorized evaluation
        for group in self.device_groups:
            if group.n_devices == 0:
                continue

            # Get terminal voltages for all devices in this group
            # node_indices is (n_devices, n_terminals)
            V_batch = voltages[group.node_indices]  # (n_devices, n_terminals)

            if group.device_type == DeviceType.VSOURCE:
                # Vectorized voltage source evaluation
                V_target = group.params.get('v', group.params.get('dc', jnp.zeros(group.n_devices)))
                I_batch, G_batch = vsource_batch(V_batch, V_target)

                # Stamp into matrix: 2-terminal device with stamps:
                # Residual: f[p] += I, f[n] -= I
                # Jacobian: G[p,p] += G, G[p,n] -= G, G[n,p] -= G, G[n,n] += G
                node_p = np.array(group.node_indices[:, 0])  # (n_devices,)
                node_n = np.array(group.node_indices[:, 1])  # (n_devices,)
                I_np = np.array(I_batch)
                G_np = np.array(G_batch)

                # Stamp residuals (KCL current sum at each node)
                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    if np_idx != self.ground_node:
                        residual[np_idx - 1] += I_np[i]
                    if nn_idx != self.ground_node:
                        residual[nn_idx - 1] -= I_np[i]

                # Stamp Jacobian conductances
                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    g = G_np[i]

                    # G[p,p] += G
                    if np_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(g)

                    # G[p,n] -= G
                    if np_idx != self.ground_node and nn_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(-g)

                    # G[n,p] -= G
                    if nn_idx != self.ground_node and np_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(-g)

                    # G[n,n] += G
                    if nn_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(g)

            elif group.device_type == DeviceType.RESISTOR:
                # Vectorized resistor evaluation
                R_batch = group.params.get('r', group.params.get('R', jnp.ones(group.n_devices) * 1000.0))
                I_batch, G_batch = resistor_batch(V_batch, R_batch)

                node_p = np.array(group.node_indices[:, 0])
                node_n = np.array(group.node_indices[:, 1])
                I_np = np.array(I_batch)
                G_np = np.array(G_batch)

                # Same stamping pattern as voltage sources
                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    if np_idx != self.ground_node:
                        residual[np_idx - 1] += I_np[i]
                    if nn_idx != self.ground_node:
                        residual[nn_idx - 1] -= I_np[i]

                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    g = G_np[i]

                    if np_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(g)
                    if np_idx != self.ground_node and nn_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(-g)
                    if nn_idx != self.ground_node and np_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(-g)
                    if nn_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(g)

            else:
                # Fallback to scalar evaluation for unsupported device types
                # (MOSFET, Verilog-A, etc. - to be added later)
                for idx, device_name in enumerate(group.device_names):
                    # Find matching device in self.devices
                    for device in self.devices:
                        if device.name == device_name:
                            dev_voltages = {}
                            for j, (term, node_idx) in enumerate(zip(device.terminals, device.node_indices)):
                                dev_voltages[term] = float(voltages[node_idx])
                            stamps = device.eval_fn(dev_voltages, device.params, context)

                            for term, current in stamps.currents.items():
                                term_idx = device.terminals.index(term)
                                node_idx = device.node_indices[term_idx]
                                if node_idx != self.ground_node:
                                    residual[node_idx - 1] += float(current)

                            for (term_i, term_j), conductance in stamps.conductances.items():
                                idx_i = device.terminals.index(term_i)
                                idx_j = device.terminals.index(term_j)
                                node_i = device.node_indices[idx_i]
                                node_j = device.node_indices[idx_j]
                                if node_i != self.ground_node and node_j != self.ground_node:
                                    rows.append(node_i - 1)
                                    cols.append(node_j - 1)
                                    values.append(float(conductance))
                            break

        # Add GMIN to diagonal
        gmin = context.gmin if hasattr(context, 'gmin') else 1e-12
        for i in range(n):
            rows.append(i)
            cols.append(i)
            values.append(gmin)

        # Convert to CSR format
        from jax_spice.analysis.sparse import build_csr_arrays

        rows_arr = np.array(rows, dtype=np.int32)
        cols_arr = np.array(cols, dtype=np.int32)
        values_arr = np.array(values, dtype=np.float64)

        data, indices, indptr = build_csr_arrays(
            rows_arr, cols_arr, values_arr, (n, n)
        )

        return (data, indices, indptr, (n, n)), residual

    def build_device_groups(self, vdd: float = 1.2) -> None:
        """Build vectorized device groups from the devices list

        Groups devices by type and creates VectorizedDeviceGroup instances
        with pre-computed node indices and parameter arrays.

        Args:
            vdd: Supply voltage for resolving 'vdd' parameter references
        """
        from collections import defaultdict

        # Group devices by type
        type_to_devices: Dict[DeviceType, List[DeviceInfo]] = defaultdict(list)

        for device in self.devices:
            # Determine device type from model name or eval_fn
            model_lower = device.model_name.lower()

            if 'vsource' in model_lower or 'vdc' in model_lower or model_lower == 'v':
                dtype = DeviceType.VSOURCE
            elif 'isource' in model_lower or 'idc' in model_lower or model_lower == 'i':
                dtype = DeviceType.ISOURCE
            elif 'resistor' in model_lower or model_lower.startswith('r'):
                dtype = DeviceType.RESISTOR
            elif 'capacitor' in model_lower or model_lower.startswith('c'):
                dtype = DeviceType.CAPACITOR
            elif 'nmos' in model_lower or 'pmos' in model_lower or 'mosfet' in model_lower:
                dtype = DeviceType.MOSFET
            elif 'psp' in model_lower:  # PSP103 MOSFET model
                dtype = DeviceType.MOSFET
            else:
                dtype = DeviceType.UNKNOWN

            type_to_devices[dtype].append(device)

        # Build VectorizedDeviceGroup for each type
        self.device_groups = []

        for dtype, devices in type_to_devices.items():
            if not devices:
                continue

            n_devices = len(devices)
            device_names = [d.name for d in devices]

            # Build node_indices array
            # All devices of same type should have same number of terminals
            n_terminals = len(devices[0].node_indices)
            node_indices = np.array(
                [d.node_indices for d in devices],
                dtype=np.int32
            )

            # Build parameter arrays
            # Collect all unique parameter names, resolving string references
            all_params: Dict[str, List[float]] = defaultdict(list)
            for device in devices:
                for key, val in device.params.items():
                    # Use eval_param_simple to resolve references like 'vdd'
                    resolved = eval_param_simple(val, vdd=vdd)
                    all_params[key].append(float(resolved))

            # For MOSFETs, add pmos flag based on model name
            if dtype == DeviceType.MOSFET:
                pmos_flags = []
                for device in devices:
                    model_lower = device.model_name.lower()
                    # Detect PMOS by model name patterns
                    is_pmos = 'pmos' in model_lower or model_lower.endswith('p')
                    pmos_flags.append(1.0 if is_pmos else 0.0)
                all_params['pmos'] = pmos_flags

            # Convert to JAX arrays
            params = {}
            for key, vals in all_params.items():
                if len(vals) == n_devices:
                    params[key] = jnp.array(vals)

            group = VectorizedDeviceGroup(
                device_type=dtype,
                n_devices=n_devices,
                device_names=device_names,
                node_indices=jnp.array(node_indices),
                params=params
            )
            self.device_groups.append(group)

    def build_gpu_residual_fn(
        self,
        vdd: float = 1.0,
        gmin: float = 1e-12,
    ) -> Callable[[Array], Array]:
        """Build pure JAX residual function for GPU execution.

        Returns a function f(V) -> residual that uses only JAX operations,
        enabling automatic differentiation and GPU-native execution.

        This replaces Python stamping loops with JAX scatter operations
        for significant speedup on GPU.

        Args:
            vdd: Supply voltage for voltage sources
            gmin: GMIN conductance for numerical stability

        Returns:
            f(V) -> residual function, where V has shape (num_nodes,)
            and residual has shape (num_nodes-1,) excluding ground

        Note:
            Requires device_groups to be populated (via build_device_groups()).
        """
        from jax_spice.devices.resistor import resistor_batch
        from jax_spice.devices.mosfet_simple import mosfet_batch
        from jax_spice.analysis.mna_gpu import (
            stamp_2terminal_residual_gpu,
            stamp_4terminal_residual_gpu,
            stamp_gmin_residual_gpu,
            build_mosfet_params_from_group,
        )

        # Pre-extract static data from device groups
        # This is done once at function build time, not during evaluation
        group_data = []
        for group in self.device_groups:
            if group.n_devices == 0:
                continue

            data = {
                'type': group.device_type,
                'n_devices': group.n_devices,
                'node_indices': group.node_indices,
                'params': group.params,
            }

            # Pre-build parameter arrays for MOSFETs
            if group.device_type == DeviceType.MOSFET:
                data['mosfet_params'] = build_mosfet_params_from_group(group)

            group_data.append(data)

        n = self.num_nodes - 1
        ground_node = self.ground_node

        def residual_fn(V: Array) -> Array:
            """Compute residual vector using GPU-native operations.

            Args:
                V: Node voltages (num_nodes,) including ground

            Returns:
                Residual vector (num_nodes-1,) excluding ground
            """
            residual = jnp.zeros(n, dtype=V.dtype)

            for data in group_data:
                dtype_enum = data['type']
                node_indices = data['node_indices']
                params = data['params']

                # Get terminal voltages for all devices
                V_batch = V[node_indices]  # (n_devices, n_terminals)

                if dtype_enum == DeviceType.RESISTOR:
                    # Resistor: I = (V_p - V_n) / R
                    R_batch = params.get('r', params.get('R', jnp.ones(data['n_devices']) * 1000.0))
                    V_diff = V_batch[:, 0] - V_batch[:, 1]  # V_p - V_n
                    I_batch = V_diff / R_batch

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]

                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.VSOURCE:
                    # Voltage source: enforce V_p - V_n = V_target
                    # Using large conductance method (penalty approach)
                    # Must match VoltageSource.G_BIG = 1e12 in vsource.py
                    V_target = params.get('v', params.get('dc', jnp.zeros(data['n_devices'])))
                    V_diff = V_batch[:, 0] - V_batch[:, 1]
                    # Current = G * (V_diff - V_target) where G is large
                    # V_target is already the resolved voltage value (e.g., 1.2V for VDD)
                    G_vsource = 1e12  # Must match VoltageSource.G_BIG
                    I_batch = G_vsource * (V_diff - V_target)

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]

                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.MOSFET:
                    # MOSFET: evaluate drain current using batched model
                    mosfet_params = data['mosfet_params']

                    # V_batch is (n_devices, 4) with [V_d, V_g, V_s, V_b]
                    Ids, _gm, _gds, _gmb = mosfet_batch(V_batch, mosfet_params)

                    node_d = node_indices[:, 0]
                    node_g = node_indices[:, 1]
                    node_s = node_indices[:, 2]
                    node_b = node_indices[:, 3]

                    residual = stamp_4terminal_residual_gpu(
                        residual, node_d, node_g, node_s, node_b,
                        Ids, ground_node
                    )

            # Add GMIN contribution
            residual = stamp_gmin_residual_gpu(residual, V, gmin, ground_node)

            return residual

        return residual_fn

    def build_parameterized_residual_fn(
        self,
        gmin: float = 1e-12,
    ) -> Callable[[Array, float], Array]:
        """Build parameterized residual function for source stepping.

        Returns a function f(V, vdd_scale) -> residual where vdd_scale
        multiplies all VDD-type voltage sources. This enables JIT-compiled
        source stepping via lax.scan.

        Args:
            gmin: GMIN conductance for numerical stability

        Returns:
            f(V, vdd_scale) -> residual function
        """
        from jax_spice.devices.resistor import resistor_batch
        from jax_spice.devices.mosfet_simple import mosfet_batch
        from jax_spice.analysis.mna_gpu import (
            stamp_2terminal_residual_gpu,
            stamp_4terminal_residual_gpu,
            stamp_gmin_residual_gpu,
            build_mosfet_params_from_group,
        )

        # Pre-extract static data, separating VDD sources from other sources
        group_data = []
        vdd_source_data = None  # Special handling for VDD sources

        for group in self.device_groups:
            if group.n_devices == 0:
                continue

            data = {
                'type': group.device_type,
                'n_devices': group.n_devices,
                'node_indices': group.node_indices,
                'params': group.params,
            }

            if group.device_type == DeviceType.MOSFET:
                data['mosfet_params'] = build_mosfet_params_from_group(group)

            # Check if this is a VDD voltage source group
            if group.device_type == DeviceType.VSOURCE:
                # Check device names for VDD pattern
                is_vdd_source = []
                for name in group.device_names:
                    name_lower = name.lower()
                    is_vdd = 'vdd' in name_lower or name_lower.startswith('v1')
                    is_vdd_source.append(is_vdd)

                data['is_vdd_source'] = jnp.array(is_vdd_source, dtype=jnp.bool_)

            group_data.append(data)

        n = self.num_nodes - 1
        ground_node = self.ground_node

        def residual_fn(V: Array, vdd_scale: float) -> Array:
            """Compute residual with parameterized VDD scaling.

            Args:
                V: Node voltages (num_nodes,) including ground
                vdd_scale: Scale factor for VDD sources (0 to 1 for stepping)

            Returns:
                Residual vector (num_nodes-1,) excluding ground
            """
            residual = jnp.zeros(n, dtype=V.dtype)

            for data in group_data:
                dtype_enum = data['type']
                node_indices = data['node_indices']
                params = data['params']

                V_batch = V[node_indices]

                if dtype_enum == DeviceType.RESISTOR:
                    R_batch = params.get('r', params.get('R', jnp.ones(data['n_devices']) * 1000.0))
                    V_diff = V_batch[:, 0] - V_batch[:, 1]
                    I_batch = V_diff / R_batch

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]

                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.VSOURCE:
                    V_target_base = params.get('v', params.get('dc', jnp.zeros(data['n_devices'])))
                    V_diff = V_batch[:, 0] - V_batch[:, 1]

                    # Scale VDD sources by vdd_scale, leave others unchanged
                    is_vdd = data.get('is_vdd_source', jnp.zeros(data['n_devices'], dtype=jnp.bool_))
                    V_target = jnp.where(is_vdd, V_target_base * vdd_scale, V_target_base)

                    G_vsource = 1e12
                    I_batch = G_vsource * (V_diff - V_target)

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]

                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.MOSFET:
                    mosfet_params = data['mosfet_params']
                    Ids, _gm, _gds, _gmb = mosfet_batch(V_batch, mosfet_params)

                    node_d = node_indices[:, 0]
                    node_g = node_indices[:, 1]
                    node_s = node_indices[:, 2]
                    node_b = node_indices[:, 3]

                    residual = stamp_4terminal_residual_gpu(
                        residual, node_d, node_g, node_s, node_b,
                        Ids, ground_node
                    )

            residual = stamp_gmin_residual_gpu(residual, V, gmin, ground_node)

            return residual

        return residual_fn

    def build_gpu_system_fns(
        self,
        vdd: float = 1.0,
        gmin: float = 1e-12,
    ) -> Tuple[Callable[[Array], Array], Callable[[Array], Array]]:
        """Build both residual and Jacobian functions for GPU execution.

        This returns a pair of pure JAX functions that can run entirely on GPU.
        The Jacobian is computed via automatic differentiation of the residual.

        Args:
            vdd: Supply voltage for voltage sources
            gmin: GMIN conductance for numerical stability

        Returns:
            Tuple of (residual_fn, jacobian_fn) where:
                residual_fn(V) -> residual vector (n-1,)
                jacobian_fn(V) -> Jacobian matrix (n-1, n-1)

        Note:
            Both functions expect V with shape (num_nodes,) including ground.
            The Jacobian is computed using jax.jacfwd which is efficient
            for functions where output dimension < input dimension.
        """
        import jax

        residual_fn = self.build_gpu_residual_fn(vdd=vdd, gmin=gmin)

        # Use jacfwd for efficient Jacobian computation
        # jacfwd is better when n_outputs < n_inputs (typically true for circuits)
        jacobian_fn = jax.jacfwd(residual_fn)

        return residual_fn, jacobian_fn

    def build_transient_residual_fn(
        self,
        gmin: float = 1e-12,
    ) -> Callable[[Array, Array, float], Array]:
        """Build vectorized transient residual function for GPU execution.

        Returns a function f(V, V_prev, dt) -> residual that uses batched
        JAX operations for GPU-friendly execution. This is the transient
        counterpart to build_gpu_residual_fn.

        Capacitor companion model (backward Euler):
            I_cap = C/dt * (V - V_prev)

        Args:
            gmin: GMIN conductance for numerical stability

        Returns:
            f(V, V_prev, dt) -> residual function

        Note:
            Requires device_groups to be populated (via build_device_groups()).
        """
        from jax_spice.devices.mosfet_simple import mosfet_batch
        from jax_spice.analysis.mna_gpu import (
            stamp_2terminal_residual_gpu,
            stamp_4terminal_residual_gpu,
            stamp_gmin_residual_gpu,
            build_mosfet_params_from_group,
        )

        # Pre-extract static data from device groups
        group_data = []
        for group in self.device_groups:
            if group.n_devices == 0:
                continue

            data = {
                'type': group.device_type,
                'n_devices': group.n_devices,
                'node_indices': group.node_indices,
                'params': group.params,
            }

            if group.device_type == DeviceType.MOSFET:
                data['mosfet_params'] = build_mosfet_params_from_group(group)

            group_data.append(data)

        n = self.num_nodes - 1
        ground_node = self.ground_node

        def residual_fn(V: Array, V_prev: Array, dt: float) -> Array:
            """Compute transient residual vector using GPU-native operations.

            Args:
                V: Current node voltages (num_nodes,) including ground
                V_prev: Previous timestep voltages (num_nodes,)
                dt: Timestep size

            Returns:
                Residual vector (num_nodes-1,) excluding ground
            """
            residual = jnp.zeros(n, dtype=V.dtype)

            for data in group_data:
                dtype_enum = data['type']
                node_indices = data['node_indices']
                params = data['params']

                # Get terminal voltages for all devices
                V_batch = V[node_indices]  # (n_devices, n_terminals)

                if dtype_enum == DeviceType.RESISTOR:
                    # Resistor: I = (V_p - V_n) / R
                    R_batch = params.get('r', params.get('R', jnp.ones(data['n_devices']) * 1000.0))
                    V_diff = V_batch[:, 0] - V_batch[:, 1]
                    I_batch = V_diff / R_batch

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]
                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.CAPACITOR:
                    # Capacitor companion model: I = C/dt * (V - V_prev)
                    C_batch = params.get('c', params.get('C', jnp.ones(data['n_devices']) * 1e-12))
                    V_prev_batch = V_prev[node_indices]

                    V_diff = V_batch[:, 0] - V_batch[:, 1]
                    V_prev_diff = V_prev_batch[:, 0] - V_prev_batch[:, 1]

                    # Backward Euler: I = C/dt * (V_curr - V_prev)
                    G_cap = C_batch / dt
                    I_batch = G_cap * (V_diff - V_prev_diff)

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]
                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.VSOURCE:
                    # Voltage source: enforce V_p - V_n = V_target
                    V_target = params.get('v', params.get('dc', jnp.zeros(data['n_devices'])))
                    V_diff = V_batch[:, 0] - V_batch[:, 1]
                    G_vsource = 1e12
                    I_batch = G_vsource * (V_diff - V_target)

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]
                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.ISOURCE:
                    # Current source: fixed current
                    I_target = params.get('i', params.get('dc', jnp.zeros(data['n_devices'])))
                    I_batch = I_target

                    node_p = node_indices[:, 0]
                    node_n = node_indices[:, 1]
                    residual = stamp_2terminal_residual_gpu(
                        residual, node_p, node_n, I_batch, ground_node
                    )

                elif dtype_enum == DeviceType.MOSFET:
                    # MOSFET: batched drain current evaluation
                    mosfet_params = data['mosfet_params']
                    Ids, _gm, _gds, _gmb = mosfet_batch(V_batch, mosfet_params)

                    node_d = node_indices[:, 0]
                    node_g = node_indices[:, 1]
                    node_s = node_indices[:, 2]
                    node_b = node_indices[:, 3]
                    residual = stamp_4terminal_residual_gpu(
                        residual, node_d, node_g, node_s, node_b,
                        Ids, ground_node
                    )

            # Add GMIN contribution
            residual = stamp_gmin_residual_gpu(residual, V, gmin, ground_node)

            return residual

        return residual_fn

    def build_sparsity_pattern(self) -> Tuple[Array, Array]:
        """Build sparsity pattern (row, col indices) for Jacobian.

        Returns sparse indices for the Jacobian matrix based on device connectivity.
        Used for efficient sparse matrix assembly.

        Returns:
            Tuple of (row_indices, col_indices) arrays
        """
        import jax.numpy as jnp

        n = self.num_nodes
        ground_node = 0

        # Collect all (row, col) pairs where Jacobian is nonzero
        # Using sets to avoid duplicates
        nonzero_pairs = set()

        for group in self.device_groups:
            if group.n_devices == 0:
                continue

            node_indices = group.node_indices  # (n_devices, n_terminals)

            if group.device_type == DeviceType.RESISTOR:
                # 2-terminal: J[p,p], J[p,n], J[n,p], J[n,n]
                for dev_idx in range(group.n_devices):
                    p, n_node = int(node_indices[dev_idx, 0]), int(node_indices[dev_idx, 1])
                    for row in [p, n_node]:
                        for col in [p, n_node]:
                            if row != ground_node and col != ground_node:
                                nonzero_pairs.add((row - 1, col - 1))

            elif group.device_type == DeviceType.VSOURCE:
                # 2-terminal penalty method: same as resistor
                for dev_idx in range(group.n_devices):
                    p, n_node = int(node_indices[dev_idx, 0]), int(node_indices[dev_idx, 1])
                    for row in [p, n_node]:
                        for col in [p, n_node]:
                            if row != ground_node and col != ground_node:
                                nonzero_pairs.add((row - 1, col - 1))

            elif group.device_type == DeviceType.MOSFET:
                # 4-terminal: d, g, s, b affect d and s currents
                for dev_idx in range(group.n_devices):
                    d, g, s, b = [int(node_indices[dev_idx, i]) for i in range(4)]
                    # Rows affected: d and s (current flows d->s)
                    # Cols affecting: d, g, s, b (all terminal voltages)
                    for row in [d, s]:
                        for col in [d, g, s, b]:
                            if row != ground_node and col != ground_node:
                                nonzero_pairs.add((row - 1, col - 1))

        # Add GMIN diagonal entries
        for i in range(n - 1):
            nonzero_pairs.add((i, i))

        # Convert to arrays
        if nonzero_pairs:
            pairs = list(nonzero_pairs)
            rows = jnp.array([p[0] for p in pairs], dtype=jnp.int32)
            cols = jnp.array([p[1] for p in pairs], dtype=jnp.int32)
        else:
            rows = jnp.array([], dtype=jnp.int32)
            cols = jnp.array([], dtype=jnp.int32)

        return rows, cols
