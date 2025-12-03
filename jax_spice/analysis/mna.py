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
import jax
import jax.numpy as jnp
from jax import Array

from jax_spice.netlist.circuit import Circuit, Instance, Model
from jax_spice.analysis.context import AnalysisContext


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
