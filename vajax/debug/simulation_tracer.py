"""Simulation tracing utilities for debugging circuit simulation issues.

This module provides tools for tracing voltage values, node mappings,
and device parameter flow through the simulation.

Usage:
    from vajax.debug.simulation_tracer import SimulationTracer

    engine = CircuitEngine(sim_path)
    engine.parse()

    tracer = SimulationTracer(engine)
    tracer.print_node_allocation()
    tracer.print_voltage_mapping('sp_diode')
    tracer.trace_device_params('sp_diode', V_test)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from vajax.analysis.node_setup import setup_internal_nodes

if TYPE_CHECKING:
    from vajax.analysis.engine import CircuitEngine


@dataclass
class NodeAllocation:
    """Information about node allocation in the circuit."""

    num_external: int
    num_internal: int
    num_total: int
    external_nodes: Dict[str, int]  # name -> index
    internal_nodes: Dict[str, Dict[str, int]]  # device_name -> {node_name -> index}

    def __str__(self) -> str:
        lines = [
            "=== Node Allocation ===",
            f"External nodes: {self.num_external}",
            f"Internal nodes: {self.num_internal}",
            f"Total nodes: {self.num_total}",
            "",
            "External node mapping:",
        ]
        for name, idx in sorted(self.external_nodes.items(), key=lambda x: x[1]):
            lines.append(f"  {idx}: {name}")

        if self.internal_nodes:
            lines.append("")
            lines.append("Internal node mapping:")
            for dev_name, nodes in sorted(self.internal_nodes.items()):
                if nodes:
                    node_str = ", ".join(f"{k}={v}" for k, v in nodes.items())
                    lines.append(f"  {dev_name}: {node_str}")

        return "\n".join(lines)


@dataclass
class VoltageMapping:
    """Information about voltage parameter mapping for a device model."""

    model_type: str
    num_devices: int
    voltage_indices: List[int]  # param indices that are voltages
    voltage_positions: List[int]  # positions in device_params
    voltage_node1: np.ndarray  # shape (n_devices, n_voltage_params)
    voltage_node2: np.ndarray  # shape (n_devices, n_voltage_params)
    voltage_param_names: List[str]  # e.g., ['V(a_int,a)', 'V(a_int,c)']
    device_names: List[str]
    node_name_map: Dict[int, str]  # index -> name

    def __str__(self) -> str:
        lines = [
            f"=== Voltage Mapping for {self.model_type} ===",
            f"Devices: {self.num_devices}",
            f"Voltage params: {self.voltage_param_names}",
            f"Voltage indices in eval params: {self.voltage_indices}",
            f"Positions in device_params: {list(self.voltage_positions)}",
            "",
        ]

        for dev_idx, dev_name in enumerate(self.device_names):
            lines.append(f"{dev_name}:")
            for v_idx, v_name in enumerate(self.voltage_param_names):
                n1 = int(self.voltage_node1[dev_idx, v_idx])
                n2 = int(self.voltage_node2[dev_idx, v_idx])
                name1 = self.node_name_map.get(n1, f"node{n1}")
                name2 = self.node_name_map.get(n2, f"node{n2}")
                lines.append(f"  {v_name} = V[{n1}] - V[{n2}] = V({name1}) - V({name2})")

        return "\n".join(lines)


@dataclass
class DeviceParamsTrace:
    """Trace of device parameters at a given voltage state."""

    model_type: str
    V: np.ndarray  # voltage vector used
    voltage_updates: np.ndarray  # computed voltage differences
    device_params_before: np.ndarray  # device_params before voltage update
    device_params_after: np.ndarray  # device_params after voltage update
    voltage_positions: List[int]
    device_names: List[str]
    voltage_param_names: List[str]

    def __str__(self) -> str:
        lines = [
            f"=== Device Params Trace for {self.model_type} ===",
            f"V vector (len={len(self.V)}): {self.V}",
            "",
            "Voltage updates per device:",
        ]

        for dev_idx, dev_name in enumerate(self.device_names):
            lines.append(f"  {dev_name}:")
            for v_idx, v_name in enumerate(self.voltage_param_names):
                v_update = self.voltage_updates[dev_idx, v_idx]
                lines.append(f"    {v_name} = {v_update:.6f}V")

        lines.append("")
        lines.append(f"device_params shape: {self.device_params_after.shape}")
        lines.append(f"Voltage positions: {self.voltage_positions}")

        return "\n".join(lines)


class SimulationTracer:
    """Tracer for debugging circuit simulation issues.

    Provides methods to inspect node allocation, voltage mappings,
    and device parameter flow through the simulation.
    """

    def __init__(self, engine: "CircuitEngine"):
        """Initialize tracer with a CircuitEngine instance.

        Args:
            engine: A CircuitEngine that has been parsed (engine.parse() called).
        """
        self.engine = engine
        self._node_allocation: Optional[NodeAllocation] = None
        self._device_internal_nodes: Optional[Dict[str, Dict[str, int]]] = None

    def get_node_allocation(self) -> NodeAllocation:
        """Get node allocation information.

        Calls setup_internal_nodes() if not already done.

        Returns:
            NodeAllocation with external and internal node mappings.
        """
        if self._node_allocation is not None:
            return self._node_allocation

        # Get external nodes from engine
        external_nodes = dict(self.engine.node_names)
        num_external = self.engine.num_nodes

        # Set up internal nodes (this may have already been done)
        n_total, device_internal_nodes = setup_internal_nodes(
            devices=self.engine.devices,
            num_nodes=self.engine.num_nodes,
            compiled_models=self.engine._compiled_models,
            device_collapse_decisions=self.engine._device_collapse_decisions,
        )
        self._device_internal_nodes = device_internal_nodes

        num_internal = n_total - num_external

        self._node_allocation = NodeAllocation(
            num_external=num_external,
            num_internal=num_internal,
            num_total=n_total,
            external_nodes=external_nodes,
            internal_nodes=device_internal_nodes,
        )

        return self._node_allocation

    def print_node_allocation(self) -> None:
        """Print node allocation information."""
        print(self.get_node_allocation())

    def get_voltage_mapping(self, model_type: str) -> VoltageMapping:
        """Get voltage parameter mapping for a device model.

        Args:
            model_type: The model type (e.g., 'sp_diode', 'psp103').

        Returns:
            VoltageMapping with node arrays and param info.
        """
        # Ensure internal nodes are set up
        alloc = self.get_node_allocation()

        # Get compiled model
        compiled = self.engine._compiled_models.get(model_type)
        if not compiled:
            raise ValueError(f"Model {model_type} not compiled")

        # Get devices of this type
        openvaf_devices = [
            d for d in self.engine.devices if d.get("model") == model_type and d.get("is_openvaf")
        ]

        if not openvaf_devices:
            raise ValueError(f"No devices of type {model_type}")

        # Get voltage indices and param names
        param_names = compiled.get("param_names", [])
        param_kinds = compiled.get("param_kinds", [])
        voltage_indices = [i for i, k in enumerate(param_kinds) if k == "voltage"]
        voltage_param_names = [param_names[i] for i in voltage_indices]

        # Build voltage_node arrays by calling _prepare_static_inputs
        # This also populates voltage_positions_in_varying in compiled dict
        ground = 0
        _, device_contexts, _, _ = self.engine._prepare_static_inputs(
            model_type, openvaf_devices, self._device_internal_nodes, ground
        )

        # Get voltage positions in device_params (populated by _prepare_static_inputs)
        voltage_positions = compiled.get("voltage_positions_in_varying", jnp.array([]))

        # Build voltage_node arrays
        voltage_node1 = np.array(
            [[n1 for n1, n2 in ctx["voltage_node_pairs"]] for ctx in device_contexts],
            dtype=np.int32,
        )
        voltage_node2 = np.array(
            [[n2 for n1, n2 in ctx["voltage_node_pairs"]] for ctx in device_contexts],
            dtype=np.int32,
        )

        # Build node name map (index -> name)
        node_name_map = {v: k for k, v in alloc.external_nodes.items()}
        for dev_name, nodes in alloc.internal_nodes.items():
            for node_name, idx in nodes.items():
                node_name_map[idx] = f"{dev_name}.{node_name}"

        device_names = [d["name"] for d in openvaf_devices]

        return VoltageMapping(
            model_type=model_type,
            num_devices=len(openvaf_devices),
            voltage_indices=voltage_indices,
            voltage_positions=list(np.array(voltage_positions)),
            voltage_node1=voltage_node1,
            voltage_node2=voltage_node2,
            voltage_param_names=voltage_param_names,
            device_names=device_names,
            node_name_map=node_name_map,
        )

    def print_voltage_mapping(self, model_type: str) -> None:
        """Print voltage parameter mapping for a device model."""
        print(self.get_voltage_mapping(model_type))

    def trace_device_params(
        self,
        model_type: str,
        V: np.ndarray,
    ) -> DeviceParamsTrace:
        """Trace device parameters at a given voltage state.

        Args:
            model_type: The model type.
            V: Voltage vector (including ground at index 0).

        Returns:
            DeviceParamsTrace showing voltage updates and device_params.
        """
        mapping = self.get_voltage_mapping(model_type)
        compiled = self.engine._compiled_models.get(model_type)

        V_jnp = jnp.array(V)

        # Compute voltage updates
        voltage_updates = V_jnp[mapping.voltage_node1] - V_jnp[mapping.voltage_node2]

        # Get device_params
        device_params = compiled.get("device_params", jnp.zeros((mapping.num_devices, 0)))
        voltage_positions = jnp.array(mapping.voltage_positions)

        # Update device_params with voltages
        device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

        return DeviceParamsTrace(
            model_type=model_type,
            V=np.array(V),
            voltage_updates=np.array(voltage_updates),
            device_params_before=np.array(device_params),
            device_params_after=np.array(device_params_updated),
            voltage_positions=mapping.voltage_positions,
            device_names=mapping.device_names,
            voltage_param_names=mapping.voltage_param_names,
        )

    def print_device_params_trace(self, model_type: str, V: np.ndarray) -> None:
        """Print device params trace at a given voltage state."""
        print(self.trace_device_params(model_type, V))

    def create_test_voltage(
        self,
        external_voltages: Dict[str, float],
        internal_voltages: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Create a voltage vector for testing.

        Args:
            external_voltages: Dict of node_name -> voltage for external nodes.
            internal_voltages: Optional dict of "dev.node" -> voltage for internal nodes.

        Returns:
            Voltage vector with ground at index 0.
        """
        alloc = self.get_node_allocation()
        V = np.zeros(alloc.num_total)

        # Set external node voltages
        for name, voltage in external_voltages.items():
            if name in alloc.external_nodes:
                idx = alloc.external_nodes[name]
                V[idx] = voltage

        # Set internal node voltages
        if internal_voltages:
            for key, voltage in internal_voltages.items():
                if "." in key:
                    dev_name, node_name = key.split(".", 1)
                    if dev_name in alloc.internal_nodes:
                        nodes = alloc.internal_nodes[dev_name]
                        if node_name in nodes:
                            V[nodes[node_name]] = voltage

        return V

    def print_simulation_summary(self) -> None:
        """Print a summary of the simulation setup."""
        alloc = self.get_node_allocation()
        print("=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(alloc)
        print()

        # Print voltage mapping for each OpenVAF model type
        model_types = set()
        for d in self.engine.devices:
            if d.get("is_openvaf"):
                model_types.add(d["model"])

        for model_type in sorted(model_types):
            print()
            try:
                self.print_voltage_mapping(model_type)
            except Exception as e:
                print(f"Error getting voltage mapping for {model_type}: {e}")


def trace_simulation(engine: "CircuitEngine") -> SimulationTracer:
    """Convenience function to create a tracer and print summary.

    Args:
        engine: A CircuitEngine that has been parsed.

    Returns:
        SimulationTracer instance for further inspection.
    """
    tracer = SimulationTracer(engine)
    tracer.print_simulation_summary()
    return tracer
