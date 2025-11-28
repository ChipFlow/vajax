"""Circuit netlist and topology management

Inspired by SAX's declarative netlist approach, but extended for nonlinear DC/transient analysis.
"""

from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
import jax.numpy as jnp
from jax import Array

from jax_spice.devices.base import Device, DeviceStamps

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext


class Circuit:
    """Circuit netlist with device instances and connections

    Example:
        ```python
        from jax_spice import Circuit, MOSFETSimple

        ckt = Circuit()

        # Add devices
        ckt.add_device("M1", MOSFETSimple(W=10e-6, L=0.25e-6, pmos=False),
                       connections={'d': 'vout', 'g': 'vin', 's': 'gnd', 'b': 'gnd'})
        ckt.add_device("M2", MOSFETSimple(W=20e-6, L=0.25e-6, pmos=True),
                       connections={'d': 'vout', 'g': 'vin', 's': 'vdd', 'b': 'vdd'})

        # Add voltage sources
        ckt.add_vsource("VDD", 'vdd', 'gnd', 2.5)
        ckt.add_vsource("VIN", 'vin', 'gnd', 1.25)

        # Ground reference
        ckt.set_ground('gnd')
        ```
    """

    def __init__(self):
        """Initialize empty circuit"""
        self.devices: Dict[str, Tuple[Device, Dict[str, str]]] = {}
        self.vsources: Dict[str, Tuple[str, str, float]] = {}
        self.nodes: Dict[str, int] = {}  # node_name -> node_index
        self.ground_node: Optional[str] = None
        self._node_counter = 0

    def add_device(self, name: str, device: Device, connections: Dict[str, str]):
        """Add a device to the circuit

        Args:
            name: Unique device name
            device: Device instance
            connections: Dict mapping device terminals to circuit nodes
                        e.g., {'d': 'vout', 'g': 'vin', 's': 'gnd', 'b': 'gnd'}
        """
        if name in self.devices:
            raise ValueError(f"Device {name} already exists")

        # Validate connections
        required_terminals = set(device.terminals)
        provided_terminals = set(connections.keys())
        if required_terminals != provided_terminals:
            raise ValueError(
                f"Device {name} requires terminals {required_terminals}, "
                f"but got {provided_terminals}"
            )

        self.devices[name] = (device, connections)

        # Register nodes
        for node in connections.values():
            if node not in self.nodes:
                self.nodes[node] = self._node_counter
                self._node_counter += 1

    def add_vsource(self, name: str, pos_node: str, neg_node: str, voltage: float):
        """Add ideal voltage source

        Args:
            name: Source name
            pos_node: Positive terminal node
            neg_node: Negative terminal node
            voltage: Source voltage (V)
        """
        self.vsources[name] = (pos_node, neg_node, voltage)

        # Register nodes
        for node in [pos_node, neg_node]:
            if node not in self.nodes:
                self.nodes[node] = self._node_counter
                self._node_counter += 1

    def set_ground(self, node_name: str):
        """Set ground reference node

        Args:
            node_name: Node to use as ground (0V reference)
        """
        if node_name not in self.nodes:
            self.nodes[node_name] = self._node_counter
            self._node_counter += 1
        self.ground_node = node_name

    def get_node_index(self, node_name: str) -> int:
        """Get integer index for a node name"""
        return self.nodes[node_name]

    def num_nodes(self) -> int:
        """Total number of nodes (including ground)"""
        return len(self.nodes)

    def num_unknowns(self) -> int:
        """Number of unknown voltages (nodes minus ground)"""
        return self.num_nodes() - (1 if self.ground_node else 0)

    def get_device_voltages(self, node_voltages: Array, device_name: str) -> Dict[str, float]:
        """Extract device terminal voltages from solution vector

        Args:
            node_voltages: Solution vector (length = num_unknowns)
            device_name: Device to query

        Returns:
            Dict mapping terminal names to voltages
        """
        device, connections = self.devices[device_name]
        ground_idx = self.get_node_index(self.ground_node) if self.ground_node else -1

        voltages = {}
        for terminal, node_name in connections.items():
            node_idx = self.get_node_index(node_name)
            if node_idx == ground_idx:
                voltages[terminal] = 0.0
            else:
                # Map from full node indices to reduced system (excluding ground)
                if ground_idx >= 0 and node_idx > ground_idx:
                    reduced_idx = node_idx - 1
                else:
                    reduced_idx = node_idx
                voltages[terminal] = float(node_voltages[reduced_idx])

        return voltages

    def stamp_device(
        self,
        node_voltages: Array,
        device_name: str,
        context: Optional["AnalysisContext"] = None,
    ) -> Tuple[Array, Array]:
        """Compute device contribution to residual and Jacobian

        Args:
            node_voltages: Current voltage solution
            device_name: Device to stamp
            context: Analysis context for device behavior modification

        Returns:
            (residual_contribution, jacobian_contribution)
            Both arrays are size (num_unknowns,) and (num_unknowns, num_unknowns)
        """
        device, connections = self.devices[device_name]

        # Get device terminal voltages
        voltages = self.get_device_voltages(node_voltages, device_name)

        # Evaluate device with context
        stamps = device.evaluate(voltages, context=context)

        # Initialize contributions
        n = self.num_unknowns()
        residual = jnp.zeros(n)
        jacobian = jnp.zeros((n, n))

        ground_idx = self.get_node_index(self.ground_node) if self.ground_node else -1

        # Stamp currents into residual
        for terminal, current in stamps.currents.items():
            node_name = connections[terminal]
            node_idx = self.get_node_index(node_name)

            if node_idx != ground_idx:
                # Map to reduced system
                if ground_idx >= 0 and node_idx > ground_idx:
                    reduced_idx = node_idx - 1
                else:
                    reduced_idx = node_idx
                residual = residual.at[reduced_idx].add(current)

        # Stamp conductances into Jacobian
        for (to_term, from_term), conductance in stamps.conductances.items():
            to_node = connections[to_term]
            from_node = connections[from_term]

            to_idx = self.get_node_index(to_node)
            from_idx = self.get_node_index(from_node)

            # Skip if either node is ground
            if to_idx == ground_idx or from_idx == ground_idx:
                continue

            # Map to reduced system
            to_reduced = to_idx - 1 if (ground_idx >= 0 and to_idx > ground_idx) else to_idx
            from_reduced = from_idx - 1 if (ground_idx >= 0 and from_idx > ground_idx) else from_idx

            jacobian = jacobian.at[to_reduced, from_reduced].add(conductance)

        return residual, jacobian

    def build_system(
        self,
        node_voltages: Array,
        context: Optional["AnalysisContext"] = None,
    ) -> Tuple[Array, Array]:
        """Build complete circuit equations: F(V) = 0 and J = dF/dV

        Args:
            node_voltages: Current voltage solution (length = num_unknowns)
            context: Analysis context for device behavior modification

        Returns:
            (residual, jacobian) where residual[i] = sum of currents leaving node i
            and jacobian[i,j] = dF[i]/dV[j]
        """
        n = self.num_unknowns()
        residual = jnp.zeros(n)
        jacobian = jnp.zeros((n, n))

        # Stamp all devices
        for device_name in self.devices.keys():
            dev_residual, dev_jacobian = self.stamp_device(
                node_voltages, device_name, context=context
            )
            residual = residual + dev_residual
            jacobian = jacobian + dev_jacobian

        # Apply voltage source constraints
        for source_name, (pos_node, neg_node, voltage) in self.vsources.items():
            pos_idx = self.get_node_index(pos_node)
            neg_idx = self.get_node_index(neg_node)
            ground_idx = self.get_node_index(self.ground_node) if self.ground_node else -1

            # Set voltage constraint: V[pos] - V[neg] = voltage
            # This replaces one row of the system
            if pos_idx != ground_idx:
                pos_reduced = pos_idx - 1 if (ground_idx >= 0 and pos_idx > ground_idx) else pos_idx

                # Clear the row and set constraint
                jacobian = jacobian.at[pos_reduced, :].set(0.0)
                jacobian = jacobian.at[pos_reduced, pos_reduced].set(1.0)

                if neg_idx != ground_idx:
                    neg_reduced = neg_idx - 1 if (ground_idx >= 0 and neg_idx > ground_idx) else neg_idx
                    jacobian = jacobian.at[pos_reduced, neg_reduced].add(-1.0)
                    residual = residual.at[pos_reduced].set(
                        node_voltages[pos_reduced] - node_voltages[neg_reduced] - voltage
                    )
                else:
                    residual = residual.at[pos_reduced].set(node_voltages[pos_reduced] - voltage)

        return residual, jacobian

    def __repr__(self):
        return (f"Circuit(devices={len(self.devices)}, "
                f"nodes={len(self.nodes)}, ground='{self.ground_node}')")
