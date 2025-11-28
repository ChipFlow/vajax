"""JAX device generated from MIR"""

from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import jit

from jax_spice.mir.translator import MIRToJAX
from jax_spice.mir.parser import parse_mir, parse_system


@dataclass
class JAXDevice:
    """A device model compiled from OpenVAF MIR snapshots

    This provides an interface similar to OSDI devices but runs
    entirely in JAX, enabling automatic differentiation and JIT compilation.
    """
    name: str
    terminals: List[str]
    parameters: Dict[str, float]
    _eval_fn: Callable = field(repr=False)
    _translator: MIRToJAX = field(repr=False)

    @classmethod
    def from_snapshots(
        cls,
        name: str,
        mir_text: str,
        system_text: str,
        default_params: Dict[str, float] = None
    ) -> "JAXDevice":
        """Create a device from MIR snapshot files

        Args:
            name: Device name
            mir_text: Contents of the *_mir.snap file
            system_text: Contents of the *_system.snap file
            default_params: Default parameter values
        """
        translator = MIRToJAX.from_snapshots(mir_text, system_text)
        eval_fn = translator.translate()

        # Get terminal names from the system
        terminals = []
        for sim_node, node_name in translator.system.unknowns.items():
            terminals.append(node_name)

        return cls(
            name=name,
            terminals=terminals,
            parameters=default_params or {},
            _eval_fn=eval_fn,
            _translator=translator
        )

    @classmethod
    def from_snapshot_files(
        cls,
        name: str,
        mir_path: str,
        system_path: str,
        default_params: Dict[str, float] = None
    ) -> "JAXDevice":
        """Create a device from MIR snapshot file paths"""
        with open(mir_path) as f:
            mir_text = f.read()
        with open(system_path) as f:
            system_text = f.read()
        return cls.from_snapshots(name, mir_text, system_text, default_params)

    def set_parameters(self, **params):
        """Set device parameters"""
        self.parameters.update(params)

    def eval(
        self,
        voltages: Dict[str, float],
        temperature: float = 300.15
    ) -> Tuple[Dict, Dict]:
        """Evaluate the device at given node voltages

        Args:
            voltages: Dict mapping node names to voltages
            temperature: Temperature in Kelvin

        Returns:
            (residuals, jacobian) tuple
        """
        # Build input array from parameters and voltages
        # The order matches the MIR function signature
        inputs = self._build_inputs(voltages, temperature)

        # Call the generated function
        residuals, jacobian = self._eval_fn(inputs)

        return residuals, jacobian

    def _build_inputs(
        self,
        voltages: Dict[str, float],
        temperature: float
    ) -> List[float]:
        """Build input array for the eval function

        The MIR function parameters are typically:
        - Branch voltages (computed from node voltages)
        - Device parameters
        - Temperature and tnom
        - Temperature coefficients
        """
        # For a simple 2-terminal device:
        # v16 = V(node0) - V(node1) = branch voltage
        # v17 = (unused or flow)
        # v18 = R (resistance parameter)
        # v19 = $temperature
        # v20 = tnom
        # v22 = zeta (temp coefficient)
        # v25, v28 = scale factors

        node_names = list(self._translator.system.unknowns.values())

        # Compute branch voltage
        if len(node_names) >= 2:
            v_branch = voltages.get(node_names[0], 0.0) - voltages.get(node_names[1], 0.0)
        else:
            v_branch = voltages.get(node_names[0], 0.0) if node_names else 0.0

        # Build inputs - this is device-specific
        # For resistor: [V, 0, R, T, tnom, zeta, scale, scale]
        inputs = [
            v_branch,                                    # v16: branch voltage
            0.0,                                         # v17: unused
            self.parameters.get('R', 1000.0),           # v18: resistance
            temperature,                                 # v19: temperature
            self.parameters.get('tnom', 300.15),        # v20: tnom
            self.parameters.get('zeta', 0.0),           # v22: zeta
            1.0,                                         # v25: scale
            1.0,                                         # v28: scale
        ]

        return inputs

    def get_stamps(
        self,
        node_indices: Dict[str, int],
        voltages: Dict[str, float],
        temperature: float = 300.15
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]:
        """Get conductance matrix stamps and RHS contributions

        This converts the device evaluation into matrix stamps suitable
        for circuit simulation.

        Args:
            node_indices: Dict mapping node names to matrix indices
            voltages: Current node voltages
            temperature: Temperature in Kelvin

        Returns:
            (G_stamps, I_stamps) where:
            - G_stamps: Dict of (row, col) -> conductance value
            - I_stamps: Dict of row -> current contribution
        """
        residuals, jacobian = self.eval(voltages, temperature)

        G_stamps = {}
        I_stamps = {}

        # Convert jacobian to G stamps
        node_map = self._translator.system.unknowns
        for (row_sim, col_sim), entry in jacobian.items():
            row_name = node_map.get(row_sim)
            col_name = node_map.get(col_sim)
            if row_name and col_name:
                row_idx = node_indices.get(row_name, -1)
                col_idx = node_indices.get(col_name, -1)
                if row_idx >= 0 and col_idx >= 0:
                    G_stamps[(row_idx, col_idx)] = entry['resist']

        # Convert residuals to I stamps
        for node_sim, res in residuals.items():
            node_name = node_map.get(node_sim)
            if node_name:
                idx = node_indices.get(node_name, -1)
                if idx >= 0:
                    I_stamps[idx] = res['resist']

        return G_stamps, I_stamps

    @property
    def num_terminals(self) -> int:
        return len(self.terminals)


def create_resistor_device(R: float = 1000.0, tnom: float = 300.15, zeta: float = 0.0) -> JAXDevice:
    """Create a simple resistor device using embedded MIR

    This is a convenience function for testing.
    """
    mir_text = """function %(v16, v17, v18, v19, v20, v22, v25, v28) {
    v6 = fconst 0x1.0000000000000p0
                                block2:
@0008                               v21 = fdiv v19, v20
@000a                               v23 = pow v21, v22
@000b                               v24 = fmul v18, v23
@0010                               v26 = fdiv v16, v24
@fffffff0                           v32 = fdiv v6, v24
                                    v38 = fmul v28, v26
                                    v29 = optbarrier v38
                                    v30 = fneg v26
                                    v33 = fneg v32
                                    v42 = fmul v28, v32
                                    v35 = optbarrier v42
                                    v45 = fmul v28, v33
                                    v37 = optbarrier v45
                                    v40 = fmul v28, v30
                                    v39 = optbarrier v40
                                    v41 = optbarrier v28
                                    v43 = optbarrier v45
                                    v46 = optbarrier v42
}
"""

    system_text = """DaeSystem {
    unknowns: {
        sim_node0: node0,
        sim_node1: node1,
    },
    residual: {
        sim_node0: Residual {
            resist: v29,
            react: v3,
        },
        sim_node1: Residual {
            resist: v39,
            react: v3,
        },
    },
    jacobian: {
        j0: MatrixEntry {
            row: sim_node0,
            col: sim_node0,
            resist: v35,
            react: v3,
        },
        j1: MatrixEntry {
            row: sim_node0,
            col: sim_node1,
            resist: v43,
            react: v3,
        },
        j2: MatrixEntry {
            row: sim_node1,
            col: sim_node0,
            resist: v37,
            react: v3,
        },
        j3: MatrixEntry {
            row: sim_node1,
            col: sim_node1,
            resist: v46,
            react: v3,
        },
    },
}
"""

    device = JAXDevice.from_snapshots(
        name="resistor",
        mir_text=mir_text,
        system_text=system_text,
        default_params={'R': R, 'tnom': tnom, 'zeta': zeta}
    )
    return device
