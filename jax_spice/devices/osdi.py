"""OSDI (Open Source Device Interface) wrapper for JAX-SPICE

Allows loading Verilog-A models compiled with OpenVAF into JAX-SPICE circuits.
The OSDI interface separates resistive (DC) and reactive (charge/ddt) components,
which aligns perfectly with our AnalysisContext approach.

Reference: OpenVAF OSDI 0.4 specification
"""

import ctypes
from ctypes import (
    POINTER, Structure, Union, c_void_p, c_char_p, c_double, c_uint32,
    c_int32, c_bool, c_size_t, pointer, cast, byref
)
from pathlib import Path
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
import numpy as np

from jax_spice.devices.base import DeviceStamps

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext

# OSDI 0.4 Constants
OSDI_VERSION_MAJOR = 0
OSDI_VERSION_MINOR = 4

# Parameter type flags
PARA_TY_MASK = 3
PARA_TY_REAL = 0
PARA_TY_INT = 1
PARA_TY_STR = 2
PARA_KIND_MASK = 3 << 30
PARA_KIND_MODEL = 0 << 30
PARA_KIND_INST = 1 << 30
PARA_KIND_OPVAR = 2 << 30

# Access flags
ACCESS_FLAG_READ = 0
ACCESS_FLAG_SET = 1
ACCESS_FLAG_INSTANCE = 4

# Calculation flags
CALC_RESIST_RESIDUAL = 1
CALC_REACT_RESIDUAL = 2
CALC_RESIST_JACOBIAN = 4
CALC_REACT_JACOBIAN = 8
CALC_NOISE = 16
CALC_OP = 32
CALC_RESIST_LIM_RHS = 64
CALC_REACT_LIM_RHS = 128
ENABLE_LIM = 256
INIT_LIM = 512

# Analysis flags
ANALYSIS_NOISE = 1024
ANALYSIS_DC = 2048
ANALYSIS_AC = 4096
ANALYSIS_TRAN = 8192
ANALYSIS_IC = 16384
ANALYSIS_STATIC = 32768
ANALYSIS_NODESET = 65536


# OSDI Structures (ctypes definitions matching osdi_0_4.h)

class OsdiSimParas(Structure):
    _fields_ = [
        ("names", POINTER(c_char_p)),
        ("vals", POINTER(c_double)),
        ("names_str", POINTER(c_char_p)),
        ("vals_str", POINTER(c_char_p)),
    ]


class OsdiSimInfo(Structure):
    _fields_ = [
        ("paras", OsdiSimParas),
        ("abstime", c_double),
        ("prev_solve", POINTER(c_double)),
        ("prev_state", POINTER(c_double)),
        ("next_state", POINTER(c_double)),
        ("flags", c_uint32),
    ]


class OsdiInitErrorPayload(Union):
    _fields_ = [
        ("parameter_id", c_uint32),
    ]


class OsdiInitError(Structure):
    _fields_ = [
        ("code", c_uint32),
        ("payload", OsdiInitErrorPayload),
    ]


class OsdiInitInfo(Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("num_errors", c_uint32),
        ("errors", POINTER(OsdiInitError)),
    ]


class OsdiNodePair(Structure):
    _fields_ = [
        ("node_1", c_uint32),
        ("node_2", c_uint32),
    ]


class OsdiJacobianEntry(Structure):
    _fields_ = [
        ("nodes", OsdiNodePair),
        ("react_ptr_off", c_uint32),
        ("flags", c_uint32),
    ]


class OsdiNode(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("units", c_char_p),
        ("residual_units", c_char_p),
        ("resist_residual_off", c_uint32),
        ("react_residual_off", c_uint32),
        ("resist_limit_rhs_off", c_uint32),
        ("react_limit_rhs_off", c_uint32),
        ("is_flow", c_bool),
    ]


class OsdiParamOpvar(Structure):
    _fields_ = [
        ("name", POINTER(c_char_p)),
        ("num_alias", c_uint32),
        ("description", c_char_p),
        ("units", c_char_p),
        ("flags", c_uint32),
        ("len", c_uint32),
    ]


class OsdiNoiseSource(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("nodes", OsdiNodePair),
    ]


class OsdiNatureRef(Structure):
    _fields_ = [
        ("ref_type", c_uint32),
        ("index", c_uint32),
    ]


class OsdiHandle(Structure):
    """Handle structure passed to OSDI functions (ngspice convention)

    kind values:
        1 = setup_model
        2 = setup_instance
        3 = eval
    """
    _fields_ = [
        ("kind", c_uint32),
        ("name", c_char_p),
    ]


# Function pointer types
AccessFunc = ctypes.CFUNCTYPE(c_void_p, c_void_p, c_void_p, c_uint32, c_uint32)
SetupModelFunc = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, POINTER(OsdiSimParas), POINTER(OsdiInitInfo))
SetupInstanceFunc = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_void_p, c_double, c_uint32, POINTER(OsdiSimParas), POINTER(OsdiInitInfo))
EvalFunc = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p, c_void_p, POINTER(OsdiSimInfo))
LoadResidualFunc = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, POINTER(c_double))
LoadJacobianFunc = ctypes.CFUNCTYPE(None, c_void_p, c_void_p)
WriteJacobianArrayFunc = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, POINTER(c_double))


class OsdiDescriptor(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("num_nodes", c_uint32),
        ("num_terminals", c_uint32),
        ("nodes", POINTER(OsdiNode)),
        ("num_jacobian_entries", c_uint32),
        ("jacobian_entries", POINTER(OsdiJacobianEntry)),
        ("num_collapsible", c_uint32),
        ("collapsible", POINTER(OsdiNodePair)),
        ("collapsed_offset", c_uint32),
        ("noise_sources", POINTER(OsdiNoiseSource)),
        ("num_noise_src", c_uint32),
        ("num_params", c_uint32),
        ("num_instance_params", c_uint32),
        ("num_opvars", c_uint32),
        ("param_opvar", POINTER(OsdiParamOpvar)),
        ("node_mapping_offset", c_uint32),
        ("jacobian_ptr_resist_offset", c_uint32),
        ("num_states", c_uint32),
        ("state_idx_off", c_uint32),
        ("bound_step_offset", c_uint32),
        ("instance_size", c_uint32),
        ("model_size", c_uint32),
        ("access", AccessFunc),
        ("setup_model", SetupModelFunc),
        ("setup_instance", SetupInstanceFunc),
        ("eval", EvalFunc),
        ("load_noise", c_void_p),  # Simplified - not using noise yet
        ("load_residual_resist", LoadResidualFunc),
        ("load_residual_react", LoadResidualFunc),
        ("load_limit_rhs_resist", LoadResidualFunc),
        ("load_limit_rhs_react", LoadResidualFunc),
        ("load_spice_rhs_dc", c_void_p),
        ("load_spice_rhs_tran", c_void_p),
        ("load_jacobian_resist", LoadJacobianFunc),
        ("load_jacobian_react", c_void_p),
        ("load_jacobian_tran", c_void_p),
        ("given_flag_model", c_void_p),
        ("given_flag_instance", c_void_p),
        ("num_resistive_jacobian_entries", c_uint32),
        ("num_reactive_jacobian_entries", c_uint32),
        ("write_jacobian_array_resist", WriteJacobianArrayFunc),
        ("write_jacobian_array_react", WriteJacobianArrayFunc),
        ("num_inputs", c_uint32),
        ("inputs", POINTER(OsdiNodePair)),
        ("load_jacobian_with_offset_resist", c_void_p),
        ("load_jacobian_with_offset_react", c_void_p),
        ("unknown_nature", POINTER(OsdiNatureRef)),
        ("residual_nature", POINTER(OsdiNatureRef)),
    ]


class OSDILibrary:
    """Manages loading and accessing an OSDI shared library"""

    def __init__(self, osdi_path: str):
        self.path = Path(osdi_path)
        if not self.path.exists():
            raise FileNotFoundError(f"OSDI library not found: {osdi_path}")

        self.lib = ctypes.CDLL(str(self.path))

        # Load global symbols
        self.num_descriptors = c_uint32.in_dll(self.lib, "OSDI_NUM_DESCRIPTORS").value
        self.version_major = c_uint32.in_dll(self.lib, "OSDI_VERSION_MAJOR").value
        self.version_minor = c_uint32.in_dll(self.lib, "OSDI_VERSION_MINOR").value

        if self.version_major != OSDI_VERSION_MAJOR or self.version_minor != OSDI_VERSION_MINOR:
            raise ValueError(
                f"OSDI version mismatch: library is {self.version_major}.{self.version_minor}, "
                f"expected {OSDI_VERSION_MAJOR}.{OSDI_VERSION_MINOR}"
            )

        # Load descriptor array
        # Note: OSDI_DESCRIPTORS is the array itself, not a pointer to it
        descriptor_size = c_uint32.in_dll(self.lib, "OSDI_DESCRIPTOR_SIZE").value
        desc_start = ctypes.c_byte.in_dll(self.lib, "OSDI_DESCRIPTORS")
        base_addr = ctypes.addressof(desc_start)

        # Cast to array of descriptors
        self.descriptors: List[OsdiDescriptor] = []
        for i in range(self.num_descriptors):
            desc_ptr = cast(base_addr + i * descriptor_size, POINTER(OsdiDescriptor))
            self.descriptors.append(desc_ptr.contents)

    def get_descriptor(self, name: Optional[str] = None) -> OsdiDescriptor:
        """Get descriptor by name, or first if name not specified"""
        if name is None:
            return self.descriptors[0]

        for desc in self.descriptors:
            if desc.name.decode() == name:
                return desc

        available = [d.name.decode() for d in self.descriptors]
        raise ValueError(f"Model '{name}' not found. Available: {available}")


class OSDIDevice:
    """OSDI device wrapper for JAX-SPICE

    Wraps a compiled Verilog-A model (OSDI format) for use in JAX-SPICE circuits.
    Uses the AnalysisContext to determine whether to include reactive (ddt) terms.

    Example:
        ```python
        diode = OSDIDevice(
            osdi_path="diode_rr.osdi",
            params={'is': 1e-14, 'n': 1.0, 'rs': 10.0},
            terminals=['anode', 'cathode']
        )

        stamps = diode.evaluate(
            voltages={'anode': 0.7, 'cathode': 0.0},
            context=AnalysisContext.dc()
        )
        ```
    """

    def __init__(
        self,
        osdi_path: str,
        params: Optional[Dict[str, float]] = None,
        model_name: Optional[str] = None,
        temperature: float = 300.0,
        terminal_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize OSDI device

        Args:
            osdi_path: Path to compiled .osdi file
            params: Model/instance parameters to override defaults
            model_name: Name of model in OSDI file (if multiple)
            temperature: Device temperature in Kelvin
            terminal_mapping: Optional mapping from OSDI node names to circuit terminal names
        """
        self.osdi_lib = OSDILibrary(osdi_path)
        self.descriptor = self.osdi_lib.get_descriptor(model_name)
        self.temperature = temperature
        self.params = params or {}

        # Extract terminal info from descriptor
        self._setup_terminals(terminal_mapping)

        # Allocate model and instance storage
        self._model_data = (ctypes.c_byte * self.descriptor.model_size)()
        self._instance_data = (ctypes.c_byte * self.descriptor.instance_size)()
        self._model_ptr = cast(self._model_data, c_void_p)
        self._instance_ptr = cast(self._instance_data, c_void_p)

        # Allocate working arrays (before setup_instance which references them)
        self._residual = (c_double * self.descriptor.num_nodes)()
        self._jacobian = (c_double * self.descriptor.num_resistive_jacobian_entries)()
        self._node_voltages = (c_double * self.descriptor.num_nodes)()

        # Build parameter name to index map
        self._param_map = self._build_param_map()

        # Set up simulation parameters (shared across all OSDI calls)
        self._setup_sim_params()

        # Set model parameters using access function BEFORE setup_model
        self._set_parameters()

        # Initialize model and instance
        self._setup_model()
        self._setup_instance()

    def _setup_terminals(self, terminal_mapping: Optional[Dict[str, str]]):
        """Set up terminal names from OSDI descriptor"""
        osdi_terminals = []
        for i in range(self.descriptor.num_terminals):
            node = self.descriptor.nodes[i]
            osdi_terminals.append(node.name.decode())

        if terminal_mapping:
            self.terminals = tuple(terminal_mapping.get(t, t) for t in osdi_terminals)
        else:
            self.terminals = tuple(osdi_terminals)

        self._osdi_terminals = osdi_terminals
        self._terminal_to_idx = {t: i for i, t in enumerate(self.terminals)}

    def _build_param_map(self) -> Dict[str, int]:
        """Build mapping from parameter names to indices"""
        param_map = {}
        for i in range(self.descriptor.num_params):
            pinfo = self.descriptor.param_opvar[i]
            name = pinfo.name[0].decode() if pinfo.name else None
            if name:
                param_map[name] = i
        return param_map

    def _setup_sim_params(self):
        """Set up simulation parameters used by OSDI $simparam calls"""
        # These are the standard simulation parameters ngspice provides
        self._sim_param_names = (c_char_p * 11)(
            b'initializeLimiting', b'gmin', b'gdev', b'tnom', b'simulatorVersion',
            b'sourceScaleFactor', b'epsmin', b'reltol', b'vntol', b'abstol', None
        )
        # Convert temperature to Celsius for tnom
        tnom_celsius = self.temperature - 273.15
        self._sim_param_vals = (c_double * 10)(
            0.0,        # initializeLimiting
            1e-12,      # gmin
            1e-12,      # gdev
            tnom_celsius,  # tnom (in Celsius)
            42.0,       # simulatorVersion
            1.0,        # sourceScaleFactor
            1e-15,      # epsmin
            1e-3,       # reltol
            1e-6,       # vntol
            1e-12,      # abstol
        )
        self._sim_paras = OsdiSimParas()
        self._sim_paras.names = self._sim_param_names
        self._sim_paras.vals = self._sim_param_vals
        self._sim_paras.names_str = None
        self._sim_paras.vals_str = None

    def _set_parameters(self):
        """Set model parameters using the access function"""
        for name, value in self.params.items():
            if name in self._param_map:
                param_idx = self._param_map[name]
                dst = self.descriptor.access(
                    None, self._model_ptr,
                    c_uint32(param_idx), c_uint32(ACCESS_FLAG_SET)
                )
                if dst:
                    double_ptr = cast(dst, POINTER(c_double))
                    double_ptr[0] = value

    def _setup_model(self):
        """Initialize OSDI model"""
        # Initialize result structure
        init_info = OsdiInitInfo()
        init_info.flags = 0
        init_info.num_errors = 0
        init_info.errors = None

        # Create handle (kind=1 for setup_model)
        handle = OsdiHandle()
        handle.kind = 1
        handle.name = b"model"

        # Call setup_model with shared sim_paras
        self.descriptor.setup_model(
            byref(handle),
            self._model_ptr,
            byref(self._sim_paras),
            byref(init_info)
        )

        if init_info.num_errors > 0:
            raise RuntimeError(f"OSDI setup_model failed with {init_info.num_errors} errors")

    def _setup_instance(self):
        """Initialize OSDI instance"""
        # Initialize result structure
        init_info = OsdiInitInfo()
        init_info.flags = 0
        init_info.num_errors = 0
        init_info.errors = None

        # Create handle (kind=2 for setup_instance)
        handle = OsdiHandle()
        handle.kind = 2
        handle.name = b"inst"

        # Temperature is passed in Kelvin to OSDI setup_instance
        temp_kelvin = self.temperature

        # Call setup_instance with shared sim_paras
        self.descriptor.setup_instance(
            byref(handle),
            self._instance_ptr,
            self._model_ptr,
            c_double(temp_kelvin),
            c_uint32(self.descriptor.num_terminals),
            byref(self._sim_paras),
            byref(init_info)
        )

        if init_info.num_errors > 0:
            raise RuntimeError(f"OSDI setup_instance failed with {init_info.num_errors} errors")

        # Set up node mapping (identity for now - no collapsing)
        node_mapping_offset = self.descriptor.node_mapping_offset
        node_mapping_ptr = cast(
            self._instance_ptr.value + node_mapping_offset,
            POINTER(c_uint32)
        )
        for i in range(self.descriptor.num_nodes):
            node_mapping_ptr[i] = i

        # Set up Jacobian pointers
        jacobian_ptr_offset = self.descriptor.jacobian_ptr_resist_offset
        jacobian_ptrs = cast(
            self._instance_ptr.value + jacobian_ptr_offset,
            POINTER(POINTER(c_double))
        )
        for i in range(self.descriptor.num_resistive_jacobian_entries):
            jacobian_ptrs[i] = cast(
                ctypes.addressof(self._jacobian) + i * ctypes.sizeof(c_double),
                POINTER(c_double)
            )

    def _set_node_voltages(self, voltages: Dict[str, float]):
        """Set node voltages from terminal voltage dict

        For OSDI devices with internal nodes (e.g., diode with series resistance),
        we need to provide initial guesses for internal nodes.
        """
        # First, set terminal voltages
        for terminal, voltage in voltages.items():
            if terminal in self._terminal_to_idx:
                idx = self._terminal_to_idx[terminal]
                self._node_voltages[idx] = voltage

        # For internal nodes, use heuristics based on node name
        # - 'internal' nodes are typically between anode and cathode, set to anode voltage
        # - State/charge nodes (like 'drr') should start at 0
        # - Flow nodes represent currents, initialize to 0
        if self.descriptor.num_terminals >= 2:
            v_high = self._node_voltages[0]  # Typically anode

            for i in range(self.descriptor.num_terminals, self.descriptor.num_nodes):
                node = self.descriptor.nodes[i]
                name = node.name.decode() if node.name else ""

                if node.is_flow:
                    # Flow nodes represent currents, initialize to 0
                    self._node_voltages[i] = 0.0
                elif "internal" in name.lower():
                    # Internal voltage node between terminals
                    self._node_voltages[i] = v_high
                else:
                    # Other nodes (state variables, etc.) start at 0
                    self._node_voltages[i] = 0.0

    def evaluate(
        self,
        voltages: Dict[str, float],
        context: Optional["AnalysisContext"] = None,
    ) -> DeviceStamps:
        """Evaluate OSDI device at given terminal voltages

        Args:
            voltages: Dict mapping terminal names to voltages
            context: Analysis context (DC skips reactive terms)

        Returns:
            DeviceStamps with currents and conductances
        """
        # Determine analysis type
        is_dc = context is None or context.is_dc

        # Set up node voltages
        self._set_node_voltages(voltages)

        # Build simulation info with shared sim_paras
        sim_info = OsdiSimInfo()
        sim_info.paras = self._sim_paras
        sim_info.abstime = 0.0
        sim_info.prev_solve = self._node_voltages
        sim_info.prev_state = None
        sim_info.next_state = None

        # Set calculation flags based on analysis type
        if is_dc:
            sim_info.flags = (
                CALC_RESIST_RESIDUAL |
                CALC_RESIST_JACOBIAN |
                CALC_OP |
                ANALYSIS_DC |
                ANALYSIS_STATIC
            )
        else:
            sim_info.flags = (
                CALC_RESIST_RESIDUAL |
                CALC_REACT_RESIDUAL |
                CALC_RESIST_JACOBIAN |
                CALC_REACT_JACOBIAN |
                ANALYSIS_TRAN
            )

        # Clear arrays
        for i in range(self.descriptor.num_nodes):
            self._residual[i] = 0.0
        for i in range(self.descriptor.num_resistive_jacobian_entries):
            self._jacobian[i] = 0.0

        # Create handle (kind=3 for eval)
        handle = OsdiHandle()
        handle.kind = 3
        handle.name = b"inst"

        # Call eval
        ret = self.descriptor.eval(
            byref(handle),
            self._instance_ptr,
            self._model_ptr,
            byref(sim_info)
        )

        # Load residuals (currents)
        self.descriptor.load_residual_resist(
            self._instance_ptr,
            self._model_ptr,
            self._residual
        )

        # Load Jacobian
        self.descriptor.load_jacobian_resist(
            self._instance_ptr,
            self._model_ptr
        )

        # Convert to DeviceStamps format
        return self._build_stamps()

    def _build_stamps(self) -> DeviceStamps:
        """Convert OSDI results to DeviceStamps"""
        # Build current dict from residuals (terminals only)
        currents = {}
        for i, terminal in enumerate(self.terminals):
            # OSDI convention: positive current flows INTO the node
            # Our convention: positive current flows OUT of the terminal
            currents[terminal] = -float(self._residual[i])

        # For 2-terminal devices with flow nodes (like diodes), the terminal residuals
        # may be 0 because current flows via internal flow nodes. Use KCL to derive
        # the proper terminal currents: sum of all terminal currents must be 0.
        if len(self.terminals) == 2:
            t1, t2 = self.terminals
            if currents[t1] == 0 and currents[t2] != 0:
                currents[t1] = -currents[t2]
            elif currents[t2] == 0 and currents[t1] != 0:
                currents[t2] = -currents[t1]

        # Build conductance dict from Jacobian entries
        conductances = {}
        for i in range(self.descriptor.num_resistive_jacobian_entries):
            entry = self.descriptor.jacobian_entries[i]
            node1 = entry.nodes.node_1
            node2 = entry.nodes.node_2

            # Only include terminal-to-terminal entries
            if node1 < len(self.terminals) and node2 < len(self.terminals):
                term1 = self.terminals[node1]
                term2 = self.terminals[node2]
                # OSDI Jacobian is dI/dV, our convention is also dI/dV
                # But we negated current above, so negate conductance too
                conductances[(term1, term2)] = -float(self._jacobian[i])

        return DeviceStamps(currents=currents, conductances=conductances)

    def get_param_info(self) -> Dict[str, Dict]:
        """Get information about available parameters"""
        params = {}
        total_params = (self.descriptor.num_params +
                       self.descriptor.num_instance_params +
                       self.descriptor.num_opvars)

        for i in range(total_params):
            pinfo = self.descriptor.param_opvar[i]
            name = pinfo.name[0].decode() if pinfo.name else f"param_{i}"
            flags = pinfo.flags

            kind = "model" if (flags & PARA_KIND_MASK) == PARA_KIND_MODEL else \
                   "instance" if (flags & PARA_KIND_MASK) == PARA_KIND_INST else \
                   "opvar"

            ptype = "real" if (flags & PARA_TY_MASK) == PARA_TY_REAL else \
                    "int" if (flags & PARA_TY_MASK) == PARA_TY_INT else \
                    "str"

            params[name] = {
                "kind": kind,
                "type": ptype,
                "description": pinfo.description.decode() if pinfo.description else "",
                "units": pinfo.units.decode() if pinfo.units else "",
            }

        return params

    def __repr__(self):
        model_name = self.descriptor.name.decode()
        return f"OSDIDevice({model_name}, terminals={self.terminals})"
