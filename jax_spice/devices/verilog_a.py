"""Verilog-A device wrapper using OpenVAF Python bindings

This module provides a JAX-compatible device interface for Verilog-A models
compiled using OpenVAF. It uses direct Python bindings to the MIR interpreter,
providing better integration than text-based snapshot parsing.
"""

# Force CPU backend on Apple Silicon to avoid Metal backend compatibility issues
# This must be done before any JAX imports
import os
import platform

if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Apple Silicon - force CPU backend to avoid Metal/GPU issues
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import jax.numpy as jnp

import sys

# Try to import openvaf_py directly first (if installed)
# If not found, add openvaf-py to path
try:
    import openvaf_py
    import openvaf_jax
except ImportError:
    openvaf_py_path = Path(__file__).parent.parent.parent / "openvaf-py"
    if openvaf_py_path.exists() and str(openvaf_py_path) not in sys.path:
        sys.path.insert(0, str(openvaf_py_path))
    import openvaf_py
    import openvaf_jax


@dataclass
class VerilogADevice:
    """A device model compiled from Verilog-A using OpenVAF

    This provides a JAX-compatible interface for Verilog-A compact models,
    enabling automatic differentiation and JIT compilation.

    Attributes:
        name: Device name (from Verilog-A module)
        va_path: Path to the source .va file
        terminals: List of terminal names
        param_names: List of parameter names (model params, voltages, hidden states)
        param_kinds: List of parameter kinds (param, voltage, hidden_state, etc.)
        default_params: Default values for model parameters
    """
    name: str
    va_path: str
    terminals: List[str]
    param_names: List[str]
    param_kinds: List[str]
    default_params: Dict[str, float]
    _module: Any = field(repr=False)
    _translator: openvaf_jax.OpenVAFToJAX = field(repr=False)
    _eval_fn: Callable = field(repr=False)

    @classmethod
    def from_va_file(
        cls,
        va_path: str,
        default_params: Optional[Dict[str, float]] = None,
        allow_analog_in_cond: bool = False
    ) -> "VerilogADevice":
        """Create a device from a Verilog-A file

        Args:
            va_path: Path to the .va file
            default_params: Optional default parameter values
            allow_analog_in_cond: Allow analog operators (limexp, ddt, idt) in
                                  conditionals. Required for some foundry models
                                  like GF130 PDK.

        Returns:
            VerilogADevice instance
        """
        modules = openvaf_py.compile_va(va_path, allow_analog_in_cond=allow_analog_in_cond)
        if not modules:
            raise ValueError(f"No modules found in {va_path}")

        module = modules[0]
        translator = openvaf_jax.OpenVAFToJAX(module)
        eval_fn = translator.translate()

        # Build default params from module
        defaults = {}
        for name, kind in zip(module.param_names, module.param_kinds):
            if kind == 'param':
                # Set sensible defaults for common parameters
                if name in ('tnom',):
                    defaults[name] = 300.0
                elif name in ('mfactor',):
                    defaults[name] = 1.0
                else:
                    defaults[name] = 0.0
            elif kind == 'temperature':
                defaults[name] = 300.15
            elif kind == 'sysfun':
                if name == 'mfactor':
                    defaults[name] = 1.0
                else:
                    defaults[name] = 0.0

        if default_params:
            defaults.update(default_params)

        return cls(
            name=module.name,
            va_path=va_path,
            terminals=list(module.nodes),
            param_names=list(module.param_names),
            param_kinds=list(module.param_kinds),
            default_params=defaults,
            _module=module,
            _translator=translator,
            _eval_fn=eval_fn
        )

    def set_parameters(self, **params):
        """Set device parameters

        Args:
            **params: Parameter name-value pairs
        """
        self.default_params.update(params)

    def get_parameter_info(self) -> List[Tuple[str, str, float]]:
        """Get information about all parameters

        Returns:
            List of (name, kind, default_value) tuples
        """
        result = []
        for name, kind in zip(self.param_names, self.param_kinds):
            default = self.default_params.get(name, 0.0)
            result.append((name, kind, default))
        return result

    def build_inputs(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        temperature: float = 300.15
    ) -> List[float]:
        """Build input array for the eval function

        Args:
            voltages: Dictionary mapping terminal/branch names to voltages
            params: Optional parameter overrides
            temperature: Temperature in Kelvin

        Returns:
            Input array for the eval function
        """
        # Merge defaults with overrides
        all_params = dict(self.default_params)
        if params:
            all_params.update(params)

        inputs = []
        for name, kind in zip(self.param_names, self.param_kinds):
            if kind == 'voltage':
                # Look up voltage by name
                inputs.append(voltages.get(name, 0.0))
            elif kind == 'temperature':
                inputs.append(temperature)
            elif kind == 'hidden_state':
                # Hidden states should be pre-computed or use defaults
                inputs.append(all_params.get(name, 0.0))
            elif kind == 'current':
                inputs.append(all_params.get(name, 0.0))
            else:
                # Model parameters
                inputs.append(all_params.get(name, 0.0))

        return inputs

    def eval(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        temperature: float = 300.15
    ) -> Tuple[Dict, Dict]:
        """Evaluate the device at given voltages

        Args:
            voltages: Dictionary mapping terminal/branch names to voltages
            params: Optional parameter overrides
            temperature: Temperature in Kelvin

        Returns:
            (residuals, jacobian) tuple where:
            - residuals: Dict[node_name, Dict[str, float]] with 'resist' and 'react' keys
            - jacobian: Dict[(row_node, col_node), Dict[str, float]]
        """
        inputs = self.build_inputs(voltages, params, temperature)
        return self._eval_fn(inputs)

    def eval_with_interpreter(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        temperature: float = 300.15
    ) -> Tuple[Any, Any]:
        """Evaluate using the MIR interpreter (for validation)

        This uses the Rust-based MIR interpreter instead of the JAX function.
        Useful for comparing results.

        Args:
            voltages: Dictionary mapping terminal/branch names to voltages
            params: Optional parameter overrides
            temperature: Temperature in Kelvin

        Returns:
            (residuals, jacobian) from the interpreter
        """
        all_params = dict(self.default_params)
        if params:
            all_params.update(params)

        # Build param dict by name
        param_dict = {}
        for name, kind in zip(self.param_names, self.param_kinds):
            if kind == 'voltage':
                param_dict[name] = voltages.get(name, 0.0)
            elif kind == 'temperature':
                param_dict[name] = temperature
            else:
                param_dict[name] = all_params.get(name, 0.0)

        return self._module.run_init_eval(param_dict)

    def get_stamps(
        self,
        node_indices: Dict[str, int],
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        temperature: float = 300.15
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]:
        """Get conductance matrix stamps and RHS contributions

        Args:
            node_indices: Dict mapping node names to matrix indices
            voltages: Current node voltages
            params: Optional parameter overrides
            temperature: Temperature in Kelvin

        Returns:
            (G_stamps, I_stamps) where:
            - G_stamps: Dict of (row, col) -> conductance value
            - I_stamps: Dict of row -> current contribution
        """
        residuals, jacobian = self.eval(voltages, params, temperature)

        G_stamps = {}
        I_stamps = {}

        # Get DAE system info for node mapping
        dae = self._translator.dae_data

        # Convert residuals to current stamps
        for sim_node, res in residuals.items():
            # Map sim_node to actual node name
            if sim_node in dae.get('unknowns', {}):
                node_name = dae['unknowns'][sim_node]
                if node_name in node_indices:
                    idx = node_indices[node_name]
                    I_stamps[idx] = float(res.get('resist', 0.0))

        # Convert jacobian to conductance stamps
        for (row_sim, col_sim), entry in jacobian.items():
            row_name = dae.get('unknowns', {}).get(row_sim)
            col_name = dae.get('unknowns', {}).get(col_sim)
            if row_name in node_indices and col_name in node_indices:
                row_idx = node_indices[row_name]
                col_idx = node_indices[col_name]
                G_stamps[(row_idx, col_idx)] = float(entry.get('resist', 0.0))

        return G_stamps, I_stamps


def compile_va(
    va_path: str,
    allow_analog_in_cond: bool = False,
    **default_params
) -> VerilogADevice:
    """Convenience function to compile a Verilog-A file

    Args:
        va_path: Path to the .va file
        allow_analog_in_cond: Allow analog operators in conditionals (for foundry models)
        **default_params: Default parameter values

    Returns:
        VerilogADevice instance
    """
    return VerilogADevice.from_va_file(
        va_path,
        default_params or None,
        allow_analog_in_cond=allow_analog_in_cond
    )
