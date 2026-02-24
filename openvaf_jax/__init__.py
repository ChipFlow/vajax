"""OpenVAF to JAX translator package.

This package provides translation from OpenVAF MIR (Mid-level IR) to JAX functions
for circuit simulation. The main entry point is the OpenVAFToJAX class.

Example usage:
    from openvaf_jax import OpenVAFToJAX
    import openvaf_py

    # Compile Verilog-A model
    modules = openvaf_py.compile_va("model.va")
    module = modules[0]

    # Create translator
    translator = OpenVAFToJAX(module)

    # Generate init function (computes cache from params)
    init_fn, init_meta = translator.translate_init(
        params={'VTO': 0.5, 'KP': 150e-6},
        temperature=300.0,
    )

    # Generate eval function (computes residuals/Jacobian)
    eval_fn, eval_meta = translator.translate_eval(
        params={'VTO': 0.5, 'KP': 150e-6},
        temperature=300.0,
    )
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("vajax.openvaf")

# Re-export key types
from .cache import cache_stats, clear_cache, exec_with_cache, get_vmapped_jit
from .mir import (
    Block,
    CFGAnalyzer,
    LoopInfo,
    MIRFunction,
    MIRInstruction,
    PhiOperand,
    PHIResolution,
    SSAAnalyzer,
    parse_mir_function,
)


class OpenVAFToJAX:
    """Translates OpenVAF MIR to JAX functions.

    This is the main entry point for the translator. It provides methods to
    generate JAX functions for device evaluation.
    """

    def __init__(self, module):
        """Initialize with a compiled VaModule from openvaf_py.

        Args:
            module: VaModule from openvaf_py.compile_va()
        """
        self.module = module

        # Parse MIR data
        self.mir_data = module.get_mir_instructions()
        self.dae_data = module.get_dae_system()
        self.init_mir_data = module.get_init_mir_instructions()

        # String constants (needed for $display support, must be parsed before MIR)
        self.str_constants = dict(module.get_str_constants())

        # Parse into structured MIR
        self.eval_mir = parse_mir_function("eval", self.mir_data, self.str_constants)
        self.init_mir = parse_mir_function("init", self.init_mir_data, self.str_constants)

        # Extract metadata
        self.params = list(self.mir_data["params"])
        self.constants = dict(self.mir_data["constants"])
        self.bool_constants = dict(self.mir_data.get("bool_constants", {}))
        self.int_constants = dict(self.mir_data.get("int_constants", {}))

        self.init_params = list(self.init_mir_data["params"])
        self.cache_mapping = list(self.init_mir_data["cache_mapping"])

        # Node collapse support
        self.collapse_decision_outputs = list(module.collapse_decision_outputs)
        self.collapsible_pairs = list(module.collapsible_pairs)

        # Build param index to value mapping for eval
        all_func_params = module.get_all_func_params()
        self.param_idx_to_val = {p[0]: f"v{p[1]}" for p in all_func_params}

        # Track feature usage
        self.uses_simparam_gmin = False
        self.uses_analysis = False
        self.analysis_type_map = {
            "dc": 0,
            "static": 0,
            "ac": 1,
            "tran": 2,
            "transient": 2,
            "noise": 3,
            "nodeset": 4,
        }

    @classmethod
    def from_cache(cls, cached_data: Dict[str, Any]) -> "OpenVAFToJAX":
        """Create translator from cached MIR data.

        This avoids re-running openvaf_py.compile_va() by loading
        previously cached MIR data and metadata.

        Args:
            cached_data: Dict with keys (see get_cache_data for full list)

        Returns:
            OpenVAFToJAX instance
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)

        # Create a mock module object with the cached data
        # This allows existing code to work without modification
        class MockModule:
            pass

        mock = MockModule()
        mock.param_names = cached_data["param_names"]
        mock.param_kinds = cached_data["param_kinds"]
        mock.nodes = cached_data["nodes"]
        mock.collapse_decision_outputs = cached_data["collapse_decision_outputs"]
        mock.collapsible_pairs = cached_data["collapsible_pairs"]
        mock.num_collapsible = cached_data["num_collapsible"]
        mock.init_param_names = cached_data["init_param_names"]
        mock.init_param_kinds = cached_data["init_param_kinds"]
        mock.name = cached_data.get("model_name", "cached_model")

        # Cached callables
        _param_defaults = cached_data["param_defaults"]
        _osdi_descriptor = cached_data["osdi_descriptor"]

        mock.get_param_defaults = lambda: _param_defaults.items()
        mock.get_osdi_descriptor = lambda: _osdi_descriptor
        mock.get_all_func_params = lambda: cached_data["all_func_params"]

        instance.module = mock

        # Load MIR data from cache
        instance.mir_data = cached_data["mir_data"]
        instance.dae_data = cached_data["dae_data"]
        instance.init_mir_data = cached_data["init_mir_data"]
        instance.str_constants = cached_data["str_constants"]

        # Parse into structured MIR
        instance.eval_mir = parse_mir_function("eval", instance.mir_data, instance.str_constants)
        instance.init_mir = parse_mir_function(
            "init", instance.init_mir_data, instance.str_constants
        )

        # Extract metadata
        instance.params = list(instance.mir_data["params"])
        instance.constants = dict(instance.mir_data["constants"])
        instance.bool_constants = dict(instance.mir_data.get("bool_constants", {}))
        instance.int_constants = dict(instance.mir_data.get("int_constants", {}))

        instance.init_params = list(instance.init_mir_data["params"])
        instance.cache_mapping = list(instance.init_mir_data["cache_mapping"])

        # Node collapse support
        instance.collapse_decision_outputs = cached_data["collapse_decision_outputs"]
        instance.collapsible_pairs = cached_data["collapsible_pairs"]

        # Build param index to value mapping
        all_func_params = cached_data["all_func_params"]
        instance.param_idx_to_val = {p[0]: f"v{p[1]}" for p in all_func_params}

        # Track feature usage (will be set during code generation)
        instance.uses_simparam_gmin = False
        instance.uses_analysis = False
        instance.analysis_type_map = {
            "dc": 0,
            "static": 0,
            "ac": 1,
            "tran": 2,
            "transient": 2,
            "noise": 3,
            "nodeset": 4,
        }

        logger.info("Created OpenVAFToJAX from cached MIR data")
        return instance

    def get_cache_data(self) -> Dict[str, Any]:
        """Get data needed to reconstruct this translator from cache.

        Returns:
            Dict with all data needed by from_cache()
        """
        if self.module is None or not hasattr(self.module, "get_mir_instructions"):
            raise ValueError("Cannot get cache data from a translator loaded from cache")

        return {
            "mir_data": self.mir_data,
            "init_mir_data": self.init_mir_data,
            "dae_data": self.dae_data,
            "str_constants": self.str_constants,
            "param_names": list(self.module.param_names),
            "param_kinds": list(self.module.param_kinds),
            "nodes": list(self.module.nodes),
            "collapse_decision_outputs": list(self.module.collapse_decision_outputs),
            "collapsible_pairs": list(self.module.collapsible_pairs),
            "num_collapsible": self.module.num_collapsible,
            "all_func_params": list(self.module.get_all_func_params()),
            "init_param_names": list(self.module.init_param_names),
            "init_param_kinds": list(self.module.init_param_kinds),
            "param_defaults": dict(self.module.get_param_defaults()),
            "osdi_descriptor": self.module.get_osdi_descriptor(),
            "model_name": self.module.name if hasattr(self.module, "name") else "model",
        }

    @classmethod
    def from_file(cls, va_path: str) -> "OpenVAFToJAX":
        """Create translator from a Verilog-A file.

        Args:
            va_path: Path to the .va file

        Returns:
            OpenVAFToJAX instance
        """
        import openvaf_py

        modules = openvaf_py.compile_va(va_path)
        if not modules:
            raise ValueError(f"No modules found in {va_path}")
        return cls(modules[0])

    def release_mir_data(self):
        """Release MIR data after code generation is complete.

        Call this after all translate_*() methods have been called to free
        memory. The translator remains usable for accessing metadata.
        """
        self.mir_data = None
        self.init_mir_data = None
        self.dae_data = None
        self.eval_mir = None
        self.init_mir = None

    def get_dae_metadata(self) -> Dict:
        """Get DAE system metadata without generating code.

        Returns the same metadata that would be included in translate_array()
        or translate_eval_array_with_cache_split() output, but without
        generating any JAX functions.

        Returns:
            Dict with:
            - 'node_names': list of residual node names
            - 'jacobian_keys': list of (row_name, col_name) tuples
            - 'terminals': list of terminal node names
            - 'internal_nodes': list of internal node names
            - 'num_terminals': number of terminals
            - 'num_internal': number of internal nodes
        """
        assert self.dae_data is not None, "dae_data released, call before release_mir_data()"
        return {
            "node_names": [res["node_name"] for res in self.dae_data["residuals"]],
            "jacobian_keys": [
                (entry["row_node_name"], entry["col_node_name"])
                for entry in self.dae_data["jacobian"]
            ],
            "terminals": self.dae_data["terminals"],
            "internal_nodes": self.dae_data["internal_nodes"],
            "num_terminals": self.dae_data["num_terminals"],
            "num_internal": self.dae_data["num_internal"],
        }

    def get_params(self, include_internal: bool = False) -> List[Dict]:
        """Get parameter metadata from the Verilog-A model.

        Returns a list of parameter info dicts, preserving the order from the .va file.
        Each dict contains:
            - 'name': Parameter name
            - 'type': 'real', 'int', or 'str'
            - 'default': Default value (None if not specified)
            - 'units': Units string from `_P(units="...")`
            - 'description': Description from `_P(info="...")` or comments
            - 'aliases': List of alternative names
            - 'is_instance': True if instance parameter
            - 'is_model_param': True if model parameter

        Args:
            include_internal: If True, include internal parameters (hidden_state, etc.)
                             Default False returns only user-facing model parameters.

        Returns:
            List of parameter info dicts.

        Example:
            >>> translator = OpenVAFToJAX.from_file("diode.va")
            >>> params = translator.get_params()
            >>> for p in params[:3]:
            ...     print(f"{p['name']}: {p['description']} (default={p['default']})")
            Is: Saturation current (default=1e-14)
            N: Emission coefficient (default=1.0)
            Rs: Ohmic resistance (default=0.0)
        """
        desc = self.module.get_osdi_descriptor()
        defaults = dict(self.module.get_param_defaults())

        params = []
        for p in desc["params"]:
            name = p["name"]

            # Skip internal params unless requested
            if not include_internal:
                # Skip system params and hidden state
                if name.startswith("$") or name == "mfactor":
                    continue

            # Type is encoded in flags: 0=real, 1=int, 2=str
            flags = p.get("flags", 0)
            type_code = flags & 3
            type_str = {0: "real", 1: "int", 2: "str"}.get(type_code, "real")

            # Get default - lookup case-insensitive
            default = defaults.get(name)
            if default is None:
                default = defaults.get(name.lower())

            params.append(
                {
                    "name": name,
                    "type": type_str,
                    "default": default,
                    "units": p.get("units", ""),
                    "description": p.get("description", ""),
                    "aliases": p.get("aliases", []),
                    "is_instance": p.get("is_instance", False),
                    "is_model_param": p.get("is_model_param", True),
                }
            )

        return params

    def print_params(self, include_internal: bool = False) -> None:
        """Print parameter table to stdout.

        Convenience method to display all parameters with their metadata
        in a formatted table.

        Args:
            include_internal: If True, include internal parameters.

        Example:
            >>> translator = OpenVAFToJAX.from_file("diode.va")
            >>> translator.print_params()
            === Parameters for diode ===
            Name                 Type   Default      Units    Description
            --------------------------------------------------------------------------------
            Is                   real   1e-14        A        Saturation current
            N                    real   1            -        Emission coefficient
            ...
        """
        params = self.get_params(include_internal=include_internal)
        model_name = self.module.name if hasattr(self.module, "name") else "model"

        print(f"\n=== Parameters for {model_name} ({len(params)} params) ===")
        print(f"{'Name':<20} {'Type':<6} {'Default':<12} {'Units':<8} Description")
        print("-" * 80)

        for p in params:
            default_str = "None"
            if p["default"] is not None:
                if isinstance(p["default"], float):
                    if abs(p["default"]) < 1e-3 or abs(p["default"]) >= 1e6:
                        default_str = f"{p['default']:.4g}"
                    else:
                        default_str = f"{p['default']}"
                else:
                    default_str = str(p["default"])

            units = p["units"] if p["units"] else "-"
            desc = p["description"][:40] if p["description"] else ""

            print(f"{p['name']:<20} {p['type']:<6} {default_str:<12} {units:<8} {desc}")

        # Show aliases if any
        aliases_found = [(p["name"], p["aliases"]) for p in params if p["aliases"]]
        if aliases_found:
            print("\nAliases:")
            for name, aliases in aliases_found:
                print(f"  {name}: {', '.join(aliases)}")

    def _get_param_info(self) -> Dict[str, Dict]:
        """Get comprehensive parameter info from OSDI descriptor and defaults.

        Internal method used by translate_init/translate_eval.
        For user code, use get_params() instead.

        Returns:
            Dict mapping param name -> param info dict
        """
        desc = self.module.get_osdi_descriptor()
        defaults = dict(self.module.get_param_defaults())

        param_info = {}
        for p in desc["params"]:
            name = p["name"]
            # Type is encoded in flags: 0=real, 1=int, 2=str
            flags = p.get("flags", 0)
            type_code = flags & 3
            type_str = {0: "real", 1: "int", 2: "str"}.get(type_code, "real")

            # Get default - lookup case-insensitive
            default = defaults.get(name)
            if default is None:
                default = defaults.get(name.lower())

            param_info[name] = {
                "name": name,
                "type": type_str,
                "default": default,
                "units": p.get("units", ""),
                "description": p.get("description", ""),
                "is_instance": p.get("is_instance", False),
                "is_model_param": p.get("is_model_param", True),
            }

        return param_info

    def _build_param_inputs(
        self,
        param_names: List[str],
        param_kinds: List[str],
        params: Dict[str, float],
        temperature: float,
        mfactor: float,
        param_info: Dict[str, Dict],
        context: str = "eval",
    ) -> Tuple[List[float], List[str]]:
        """Build validated input array for init or eval function.

        Args:
            param_names: List of param names (from module.init_param_names or param_names)
            param_kinds: List of param kinds
            params: User-provided params dict (case-insensitive)
            temperature: Device temperature in Kelvin
            mfactor: Device multiplier
            param_info: Dict from _get_param_info()
            context: 'init' or 'eval' for error messages

        Returns:
            Tuple of (inputs_list, warnings_list)
        """
        # Build case-insensitive param lookup
        params_lower = {k.lower(): v for k, v in params.items()}

        # Sentinel value for TEMP/TNOM meaning "use $temperature"
        SENTINEL = 1e21

        warnings = []
        inputs = []

        for name, kind in zip(param_names, param_kinds):
            name_lower = name.lower()
            value = None

            # Handle by kind first
            if kind == "voltage":
                # Voltage params are set at runtime, use 0.0 placeholder
                value = 0.0
            elif kind == "temperature" or name == "$temperature":
                # System temperature
                value = temperature
            elif kind == "sysfun" and name == "mfactor":
                value = mfactor
            elif kind in ("hidden_state", "current"):
                # Placeholders - filled by init or eval
                value = 0.0
            elif kind == "implicit_unknown":
                # Internal node voltage for implicit equations
                # These are set at runtime from the voltage array, same as 'voltage'
                value = 0.0
            elif kind == "param_given":
                # Check if the corresponding param was explicitly set
                # param_given names are like "VTO_given" -> check for "VTO"
                base_name = name.replace("_given", "").lower()
                value = 1.0 if base_name in params_lower else 0.0
            elif kind == "port_connected":
                # Assume all ports are connected (LRM 9.19)
                # TODO: Add support for optional ports by tracking connections
                warnings.append(
                    f"{context}: $port_connected({name}) assumed true - optional ports not supported"
                )
                value = 1.0
            elif kind == "abstime":
                # Absolute simulation time (LRM 9.7)
                # For DC analysis, abstime=0.0. For transient, caller must update this.
                warnings.append(
                    f"{context}: $abstime used - defaults to 0.0 (DC). For transient, update input array."
                )
                value = 0.0
            else:
                # Regular param - check user params first
                if name in params:
                    value = float(params[name])
                elif name_lower in params_lower:
                    value = float(params_lower[name_lower])
                else:
                    # Check for default from OSDI info
                    info = param_info.get(name) or param_info.get(name.upper())
                    if info and info["default"] is not None:
                        value = float(info["default"])
                    else:
                        # Special handling for TEMP/TNOM
                        if name_lower in ("temp", "tnom"):
                            value = SENTINEL
                        elif kind == "unknown":
                            warnings.append(
                                f"{context}: '{name}' (kind={kind}) has no value, using 0.0"
                            )
                            value = 0.0
                        elif kind == "sysfun":
                            # Other system functions default to appropriate values
                            if "scale" in name_lower:
                                value = 1.0
                            elif "shrink" in name_lower:
                                value = 0.0
                            else:
                                warnings.append(
                                    f"{context}: sysfun '{name}' has no handler, using 0.0"
                                )
                                value = 0.0
                        else:
                            warnings.append(
                                f"{context}: param '{name}' (kind={kind}) has no value or default, using 0.0"
                            )
                            value = 0.0

            assert value is not None, f"Failed to compute value for {name}"
            inputs.append(value)

        return inputs, warnings

    def translate_init(
        self,
        params: Dict[str, float] = None,
        temperature: float = 300.0,
        mfactor: float = 1.0,
        debug: bool = False,
    ) -> Tuple[Callable, Dict]:
        """Generate init function with validated parameters.

        This is the main API for generating init functions. It validates parameters,
        handles special cases (sentinel values, $temperature, mfactor), and returns
        a function plus comprehensive metadata.

        Args:
            params: Dict of parameter name -> value. Case-insensitive lookup.
            temperature: Device temperature in Kelvin (default 300K).
            mfactor: Device multiplier (default 1.0).
            debug: If True, print parameter table for debugging.

        Returns:
            Tuple of (init_fn, metadata)

            init_fn signature:
                init_fn(inputs: Array[N_init]) -> (cache: Array[N_cache], collapse: Array[N_collapse])

            metadata dict:
                - 'param_names': list of init param names
                - 'param_kinds': list of init param kinds
                - 'init_inputs': validated input array (ready to use)
                - 'param_info': list of {name, type, default, value, units, desc}
                - 'cache_size': number of cached values
                - 'warnings': validation warnings
        """
        from .codegen.function_builder import InitFunctionBuilder

        assert self.init_mir is not None, "init_mir released, call before release_mir_data()"

        if params is None:
            params = {}

        # Get OSDI param info
        param_info = self._get_param_info()

        # Get init param names/kinds
        init_param_names = list(self.module.init_param_names)
        init_param_kinds = list(self.module.init_param_kinds)

        # Build validated inputs
        init_inputs, warnings = self._build_param_inputs(
            init_param_names,
            init_param_kinds,
            params,
            temperature,
            mfactor,
            param_info,
            context="init",
        )

        # Build detailed param info list for metadata
        detailed_param_info = []
        for i, (name, kind) in enumerate(zip(init_param_names, init_param_kinds)):
            info = param_info.get(name) or param_info.get(name.upper()) or {}
            detailed_param_info.append(
                {
                    "name": name,
                    "kind": kind,
                    "type": info.get("type", "real"),
                    "default": info.get("default"),
                    "value": init_inputs[i],
                    "units": info.get("units", ""),
                    "description": info.get("description", ""),
                }
            )

        # Debug output
        if debug:
            print(f"\n=== Init Parameters ({len(init_param_names)}) ===")
            print(
                f"{'Name':<20} {'Kind':<15} {'Type':<6} {'Default':<12} {'Value':<12} {'Units':<8} Description"
            )
            print("-" * 100)
            for p in detailed_param_info:
                default_str = f"{p['default']:.4g}" if p["default"] is not None else "None"
                value_str = (
                    f"{p['value']:.4g}" if isinstance(p["value"], float) else str(p["value"])
                )
                print(
                    f"{p['name']:<20} {p['kind']:<15} {p['type']:<6} {default_str:<12} {value_str:<12} {p['units']:<8} {p['description'][:30]}"
                )
            if warnings:
                print(f"\nWarnings: {len(warnings)}")
                for w in warnings:
                    print(f"  - {w}")

        # Generate code
        t0 = time.perf_counter()
        logger.info("    translate_init: generating code...")

        n_init_params = len(init_param_names)
        all_indices = list(range(n_init_params))

        builder = InitFunctionBuilder(
            self.init_mir, self.cache_mapping, self.collapse_decision_outputs
        )
        fn_name, code_lines = builder.build_simple(all_indices)

        # Collect codegen warnings (unknown $simparam, $discontinuity, etc.)
        warnings.extend(builder.codegen_warnings)

        t1 = time.perf_counter()
        logger.info(
            f"    translate_init: code generated ({len(code_lines)} lines) in {t1 - t0:.1f}s"
        )

        code = "\n".join(code_lines)

        # Compile with caching
        logger.info("    translate_init: exec()...")
        init_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_init: exec() done in {t2 - t1:.1f}s")

        metadata = {
            "param_names": init_param_names,
            "param_kinds": init_param_kinds,
            "init_inputs": init_inputs,
            "param_info": detailed_param_info,
            "cache_size": len(self.cache_mapping),
            "cache_mapping": self.cache_mapping,
            "collapsible_pairs": self.collapsible_pairs,
            "collapse_decision_outputs": self.collapse_decision_outputs,
            "temperature": temperature,
            "mfactor": mfactor,
            "warnings": warnings,
        }

        return init_fn, metadata

    def translate_eval(
        self,
        params: Dict[str, float] = None,
        temperature: float = 300.0,
        mfactor: float = 1.0,
        debug: bool = False,
        cache_split: Optional[Tuple[List[int], List[int]]] = None,
        sccp_known_values: Optional[Dict[str, Any]] = None,
        propagate_constants: bool = True,
    ) -> Tuple[Callable, Dict]:
        """Generate eval function with validated parameters.

        This is the main API for generating eval functions. It validates parameters,
        computes voltage/shared indices, and returns a function plus comprehensive metadata.

        Args:
            params: Dict of parameter name -> value. Case-insensitive lookup.
            temperature: Device temperature in Kelvin (default 300K).
            mfactor: Device multiplier (default 1.0).
            debug: If True, print parameter table for debugging.
            cache_split: Optional tuple of (shared_cache_indices, varying_cache_indices)
                         for advanced cache splitting optimization.
            sccp_known_values: Optional dict mapping MIR value IDs to constant values
                              for SCCP-based dead code elimination. Used to eliminate
                              branches based on compile-time known values like TYPE=1.
                              Example: {'v51588': 1} for PSP102 NMOS.
            propagate_constants: If True (default), automatically propagate all model
                                parameters to SCCP for constant folding. This enables
                                dead code elimination for any computation that only
                                depends on model params (not voltages).

        Returns:
            Tuple of (eval_fn, metadata)

            eval_fn signature:
                eval_fn(shared_params, varying_params, cache, simparams, limit_state_in, limit_funcs)
                    -> (res_resist, res_react, jac_resist, jac_react,
                        lim_rhs_resist, lim_rhs_react, ss_resist, ss_react, limit_state_out)

            simparams array layout:
                - simparams[0] = analysis_type (0=DC, 1=AC, 2=transient, 3=noise)
                - simparams[1] = gmin (minimum conductance for convergence)

            metadata dict:
                - 'param_names': list of eval param names
                - 'param_kinds': list of eval param kinds
                - 'shared_inputs': validated shared param array (ready to use)
                - 'shared_indices': indices of shared params (non-voltage)
                - 'voltage_indices': indices of voltage params
                - 'param_info': list of {name, type, default, value, units, desc}
                - 'node_names': list of node names for voltage mapping
                - 'jacobian_keys': list of (row, col) tuples
                - 'simparams_layout': dict describing simparams array layout
                - 'warnings': validation warnings
        """
        from .codegen.function_builder import EvalFunctionBuilder

        assert self.eval_mir is not None, "eval_mir released, call before release_mir_data()"
        assert self.dae_data is not None, "dae_data released, call before release_mir_data()"

        if params is None:
            params = {}

        # Get OSDI param info
        param_info = self._get_param_info()

        # Get eval param names/kinds
        eval_param_names = list(self.module.param_names)
        eval_param_kinds = list(self.module.param_kinds)

        # Compute voltage and shared indices
        # Both 'voltage' and 'implicit_unknown' are node voltages provided at runtime
        voltage_kinds = ("voltage", "implicit_unknown")
        voltage_indices = [i for i, k in enumerate(eval_param_kinds) if k in voltage_kinds]

        # Identify params that should come from simparams array instead of shared_params
        # These are runtime environment values: $abstime, mfactor
        simparam_params: Dict[int, str] = {}
        for i, (name, kind) in enumerate(zip(eval_param_names, eval_param_kinds)):
            if kind == "abstime":
                simparam_params[i] = "$abstime"
            elif kind == "sysfun" and name == "mfactor":
                simparam_params[i] = "$mfactor"

        # shared_indices excludes voltage params AND simparam params
        shared_indices = [
            i
            for i, k in enumerate(eval_param_kinds)
            if k not in voltage_kinds and i not in simparam_params
        ]

        # Build validated inputs for shared params only
        # (voltage params are provided at runtime, simparam params come from simparams array)
        shared_param_names = [eval_param_names[i] for i in shared_indices]
        shared_param_kinds = [eval_param_kinds[i] for i in shared_indices]

        shared_inputs, warnings = self._build_param_inputs(
            shared_param_names,
            shared_param_kinds,
            params,
            temperature,
            mfactor,
            param_info,
            context="eval",
        )

        # Build detailed param info list
        detailed_param_info = []
        for i, (name, kind) in enumerate(zip(eval_param_names, eval_param_kinds)):
            info = param_info.get(name) or param_info.get(name.upper()) or {}
            # Value depends on whether this is shared, voltage, or simparam
            if kind in voltage_kinds:
                value = None  # Set at runtime from voltage array
            elif i in simparam_params:
                value = "(simparam)"  # Comes from simparams array at runtime
            else:
                shared_idx = shared_indices.index(i)
                value = shared_inputs[shared_idx]
            detailed_param_info.append(
                {
                    "name": name,
                    "kind": kind,
                    "type": info.get("type", "real"),
                    "default": info.get("default"),
                    "value": value,
                    "units": info.get("units", ""),
                    "description": info.get("description", ""),
                }
            )

        # Debug output
        if debug:
            print(f"\n=== Eval Parameters ({len(eval_param_names)}) ===")
            print(f"  Shared: {len(shared_indices)}, Voltage: {len(voltage_indices)}")
            print(f"\n{'Name':<25} {'Kind':<15} {'Type':<6} {'Default':<12} {'Value':<12}")
            print("-" * 80)
            for p in detailed_param_info:
                default_str = f"{p['default']:.4g}" if p["default"] is not None else "None"
                if p["value"] is None:
                    value_str = "(runtime)"
                elif isinstance(p["value"], float):
                    value_str = f"{p['value']:.4g}"
                else:
                    value_str = str(p["value"])
                print(
                    f"{p['name']:<25} {p['kind']:<15} {p['type']:<6} {default_str:<12} {value_str:<12}"
                )
            if warnings:
                print(f"\nWarnings: {len(warnings)}")
                for w in warnings:
                    print(f"  - {w}")

        # Parse cache_split
        # When no cache_split provided, all cache entries are varying (per-device)
        if cache_split is not None:
            shared_cache_indices, varying_cache_indices = cache_split
        else:
            # Default: all cache indices are varying (per-device)
            shared_cache_indices = []
            varying_cache_indices = list(range(len(self.cache_mapping)))

        # Build SCCP known values for constant propagation
        # This maps MIR value IDs to their constant values for shared params
        effective_sccp_values: Optional[Dict[str, Any]] = sccp_known_values
        if propagate_constants:
            # Start with explicitly provided values
            effective_sccp_values = dict(sccp_known_values) if sccp_known_values else {}

            # Add all shared params with known values
            for j, orig_idx in enumerate(shared_indices):
                # Get MIR value ID for this param
                value_id = self.param_idx_to_val.get(orig_idx)
                if value_id and value_id not in effective_sccp_values:
                    # Get the computed value for this param
                    param_value = shared_inputs[j]
                    if param_value is not None:
                        effective_sccp_values[value_id] = param_value

            if debug and effective_sccp_values:
                print(f"\n=== SCCP Constants ({len(effective_sccp_values)}) ===")
                n_explicit = len(sccp_known_values) if sccp_known_values else 0
                n_auto = len(effective_sccp_values) - n_explicit
                print(f"  Explicit: {n_explicit}, Auto from params: {n_auto}")

        # Generate code
        t0 = time.perf_counter()
        has_sccp = effective_sccp_values is not None and len(effective_sccp_values) > 0
        logger.info(f"    translate_eval: generating code (sccp={has_sccp})...")

        builder = EvalFunctionBuilder(
            self.eval_mir,
            self.dae_data,
            self.cache_mapping,
            self.param_idx_to_val,
            sccp_known_values=effective_sccp_values,
            eval_param_names=eval_param_names,
        )
        fn_name, code_lines = builder.build_with_cache_split(
            shared_indices,
            voltage_indices,
            shared_cache_indices,
            varying_cache_indices,
            simparam_params=simparam_params,
        )

        # Collect codegen warnings (unknown $simparam, $discontinuity, etc.)
        warnings.extend(builder.codegen_warnings)

        t1 = time.perf_counter()
        logger.info(
            f"    translate_eval: code generated ({len(code_lines)} lines) in {t1 - t0:.1f}s"
        )

        code = "\n".join(code_lines)

        # Compile with caching
        logger.info("    translate_eval: exec()...")
        eval_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_eval: exec() done in {t2 - t1:.1f}s")

        # Build metadata
        node_names = [res["node_name"] for res in self.dae_data["residuals"]]
        node_indices = [res["node_idx"] for res in self.dae_data["residuals"]]
        jacobian_keys = [
            (entry["row_node_name"], entry["col_node_name"]) for entry in self.dae_data["jacobian"]
        ]
        jacobian_indices = [
            (entry["row_node_idx"], entry["col_node_idx"]) for entry in self.dae_data["jacobian"]
        ]

        # Get simparam metadata from builder (dynamically tracked during codegen)
        simparam_meta = builder.simparam_metadata

        # Build simparams_layout from tracked simparams for documentation
        simparams_layout = {}
        for name, idx in simparam_meta.get("simparam_indices", {}).items():
            # Add description for known simparams
            descriptions = {
                "$analysis_type": "0=DC, 1=AC, 2=transient, 3=noise",
                "gmin": "minimum conductance for convergence (S)",
                "abstol": "absolute current tolerance (A)",
                "vntol": "absolute voltage tolerance (V)",
                "reltol": "relative tolerance",
                "tnom": "nominal temperature (K)",
                "scale": "scale factor",
                "shrink": "shrink factor",
                "imax": "branch current limit (A)",
                "$abstime": "absolute simulation time (s)",
                "$mfactor": "device multiplicity factor",
            }
            simparams_layout[idx] = {
                "name": name,
                "description": descriptions.get(name, f'$simparam("{name}")'),
            }

        # Collect SCCP statistics if available
        sccp_stats = {}
        if builder.sccp is not None:
            sccp = builder.sccp
            total_constants = sum(1 for v in sccp.lattice.values() if v.is_constant())
            mir_constants = (
                len(self.eval_mir.constants)
                + len(self.eval_mir.int_constants)
                + len(self.eval_mir.bool_constants)
            )
            # Constants discovered through propagation (not from MIR or known_values)
            builtin_count = len(sccp.BUILTIN_CONSTANTS)
            known_count = len(effective_sccp_values) if effective_sccp_values else 0
            computed_constants = total_constants - mir_constants - builtin_count - known_count
            sccp_stats = {
                "total_blocks": len(self.eval_mir.blocks),
                "reachable_blocks": len(sccp.visited_blocks),
                "dead_blocks": len(sccp.get_dead_blocks()),
                "total_constants": total_constants,
                "mir_constants": mir_constants,
                "param_constants": known_count,
                "computed_constants": max(0, computed_constants),
            }
            if debug and sccp_stats:
                print("\n=== SCCP Results ===")
                print(
                    f"  Blocks: {sccp_stats['reachable_blocks']}/{sccp_stats['total_blocks']} reachable, {sccp_stats['dead_blocks']} dead"
                )
                print(f"  Constants: {sccp_stats['total_constants']} total")
                print(f"    MIR constants: {sccp_stats['mir_constants']}")
                print(f"    Param constants: {sccp_stats['param_constants']}")
                print(f"    Computed (propagated): {sccp_stats['computed_constants']}")

        metadata = {
            "param_names": eval_param_names,
            "param_kinds": eval_param_kinds,
            "shared_inputs": shared_inputs,
            "shared_indices": shared_indices,
            "voltage_indices": voltage_indices,
            "param_info": detailed_param_info,
            "node_names": node_names,
            "node_indices": node_indices,
            "jacobian_keys": jacobian_keys,
            "jacobian_indices": jacobian_indices,
            "terminals": self.dae_data["terminals"],
            "internal_nodes": self.dae_data["internal_nodes"],
            "num_terminals": self.dae_data["num_terminals"],
            "num_internal": self.dae_data["num_internal"],
            "temperature": temperature,
            "mfactor": mfactor,
            "simparams_layout": simparams_layout,
            # Simparam metadata for building simparams array
            "simparams_used": simparam_meta.get("simparams_used", ["$analysis_type"]),
            "simparam_indices": simparam_meta.get("simparam_indices", {"$analysis_type": 0}),
            "simparam_count": simparam_meta.get("simparam_count", 1),
            "sccp_stats": sccp_stats,
            "warnings": warnings,
        }

        return eval_fn, metadata

    def translate(
        self,
        params: Dict[str, float] = None,
        temperature: float = 300.0,
    ) -> Callable:
        """Generate a single evaluation function (legacy API).

        This is a backward-compatible wrapper around translate_init() and
        translate_eval(). For new code, prefer using those methods directly
        for better control over init/eval separation and caching.

        Args:
            params: Dict of parameter name -> value. Case-insensitive lookup.
            temperature: Device temperature in Kelvin (default 300K).

        Returns:
            eval_fn with signature:
                eval_fn(inputs: List[float]) -> (residuals_dict, jacobian_dict)

            where:
                - inputs is a list indexed by module.param_names order
                - residuals_dict maps node_name -> {'resist': float, 'react': float}
                - jacobian_dict maps (row, col) -> {'resist': float, 'react': float}
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        if params is None:
            params = {}

        # Pre-compile with propagate_constants=False so params can change at runtime
        init_fn, init_meta = self.translate_init(params=params, temperature=temperature)
        eval_fn, eval_meta = self.translate_eval(
            params=params, temperature=temperature, propagate_constants=False
        )
        init_fn = jax.jit(init_fn)
        eval_fn = jax.jit(eval_fn)

        # Build mapping from user param index to init param index
        init_param_names = list(self.module.init_param_names)
        user_param_names = list(self.module.param_names)
        user_indices = []
        init_indices = []
        for init_idx, init_name in enumerate(init_param_names):
            if init_name in user_param_names:
                user_idx = user_param_names.index(init_name)
                user_indices.append(user_idx)
                init_indices.append(init_idx)
        user_indices_arr = jnp.array(user_indices, dtype=jnp.int32) if user_indices else None
        init_indices_arr = jnp.array(init_indices, dtype=jnp.int32) if init_indices else None

        # Store metadata for the wrapper
        shared_indices = eval_meta["shared_indices"]
        voltage_indices = eval_meta["voltage_indices"]
        node_names = eval_meta["node_names"]
        jacobian_keys = eval_meta["jacobian_keys"]
        default_init_inputs = jnp.array(init_meta["init_inputs"])
        param_names = self.module.param_names
        param_kinds = self.module.param_kinds

        def wrapper(inputs: List[float]) -> Tuple[Dict, Dict]:
            """Evaluate the model (legacy dict interface)."""
            inputs_arr = jnp.asarray(inputs)

            # Extract mfactor for simparams
            mfactor = 1.0
            for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
                if kind == "sysfun" and name == "mfactor":
                    mfactor = float(inputs[i])
                    break

            # Build init inputs from user inputs
            init_inputs = default_init_inputs
            if user_indices_arr is not None:
                user_values = inputs_arr[user_indices_arr]
                init_inputs = init_inputs.at[init_indices_arr].set(user_values)

            # Run init to get cache
            cache, _ = init_fn(init_inputs)

            # Build eval inputs
            shared_params = (
                inputs_arr[jnp.array(shared_indices)] if shared_indices else jnp.array([])
            )
            varying_params = (
                inputs_arr[jnp.array(voltage_indices)] if voltage_indices else jnp.array([])
            )

            # Run eval with simparams [analysis_type, mfactor, gmin]
            simparams = jnp.array([0.0, mfactor, 1e-12])
            # Uniform interface: always pass limit_state_in (zeros when not using limits)
            limit_state_in = jnp.zeros(1)  # Minimal dummy array
            limit_funcs = {}  # Empty dict - limit functions not used
            # Cache split: shared_cache is empty, all cache in device_cache
            shared_cache = jnp.array([])
            device_cache = cache
            result = eval_fn(
                shared_params,
                varying_params,
                shared_cache,
                device_cache,
                simparams,
                limit_state_in,
                limit_funcs,
            )
            res_resist, res_react, jac_resist, jac_react = result[:4]

            # Convert to dicts (NumPy for performance)
            res_resist_np = np.asarray(res_resist)
            res_react_np = np.asarray(res_react)
            jac_resist_np = np.asarray(jac_resist)
            jac_react_np = np.asarray(jac_react)

            residuals = {
                name: {"resist": res_resist_np[i], "react": res_react_np[i]}
                for i, name in enumerate(node_names)
            }
            jacobian = {
                key: {"resist": jac_resist_np[i], "react": jac_react_np[i]}
                for i, key in enumerate(jacobian_keys)
            }
            return residuals, jacobian

        return wrapper

    def translate_init_array(self) -> Tuple[Callable, Dict]:
        """Generate a vmappable init function (internal API for engine).

        For user code, prefer translate_init() which handles param validation.

        Returns a function with signature:
            init_fn(inputs: Array[N_init]) -> (cache: Array[N_cache], collapse_decisions: Array[N_collapse])

        Also returns metadata dict with:
            - 'param_names': list of init param names
            - 'param_kinds': list of init param kinds
            - 'cache_size': number of cached values
            - 'cache_mapping': list of {init_value, eval_param} dicts

        This function computes all cached values that eval needs.
        """
        from .codegen.function_builder import InitFunctionBuilder

        assert self.init_mir is not None, "init_mir released, call before release_mir_data()"

        t0 = time.perf_counter()
        logger.info("    translate_init_array: generating code...")

        # Build init function with all params as "device" params (no shared)
        # This produces init_fn(device_params) instead of init_fn_split(shared, device)
        n_init_params = len(self.init_params)
        all_indices = list(range(n_init_params))

        builder = InitFunctionBuilder(
            self.init_mir, self.cache_mapping, self.collapse_decision_outputs
        )
        fn_name, code_lines = builder.build_simple(all_indices)

        t1 = time.perf_counter()
        logger.info(
            f"    translate_init_array: code generated ({len(code_lines)} lines) in {t1 - t0:.1f}s"
        )

        code = "\n".join(code_lines)
        logger.info(f"    translate_init_array: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate_init_array: exec()...")
        init_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_init_array: exec() done in {t2 - t1:.1f}s")

        # Build metadata
        param_defaults = {}
        if hasattr(self.module, "get_param_defaults"):
            param_defaults = dict(self.module.get_param_defaults())

        metadata = {
            "param_names": list(self.module.init_param_names),
            "param_kinds": list(self.module.init_param_kinds),
            "cache_size": len(self.cache_mapping),
            "cache_mapping": self.cache_mapping,
            "param_defaults": param_defaults,
            "collapsible_pairs": self.collapsible_pairs,
            "collapse_decision_outputs": self.collapse_decision_outputs,
        }

        return init_fn, metadata

    def translate_init_array_split(
        self, shared_indices: List[int], varying_indices: List[int], init_to_eval: List[int]
    ) -> Tuple[Callable, Dict]:
        """Generate a vmappable init function with split shared/device params (internal API).

        For user code, prefer translate_init() which handles param validation.

        This is an optimized version for batched device evaluation that reduces
        memory by separating constant parameters (shared across all devices) from
        varying parameters (different per device).

        Args:
            shared_indices: Eval param indices that are constant across all devices
            varying_indices: Eval param indices that vary per device
            init_to_eval: Mapping from init param index to eval param index

        Returns:
            Tuple of (init_fn, metadata)

        The function has signature:
            init_fn_split(shared_params: Array[N_shared], device_params: Array[N_varying])
                -> (cache: Array[N_cache], collapse_decisions: Array[N_collapse])

        Should be vmapped with in_axes=(None, 0) so that:
        - shared_params broadcasts (not sliced)
        - device_params is mapped over axis 0
        """
        from .codegen.function_builder import InitFunctionBuilder

        assert self.init_mir is not None, "init_mir released, call before release_mir_data()"

        t0 = time.perf_counter()
        logger.info("    translate_init_array_split: generating code...")

        # Build the init function
        builder = InitFunctionBuilder(
            self.init_mir, self.cache_mapping, self.collapse_decision_outputs
        )
        fn_name, code_lines = builder.build_split(shared_indices, varying_indices, init_to_eval)

        t1 = time.perf_counter()
        logger.info(
            f"    translate_init_array_split: code generated ({len(code_lines)} lines) in {t1 - t0:.1f}s"
        )

        code = "\n".join(code_lines)
        logger.info(f"    translate_init_array_split: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate_init_array_split: exec()...")
        init_fn, code_hash = exec_with_cache(code, fn_name, return_hash=True)
        t2 = time.perf_counter()
        logger.info(f"    translate_init_array_split: exec() done in {t2 - t1:.1f}s")

        # Build metadata
        param_defaults = {}
        if hasattr(self.module, "get_param_defaults"):
            param_defaults = dict(self.module.get_param_defaults())

        metadata = {
            "param_names": list(self.module.init_param_names),
            "param_kinds": list(self.module.init_param_kinds),
            "cache_size": len(self.cache_mapping),
            "cache_mapping": self.cache_mapping,
            "param_defaults": param_defaults,
            "collapsible_pairs": self.collapsible_pairs,
            "collapse_decision_outputs": self.collapse_decision_outputs,
            "shared_indices": shared_indices,
            "varying_indices": varying_indices,
            "code_hash": code_hash,
        }

        return init_fn, metadata

    def translate_eval_array_with_cache_split(
        self,
        shared_indices: List[int],
        varying_indices: List[int],
        shared_cache_indices: Optional[List[int]] = None,
        varying_cache_indices: Optional[List[int]] = None,
        use_limit_functions: bool = False,
        limit_param_map: Optional[Dict[int, Tuple[str, str]]] = None,
    ) -> Tuple[Callable, Dict]:
        """Generate a vmappable eval function with split params and cache (internal API).

        For user code, prefer translate_eval() which handles param validation.

        Args:
            shared_indices: Original param indices that are constant across all devices
            varying_indices: Original param indices that vary per device (including voltages)
            shared_cache_indices: Cache column indices that are constant across devices (optional)
            varying_cache_indices: Cache column indices that vary per device (optional)
            use_limit_functions: If True, generate calls to limit_funcs['pnjlim'] etc.
                                When enabled, the generated function has an additional
                                'limit_funcs' parameter that should be a dict like:
                                {'pnjlim': pnjlim_fn, 'fetlim': fetlim_fn}
            limit_param_map: Dict mapping original param indices to (kind, name) tuples
                            for limit-related params (prev_state, enable_lim, new_state,
                            enable_integration). Excluded from shared/device params.

        Returns:
            Tuple of (eval_fn, metadata)

        Function signature (if cache is split):
            eval_fn(shared_params, device_params, shared_cache, device_cache, simparams[, limit_funcs])
                -> (res_resist, res_react, jac_resist, jac_react,
                    lim_rhs_resist, lim_rhs_react,
                    small_signal_resist, small_signal_react)

        Or (if cache is not split):
            eval_fn(shared_params, device_params, cache, simparams[, limit_funcs])
                -> (res_resist, res_react, jac_resist, jac_react,
                    lim_rhs_resist, lim_rhs_react,
                    small_signal_resist, small_signal_react)

        Should be vmapped with in_axes=(None, 0, None, 0) for split cache
        or in_axes=(None, 0, 0) for unsplit cache.
        """
        from .codegen.function_builder import EvalFunctionBuilder

        assert self.eval_mir is not None, "eval_mir released, call before release_mir_data()"
        assert self.dae_data is not None, "dae_data released, call before release_mir_data()"

        t0 = time.perf_counter()
        logger.info(
            f"    translate_eval_array_with_cache_split: generating code (limit_funcs={use_limit_functions})..."
        )

        # Build the eval function
        eval_param_names = list(self.module.param_names)
        builder = EvalFunctionBuilder(
            self.eval_mir,
            self.dae_data,
            self.cache_mapping,
            self.param_idx_to_val,
            eval_param_names=eval_param_names,
        )
        fn_name, code_lines = builder.build_with_cache_split(
            shared_indices,
            varying_indices,
            shared_cache_indices,
            varying_cache_indices,
            use_limit_functions=use_limit_functions,
            limit_param_map=limit_param_map,
        )

        t1 = time.perf_counter()
        logger.info(
            f"    translate_eval_array_with_cache_split: code generated ({len(code_lines)} lines) in {t1 - t0:.1f}s"
        )

        code = "\n".join(code_lines)
        logger.info(f"    translate_eval_array_with_cache_split: code size = {len(code)} chars")

        # Dump generated code when OPENVAF_JAX_DUMP_CODE=1 is set
        if os.environ.get("OPENVAF_JAX_DUMP_CODE"):
            import hashlib

            dump_dir = os.path.expanduser("~/.cache/vajax/openvaf_codegen_dump")
            os.makedirs(dump_dir, exist_ok=True)
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:8]
            dump_path = os.path.join(dump_dir, f"{fn_name}_{code_hash}_{len(code_lines)}lines.py")
            with open(dump_path, "w") as df:
                df.write(f"# limit_param_map={limit_param_map}\n")
                df.write(f"# use_limit_functions={use_limit_functions}\n")
                df.write(f"# shared_indices={shared_indices}\n")
                df.write(f"# varying_indices={varying_indices}\n")
                df.write(code)
            logger.info(f"    Generated code dumped to {dump_path}")

        # Compile with caching
        logger.info("    translate_eval_array_with_cache_split: exec()...")
        eval_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_eval_array_with_cache_split: exec() done in {t2 - t1:.1f}s")

        # Build metadata using v2 API for clean node names
        node_names = [res["node_name"] for res in self.dae_data["residuals"]]
        node_indices = [res["node_idx"] for res in self.dae_data["residuals"]]
        jacobian_keys = [
            (entry["row_node_name"], entry["col_node_name"]) for entry in self.dae_data["jacobian"]
        ]
        jacobian_indices = [
            (entry["row_node_idx"], entry["col_node_idx"]) for entry in self.dae_data["jacobian"]
        ]
        cache_to_param = [m["eval_param"] for m in self.cache_mapping]

        # Get simparam metadata from builder
        simparam_meta = builder.simparam_metadata

        metadata = {
            "node_names": node_names,
            "node_indices": node_indices,
            "jacobian_keys": jacobian_keys,
            "jacobian_indices": jacobian_indices,
            "terminals": self.dae_data["terminals"],
            "internal_nodes": self.dae_data["internal_nodes"],
            "num_terminals": self.dae_data["num_terminals"],
            "num_internal": self.dae_data["num_internal"],
            "cache_to_param_mapping": cache_to_param,
            "uses_simparam_gmin": self.uses_simparam_gmin,
            "uses_analysis": self.uses_analysis,
            "analysis_type_map": self.analysis_type_map,
            "shared_indices": shared_indices,
            "varying_indices": varying_indices,
            "shared_cache_indices": shared_cache_indices,
            "varying_cache_indices": varying_cache_indices,
            "use_limit_functions": use_limit_functions,
            "limit_metadata": builder.limit_metadata if use_limit_functions else None,
            # Simparam metadata for building the simparams array
            "simparams_used": simparam_meta.get("simparams_used", ["$analysis_type"]),
            "simparam_indices": simparam_meta.get("simparam_indices", {"$analysis_type": 0}),
            "simparam_count": simparam_meta.get("simparam_count", 1),
        }

        return eval_fn, metadata


__all__ = [
    "OpenVAFToJAX",
    "MIRFunction",
    "MIRInstruction",
    "Block",
    "PhiOperand",
    "CFGAnalyzer",
    "LoopInfo",
    "SSAAnalyzer",
    "PHIResolution",
    "parse_mir_function",
    "exec_with_cache",
    "get_vmapped_jit",
    "clear_cache",
    "cache_stats",
]
