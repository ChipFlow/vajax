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
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("jax_spice.openvaf")

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
        self.eval_mir = parse_mir_function('eval', self.mir_data, self.str_constants)
        self.init_mir = parse_mir_function('init', self.init_mir_data, self.str_constants)

        # Extract metadata
        self.params = list(self.mir_data['params'])
        self.constants = dict(self.mir_data['constants'])
        self.bool_constants = dict(self.mir_data.get('bool_constants', {}))
        self.int_constants = dict(self.mir_data.get('int_constants', {}))

        self.init_params = list(self.init_mir_data['params'])
        self.cache_mapping = list(self.init_mir_data['cache_mapping'])

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
            'dc': 0, 'static': 0,
            'ac': 1,
            'tran': 2, 'transient': 2,
            'noise': 3,
            'nodeset': 4,
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
            'node_names': [res['node_name'] for res in self.dae_data['residuals']],
            'jacobian_keys': [
                (entry['row_node_name'], entry['col_node_name'])
                for entry in self.dae_data['jacobian']
            ],
            'terminals': self.dae_data['terminals'],
            'internal_nodes': self.dae_data['internal_nodes'],
            'num_terminals': self.dae_data['num_terminals'],
            'num_internal': self.dae_data['num_internal'],
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
        for p in desc['params']:
            name = p['name']

            # Skip internal params unless requested
            if not include_internal:
                # Skip system params and hidden state
                if name.startswith('$') or name == 'mfactor':
                    continue

            # Type is encoded in flags: 0=real, 1=int, 2=str
            flags = p.get('flags', 0)
            type_code = flags & 3
            type_str = {0: 'real', 1: 'int', 2: 'str'}.get(type_code, 'real')

            # Get default - lookup case-insensitive
            default = defaults.get(name)
            if default is None:
                default = defaults.get(name.lower())

            params.append({
                'name': name,
                'type': type_str,
                'default': default,
                'units': p.get('units', ''),
                'description': p.get('description', ''),
                'aliases': p.get('aliases', []),
                'is_instance': p.get('is_instance', False),
                'is_model_param': p.get('is_model_param', True),
            })

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
        model_name = self.module.name if hasattr(self.module, 'name') else 'model'

        print(f"\n=== Parameters for {model_name} ({len(params)} params) ===")
        print(f"{'Name':<20} {'Type':<6} {'Default':<12} {'Units':<8} Description")
        print("-" * 80)

        for p in params:
            default_str = "None"
            if p['default'] is not None:
                if isinstance(p['default'], float):
                    if abs(p['default']) < 1e-3 or abs(p['default']) >= 1e6:
                        default_str = f"{p['default']:.4g}"
                    else:
                        default_str = f"{p['default']}"
                else:
                    default_str = str(p['default'])

            units = p['units'] if p['units'] else '-'
            desc = p['description'][:40] if p['description'] else ''

            print(f"{p['name']:<20} {p['type']:<6} {default_str:<12} {units:<8} {desc}")

        # Show aliases if any
        aliases_found = [(p['name'], p['aliases']) for p in params if p['aliases']]
        if aliases_found:
            print(f"\nAliases:")
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
        for p in desc['params']:
            name = p['name']
            # Type is encoded in flags: 0=real, 1=int, 2=str
            flags = p.get('flags', 0)
            type_code = flags & 3
            type_str = {0: 'real', 1: 'int', 2: 'str'}.get(type_code, 'real')

            # Get default - lookup case-insensitive
            default = defaults.get(name)
            if default is None:
                default = defaults.get(name.lower())

            param_info[name] = {
                'name': name,
                'type': type_str,
                'default': default,
                'units': p.get('units', ''),
                'description': p.get('description', ''),
                'is_instance': p.get('is_instance', False),
                'is_model_param': p.get('is_model_param', True),
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
            if kind == 'voltage':
                # Voltage params are set at runtime, use 0.0 placeholder
                value = 0.0
            elif kind == 'temperature' or name == '$temperature':
                # System temperature
                value = temperature
            elif kind == 'sysfun' and name == 'mfactor':
                value = mfactor
            elif kind in ('hidden_state', 'current'):
                # Placeholders - filled by init or eval
                value = 0.0
            elif kind == 'implicit_unknown':
                # Internal node voltage for implicit equations
                # These are set at runtime from the voltage array, same as 'voltage'
                value = 0.0
            elif kind == 'param_given':
                # Check if the corresponding param was explicitly set
                # param_given names are like "VTO_given" -> check for "VTO"
                base_name = name.replace('_given', '').lower()
                value = 1.0 if base_name in params_lower else 0.0
            elif kind == 'port_connected':
                # Assume all ports are connected (LRM 9.19)
                # TODO: Add support for optional ports by tracking connections
                warnings.append(f"{context}: $port_connected({name}) assumed true - optional ports not supported")
                value = 1.0
            elif kind == 'abstime':
                # Absolute simulation time (LRM 9.7)
                # For DC analysis, abstime=0.0. For transient, caller must update this.
                warnings.append(f"{context}: $abstime used - defaults to 0.0 (DC). For transient, update input array.")
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
                    if info and info['default'] is not None:
                        value = float(info['default'])
                    else:
                        # Special handling for TEMP/TNOM
                        if name_lower in ('temp', 'tnom'):
                            value = SENTINEL
                        elif kind == 'unknown':
                            warnings.append(f"{context}: '{name}' (kind={kind}) has no value, using 0.0")
                            value = 0.0
                        elif kind == 'sysfun':
                            # Other system functions default to appropriate values
                            if 'scale' in name_lower:
                                value = 1.0
                            elif 'shrink' in name_lower:
                                value = 0.0
                            else:
                                warnings.append(f"{context}: sysfun '{name}' has no handler, using 0.0")
                                value = 0.0
                        else:
                            warnings.append(f"{context}: param '{name}' (kind={kind}) has no value or default, using 0.0")
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
            init_param_names, init_param_kinds,
            params, temperature, mfactor, param_info,
            context="init"
        )

        # Build detailed param info list for metadata
        detailed_param_info = []
        for i, (name, kind) in enumerate(zip(init_param_names, init_param_kinds)):
            info = param_info.get(name) or param_info.get(name.upper()) or {}
            detailed_param_info.append({
                'name': name,
                'kind': kind,
                'type': info.get('type', 'real'),
                'default': info.get('default'),
                'value': init_inputs[i],
                'units': info.get('units', ''),
                'description': info.get('description', ''),
            })

        # Debug output
        if debug:
            print(f"\n=== Init Parameters ({len(init_param_names)}) ===")
            print(f"{'Name':<20} {'Kind':<15} {'Type':<6} {'Default':<12} {'Value':<12} {'Units':<8} Description")
            print("-" * 100)
            for p in detailed_param_info:
                default_str = f"{p['default']:.4g}" if p['default'] is not None else "None"
                value_str = f"{p['value']:.4g}" if isinstance(p['value'], float) else str(p['value'])
                print(f"{p['name']:<20} {p['kind']:<15} {p['type']:<6} {default_str:<12} {value_str:<12} {p['units']:<8} {p['description'][:30]}")
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
            self.init_mir,
            self.cache_mapping,
            self.collapse_decision_outputs
        )
        fn_name, code_lines = builder.build_simple(all_indices)

        # Collect codegen warnings (unknown $simparam, $discontinuity, etc.)
        warnings.extend(builder.codegen_warnings)

        t1 = time.perf_counter()
        logger.info(f"    translate_init: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)

        # Compile with caching
        logger.info("    translate_init: exec()...")
        init_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_init: exec() done in {t2-t1:.1f}s")

        metadata = {
            'param_names': init_param_names,
            'param_kinds': init_param_kinds,
            'init_inputs': init_inputs,
            'param_info': detailed_param_info,
            'cache_size': len(self.cache_mapping),
            'cache_mapping': self.cache_mapping,
            'collapsible_pairs': self.collapsible_pairs,
            'collapse_decision_outputs': self.collapse_decision_outputs,
            'temperature': temperature,
            'mfactor': mfactor,
            'warnings': warnings,
        }

        return init_fn, metadata

    def translate_eval(
        self,
        params: Dict[str, float] = None,
        temperature: float = 300.0,
        mfactor: float = 1.0,
        debug: bool = False,
        cache_split: Optional[Tuple[List[int], List[int]]] = None,
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

        Returns:
            Tuple of (eval_fn, metadata)

            eval_fn signature:
                eval_fn(shared_params, varying_params, cache, simparams)
                    -> (res_resist, res_react, jac_resist, jac_react,
                        lim_rhs_resist, lim_rhs_react, ss_resist, ss_react)

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
        voltage_kinds = ('voltage', 'implicit_unknown')
        voltage_indices = [i for i, k in enumerate(eval_param_kinds) if k in voltage_kinds]
        shared_indices = [i for i, k in enumerate(eval_param_kinds) if k not in voltage_kinds]

        # Build validated inputs for shared params only
        # (voltage params are provided at runtime)
        shared_param_names = [eval_param_names[i] for i in shared_indices]
        shared_param_kinds = [eval_param_kinds[i] for i in shared_indices]

        shared_inputs, warnings = self._build_param_inputs(
            shared_param_names, shared_param_kinds,
            params, temperature, mfactor, param_info,
            context="eval"
        )

        # Build detailed param info list
        detailed_param_info = []
        for i, (name, kind) in enumerate(zip(eval_param_names, eval_param_kinds)):
            info = param_info.get(name) or param_info.get(name.upper()) or {}
            # Value depends on whether this is shared or voltage
            if kind in voltage_kinds:
                value = None  # Set at runtime from voltage array
            else:
                shared_idx = shared_indices.index(i)
                value = shared_inputs[shared_idx]
            detailed_param_info.append({
                'name': name,
                'kind': kind,
                'type': info.get('type', 'real'),
                'default': info.get('default'),
                'value': value,
                'units': info.get('units', ''),
                'description': info.get('description', ''),
            })

        # Debug output
        if debug:
            print(f"\n=== Eval Parameters ({len(eval_param_names)}) ===")
            print(f"  Shared: {len(shared_indices)}, Voltage: {len(voltage_indices)}")
            print(f"\n{'Name':<25} {'Kind':<15} {'Type':<6} {'Default':<12} {'Value':<12}")
            print("-" * 80)
            for p in detailed_param_info:
                default_str = f"{p['default']:.4g}" if p['default'] is not None else "None"
                if p['value'] is None:
                    value_str = "(runtime)"
                elif isinstance(p['value'], float):
                    value_str = f"{p['value']:.4g}"
                else:
                    value_str = str(p['value'])
                print(f"{p['name']:<25} {p['kind']:<15} {p['type']:<6} {default_str:<12} {value_str:<12}")
            if warnings:
                print(f"\nWarnings: {len(warnings)}")
                for w in warnings:
                    print(f"  - {w}")

        # Parse cache_split
        shared_cache_indices = None
        varying_cache_indices = None
        if cache_split is not None:
            shared_cache_indices, varying_cache_indices = cache_split

        # Generate code
        t0 = time.perf_counter()
        use_cache_split = shared_cache_indices is not None
        logger.info(f"    translate_eval: generating code (cache_split={use_cache_split})...")

        builder = EvalFunctionBuilder(
            self.eval_mir,
            self.dae_data,
            self.cache_mapping,
            self.param_idx_to_val
        )
        fn_name, code_lines = builder.build_with_cache_split(
            shared_indices, voltage_indices,
            shared_cache_indices, varying_cache_indices
        )

        # Collect codegen warnings (unknown $simparam, $discontinuity, etc.)
        warnings.extend(builder.codegen_warnings)

        t1 = time.perf_counter()
        logger.info(f"    translate_eval: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)

        # Compile with caching
        logger.info("    translate_eval: exec()...")
        eval_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_eval: exec() done in {t2-t1:.1f}s")

        # Build metadata
        node_names = [res['node_name'] for res in self.dae_data['residuals']]
        node_indices = [res['node_idx'] for res in self.dae_data['residuals']]
        jacobian_keys = [
            (entry['row_node_name'], entry['col_node_name'])
            for entry in self.dae_data['jacobian']
        ]
        jacobian_indices = [
            (entry['row_node_idx'], entry['col_node_idx'])
            for entry in self.dae_data['jacobian']
        ]

        # simparams layout documentation
        simparams_layout = {
            0: {'name': 'analysis_type', 'description': '0=DC, 1=AC, 2=transient, 3=noise'},
            1: {'name': 'gmin', 'description': 'minimum conductance for convergence'},
        }

        metadata = {
            'param_names': eval_param_names,
            'param_kinds': eval_param_kinds,
            'shared_inputs': shared_inputs,
            'shared_indices': shared_indices,
            'voltage_indices': voltage_indices,
            'param_info': detailed_param_info,
            'node_names': node_names,
            'node_indices': node_indices,
            'jacobian_keys': jacobian_keys,
            'jacobian_indices': jacobian_indices,
            'terminals': self.dae_data['terminals'],
            'internal_nodes': self.dae_data['internal_nodes'],
            'num_terminals': self.dae_data['num_terminals'],
            'num_internal': self.dae_data['num_internal'],
            'temperature': temperature,
            'mfactor': mfactor,
            'use_cache_split': use_cache_split,
            'simparams_layout': simparams_layout,
            'warnings': warnings,
        }

        return eval_fn, metadata

    def translate(self) -> Callable:
        """Generate a JAX function from MIR (legacy interface).

        DEPRECATED: Use translate_init() and translate_eval() instead.

        Returns a function with signature:
            f(inputs: List[float]) -> (residuals: Dict, jacobian: Dict)

        Where residuals is a dict mapping node names to dicts with 'resist' and 'react' keys,
        and jacobian is a dict mapping (row, col) tuples to dicts with 'resist' and 'react' keys.

        The inputs should be ordered according to module.param_names
        """
        # Get array-based function and metadata
        eval_fn, metadata = self.translate_array()
        node_names = metadata['node_names']
        jacobian_keys = metadata['jacobian_keys']

        def dict_wrapper(inputs):
            res_resist, res_react, jac_resist, jac_react = eval_fn(inputs)[:4]
            # Convert arrays to nested dicts with resist/react keys
            residuals = {
                name: {'resist': res_resist[i], 'react': res_react[i]}
                for i, name in enumerate(node_names)
            }
            jacobian = {
                key: {'resist': jac_resist[i], 'react': jac_react[i]}
                for i, key in enumerate(jacobian_keys)
            }
            return residuals, jacobian

        return dict_wrapper

    def translate_array(self) -> Tuple[Callable, Dict]:
        """Generate a JAX function that returns arrays (vmap-compatible).

        DEPRECATED: Use translate_init() and translate_eval() instead.

        Returns a function with signature:
            f(inputs: Array[N]) -> (residuals: Array[num_nodes], jacobian: Array[num_jac_entries])

        The inputs array should be ordered according to module.param_names.

        Also returns metadata dict with:
            - 'node_names': list of node names in residual array order
            - 'jacobian_keys': list of (row, col) tuples in jacobian array order
        """
        import jax.numpy as jnp

        # Generate init and eval functions
        init_fn, init_metadata = self.translate_init_array()

        # Use all indices as varying (no shared/device split for legacy interface)
        n_eval_params = len(self.params)
        all_indices = list(range(n_eval_params))

        eval_fn, eval_metadata = self.translate_eval_array_with_cache_split(
            shared_indices=[],
            varying_indices=all_indices,
        )

        # Build mapping from user param index -> init param index
        # User params are module.param_names, init params are module.init_param_names
        user_param_names = list(self.module.param_names)
        user_param_value_ids = list(self.module.param_value_indices)
        init_param_names = list(self.module.init_param_names)
        init_param_value_ids = list(self.module.init_param_value_indices)

        # Map user param index -> init param index (by matching value IDs or names)
        user_to_init_mapping = []  # list of (user_idx, init_idx) for each init param
        for init_idx, (init_name, init_vid) in enumerate(zip(init_param_names, init_param_value_ids)):
            # Try to find matching user param by value ID first
            if init_vid in user_param_value_ids:
                user_idx = user_param_value_ids.index(init_vid)
                user_to_init_mapping.append((user_idx, init_idx))
            # Otherwise try by name (for param_given which has different value ID)
            elif init_name in user_param_names:
                user_idx = user_param_names.index(init_name)
                user_to_init_mapping.append((user_idx, init_idx))
            # Else default to 0 (will be handled below)

        # Build mapping from user param index -> eval MIR param index
        # all_func_params gives (mir_idx, value_id) pairs
        all_func_params = list(self.module.get_all_func_params())
        value_id_to_mir_idx = {p[1]: p[0] for p in all_func_params}
        user_to_eval_mapping = []  # list of (user_idx, mir_idx)
        for user_idx, vid in enumerate(user_param_value_ids):
            mir_idx = value_id_to_mir_idx.get(vid)
            if mir_idx is not None:
                user_to_eval_mapping.append((user_idx, mir_idx))

        n_init_params = len(init_param_names)
        n_user_params = len(user_param_names)

        # Create combined function with proper mapping
        def combined_fn(inputs):
            inputs_arr = jnp.asarray(inputs)

            # Build init inputs array
            init_inputs = jnp.zeros(n_init_params)
            for user_idx, init_idx in user_to_init_mapping:
                init_inputs = init_inputs.at[init_idx].set(inputs_arr[user_idx])

            # Run init to get cache
            cache, _collapse = init_fn(init_inputs)

            # Build eval inputs array (device_params)
            device_params = jnp.zeros(n_eval_params)
            for user_idx, mir_idx in user_to_eval_mapping:
                device_params = device_params.at[mir_idx].set(inputs_arr[user_idx])

            # Run eval with empty shared params, device_params, and cache
            # Returns: (res_resist, res_react, jac_resist, jac_react,
            #           lim_rhs_resist, lim_rhs_react, small_signal_resist, small_signal_react)
            result = eval_fn([], device_params, cache)
            res_resist, res_react, jac_resist, jac_react = result[:4]

            # Return separate resist and react arrays
            return res_resist, res_react, jac_resist, jac_react

        # Build combined metadata
        metadata = {
            'node_names': eval_metadata['node_names'],
            'jacobian_keys': eval_metadata['jacobian_keys'],
            'init_param_names': init_metadata['param_names'],
            'eval_param_names': user_param_names,
        }

        return combined_fn, metadata

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
            self.init_mir,
            self.cache_mapping,
            self.collapse_decision_outputs
        )
        fn_name, code_lines = builder.build_simple(all_indices)

        t1 = time.perf_counter()
        logger.info(f"    translate_init_array: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)
        logger.info(f"    translate_init_array: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate_init_array: exec()...")
        init_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_init_array: exec() done in {t2-t1:.1f}s")

        # Build metadata
        param_defaults = {}
        if hasattr(self.module, 'get_param_defaults'):
            param_defaults = dict(self.module.get_param_defaults())

        metadata = {
            'param_names': list(self.module.init_param_names),
            'param_kinds': list(self.module.init_param_kinds),
            'cache_size': len(self.cache_mapping),
            'cache_mapping': self.cache_mapping,
            'param_defaults': param_defaults,
            'collapsible_pairs': self.collapsible_pairs,
            'collapse_decision_outputs': self.collapse_decision_outputs,
        }

        return init_fn, metadata

    def translate_init_array_split(
        self,
        shared_indices: List[int],
        varying_indices: List[int],
        init_to_eval: List[int]
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
            self.init_mir,
            self.cache_mapping,
            self.collapse_decision_outputs
        )
        fn_name, code_lines = builder.build_split(
            shared_indices, varying_indices, init_to_eval
        )

        t1 = time.perf_counter()
        logger.info(f"    translate_init_array_split: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)
        logger.info(f"    translate_init_array_split: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate_init_array_split: exec()...")
        init_fn, code_hash = exec_with_cache(code, fn_name, return_hash=True)
        t2 = time.perf_counter()
        logger.info(f"    translate_init_array_split: exec() done in {t2-t1:.1f}s")

        # Build metadata
        param_defaults = {}
        if hasattr(self.module, 'get_param_defaults'):
            param_defaults = dict(self.module.get_param_defaults())

        metadata = {
            'param_names': list(self.module.init_param_names),
            'param_kinds': list(self.module.init_param_kinds),
            'cache_size': len(self.cache_mapping),
            'cache_mapping': self.cache_mapping,
            'param_defaults': param_defaults,
            'collapsible_pairs': self.collapsible_pairs,
            'collapse_decision_outputs': self.collapse_decision_outputs,
            'shared_indices': shared_indices,
            'varying_indices': varying_indices,
            'code_hash': code_hash,
        }

        return init_fn, metadata

    def translate_eval_array_with_cache_split(
        self,
        shared_indices: List[int],
        varying_indices: List[int],
        shared_cache_indices: Optional[List[int]] = None,
        varying_cache_indices: Optional[List[int]] = None
    ) -> Tuple[Callable, Dict]:
        """Generate a vmappable eval function with split params and cache (internal API).

        For user code, prefer translate_eval() which handles param validation.

        Args:
            shared_indices: Original param indices that are constant across all devices
            varying_indices: Original param indices that vary per device (including voltages)
            shared_cache_indices: Cache column indices that are constant across devices (optional)
            varying_cache_indices: Cache column indices that vary per device (optional)

        Returns:
            Tuple of (eval_fn, metadata)

        Function signature (if cache is split):
            eval_fn(shared_params, device_params, shared_cache, device_cache)
                -> (res_resist, res_react, jac_resist, jac_react,
                    lim_rhs_resist, lim_rhs_react,
                    small_signal_resist, small_signal_react)

        Or (if cache is not split):
            eval_fn(shared_params, device_params, cache)
                -> (res_resist, res_react, jac_resist, jac_react,
                    lim_rhs_resist, lim_rhs_react,
                    small_signal_resist, small_signal_react)

        Should be vmapped with in_axes=(None, 0, None, 0) for split cache
        or in_axes=(None, 0, 0) for unsplit cache.
        """
        from .codegen.function_builder import EvalFunctionBuilder

        assert self.eval_mir is not None, "eval_mir released, call before release_mir_data()"
        assert self.dae_data is not None, "dae_data released, call before release_mir_data()"

        use_cache_split = shared_cache_indices is not None and varying_cache_indices is not None

        t0 = time.perf_counter()
        logger.info(f"    translate_eval_array_with_cache_split: generating code (cache_split={use_cache_split})...")

        # Build the eval function
        builder = EvalFunctionBuilder(
            self.eval_mir,
            self.dae_data,
            self.cache_mapping,
            self.param_idx_to_val
        )
        fn_name, code_lines = builder.build_with_cache_split(
            shared_indices, varying_indices,
            shared_cache_indices, varying_cache_indices
        )

        t1 = time.perf_counter()
        logger.info(f"    translate_eval_array_with_cache_split: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)
        logger.info(f"    translate_eval_array_with_cache_split: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate_eval_array_with_cache_split: exec()...")
        eval_fn = exec_with_cache(code, fn_name)
        t2 = time.perf_counter()
        logger.info(f"    translate_eval_array_with_cache_split: exec() done in {t2-t1:.1f}s")

        # Build metadata using v2 API for clean node names
        node_names = [res['node_name'] for res in self.dae_data['residuals']]
        node_indices = [res['node_idx'] for res in self.dae_data['residuals']]
        jacobian_keys = [
            (entry['row_node_name'], entry['col_node_name'])
            for entry in self.dae_data['jacobian']
        ]
        jacobian_indices = [
            (entry['row_node_idx'], entry['col_node_idx'])
            for entry in self.dae_data['jacobian']
        ]
        cache_to_param = [m['eval_param'] for m in self.cache_mapping]

        metadata = {
            'node_names': node_names,
            'node_indices': node_indices,
            'jacobian_keys': jacobian_keys,
            'jacobian_indices': jacobian_indices,
            'terminals': self.dae_data['terminals'],
            'internal_nodes': self.dae_data['internal_nodes'],
            'num_terminals': self.dae_data['num_terminals'],
            'num_internal': self.dae_data['num_internal'],
            'cache_to_param_mapping': cache_to_param,
            'uses_simparam_gmin': self.uses_simparam_gmin,
            'uses_analysis': self.uses_analysis,
            'analysis_type_map': self.analysis_type_map,
            'shared_indices': shared_indices,
            'varying_indices': varying_indices,
            'use_cache_split': use_cache_split,
            'shared_cache_indices': shared_cache_indices if use_cache_split else None,
            'varying_cache_indices': varying_cache_indices if use_cache_split else None,
        }

        return eval_fn, metadata

__all__ = [
    'OpenVAFToJAX',
    'MIRFunction',
    'MIRInstruction',
    'Block',
    'PhiOperand',
    'CFGAnalyzer',
    'LoopInfo',
    'SSAAnalyzer',
    'PHIResolution',
    'parse_mir_function',
    'exec_with_cache',
    'get_vmapped_jit',
    'clear_cache',
    'cache_stats',
]
