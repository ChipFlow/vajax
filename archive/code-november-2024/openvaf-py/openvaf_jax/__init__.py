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
    init_fn, init_meta = translator.translate_init_array_split(
        shared_indices=[...],
        varying_indices=[...],
        init_to_eval=[...]
    )

    # Generate eval function (computes residuals/Jacobian)
    eval_fn, eval_meta = translator.translate_eval_array_with_cache_split(
        shared_indices=[...],
        varying_indices=[...]
    )
"""

import logging
import time
from typing import Dict, List, Tuple, Callable, Any, Optional

logger = logging.getLogger("jax_spice.openvaf")

# Re-export key types
from .mir import (
    MIRFunction, MIRInstruction, Block, PhiOperand,
    CFGAnalyzer, LoopInfo,
    SSAAnalyzer, PHIResolution,
    parse_mir_function,
)
from .cache import exec_with_cache, get_vmapped_jit, clear_cache, cache_stats


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

        # Parse into structured MIR
        self.eval_mir = parse_mir_function('eval', self.mir_data)
        self.init_mir = parse_mir_function('init', self.init_mir_data)

        # Extract metadata
        self.params = list(self.mir_data['params'])
        self.constants = dict(self.mir_data['constants'])
        self.bool_constants = dict(self.mir_data.get('bool_constants', {}))
        self.int_constants = dict(self.mir_data.get('int_constants', {}))

        self.init_params = list(self.init_mir_data['params'])
        self.cache_mapping = list(self.init_mir_data['cache_mapping'])

        # String constants
        self.str_constants = dict(module.get_str_constants())

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

    def translate(self) -> Callable:
        """Generate a JAX function from MIR (legacy interface).

        This method provides backward compatibility with the old interface.
        For new code, prefer translate_init_array_split() and
        translate_eval_array_with_cache_split() for better GPU performance.

        Returns a function with signature:
            f(inputs: List[float]) -> (residuals: Dict, jacobian: Dict)

        The inputs should be ordered according to module.param_names
        """
        # Use old implementation for backward compatibility
        import sys
        import os
        old_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, old_path)
        try:
            import openvaf_jax_old
            old_translator = openvaf_jax_old.OpenVAFToJAX(self.module)
            return old_translator.translate()
        finally:
            sys.path.remove(old_path)

    def translate_array(self) -> Tuple[Callable, Dict]:
        """Generate a JAX function that returns arrays (vmap-compatible).

        This method provides backward compatibility with CircuitEngine.
        For new code, prefer translate_init_array_split() and
        translate_eval_array_with_cache_split() for better GPU performance.

        Returns a function with signature:
            f(inputs: Array[N]) -> (residuals: Array[num_nodes], jacobian: Array[num_jac_entries])

        Also returns metadata dict with:
            - 'node_names': list of node names in residual array order
            - 'jacobian_keys': list of (row, col) tuples in jacobian array order
        """
        # Use old implementation for backward compatibility
        import sys
        import os
        old_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, old_path)
        try:
            import openvaf_jax_old
            old_translator = openvaf_jax_old.OpenVAFToJAX(self.module)
            return old_translator.translate_array()
        finally:
            sys.path.remove(old_path)

    def translate_init_array(self) -> Tuple[Callable, Dict]:
        """Generate a vmappable init function.

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
        """Generate a vmappable init function that takes split shared/device params.

        This is an optimized version that reduces memory by separating constant
        parameters (shared across all devices) from varying parameters (different
        per device).

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
        """Generate a vmappable eval function that takes split shared/device params and cache.

        Args:
            shared_indices: Original param indices that are constant across all devices
            varying_indices: Original param indices that vary per device (including voltages)
            shared_cache_indices: Cache column indices that are constant across devices (optional)
            varying_cache_indices: Cache column indices that vary per device (optional)

        Returns:
            Tuple of (eval_fn, metadata)

        Function signature (if cache is split):
            eval_fn(shared_params, device_params, shared_cache, device_cache)
                -> (res_resist, res_react, jac_resist, jac_react)

        Or (if cache is not split):
            eval_fn(shared_params, device_params, cache)
                -> (res_resist, res_react, jac_resist, jac_react)

        Should be vmapped with in_axes=(None, 0, None, 0) for split cache
        or in_axes=(None, 0, 0) for unsplit cache.
        """
        from .codegen.function_builder import EvalFunctionBuilder

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
