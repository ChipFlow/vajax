"""OpenVAF to JAX translator

Compiles Verilog-A models to JAX functions using openvaf-py.
"""

# Force CPU backend on Apple Silicon to avoid Metal backend compatibility issues
# This must be done before any JAX imports
import os
import platform
import logging

if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Apple Silicon - force CPU backend to avoid Metal/GPU issues
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ast
from typing import Dict, List, Callable, Any, Tuple, Set, Optional, Union
from dataclasses import dataclass
import openvaf_py

# Import AST builder utilities
from openvaf_ast import (
    name as ast_name, const as ast_const, binop, unaryop, compare,
    call as ast_call, attr, subscript, list_expr, tuple_expr,
    jnp_call, jnp_where, jnp_float64, jnp_bool, jnp_inf, jnp_nan,
    lax_call, lax_while_loop, safe_divide, nested_where,
    assign, assign_tuple, function_def, return_stmt, import_stmt,
    import_from, expr_stmt, pass_stmt,
    ASTBuilder, ExpressionBuilder,
)
from openvaf_ast.statements import build_module, fix_and_compile

# Use jax_spice logger (inherits memory logging config when enabled)
logger = logging.getLogger("jax_spice.openvaf")

# Module-level cache for exec'd functions (keyed by code hash)
# This allows JAX to reuse JIT-compiled functions across translator instances
_exec_fn_cache: Dict[str, Callable] = {}

# Module-level cache for vmapped+jit'd functions (keyed by (code_hash, in_axes))
# This avoids repeated JIT compilation for the same function with same vmap axes
_vmapped_jit_cache: Dict[Tuple[str, Tuple], Callable] = {}


def _exec_with_cache(code: str, fn_name: str, return_hash: bool = False) -> Union[Callable, Tuple[Callable, str]]:
    """Execute code and cache the resulting function by code hash.

    This dramatically speeds up repeated compilations (e.g., re-parsing the same
    circuit) by reusing previously exec'd functions. JAX can then reuse its
    JIT-compiled versions.

    Args:
        code: Python source code to execute
        fn_name: Name of function to extract from executed code
        return_hash: If True, return (function, code_hash) tuple

    Returns:
        If return_hash=False: Just the function
        If return_hash=True: Tuple of (function, code_hash)
    """
    import hashlib
    import jax.numpy as jnp
    from jax import lax

    code_hash = hashlib.sha256(code.encode()).hexdigest()

    if code_hash in _exec_fn_cache:
        logger.debug(f"    {fn_name}: using cached function (hash={code_hash[:8]})")
        fn = _exec_fn_cache[code_hash]
        return (fn, code_hash) if return_hash else fn

    local_ns = {'jnp': jnp, 'lax': lax}
    exec(code, local_ns)
    fn = local_ns[fn_name]

    _exec_fn_cache[code_hash] = fn
    logger.debug(f"    {fn_name}: cached new function (hash={code_hash[:8]})")
    return (fn, code_hash) if return_hash else fn


def get_vmapped_jit(code_hash: str, fn: Callable, in_axes: Tuple) -> Callable:
    """Get a cached vmapped+jit'd version of a function.

    This caches the entire jax.jit(jax.vmap(fn, in_axes=in_axes)) result,
    avoiding repeated JIT compilation for the same function.
    """
    import jax

    cache_key = (code_hash, in_axes)

    if cache_key in _vmapped_jit_cache:
        logger.debug(f"    vmapped_jit: using cached (hash={code_hash[:8]}, in_axes={in_axes})")
        return _vmapped_jit_cache[cache_key]

    vmapped_jit_fn = jax.jit(jax.vmap(fn, in_axes=in_axes))
    _vmapped_jit_cache[cache_key] = vmapped_jit_fn
    logger.debug(f"    vmapped_jit: cached new (hash={code_hash[:8]}, in_axes={in_axes})")
    return vmapped_jit_fn


def clear_exec_cache():
    """Clear the exec function cache (useful for testing or memory management)."""
    global _exec_fn_cache, _vmapped_jit_cache
    _exec_fn_cache.clear()
    _vmapped_jit_cache.clear()


@dataclass
class CompiledDevice:
    """A compiled Verilog-A device with JAX evaluation function"""
    name: str
    module_name: str
    nodes: List[str]
    param_names: List[str]
    param_kinds: List[str]
    eval_fn: Callable
    num_residuals: int
    num_jacobian: int


def _jax_bool_repr(value: bool) -> str:
    """Return a JAX-compatible representation of a boolean.

    Using jnp.bool_() instead of Python True/False ensures the value
    can be traced by JAX JIT compilation.
    """
    return f"jnp.bool_({value})"


class OpenVAFToJAX:
    """Translates OpenVAF MIR to JAX functions"""

    # Opcode lookup tables for fast dispatch (Optimization 1)
    # Binary arithmetic ops: opcode -> operator string
    _BINARY_ARITH_OPS = {
        'fadd': '+', 'fsub': '-', 'fmul': '*',
        'iadd': '+', 'isub': '-', 'imul': '*',
        'irem': '%', 'idiv': '//',
        'iand': '&', 'ior': '|', 'ixor': '^',
    }

    # Comparison ops: opcode -> operator string
    _COMPARE_OPS = {
        'feq': '==', 'fne': '!=', 'flt': '<', 'fgt': '>', 'fle': '<=', 'fge': '>=',
        'ieq': '==', 'ine': '!=', 'ilt': '<', 'igt': '>', 'ile': '<=', 'ige': '>=',
        'beq': '==', 'bne': '!=',
    }

    # Unary jnp functions: opcode -> jnp method name (same name)
    # Note: 'sqrt' is handled specially with safe_sqrt to clamp negative inputs
    _UNARY_JNP_SAME = {'exp', 'floor', 'ceil', 'sin', 'cos', 'tan',
                       'sinh', 'cosh', 'tanh', 'hypot'}

    # Unary jnp functions: opcode -> jnp method name (different name)
    # Note: 'ln' is handled specially with safe_log to clamp non-positive inputs
    _UNARY_JNP_MAP = {
        'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
        'asinh': 'arcsinh', 'acosh': 'arccosh', 'atanh': 'arctanh',
    }

    # Binary jnp functions: opcode -> jnp method name
    _BINARY_JNP_MAP = {
        'pow': 'power',
        'atan2': 'arctan2',
    }

    def __init__(self, module):
        """Initialize with a compiled VaModule from openvaf_py

        Args:
            module: VaModule from openvaf_py.compile_va()
        """
        self.module = module
        self.mir_data = module.get_mir_instructions()
        self.dae_data = module.get_dae_system()  # v2 format with clean names
        self.init_mir_data = module.get_init_mir_instructions()

        # Build value tracking
        self.constants = dict(self.mir_data['constants'])
        self.bool_constants = dict(self.mir_data.get('bool_constants', {}))
        self.int_constants = dict(self.mir_data.get('int_constants', {}))
        self.params = list(self.mir_data['params'])

        # String constants (resolved from Spur keys to actual strings)
        # Maps operand name (e.g., "v123") to actual string (e.g., "gmin")
        self.str_constants = dict(module.get_str_constants())

        # Track if this model uses $simparam("gmin")
        self.uses_simparam_gmin = False

        # Track if this model uses analysis() function
        self.uses_analysis = False
        # Analysis types mapped to integers:
        # 0: dc/static, 1: ac, 2: tran, 3: noise
        self.analysis_type_map = {
            'dc': 0, 'static': 0,
            'ac': 1,
            'tran': 2, 'transient': 2,
            'noise': 3,
            'nodeset': 4,
        }

        # Init function data
        self.init_constants = dict(self.init_mir_data['constants'])
        self.init_bool_constants = dict(self.init_mir_data.get('bool_constants', {}))
        self.init_int_constants = dict(self.init_mir_data.get('int_constants', {}))
        self.init_params = list(self.init_mir_data['params'])
        self.cache_mapping = list(self.init_mir_data['cache_mapping'])

        # Node collapse support
        # collapse_decision_outputs: List of (eq_index, value_name) tuples
        # Each tuple maps an implicit equation to the init value that decides if it collapses
        self.collapse_decision_outputs = list(module.collapse_decision_outputs)
        self.collapsible_pairs = list(module.collapsible_pairs)

        # Cached maps for PHI condition lookup optimization
        # Built lazily on first use
        self._eval_succ_pair_map: Optional[Dict[frozenset, List[str]]] = None
        self._init_succ_pair_map: Optional[Dict[frozenset, List[str]]] = None

    def _get_eval_succ_pair_map(self) -> Dict[frozenset, List[str]]:
        """Get cached successor pair map for eval blocks."""
        if self._eval_succ_pair_map is None:
            blocks = self.mir_data.get('blocks', {})
            self._eval_succ_pair_map = self._build_succ_pair_map(blocks)
        return self._eval_succ_pair_map

    def _get_init_succ_pair_map(self) -> Dict[frozenset, List[str]]:
        """Get cached successor pair map for init blocks."""
        if self._init_succ_pair_map is None:
            blocks = self.init_mir_data.get('blocks', {})
            self._init_succ_pair_map = self._build_succ_pair_map(blocks)
        return self._init_succ_pair_map

    @classmethod
    def from_file(cls, va_path: str) -> "OpenVAFToJAX":
        """Create translator from a Verilog-A file

        Args:
            va_path: Path to the .va file

        Returns:
            OpenVAFToJAX instance
        """
        modules = openvaf_py.compile_va(va_path)
        if not modules:
            raise ValueError(f"No modules found in {va_path}")
        return cls(modules[0])

    def release_mir_data(self):
        """Release MIR data after code generation is complete.

        Call this after all translate_*() methods have been called to free
        ~28MB of memory for complex models like PSP103.

        The translator remains usable for accessing metadata (param_names, etc.)
        but cannot generate new code after this is called.
        """
        self.mir_data = None
        self.init_mir_data = None
        self.dae_data = None
        # Keep the derived data (constants, params) since they're small and may be needed
        # Only free the large MIR instruction data

    def translate(self) -> Callable:
        """Generate a JAX function from the MIR

        Returns a function with signature:
            f(inputs: List[float]) -> (residuals: Dict, jacobian: Dict)

        The inputs should be ordered according to self.params
        """
        import time
        t0 = time.perf_counter()
        logger.info("    translate: generating code...")
        code_lines = self._generate_code()
        t1 = time.perf_counter()
        logger.info(f"    translate: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)
        logger.info(f"    translate: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate: exec()...")
        fn = _exec_with_cache(code, 'device_eval')
        t2 = time.perf_counter()
        logger.info(f"    translate: exec() done in {t2-t1:.1f}s")
        return fn

    def translate_array(self) -> Tuple[Callable, Dict]:
        """Generate a JAX function that returns arrays (vmap-compatible)

        Returns a function with signature:
            f(inputs: Array[N]) -> (residuals: Array[num_nodes], jacobian: Array[num_jac_entries])

        Also returns metadata dict with:
            - 'node_names': list of node names in residual array order
            - 'jacobian_keys': list of (row, col) tuples in jacobian array order

        This output format is compatible with jax.vmap for batched evaluation.
        """
        import time

        # Profile code generation if OPENVAF_PROFILE=1
        if os.environ.get('OPENVAF_PROFILE') == '1':
            import cProfile
            import pstats
            import io
            logger.info("    translate_array: PROFILING ENABLED")
            profiler = cProfile.Profile()
            profiler.enable()

        t0 = time.perf_counter()
        logger.info("    translate_array: generating code...")
        code_lines = self._generate_code_array()
        t1 = time.perf_counter()
        logger.info(f"    translate_array: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        # Print profile results
        if os.environ.get('OPENVAF_PROFILE') == '1':
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            logger.info("    === CODE GENERATION PROFILE ===")
            logger.info(s.getvalue())

        code = '\n'.join(code_lines)
        logger.info(f"    translate_array: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate_array: exec()...")
        fn = _exec_with_cache(code, 'device_eval_array')
        t2 = time.perf_counter()
        logger.info(f"    translate_array: exec() done in {t2-t1:.1f}s")

        # Build metadata using v2 API for clean node names
        # V2 format: list of dicts with node_name, node_idx, etc.
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

        metadata = {
            'node_names': node_names,  # Clean names: ['D', 'G', 'S', 'B', 'NOI', ...]
            'node_indices': node_indices,  # Indices: [0, 1, 2, 3, 4, ...]
            'jacobian_keys': jacobian_keys,  # Clean: [('D', 'D'), ('D', 'DI'), ...]
            'jacobian_indices': jacobian_indices,  # Indices: [(0, 0), (0, 7), ...]
            'terminals': self.dae_data['terminals'],  # ['D', 'G', 'S', 'B']
            'internal_nodes': self.dae_data['internal_nodes'],  # ['NOI', 'GP', ...]
            'num_terminals': self.dae_data['num_terminals'],
            'num_internal': self.dae_data['num_internal'],
            'uses_simparam_gmin': self.uses_simparam_gmin,
            'uses_analysis': self.uses_analysis,
            'analysis_type_map': self.analysis_type_map,
        }

        return fn, metadata

    def translate_init_array(self) -> Tuple[Callable, Dict]:
        """Generate a standalone vmappable init function.

        Returns a function with signature:
            init_fn(inputs: Array[N_init]) -> cache: Array[N_cache]

        Also returns metadata dict with:
            - 'param_names': list of init param names
            - 'param_kinds': list of init param kinds
            - 'cache_size': number of cached values
            - 'cache_mapping': list of {init_value, eval_param} dicts

        This function computes all hidden_state/cached values that eval needs.
        Call this once per simulation, then pass cache to eval_with_cache().
        """
        import time

        t0 = time.perf_counter()
        logger.info("    translate_init_array: generating code...")
        code_lines = self._generate_init_code_array()
        t1 = time.perf_counter()
        logger.info(f"    translate_init_array: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)
        logger.info(f"    translate_init_array: code size = {len(code)} chars")

        # Compile with caching
        logger.info("    translate_init_array: exec()...")
        init_fn = _exec_with_cache(code, 'init_fn')
        t2 = time.perf_counter()
        logger.info(f"    translate_init_array: exec() done in {t2-t1:.1f}s")

        # Build metadata
        # Get parameter defaults from openvaf-py (extracted from Verilog-A source)
        param_defaults = {}
        if hasattr(self.module, 'get_param_defaults'):
            param_defaults = dict(self.module.get_param_defaults())

        # Cache size includes both cache_mapping values AND hidden_state values
        hidden_state_count = len(getattr(self, '_hidden_state_cache_mapping', []))
        total_cache_size = len(self.cache_mapping) + hidden_state_count

        metadata = {
            'param_names': list(self.module.init_param_names),
            'param_kinds': list(self.module.init_param_kinds),
            'cache_size': total_cache_size,  # Extended to include hidden_state
            'cache_mapping': self.cache_mapping,
            'hidden_state_cache_mapping': getattr(self, '_hidden_state_cache_mapping', []),
            'param_defaults': param_defaults,
            # Node collapse support
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

        This is an optimized version of translate_init_array that reduces memory
        by separating constant parameters (shared across all devices) from varying
        parameters (different per device).

        Args:
            shared_indices: Eval param indices that are constant across all devices
            varying_indices: Eval param indices that vary per device
            init_to_eval: Mapping from init param index to eval param index

        Returns a function with signature:
            init_fn_split(shared_params: Array[N_shared], device_params: Array[N_varying])
                -> (cache: Array[N_cache], collapse_decisions: Array[N_collapse])

        The function should be vmapped with in_axes=(None, 0) so that:
        - shared_params broadcasts (not sliced)
        - device_params is mapped over axis 0
        """
        import time

        t0 = time.perf_counter()
        logger.info("    translate_init_array_split: generating code...")
        code_lines = self._generate_init_code_array_split(
            shared_indices, varying_indices, init_to_eval
        )
        t1 = time.perf_counter()
        logger.info(f"    translate_init_array_split: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)
        logger.info(f"    translate_init_array_split: code size = {len(code)} chars")

        # Compile with caching (reuses function if code is identical)
        logger.info("    translate_init_array_split: exec()...")
        init_fn, code_hash = _exec_with_cache(code, 'init_fn_split', return_hash=True)
        t2 = time.perf_counter()
        logger.info(f"    translate_init_array_split: exec() done in {t2-t1:.1f}s")

        # Build metadata
        param_defaults = {}
        if hasattr(self.module, 'get_param_defaults'):
            param_defaults = dict(self.module.get_param_defaults())

        # Cache size includes both cache_mapping values AND hidden_state values
        hidden_state_count = len(getattr(self, '_hidden_state_cache_mapping', []))
        total_cache_size = len(self.cache_mapping) + hidden_state_count

        metadata = {
            'param_names': list(self.module.init_param_names),
            'param_kinds': list(self.module.init_param_kinds),
            'cache_size': total_cache_size,  # Extended to include hidden_state
            'cache_mapping': self.cache_mapping,
            'hidden_state_cache_mapping': getattr(self, '_hidden_state_cache_mapping', []),
            'param_defaults': param_defaults,
            'collapsible_pairs': self.collapsible_pairs,
            'collapse_decision_outputs': self.collapse_decision_outputs,
            'shared_indices': shared_indices,
            'varying_indices': varying_indices,
            'code_hash': code_hash,  # For vmapped+jit caching
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

        This is an optimized version of translate_eval_array_with_cache that reduces
        HLO slice operations by separating constant parameters (shared across all devices)
        from varying parameters (different per device). Also supports cache splitting.

        Args:
            shared_indices: Original param indices that are constant across all devices
            varying_indices: Original param indices that vary per device (including voltages)
            shared_cache_indices: Cache column indices that are constant across devices (optional)
            varying_cache_indices: Cache column indices that vary per device (optional)

        Returns a function with signature (if cache is split):
            eval_fn(shared_params: Array[N_shared],
                    device_params: Array[N_varying],
                    shared_cache: Array[N_shared_cache],
                    device_cache: Array[N_varying_cache])
                -> (res_resist, res_react, jac_resist, jac_react)

        Or (if cache is not split):
            eval_fn(shared_params: Array[N_shared],
                    device_params: Array[N_varying],
                    cache: Array[N_cache])
                -> (res_resist, res_react, jac_resist, jac_react)

        The function should be vmapped with in_axes=(None, 0, None, 0) for split cache
        or in_axes=(None, 0, 0) for unsplit cache.
        """
        import time

        use_cache_split = shared_cache_indices is not None and varying_cache_indices is not None

        t0 = time.perf_counter()
        logger.info(f"    translate_eval_array_with_cache_split: generating code (cache_split={use_cache_split})...")
        code_lines = self._generate_eval_code_with_cache_split(
            shared_indices, varying_indices, shared_cache_indices, varying_cache_indices
        )
        t1 = time.perf_counter()
        logger.info(f"    translate_eval_array_with_cache_split: code generated ({len(code_lines)} lines) in {t1-t0:.1f}s")

        code = '\n'.join(code_lines)
        logger.info(f"    translate_eval_array_with_cache_split: code size = {len(code)} chars")

        # Compile with caching
        fn_name = 'eval_fn_with_cache_split_cache' if use_cache_split else 'eval_fn_with_cache_split'
        logger.info("    translate_eval_array_with_cache_split: exec()...")
        eval_fn = _exec_with_cache(code, fn_name)
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
            'node_names': node_names,  # Clean names: ['D', 'G', 'S', 'B', 'NOI', ...]
            'node_indices': node_indices,  # Indices: [0, 1, 2, 3, 4, ...]
            'jacobian_keys': jacobian_keys,  # Clean: [('D', 'D'), ('D', 'DI'), ...]
            'jacobian_indices': jacobian_indices,  # Indices: [(0, 0), (0, 7), ...]
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

    def _generate_init_code_array(self) -> List[str]:
        """Generate standalone init function code returning cache array.

        The init function computes all cached values from parameters.
        These values are geometry calculations, temperature adjustments, etc.
        that don't depend on voltages and can be computed once per simulation.
        """
        lines = []
        lines.append("def init_fn(inputs):")
        lines.append("    import jax.numpy as jnp")
        lines.append("    from jax import lax")
        lines.append("")
        lines.append("    # Shared constants (defined once to reduce HLO size)")
        lines.append("    _ZERO = jnp.float64(0.0)")
        lines.append("    _ONE = jnp.float64(1.0)")
        lines.append("")

        # Initialize init constants
        lines.append("    # Init constants")
        for name, value in self.init_constants.items():
            if value == float('inf'):
                lines.append(f"    {name} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {name} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {name} = jnp.nan")
            else:
                lines.append(f"    {name} = {repr(value)}")

        # Boolean constants
        lines.append("    # Boolean constants")
        for name, value in self.init_bool_constants.items():
            lines.append(f"    {name} = {_jax_bool_repr(value)}")

        # Int constants
        lines.append("    # Int constants")
        for name, value in self.init_int_constants.items():
            lines.append(f"    {name} = {repr(value)}")

        # Ensure v3 exists (commonly used for zero)
        if 'v3' not in self.init_constants:
            lines.append("    v3 = _ZERO")

        lines.append("")

        # Map init params from inputs
        lines.append("    # Init parameters from inputs")
        for i, param in enumerate(self.init_params):
            lines.append(f"    {param} = inputs[{i}]")

        lines.append("")

        # Track defined variables
        init_defined: Set[str] = set(self.init_constants.keys())
        init_defined.update(self.init_bool_constants.keys())
        init_defined.update(self.init_int_constants.keys())
        init_defined.update(self.init_params)
        init_defined.add('v3')

        # Process init blocks in topological order
        init_block_order = self._topological_sort_init_blocks()
        init_by_block = self._group_init_instructions_by_block()

        lines.append("    # Init function computation")
        for item in init_block_order:
            if isinstance(item, tuple) and item[0] == 'loop':
                # Handle loop structure - reuse existing loop generation
                # but without the init_ prefix since we're in standalone init
                _, header, loop_blocks, exit_blocks = item
                loop_lines = self._generate_standalone_init_loop(
                    header, loop_blocks, exit_blocks,
                    init_by_block, init_defined
                )
                lines.extend(loop_lines)
            else:
                # Regular block
                block_name = item
                lines.append(f"")
                lines.append(f"    # {block_name}")

                for inst in init_by_block.get(block_name, []):
                    expr = self._translate_standalone_init_instruction(inst, init_defined)
                    if expr and 'result' in inst:
                        result = inst['result']
                        lines.append(f"    {result} = {expr}")
                        init_defined.add(result)

        lines.append("")

        # Build cache output array from cache_mapping
        # cache_mapping contains: {init_value: 'v123', eval_param: 45}
        # We output values in the order they appear in cache_mapping
        lines.append("    # Build cache array from computed values")
        cache_vals = []
        unmapped_cache_vals = []
        for mapping in self.cache_mapping:
            init_val = mapping['init_value']  # e.g., 'v123'
            if init_val in init_defined:
                cache_vals.append(init_val)
            else:
                # Value not computed - use _ZERO as fallback
                cache_vals.append('_ZERO')
                unmapped_cache_vals.append((init_val, mapping.get('eval_param', '?')))

        if unmapped_cache_vals:
            logger.warning(
                f"translate_init_array: {len(unmapped_cache_vals)} cache values not computed by init! "
                f"These will be 0.0. First 5: {unmapped_cache_vals[:5]}"
            )

        # Also include hidden_state values that init computes
        # These are values where init computes vN that eval expects as hidden_state
        lines.append("    # Add hidden_state values (computed by init, expected by eval)")
        hidden_state_vals = []
        unmapped_hidden_state = []
        param_kinds = list(self.module.param_kinds)
        param_names = list(self.module.param_names)
        for idx, kind in enumerate(param_kinds):
            if kind == 'hidden_state' and idx < len(self.params):
                eval_var = self.params[idx]  # e.g., 'v177' for TOXO_i
                # Init computes same var ID directly (no prefix)
                if eval_var in init_defined:
                    hidden_state_vals.append((eval_var, idx))
                else:
                    # Track unmapped hidden_state - will get 0.0 from shared_params!
                    pname = param_names[idx] if idx < len(param_names) else f'param_{idx}'
                    unmapped_hidden_state.append((idx, pname, eval_var))

        # NOTE ON HIDDEN_STATE PARAMS AND OPENVAF INLINING:
        # =================================================
        # OpenVAF's optimizer aggressively inlines hidden_state computations.
        # Analysis of MIR (Mid-level IR) shows that hidden_state params are
        # NEVER actually used in the eval function - they're all inlined into
        # the cache values computed by init.
        #
        # Tested models (all show 0% hidden_state usage in eval MIR):
        #   - resistor:  1 hidden_state param  -> 0 used
        #   - capacitor: 2 hidden_state params -> 0 used
        #   - diode:     16 hidden_state params -> 0 used
        #   - bsim4:     2330 hidden_state params -> 0 used
        #   - psp103:    1705 hidden_state params -> 0 used
        #
        # For complex models like PSP103, the eval function only reads:
        #   - voltages (13 params)
        #   - cache values from init (~462 values)
        #   - mfactor (1 param)
        #
        # The warning below is INFORMATIONAL only - unmapped hidden_state params
        # don't cause bugs because they're never actually read by eval.
        #
        if unmapped_hidden_state:
            # Log _i params that are unmapped - informational, not critical
            # (OpenVAF inlines these into cache, so they're never read)
            critical_unmapped = [(idx, name) for idx, name, _ in unmapped_hidden_state
                                 if name.endswith('_i')]
            if critical_unmapped:
                logger.debug(
                    f"INFO: {len(critical_unmapped)} instance-computed params (_i suffix) "
                    f"not computed by init. These are inlined by OpenVAF's optimizer "
                    f"and not actually used by eval. "
                    f"Examples: {[name for _, name in critical_unmapped[:5]]}"
                )
            logger.debug(
                f"hidden_state params: {len(hidden_state_vals)} mapped, "
                f"{len(unmapped_hidden_state)} unmapped (inlined by optimizer)"
            )

        # Store hidden_state mapping for eval to use
        self._hidden_state_cache_mapping = hidden_state_vals

        # Append hidden_state values to cache
        for eval_var, idx in hidden_state_vals:
            cache_vals.append(eval_var)

        if cache_vals:
            lines.append(f"    cache = jnp.array([{', '.join(cache_vals)}])")
        else:
            lines.append("    cache = jnp.array([])")

        # Build collapse decision array from collapse_decision_outputs
        # Each entry is a boolean indicating if that collapsible pair should collapse
        # Format: (pair_idx, condition_str) where condition_str may start with "!" for negation
        lines.append("")
        lines.append("    # Build collapse decision array")
        collapse_vals = []
        for pair_idx, val_name in self.collapse_decision_outputs:
            # Check if negation is needed (indicated by ! prefix)
            if val_name.startswith('!'):
                actual_val = val_name[1:]  # Remove ! prefix
                negate = True
            else:
                actual_val = val_name
                negate = False

            if actual_val in init_defined:
                # Convert boolean to float for array compatibility
                if negate:
                    # Collapse happens when condition is FALSE
                    collapse_vals.append(f"jnp.float32(jnp.logical_not({actual_val}))")
                else:
                    collapse_vals.append(f"jnp.float32({actual_val})")
            else:
                # Value not computed - default to True (collapse) if negated, else False
                # This handles the case where condition is always TRUE (never collapse)
                # or always FALSE (always collapse)
                collapse_vals.append("_ONE" if negate else "_ZERO")

        if collapse_vals:
            lines.append(f"    collapse_decisions = jnp.array([{', '.join(collapse_vals)}])")
        else:
            lines.append("    collapse_decisions = jnp.array([])")

        lines.append("    return cache, collapse_decisions")
        return lines

    def _generate_init_code_array_split(
        self,
        shared_indices: List[int],
        varying_indices: List[int],
        init_to_eval: List[int]
    ) -> List[str]:
        """Generate split init function code with shared/device params.

        Similar to _generate_init_code_array but takes split parameters:
        - shared_params: constant params (broadcasted, not sliced)
        - device_params: varying params (per-device)

        This reduces memory by not replicating shared values across all devices.
        """
        # Build index mappings
        shared_set = set(shared_indices)
        varying_set = set(varying_indices)
        shared_to_pos = {idx: pos for pos, idx in enumerate(shared_indices)}
        varying_to_pos = {idx: pos for pos, idx in enumerate(varying_indices)}

        lines = []
        lines.append("def init_fn_split(shared_params, device_params):")
        lines.append("    import jax.numpy as jnp")
        lines.append("    from jax import lax")
        lines.append("")
        lines.append("    # Shared constants (defined once to reduce HLO size)")
        lines.append("    _ZERO = jnp.float64(0.0)")
        lines.append("    _ONE = jnp.float64(1.0)")
        lines.append("")

        # Initialize init constants
        lines.append("    # Init constants")
        for name, value in self.init_constants.items():
            if value == float('inf'):
                lines.append(f"    {name} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {name} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {name} = jnp.nan")
            else:
                lines.append(f"    {name} = {repr(value)}")

        # Boolean constants
        lines.append("    # Boolean constants")
        for name, value in self.init_bool_constants.items():
            lines.append(f"    {name} = {_jax_bool_repr(value)}")

        # Int constants
        lines.append("    # Int constants")
        for name, value in self.init_int_constants.items():
            lines.append(f"    {name} = {repr(value)}")

        # Ensure v3 exists (commonly used for zero)
        if 'v3' not in self.init_constants:
            lines.append("    v3 = _ZERO")

        lines.append("")

        # Map init params from split arrays
        lines.append("    # Init parameters from split arrays (shared/device)")
        for init_idx, param in enumerate(self.init_params):
            eval_idx = init_to_eval[init_idx] if init_idx < len(init_to_eval) else -1
            if eval_idx in shared_set:
                # Param is in shared_params
                pos = shared_to_pos[eval_idx]
                lines.append(f"    {param} = shared_params[{pos}]")
            elif eval_idx in varying_set:
                # Param is in device_params
                pos = varying_to_pos[eval_idx]
                lines.append(f"    {param} = device_params[{pos}]")
            else:
                # Param not found in either - use zero (shouldn't happen)
                lines.append(f"    {param} = _ZERO  # fallback: eval_idx={eval_idx} not in split")

        lines.append("")

        # Track defined variables
        init_defined: Set[str] = set(self.init_constants.keys())
        init_defined.update(self.init_bool_constants.keys())
        init_defined.update(self.init_int_constants.keys())
        init_defined.update(self.init_params)
        init_defined.add('v3')

        # Process init blocks in topological order
        init_block_order = self._topological_sort_init_blocks()
        init_by_block = self._group_init_instructions_by_block()

        lines.append("    # Init function computation")
        for item in init_block_order:
            if isinstance(item, tuple) and item[0] == 'loop':
                _, header, loop_blocks, exit_blocks = item
                loop_lines = self._generate_standalone_init_loop(
                    header, loop_blocks, exit_blocks,
                    init_by_block, init_defined
                )
                lines.extend(loop_lines)
            else:
                block_name = item
                lines.append(f"")
                lines.append(f"    # {block_name}")

                for inst in init_by_block.get(block_name, []):
                    expr = self._translate_standalone_init_instruction(inst, init_defined)
                    if expr and 'result' in inst:
                        result = inst['result']
                        lines.append(f"    {result} = {expr}")
                        init_defined.add(result)

        lines.append("")

        # Build cache output array
        lines.append("    # Build cache array from computed values")
        cache_vals = []
        unmapped_cache_vals = []
        for mapping in self.cache_mapping:
            init_val = mapping['init_value']
            if init_val in init_defined:
                cache_vals.append(init_val)
            else:
                cache_vals.append('_ZERO')
                unmapped_cache_vals.append((init_val, mapping.get('eval_param', '?')))

        if unmapped_cache_vals:
            logger.warning(
                f"translate_init_array_split: {len(unmapped_cache_vals)} cache values not computed! "
                f"These will be 0.0. First 5: {unmapped_cache_vals[:5]}"
            )

        # Also include hidden_state values that init computes
        # These are values where init computes vN that eval expects as hidden_state
        lines.append("    # Add hidden_state values (computed by init, expected by eval)")
        hidden_state_vals = []
        unmapped_hidden_state = []
        param_kinds = list(self.module.param_kinds)
        param_names = list(self.module.param_names)
        for idx, kind in enumerate(param_kinds):
            if kind == 'hidden_state' and idx < len(self.params):
                eval_var = self.params[idx]  # e.g., 'v177' for TOXO_i
                # Init computes same var ID directly (no prefix)
                if eval_var in init_defined:
                    hidden_state_vals.append((eval_var, idx))
                else:
                    pname = param_names[idx] if idx < len(param_names) else f'param_{idx}'
                    unmapped_hidden_state.append((idx, pname, eval_var))

        # Log unmapped hidden_state params (informational - OpenVAF inlines these)
        # OpenVAF's optimizer aggressively inlines hidden_state computations into cache
        # values, so these params are never actually read by eval. Setting to 0.0 is safe.
        if unmapped_hidden_state:
            i_suffix_count = sum(1 for _, name, _ in unmapped_hidden_state if name.endswith('_i'))
            logger.debug(
                f"hidden_state params: {len(hidden_state_vals)} mapped, "
                f"{len(unmapped_hidden_state)} unmapped ({i_suffix_count} with _i suffix). "
                f"Unmapped params are inlined by OpenVAF and not read by eval."
            )

        # Store hidden_state mapping for eval to use
        self._hidden_state_cache_mapping = hidden_state_vals

        # Append hidden_state values to cache
        for eval_var, idx in hidden_state_vals:
            cache_vals.append(eval_var)

        if cache_vals:
            lines.append(f"    cache = jnp.array([{', '.join(cache_vals)}])")
        else:
            lines.append("    cache = jnp.array([])")

        # Build collapse decision array
        lines.append("")
        lines.append("    # Build collapse decision array")
        collapse_vals = []
        for pair_idx, val_name in self.collapse_decision_outputs:
            if val_name.startswith('!'):
                actual_val = val_name[1:]
                negate = True
            else:
                actual_val = val_name
                negate = False

            if actual_val in init_defined:
                if negate:
                    collapse_vals.append(f"jnp.float32(jnp.logical_not({actual_val}))")
                else:
                    collapse_vals.append(f"jnp.float32({actual_val})")
            else:
                collapse_vals.append("_ONE" if negate else "_ZERO")

        if collapse_vals:
            lines.append(f"    collapse_decisions = jnp.array([{', '.join(collapse_vals)}])")
        else:
            lines.append("    collapse_decisions = jnp.array([])")

        lines.append("    return cache, collapse_decisions")
        return lines

    def _generate_eval_code_with_cache_split(
        self,
        shared_indices: List[int],
        varying_indices: List[int],
        shared_cache_indices: Optional[List[int]] = None,
        varying_cache_indices: Optional[List[int]] = None
    ) -> List[str]:
        """Generate eval function code that takes split shared/device params and optionally split cache.

        This optimization reduces HLO slice operations by separating:
        - shared_params: constant across all devices (broadcast, not sliced)
        - device_params: varies per device (sliced via vmap)
        - shared_cache: cache values constant across devices (broadcast, not sliced)
        - device_cache: cache values that vary per device (sliced via vmap)

        Args:
            shared_indices: Original param indices that are constant across devices
            varying_indices: Original param indices that vary per device (including voltages)
            shared_cache_indices: Original cache indices constant across devices (optional)
            varying_cache_indices: Original cache indices that vary per device (optional)

        Returns:
            List of code lines for the split eval function
        """
        use_cache_split = shared_cache_indices is not None and varying_cache_indices is not None

        # Build reverse mappings: original_idx -> (source, new_idx)
        # source is 'shared' or 'device'
        idx_mapping: Dict[int, Tuple[str, int]] = {}
        for new_idx, orig_idx in enumerate(shared_indices):
            idx_mapping[orig_idx] = ('shared', new_idx)
        for new_idx, orig_idx in enumerate(varying_indices):
            idx_mapping[orig_idx] = ('device', new_idx)

        # Build cache mappings if using split cache
        cache_idx_mapping: Dict[int, Tuple[str, int]] = {}
        if use_cache_split:
            for new_idx, orig_idx in enumerate(shared_cache_indices):
                cache_idx_mapping[orig_idx] = ('shared_cache', new_idx)
            for new_idx, orig_idx in enumerate(varying_cache_indices):
                cache_idx_mapping[orig_idx] = ('device_cache', new_idx)

        lines = []
        if use_cache_split:
            lines.append("def eval_fn_with_cache_split_cache(shared_params, device_params, shared_cache, device_cache):")
        else:
            lines.append("def eval_fn_with_cache_split(shared_params, device_params, cache):")
        lines.append("    import jax.numpy as jnp")
        lines.append("    from jax import lax")
        lines.append("")
        lines.append("    # Shared constants (defined once to reduce HLO size)")
        lines.append("    _ZERO = jnp.float64(0.0)")
        lines.append("    _ONE = jnp.float64(1.0)")
        lines.append("")

        # Initialize constants for eval function
        lines.append("    # Constants (eval function)")
        for name, value in self.constants.items():
            if value == float('inf'):
                lines.append(f"    {name} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {name} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {name} = jnp.nan")
            else:
                lines.append(f"    {name} = {repr(value)}")

        # Boolean constants
        lines.append("    # Boolean constants")
        for name, value in self.bool_constants.items():
            lines.append(f"    {name} = {_jax_bool_repr(value)}")

        # Int constants
        lines.append("    # Int constants")
        for name, value in self.int_constants.items():
            if name not in self.constants:
                lines.append(f"    {name} = {repr(value)}")

        # Ensure v3 exists
        if 'v3' not in self.constants:
            lines.append("    v3 = _ZERO")

        lines.append("")

        # Map function parameters from split inputs
        # IMPORTANT: Skip hidden_state params here - they're computed by inline init
        lines.append("    # Input parameters (from split shared/device params)")
        param_kinds = list(self.module.param_kinds)
        num_named_params = len(self.module.param_names)
        hidden_state_indices = set()  # Track which indices are hidden_state
        for i, param in enumerate(self.params[:num_named_params]):
            kind = param_kinds[i] if i < len(param_kinds) else None
            if kind == 'hidden_state':
                # Skip - will be assigned from init computation later
                hidden_state_indices.add(i)
                lines.append(f"    # {param} = hidden_state (assigned from init below)")
            elif i in idx_mapping:
                source, new_idx = idx_mapping[i]
                if source == 'shared':
                    lines.append(f"    {param} = shared_params[{new_idx}]")
                else:
                    lines.append(f"    {param} = device_params[{new_idx}]")
            else:
                # Fallback - shouldn't happen if indices are complete
                lines.append(f"    {param} = _ZERO  # unmapped index {i}")

        # Derivative selector params default to 0
        lines.append("    # Derivative selector params (default to 0)")
        for param in self.params[num_named_params:]:
            lines.append(f"    {param} = 0")

        lines.append("")

        # Map cache values to eval param slots
        all_func_params = self.module.get_all_func_params()
        param_idx_to_val = {p[0]: f"v{p[1]}" for p in all_func_params}

        if use_cache_split:
            lines.append("    # Cache values from split shared/device cache arrays")
            for cache_idx, mapping in enumerate(self.cache_mapping):
                eval_param_idx = mapping['eval_param']
                eval_val = param_idx_to_val.get(eval_param_idx, f"cached_{eval_param_idx}")
                if cache_idx in cache_idx_mapping:
                    source, new_idx = cache_idx_mapping[cache_idx]
                    lines.append(f"    {eval_val} = {source}[{new_idx}]")
                else:
                    lines.append(f"    {eval_val} = _ZERO  # unmapped cache index {cache_idx}")
        else:
            lines.append("    # Cache values from init function")
            for cache_idx, mapping in enumerate(self.cache_mapping):
                eval_param_idx = mapping['eval_param']
                eval_val = param_idx_to_val.get(eval_param_idx, f"cached_{eval_param_idx}")
                lines.append(f"    {eval_val} = cache[{cache_idx}]")

        # === COMPUTE HIDDEN_STATE VALUES DIRECTLY ===
        # Instead of reading from cache (which has wrong values due to MIR mismatch),
        # we compute hidden_state values directly in eval using init blocks.
        # This duplicates work but ensures correctness.
        lines.append("")
        lines.append("    # Init function computation for hidden_state values")
        init_defined = set()

        # Add prefixed init constants
        lines.append("    # Init constants (prefixed)")
        for name, value in self.init_constants.items():
            prefixed = f"init_{name}"
            if value == float('inf'):
                lines.append(f"    {prefixed} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {prefixed} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {prefixed} = jnp.nan")
            else:
                lines.append(f"    {prefixed} = {repr(value)}")
            init_defined.add(prefixed)

        # Map init params from eval inputs (using idx_mapping for split params)
        # IMPORTANT: Use the actual input arrays, not eval param variables
        # This ensures init params get actual input values, not potentially uninitialized eval vars
        init_param_mapping = self._build_init_param_mapping()
        for init_param, eval_idx in init_param_mapping.items():
            if eval_idx is not None and eval_idx in idx_mapping:
                prefixed = f"init_{init_param}"
                source, new_idx = idx_mapping[eval_idx]
                if source == 'shared':
                    lines.append(f"    {prefixed} = shared_params[{new_idx}]")
                else:
                    lines.append(f"    {prefixed} = device_params[{new_idx}]")
                init_defined.add(prefixed)

        # Process init instructions in block order
        init_block_order = self._topological_sort_init_blocks()
        init_by_block = self._group_init_instructions_by_block()

        for item in init_block_order:
            if isinstance(item, tuple) and item[0] == 'loop':
                # Handle loop structure
                _, header, loop_blocks, exit_blocks = item
                loop_lines = self._generate_init_loop(
                    header, loop_blocks, exit_blocks,
                    init_by_block, init_defined
                )
                lines.extend(loop_lines)
            else:
                # Regular block
                block_name = item
                lines.append(f"")
                lines.append(f"    # init {block_name}")

                for inst in init_by_block.get(block_name, []):
                    expr = self._translate_init_instruction(inst, init_defined)
                    if expr and 'result' in inst:
                        prefixed_result = f"init_{inst['result']}"
                        lines.append(f"    {prefixed_result} = {expr}")
                        init_defined.add(prefixed_result)

        lines.append("")

        # Map hidden_state values from init to eval params using value-number matching
        lines.append("    # Hidden state values from init -> eval")
        hidden_state_assignments = self._build_hidden_state_assignments(init_defined)
        for eval_var, init_var in hidden_state_assignments:
            lines.append(f"    {eval_var} = {init_var}")
        if hidden_state_assignments:
            logger.info(f"    Generated {len(hidden_state_assignments)} hidden_state assignments from init blocks")

        lines.append("")

        # Track defined variables
        defined_vars: Set[str] = set(self.constants.keys())
        defined_vars.update(self.bool_constants.keys())
        defined_vars.update(self.int_constants.keys())
        defined_vars.update(self.params)
        defined_vars.add('v3')

        # Add cache-assigned variables to defined set
        for mapping in self.cache_mapping:
            eval_param_idx = mapping['eval_param']
            eval_val = param_idx_to_val.get(eval_param_idx, f"cached_{eval_param_idx}")
            defined_vars.add(eval_val)

        # Add init-defined variables (prefixed) to defined set
        defined_vars.update(init_defined)

        # Add hidden_state assigned variables to defined set
        for eval_var, init_var in hidden_state_assignments:
            defined_vars.add(eval_var)

        # Process eval blocks in topological order (no init blocks) - same as original
        block_order = self._topological_sort()
        eval_by_block = self._group_eval_instructions_by_block()

        lines.append("    # Eval function computation")
        for item in block_order:
            if isinstance(item, tuple) and item[0] == 'loop':
                _, header, loop_blocks, exit_blocks = item
                loop_lines = self._generate_eval_loop(
                    header, loop_blocks, exit_blocks,
                    eval_by_block, defined_vars
                )
                lines.extend(loop_lines)
            else:
                block_name = item
                lines.append(f"")
                lines.append(f"    # {block_name}")

                for inst in eval_by_block.get(block_name, []):
                    expr = self._translate_instruction(inst, defined_vars)
                    if expr and 'result' in inst:
                        lines.append(f"    {inst['result']} = {expr}")
                        defined_vars.add(inst['result'])

        lines.append("")

        # Build array outputs - same as original
        lines.append("    # Build output arrays (vmap-compatible)")

        residual_resist_exprs = []
        residual_react_exprs = []
        for res in self.dae_data['residuals']:
            resist_var = self._mir_to_code_var(res['resist_var'])
            react_var = self._mir_to_code_var(res['react_var'])
            resist_val = resist_var if resist_var in defined_vars else '_ZERO'
            react_val = react_var if react_var in defined_vars else '_ZERO'
            residual_resist_exprs.append(resist_val)
            residual_react_exprs.append(react_val)
        lines.append(f"    residuals_resist = jnp.array([{', '.join(residual_resist_exprs)}])")
        lines.append(f"    residuals_react = jnp.array([{', '.join(residual_react_exprs)}])")

        jacobian_resist_exprs = []
        jacobian_react_exprs = []
        for entry in self.dae_data['jacobian']:
            resist_var = self._mir_to_code_var(entry['resist_var'])
            react_var = self._mir_to_code_var(entry['react_var'])
            resist_val = resist_var if resist_var in defined_vars else '_ZERO'
            react_val = react_var if react_var in defined_vars else '_ZERO'
            jacobian_resist_exprs.append(resist_val)
            jacobian_react_exprs.append(react_val)
        lines.append(f"    jacobian_resist = jnp.array([{', '.join(jacobian_resist_exprs)}])")
        lines.append(f"    jacobian_react = jnp.array([{', '.join(jacobian_react_exprs)}])")

        lines.append("    return residuals_resist, residuals_react, jacobian_resist, jacobian_react")

        # Post-process: Replace 'inputs[-1]' and 'inputs[-2]' with 'device_params[-1]' and 'device_params[-2]'
        # This is needed because _translate_instruction generates code assuming the original 'inputs' array,
        # but in split mode, gmin and analysis_type are in device_params (as the last columns)
        fixed_lines = []
        for line in lines:
            line = line.replace('inputs[-1]', 'device_params[-1]')
            line = line.replace('inputs[-2]', 'device_params[-2]')
            fixed_lines.append(line)

        return fixed_lines

    def _translate_standalone_init_instruction(self, inst: dict, defined_vars: Set[str]) -> Optional[str]:
        """Translate an init instruction without init_ prefix (for standalone init function)"""

        def get_operand(op: str) -> str:
            if op in defined_vars:
                return op
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return _jax_bool_repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            return op

        # Handle PHI nodes for init function
        opcode = inst.get('opcode', '').lower()
        if opcode == 'phi':
            phi_ops = inst.get('phi_operands', [])
            phi_block = inst.get('block', '')
            if phi_ops and len(phi_ops) >= 2:
                pred_blocks = [op['block'] for op in phi_ops]
                val_by_block = {op['block']: get_operand(op['value']) for op in phi_ops}

                if len(pred_blocks) > 2:
                    return self._build_standalone_init_multi_way_phi(phi_block, phi_ops, val_by_block, get_operand)

                cond_info = self._get_standalone_init_phi_condition(phi_block, pred_blocks)
                if cond_info:
                    cond_var, true_block, false_block = cond_info
                    true_val = val_by_block.get(true_block, '_ZERO')
                    false_val = val_by_block.get(false_block, '_ZERO')
                    return f"jnp.where({cond_var}, {true_val}, {false_val})"
                else:
                    return get_operand(phi_ops[0]['value'])
            elif phi_ops:
                return get_operand(phi_ops[0]['value'])
            return '_ZERO'

        return self._translate_instruction_impl(inst, get_operand)

    def _get_standalone_init_phi_condition(self, phi_block: str, pred_blocks: List[str]) -> Optional[Tuple[str, str, str]]:
        """Get PHI condition for standalone init function (no prefix)"""
        blocks = self.init_mir_data.get('blocks', {})
        branch_conds = self._build_init_branch_conditions()
        return self._get_phi_condition_impl(
            phi_block, pred_blocks, blocks, branch_conds,
            prefix='', succ_pair_map=self._get_init_succ_pair_map()
        )

    def _build_standalone_init_multi_way_phi(self, phi_block: str, phi_ops: List[dict],
                                              val_by_block: Dict[str, str],
                                              get_operand: Callable[[str], str]) -> str:
        """Build nested jnp.where for multi-way PHI in standalone init (no prefix)"""
        blocks = self.init_mir_data.get('blocks', {})
        branch_conds = self._build_init_branch_conditions()
        pred_blocks = [op['block'] for op in phi_ops]

        result = self._build_standalone_init_nested_where(phi_block, pred_blocks, val_by_block, blocks, branch_conds)
        return result if result else val_by_block.get(pred_blocks[0], '_ZERO')

    def _build_standalone_init_nested_where(self, phi_block: str, pred_blocks: List[str],
                                             pred_to_val: Dict[str, str], blocks: Dict,
                                             branch_conds: Dict) -> Optional[str]:
        """Build nested where for standalone init (no prefix on conditions)"""
        if len(pred_blocks) == 1:
            return pred_to_val.get(pred_blocks[0], '_ZERO')

        if len(pred_blocks) == 2:
            cond_info = self._get_standalone_init_phi_condition(phi_block, pred_blocks)
            if cond_info:
                cond_var, true_block, false_block = cond_info
                true_val = pred_to_val.get(true_block, '_ZERO')
                false_val = pred_to_val.get(false_block, '_ZERO')
                return f"jnp.where({cond_var}, {true_val}, {false_val})"
            return pred_to_val.get(pred_blocks[0], '_ZERO')

        # For >2 predecessors
        for block_name in blocks:
            if block_name not in branch_conds:
                continue

            succs = blocks[block_name].get('successors', [])
            if len(succs) != 2:
                continue

            direct_pred = None
            indirect_succ = None
            for succ in succs:
                if succ in pred_blocks:
                    direct_pred = succ
                else:
                    indirect_succ = succ

            if direct_pred and indirect_succ:
                cond_var, is_true = branch_conds[block_name][direct_pred]
                direct_val = pred_to_val.get(direct_pred, '_ZERO')
                remaining_preds = [p for p in pred_blocks if p != direct_pred]

                remaining_expr = self._build_standalone_init_nested_where(
                    phi_block, remaining_preds, pred_to_val, blocks, branch_conds
                )

                if remaining_expr:
                    if is_true:
                        return f"jnp.where({cond_var}, {direct_val}, {remaining_expr})"
                    else:
                        return f"jnp.where({cond_var}, {remaining_expr}, {direct_val})"

        return None

    def _generate_standalone_init_loop(self, header: str, loop_blocks: Set[str],
                                        exit_blocks: List[str], init_by_block: Dict[str, List[dict]],
                                        init_defined: Set[str]) -> List[str]:
        """Generate loop for standalone init function (no init_ prefix)"""
        lines = []
        lines.append("")
        lines.append(f"    # Loop: {header} with blocks {sorted(loop_blocks)}")

        header_insts = init_by_block.get(header, [])
        phi_nodes = [inst for inst in header_insts if inst.get('opcode', '').lower() == 'phi']

        loop_carried = []
        for phi in phi_nodes:
            result = phi.get('result', '')
            phi_ops = phi.get('phi_operands', [])
            init_val = None
            loop_val = None
            for op in phi_ops:
                if op['block'] in loop_blocks:
                    loop_val = op['value']
                else:
                    init_val = op['value']
            if result and init_val and loop_val:
                loop_carried.append((result, init_val, loop_val))

        if not loop_carried:
            lines.append("    # WARNING: No loop-carried values found, skipping loop")
            return lines

        condition_inst = None
        body_insts = []

        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'br' and 'condition' in inst:
                condition_inst = inst
            elif op != 'phi':
                body_insts.append(inst)

        for block in sorted(loop_blocks):
            if block != header:
                for inst in init_by_block.get(block, []):
                    op = inst.get('opcode', '').lower()
                    if op not in ('br', 'jmp'):
                        body_insts.append(inst)

        def get_operand(op: str, local_vars: Set[str] = None) -> str:
            if local_vars is None:
                local_vars = set()
            if op in local_vars:
                return op
            if op in init_defined:
                return op
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return _jax_bool_repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            return op

        # Initial state
        init_state_parts = [get_operand(init_val) for _, init_val, _ in loop_carried]
        lines.append(f"    _loop_state_init = ({', '.join(init_state_parts)},)")

        # Condition function
        lines.append("")
        lines.append("    def _loop_cond(_loop_state):")
        state_vars = [lc[0] for lc in loop_carried]
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_standalone_init_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        if condition_inst:
            cond_var = condition_inst.get('condition', '')
            if cond_var in local_vars:
                lines.append(f"        return {cond_var}")
            else:
                lines.append(f"        return {get_operand(cond_var, local_vars)}")
        else:
            lines.append("        return False")

        # Body function
        lines.append("")
        lines.append("    def _loop_body(_loop_state):")
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_standalone_init_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        for inst in body_insts:
            result = inst.get('result', '')
            expr = self._translate_standalone_init_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        new_state_parts = []
        for _, _, loop_val in loop_carried:
            if loop_val in local_vars:
                new_state_parts.append(loop_val)
            else:
                new_state_parts.append(get_operand(loop_val, local_vars))

        lines.append(f"        return ({', '.join(new_state_parts)},)")

        # Call while_loop
        lines.append("")
        lines.append("    _loop_result = lax.while_loop(_loop_cond, _loop_body, _loop_state_init)")

        for i, (result, _, _) in enumerate(loop_carried):
            lines.append(f"    {result} = _loop_result[{i}]")
            init_defined.add(result)

        return lines

    def _translate_standalone_init_loop_instruction(self, inst: dict, local_vars: Set[str],
                                                     init_defined: Set[str]) -> Optional[str]:
        """Translate instruction for standalone init loop (no prefix)"""
        def get_operand(op: str) -> str:
            if op in local_vars:
                return op
            if op in init_defined:
                return op
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return _jax_bool_repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            return op

        return self._translate_instruction_impl(inst, get_operand)

    def _generate_core_code(self, func_name: str = "device_eval") -> Tuple[List[str], Set[str]]:
        """Generate core computation code shared by both output formats.

        Returns:
            (lines, defined_vars) - code lines and set of defined variable names
        """
        import time
        t0 = time.perf_counter()
        logger.info("      _generate_core_code: starting...")

        lines = []
        lines.append(f"def {func_name}(inputs):")
        lines.append("    import jax.numpy as jnp")
        lines.append("    from jax import lax")
        lines.append("")
        lines.append("    # Shared constants (defined once to reduce HLO size)")
        lines.append("    _ZERO = jnp.float64(0.0)")
        lines.append("    _ONE = jnp.float64(1.0)")
        lines.append("")

        # Only use eval constants (init constants are prefixed separately)
        # Initialize constants for eval function
        lines.append("    # Constants (eval function)")
        for name, value in self.constants.items():
            if value == float('inf'):
                lines.append(f"    {name} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {name} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {name} = jnp.nan")
            else:
                lines.append(f"    {name} = {repr(value)}")

        # Boolean constants - use jnp.bool_() for JIT compatibility
        lines.append("    # Boolean constants")
        for name, value in self.bool_constants.items():
            lines.append(f"    {name} = {_jax_bool_repr(value)}")

        # Int constants (from both eval and init MIR)
        lines.append("    # Int constants")
        # First add eval int constants
        for name, value in self.int_constants.items():
            if name not in self.constants:
                lines.append(f"    {name} = {repr(value)}")
        # Then add any init int constants not already defined
        for name, value in self.init_int_constants.items():
            if name not in self.constants and name not in self.int_constants:
                lines.append(f"    {name} = {repr(value)}")

        # Ensure v3 exists (commonly used for zero)
        if 'v3' not in self.constants:
            lines.append("    v3 = _ZERO")

        lines.append("")

        logger.info(f"      _generate_core_code: constants done ({len(lines)} lines)")

        # Map function parameters to inputs
        # Named eval params from user inputs, derivative selectors default to 0
        lines.append("    # Input parameters (eval function)")
        num_named_params = len(self.module.param_names)
        for i, param in enumerate(self.params[:num_named_params]):
            lines.append(f"    {param} = inputs[{i}]")

        # Derivative selector params (used internally for Jacobian computation)
        # Default to 0 for DC analysis (no derivatives enabled)
        lines.append("    # Derivative selector params (default to 0)")
        for param in self.params[num_named_params:]:
            lines.append(f"    {param} = 0")

        # Process init function first to compute cached values
        # Use init_ prefix to avoid name collisions with eval function
        lines.append("    # Init function computation")
        init_defined = set()

        # Add prefixed init constants
        lines.append("    # Init constants (prefixed)")
        for name, value in self.init_constants.items():
            prefixed = f"init_{name}"
            if value == float('inf'):
                lines.append(f"    {prefixed} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {prefixed} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {prefixed} = jnp.nan")
            else:
                lines.append(f"    {prefixed} = {repr(value)}")
            init_defined.add(prefixed)

        # Map init params from inputs (they overlap with eval params)
        # Init params are: R, $temperature, tnom, zeta, mfactor (for resistor)
        # These correspond to certain eval params
        init_param_mapping = self._build_init_param_mapping()
        for init_param, eval_idx in init_param_mapping.items():
            if eval_idx is not None:
                prefixed = f"init_{init_param}"
                lines.append(f"    {prefixed} = inputs[{eval_idx}]")
                init_defined.add(prefixed)

        # Process init instructions in block order (like eval function)
        # This ensures variables are defined before use
        init_block_order = self._topological_sort_init_blocks()
        init_by_block = self._group_init_instructions_by_block()

        for item in init_block_order:
            if isinstance(item, tuple) and item[0] == 'loop':
                # Handle loop structure
                _, header, loop_blocks, exit_blocks = item
                loop_lines = self._generate_init_loop(
                    header, loop_blocks, exit_blocks,
                    init_by_block, init_defined
                )
                lines.extend(loop_lines)
            else:
                # Regular block
                block_name = item
                lines.append(f"")
                lines.append(f"    # init {block_name}")

                for inst in init_by_block.get(block_name, []):
                    expr = self._translate_init_instruction(inst, init_defined)
                    if expr and 'result' in inst:
                        prefixed_result = f"init_{inst['result']}"
                        lines.append(f"    {prefixed_result} = {expr}")
                        init_defined.add(prefixed_result)

        lines.append("")

        # Map cached values from init to eval params
        # Need to find the actual Value names for cached params
        lines.append("    # Cached values from init -> eval params")
        all_func_params = self.module.get_all_func_params()
        param_idx_to_val = {p[0]: f"v{p[1]}" for p in all_func_params}

        for mapping in self.cache_mapping:
            init_val = f"init_{mapping['init_value']}"
            eval_param_idx = mapping['eval_param']
            # Look up the actual Value name for this param index
            eval_val = param_idx_to_val.get(eval_param_idx, f"cached_{eval_param_idx}")
            if init_val in init_defined:
                lines.append(f"    {eval_val} = {init_val}")

        lines.append("")

        # Map hidden_state values from init to eval params
        # Uses value-number matching: if eval vN has init_vN computed, assign vN = init_vN
        # Note: The 1e40 "Jacobian explosion" from NOI nodes is now handled separately
        # by masking in runner.py (huge Jacobian entries are clamped)
        lines.append("    # Hidden state values from init -> eval")
        hidden_state_assignments = self._build_hidden_state_assignments(init_defined)
        for eval_var, init_var in hidden_state_assignments:
            lines.append(f"    {eval_var} = {init_var}")
        if hidden_state_assignments:
            logger.info(f"      _generate_core_code: {len(hidden_state_assignments)} hidden_state assignments")

        lines.append("")

        logger.info(f"      _generate_core_code: init done ({len(lines)} lines)")

        # Process eval blocks in topological order
        block_order = self._topological_sort()
        defined_vars: Set[str] = set(self.constants.keys())
        defined_vars.update(self.params)
        defined_vars.update(init_defined)  # Include init-computed values
        defined_vars.add('v3')

        lines.append("    # Eval function computation")

        # Group instructions by block for efficient access
        eval_by_block = self._group_eval_instructions_by_block()

        for item in block_order:
            if isinstance(item, tuple) and item[0] == 'loop':
                # Handle loop structure
                _, header, loop_blocks, exit_blocks = item
                loop_lines = self._generate_eval_loop(
                    header, loop_blocks, exit_blocks,
                    eval_by_block, defined_vars
                )
                lines.extend(loop_lines)
            else:
                # Regular block
                block_name = item
                lines.append(f"")
                lines.append(f"    # {block_name}")

                # Get instructions for this block
                for inst in eval_by_block.get(block_name, []):
                    expr = self._translate_instruction(inst, defined_vars)
                    if expr and 'result' in inst:
                        lines.append(f"    {inst['result']} = {expr}")
                        defined_vars.add(inst['result'])

        lines.append("")

        t1 = time.perf_counter()
        logger.info(f"      _generate_core_code: eval blocks done ({len(lines)} lines in {t1-t0:.1f}s)")

        return lines, defined_vars

    def _mir_to_code_var(self, mir_var: str) -> str:
        """Convert mir_XX to vXX for code generation lookup."""
        if mir_var.startswith('mir_'):
            return 'v' + mir_var[4:]
        return mir_var

    def _generate_code(self) -> List[str]:
        """Generate JAX function code with dict outputs."""
        lines, defined_vars = self._generate_core_code("device_eval")

        # Build dict output expressions
        lines.append("    # Build outputs (dict format)")
        lines.append("    residuals = {")
        for res in self.dae_data['residuals']:
            node = res['node_name']
            resist_var = self._mir_to_code_var(res['resist_var'])
            react_var = self._mir_to_code_var(res['react_var'])
            resist_val = resist_var if resist_var in defined_vars else '_ZERO'
            react_val = react_var if react_var in defined_vars else '_ZERO'
            lines.append(f"        '{node}': {{'resist': {resist_val}, 'react': {react_val}}},")
        lines.append("    }")

        lines.append("    jacobian = {")
        for entry in self.dae_data['jacobian']:
            key = f"('{entry['row_node_name']}', '{entry['col_node_name']}')"
            resist_var = self._mir_to_code_var(entry['resist_var'])
            react_var = self._mir_to_code_var(entry['react_var'])
            resist_val = resist_var if resist_var in defined_vars else '_ZERO'
            react_val = react_var if react_var in defined_vars else '_ZERO'
            lines.append(f"        {key}: {{'resist': {resist_val}, 'react': {react_val}}},")
        lines.append("    }")

        lines.append("    return residuals, jacobian")
        return lines

    def _generate_code_array(self) -> List[str]:
        """Generate JAX function code with array outputs (vmap-compatible).

        Returns function with signature:
            f(inputs) -> (residuals_resist, residuals_react, jacobian_resist, jacobian_react)

        For DC analysis: use only resistive terms
        For transient: f_total = f_resist + Q_react/dt, J_total = J_resist + C_react/dt
        """
        lines, defined_vars = self._generate_core_code("device_eval_array")

        # Build array output expressions
        lines.append("    # Build output arrays (vmap-compatible)")
        lines.append("    # Resistive: currents and conductances")
        lines.append("    # Reactive: charges and capacitances (for transient analysis)")

        # Residuals arrays - one entry per node (resistive and reactive)
        residual_resist_exprs = []
        residual_react_exprs = []
        for res in self.dae_data['residuals']:
            resist_var = self._mir_to_code_var(res['resist_var'])
            react_var = self._mir_to_code_var(res['react_var'])
            resist_val = resist_var if resist_var in defined_vars else '_ZERO'
            react_val = react_var if react_var in defined_vars else '_ZERO'
            residual_resist_exprs.append(resist_val)
            residual_react_exprs.append(react_val)
        lines.append(f"    residuals_resist = jnp.array([{', '.join(residual_resist_exprs)}])")
        lines.append(f"    residuals_react = jnp.array([{', '.join(residual_react_exprs)}])")

        # Jacobian arrays - one entry per (row, col) pair (resistive and reactive)
        jacobian_resist_exprs = []
        jacobian_react_exprs = []
        for entry in self.dae_data['jacobian']:
            resist_var = self._mir_to_code_var(entry['resist_var'])
            react_var = self._mir_to_code_var(entry['react_var'])
            resist_val = resist_var if resist_var in defined_vars else '_ZERO'
            react_val = react_var if react_var in defined_vars else '_ZERO'
            jacobian_resist_exprs.append(resist_val)
            jacobian_react_exprs.append(react_val)
        lines.append(f"    jacobian_resist = jnp.array([{', '.join(jacobian_resist_exprs)}])")
        lines.append(f"    jacobian_react = jnp.array([{', '.join(jacobian_react_exprs)}])")

        lines.append("    return residuals_resist, residuals_react, jacobian_resist, jacobian_react")
        return lines

    def _build_init_param_mapping(self) -> Dict[str, Optional[int]]:
        """Build mapping from init params to eval input indices

        Init params (like R, $temperature, tnom) need to come from the inputs.
        We find which eval params correspond to each init param.

        Important: We match by BOTH name AND kind to handle cases where the
        same name appears multiple times (e.g., 'capacitance' appears once
        with kind='param_given' and once with kind='param').
        """
        mapping = {}

        # Get init param names and kinds
        init_param_names = list(self.module.init_param_names)
        init_param_kinds = list(self.module.init_param_kinds)
        eval_param_names = list(self.module.param_names)
        eval_param_kinds = list(self.module.param_kinds)

        for i, init_name in enumerate(init_param_names):
            # Find matching eval param by both name AND kind
            init_param_val = self.init_params[i] if i < len(self.init_params) else None
            init_kind = init_param_kinds[i] if i < len(init_param_kinds) else None
            if init_param_val and init_kind:
                # Look for this param name AND kind in eval params
                eval_idx = None
                for j, (eval_name, eval_kind) in enumerate(zip(eval_param_names, eval_param_kinds)):
                    if eval_name == init_name and eval_kind == init_kind:
                        eval_idx = j
                        break

                if eval_idx is not None:
                    # Get the eval param's value name
                    eval_param_val = self.params[eval_idx] if eval_idx < len(self.params) else None
                    if eval_param_val:
                        # Map init value name to eval input index
                        mapping[init_param_val] = eval_idx

        return mapping

    def _build_hidden_state_assignments(self, init_defined: Set[str]) -> List[Tuple[str, str]]:
        """Build assignments from init-computed hidden_state values to eval params.

        Hidden_state params are computed by the init function (via CLIP_LOW macros
        that expand to max(value, min_val) patterns) and need to flow to eval.

        Uses multiple matching strategies:
        1. Direct value number match: init_vN for eval vN
        2. Name-based match: MULT_i in eval corresponds to MULT computed in init

        Args:
            init_defined: Set of init variables that have been defined

        Returns:
            List of (eval_var, init_var) tuples for assignments
        """
        assignments = []
        assigned_eval_vars = set()

        # Get metadata
        eval_param_kinds = list(self.module.param_kinds)
        eval_param_names = list(self.module.param_names)
        init_param_names = list(self.module.init_param_names)
        init_param_kinds = list(self.module.init_param_kinds)

        # Strategy 1: Direct value number matching
        for idx, kind in enumerate(eval_param_kinds):
            if kind == 'hidden_state' and idx < len(self.params):
                eval_var = self.params[idx]

                # Try direct value number match: init_v177 for eval v177
                init_var = f"init_{eval_var}"

                if init_var in init_defined:
                    assignments.append((eval_var, init_var))
                    assigned_eval_vars.add(idx)

        value_number_count = len(assignments)

        # Strategy 2: Name-based matching for unassigned hidden_state params
        # Match by stripping _i, _p, _e, _m, _t, _n suffixes
        # Build a map from init param names to their variable names
        init_name_to_var = {}
        for i, init_name in enumerate(init_param_names):
            if i < len(self.init_params):
                init_var = self.init_params[i]
                init_name_lower = init_name.lower()
                # Prefer params with kind='param' over 'param_given'
                if init_name_lower not in init_name_to_var:
                    init_name_to_var[init_name_lower] = f"init_{init_var}"
                elif i < len(init_param_kinds) and init_param_kinds[i] == 'param':
                    # Replace param_given with param
                    init_name_to_var[init_name_lower] = f"init_{init_var}"

        for idx, kind in enumerate(eval_param_kinds):
            if kind == 'hidden_state' and idx < len(self.params) and idx not in assigned_eval_vars:
                eval_var = self.params[idx]
                eval_name = eval_param_names[idx] if idx < len(eval_param_names) else None

                if eval_name:
                    # Strip common suffixes to get base name
                    base_name = eval_name.lower()
                    for suffix in ['_i', '_p', '_e', '_m', '_t', '_n']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break

                    # Look for matching init param
                    init_var = init_name_to_var.get(base_name)
                    if init_var and init_var in init_defined:
                        assignments.append((eval_var, init_var))
                        assigned_eval_vars.add(idx)

        name_match_count = len(assignments) - value_number_count

        # Log statistics
        total_hidden_state = sum(1 for k in eval_param_kinds if k == 'hidden_state')
        logger.info(f"    Hidden state assignments: {len(assignments)}/{total_hidden_state} "
                    f"(value_number={value_number_count}, name_match={name_match_count})")

        return assignments

    def _topological_sort(self) -> List[str]:
        """Sort eval blocks in topological order"""
        return self._topological_sort_blocks(self.mir_data.get('blocks', {}))

    def _topological_sort_init_blocks(self) -> List[str]:
        """Sort init blocks in topological order"""
        return self._topological_sort_blocks(self.init_mir_data.get('blocks', {}))

    def _topological_sort_blocks(self, blocks: Dict) -> List[str]:
        """Sort blocks in topological order using Kahn's algorithm, handling loops.

        Uses Kahn's algorithm to ensure a block is only processed after ALL its
        predecessors have been processed. Loops are collapsed into single nodes.

        Returns a list where each element is either:
        - A block name (str) for non-loop blocks
        - A tuple ('loop', header_block, body_blocks, exit_blocks) for loops
        """
        if not blocks:
            return []

        # First, find loops (SCCs with >1 node)
        loops = self._find_loops(blocks)

        # Create mapping from block to its loop (if any)
        block_to_loop: Dict[str, int] = {}
        for i, loop in enumerate(loops):
            for block in loop:
                block_to_loop[block] = i

        # Build condensed graph:
        # - Each non-loop block is a node
        # - Each loop is collapsed into a single node (represented by its header)
        # The "condensed" node name is either the block name or f"loop_{i}" for loops

        def get_condensed_node(block: str) -> str:
            if block in block_to_loop:
                return f"loop_{block_to_loop[block]}"
            return block

        # Compute in-degrees for condensed graph (Kahn's algorithm)
        in_degree: Dict[str, int] = {}
        condensed_successors: Dict[str, Set[str]] = {}

        # Initialize all nodes
        for block in blocks:
            cn = get_condensed_node(block)
            if cn not in in_degree:
                in_degree[cn] = 0
                condensed_successors[cn] = set()

        # Build edges in condensed graph
        for block, data in blocks.items():
            cn = get_condensed_node(block)
            for succ in data.get('successors', []):
                if succ not in blocks:
                    continue
                succ_cn = get_condensed_node(succ)
                if cn != succ_cn:  # Skip edges within same loop
                    if succ_cn not in condensed_successors[cn]:
                        condensed_successors[cn].add(succ_cn)
                        in_degree[succ_cn] = in_degree.get(succ_cn, 0) + 1

        # Kahn's algorithm: process nodes with in-degree 0
        queue = [cn for cn, deg in in_degree.items() if deg == 0]
        # Sort for deterministic order
        queue.sort(key=lambda x: (
            not x.startswith('loop_'),  # Loops last among same in-degree
            int(x.replace('block', '').replace('loop_', '')) if x.replace('block', '').replace('loop_', '').isdigit() else 0
        ))

        result = []
        processed = set()

        while queue:
            cn = queue.pop(0)
            if cn in processed:
                continue
            processed.add(cn)

            # Add to result
            if cn.startswith('loop_'):
                loop_idx = int(cn.replace('loop_', ''))
                loop = loops[loop_idx]
                header = self._find_loop_header(loop, blocks)
                exits = self._find_loop_exits(loop, blocks)
                result.append(('loop', header, loop, exits))
            else:
                result.append(cn)

            # Decrease in-degree of successors
            for succ_cn in condensed_successors.get(cn, []):
                in_degree[succ_cn] -= 1
                if in_degree[succ_cn] == 0:
                    queue.append(succ_cn)

            # Keep queue sorted for determinism
            queue.sort(key=lambda x: (
                not x.startswith('loop_'),
                int(x.replace('block', '').replace('loop_', '')) if x.replace('block', '').replace('loop_', '').isdigit() else 0
            ))

        # Handle any remaining blocks (shouldn't happen in well-formed CFG)
        for cn in in_degree:
            if cn not in processed:
                if cn.startswith('loop_'):
                    loop_idx = int(cn.replace('loop_', ''))
                    loop = loops[loop_idx]
                    header = self._find_loop_header(loop, blocks)
                    exits = self._find_loop_exits(loop, blocks)
                    result.append(('loop', header, loop, exits))
                else:
                    result.append(cn)

        return result

    def _find_loops(self, blocks: Dict) -> List[Set[str]]:
        """Find all loops (SCCs with >1 node) using iterative Tarjan's algorithm

        Uses an explicit call stack to avoid recursion depth limits on
        complex control flow graphs with many nested blocks.
        """
        index_counter = 0
        tarjan_stack = []  # Stack for SCC detection
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []

        # Iterative DFS using explicit call stack
        # Each stack frame: (node, successor_iter, phase)
        # phase 0: initial visit
        # phase 1: after recursive call returned
        for start in blocks:
            if start in index:
                continue

            # Call stack: (node, iterator over successors, phase, current_child)
            call_stack = [(start, None, 0, None)]

            while call_stack:
                v, succ_iter, phase, child = call_stack.pop()

                if phase == 0:
                    # Initial visit of node v
                    index[v] = index_counter
                    lowlinks[v] = index_counter
                    index_counter += 1
                    tarjan_stack.append(v)
                    on_stack[v] = True

                    # Get successors
                    successors = [w for w in blocks.get(v, {}).get('successors', [])
                                  if w in blocks]
                    succ_iter = iter(successors)

                    # Push back with phase 1 to process successors
                    call_stack.append((v, succ_iter, 1, None))

                elif phase == 1:
                    # After returning from child (if any), update lowlink
                    if child is not None:
                        lowlinks[v] = min(lowlinks[v], lowlinks[child])

                    # Process next successor
                    try:
                        w = next(succ_iter)
                        if w not in index:
                            # Push current frame back to continue later
                            call_stack.append((v, succ_iter, 1, w))
                            # Push new frame to visit w
                            call_stack.append((w, None, 0, None))
                        elif on_stack.get(w, False):
                            lowlinks[v] = min(lowlinks[v], index[w])
                            # Push back to try next successor
                            call_stack.append((v, succ_iter, 1, None))
                        else:
                            # w already processed, try next successor
                            call_stack.append((v, succ_iter, 1, None))
                    except StopIteration:
                        # All successors processed, check if v is root of SCC
                        if lowlinks[v] == index[v]:
                            scc = set()
                            while True:
                                w = tarjan_stack.pop()
                                on_stack[w] = False
                                scc.add(w)
                                if w == v:
                                    break
                            sccs.append(scc)

        # Return only non-trivial SCCs (loops with >1 node)
        return [scc for scc in sccs if len(scc) > 1]

    def _find_loop_header(self, loop: Set[str], blocks: Dict) -> str:
        """Find the loop header (entry point from outside the loop)"""
        for block in loop:
            preds = blocks.get(block, {}).get('predecessors', [])
            for pred in preds:
                if pred not in loop:
                    return block
        # Fallback: return block with lowest number
        return min(loop, key=lambda x: int(x.replace('block', '')) if x.startswith('block') else 0)

    def _find_loop_exits(self, loop: Set[str], blocks: Dict) -> List[str]:
        """Find blocks that exit the loop (successors outside the loop)"""
        exits = []
        for block in loop:
            succs = blocks.get(block, {}).get('successors', [])
            for succ in succs:
                if succ not in loop and succ not in exits:
                    exits.append(succ)
        return exits

    def _generate_init_loop(self, header: str, loop_blocks: Set[str],
                            exit_blocks: List[str], init_by_block: Dict[str, List[dict]],
                            init_defined: Set[str]) -> List[str]:
        """Generate JAX code for a loop using jax.lax.while_loop

        Args:
            header: The loop header block name
            loop_blocks: Set of all blocks in the loop
            exit_blocks: List of blocks that are exited to
            init_by_block: Dict mapping block names to their instructions
            init_defined: Set of already-defined variable names (will be updated)

        Returns:
            List of code lines to add
        """
        lines = []
        lines.append("")
        lines.append(f"    # Loop: {header} with blocks {sorted(loop_blocks)}")

        # Find PHI nodes in the header - these are loop-carried values
        header_insts = init_by_block.get(header, [])
        phi_nodes = [inst for inst in header_insts if inst.get('opcode', '').lower() == 'phi']

        # Get loop-carried variables: PHI results and their incoming values from the loop
        loop_carried = []  # (result_var, init_value, loop_value)
        for phi in phi_nodes:
            result = phi.get('result', '')
            phi_ops = phi.get('phi_operands', [])
            init_val = None
            loop_val = None
            for op in phi_ops:
                if op['block'] in loop_blocks:
                    loop_val = op['value']
                else:
                    init_val = op['value']
            if result and init_val and loop_val:
                loop_carried.append((result, init_val, loop_val))

        if not loop_carried:
            # No loop-carried values - something is wrong
            lines.append("    # WARNING: No loop-carried values found, skipping loop")
            return lines

        # Find the condition and body instructions
        condition_inst = None
        body_insts = []

        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'br' and 'condition' in inst:
                condition_inst = inst
            elif op != 'phi':
                body_insts.append(inst)

        # Also get instructions from other loop blocks (the body)
        for block in sorted(loop_blocks):
            if block != header:
                for inst in init_by_block.get(block, []):
                    op = inst.get('opcode', '').lower()
                    if op not in ('br', 'jmp'):
                        body_insts.append(inst)

        # Helper to get operand with init_ prefix
        def get_operand(op: str, local_vars: Set[str] = None) -> str:
            if local_vars is None:
                local_vars = set()
            # Check local loop variables first (no prefix)
            if op in local_vars:
                return op
            prefixed = f"init_{op}"
            if prefixed in init_defined:
                return prefixed
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return _jax_bool_repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            return prefixed

        # Generate initial state tuple
        init_state_parts = []
        for result, init_val, _ in loop_carried:
            init_state_parts.append(get_operand(init_val))

        lines.append(f"    _loop_state_init = ({', '.join(init_state_parts)},)")

        # Generate condition function
        lines.append("")
        lines.append("    def _loop_cond(_loop_state):")

        # Unpack state
        state_vars = [lc[0] for lc in loop_carried]  # Use original var names inside loop
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        # Generate condition computation (instructions before the branch in header)
        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Return the condition
        if condition_inst:
            cond_var = condition_inst.get('condition', '')
            # Check if condition is in local vars or needs prefix
            if cond_var in local_vars:
                lines.append(f"        return {cond_var}")
            else:
                lines.append(f"        return {get_operand(cond_var, local_vars)}")
        else:
            lines.append("        return False")

        # Generate body function
        lines.append("")
        lines.append("    def _loop_body(_loop_state):")
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        # Recompute header instructions (needed for body)
        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Generate body instructions
        for inst in body_insts:
            result = inst.get('result', '')
            expr = self._translate_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Return new state (the loop_val from each PHI)
        new_state_parts = []
        for _, _, loop_val in loop_carried:
            if loop_val in local_vars:
                new_state_parts.append(loop_val)
            else:
                new_state_parts.append(get_operand(loop_val, local_vars))

        lines.append(f"        return ({', '.join(new_state_parts)},)")

        # Call while_loop
        lines.append("")
        lines.append("    _loop_result = lax.while_loop(_loop_cond, _loop_body, _loop_state_init)")

        # Unpack results to prefixed variables
        for i, (result, _, _) in enumerate(loop_carried):
            prefixed = f"init_{result}"
            lines.append(f"    {prefixed} = _loop_result[{i}]")
            init_defined.add(prefixed)

        return lines

    def _generate_eval_loop(self, header: str, loop_blocks: Set[str],
                            exit_blocks: List[str], eval_by_block: Dict[str, List[dict]],
                            defined_vars: Set[str]) -> List[str]:
        """Generate JAX code for a loop in the eval function using jax.lax.while_loop

        Args:
            header: The loop header block name
            loop_blocks: Set of all blocks in the loop
            exit_blocks: List of blocks that are exited to
            eval_by_block: Dict mapping block names to their instructions
            defined_vars: Set of already-defined variable names (will be updated)

        Returns:
            List of code lines to add
        """
        lines = []
        lines.append("")
        lines.append(f"    # Loop: {header} with blocks {sorted(loop_blocks)}")

        # Find PHI nodes in the header - these are loop-carried values
        header_insts = eval_by_block.get(header, [])
        phi_nodes = [inst for inst in header_insts if inst.get('opcode', '').lower() == 'phi']

        # Get loop-carried variables: PHI results and their incoming values from the loop
        loop_carried = []  # (result_var, init_value, loop_value)
        for phi in phi_nodes:
            result = phi.get('result', '')
            phi_ops = phi.get('phi_operands', [])
            init_val = None
            loop_val = None
            for op in phi_ops:
                if op['block'] in loop_blocks:
                    loop_val = op['value']
                else:
                    init_val = op['value']
            if result and init_val and loop_val:
                loop_carried.append((result, init_val, loop_val))

        if not loop_carried:
            # No loop-carried values - something is wrong
            lines.append("    # WARNING: No loop-carried values found, skipping loop")
            return lines

        # Find the condition and body instructions
        condition_inst = None
        body_insts = []

        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'br' and 'condition' in inst:
                condition_inst = inst
            elif op != 'phi':
                body_insts.append(inst)

        # Also get instructions from other loop blocks (the body)
        for block in sorted(loop_blocks):
            if block != header:
                for inst in eval_by_block.get(block, []):
                    op = inst.get('opcode', '').lower()
                    if op not in ('br', 'jmp'):
                        body_insts.append(inst)

        # Helper to get operand with proper resolution
        def get_operand(op: str, local_vars: Set[str] = None) -> str:
            if local_vars is None:
                local_vars = set()
            # Check local loop variables first (no prefix)
            if op in local_vars:
                return op
            if op in defined_vars:
                return op
            if op in self.constants:
                return repr(self.constants[op])
            return op

        # Generate initial state tuple
        init_state_parts = []
        for result, init_val, _ in loop_carried:
            init_state_parts.append(get_operand(init_val))

        lines.append(f"    _loop_state_init = ({', '.join(init_state_parts)},)")

        # Generate condition function
        lines.append("")
        lines.append("    def _loop_cond(_loop_state):")

        # Unpack state
        state_vars = [lc[0] for lc in loop_carried]  # Use original var names inside loop
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        # Generate condition computation (instructions before the branch in header)
        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_eval_loop_instruction(inst, local_vars, defined_vars)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Return the condition
        if condition_inst:
            cond_var = condition_inst.get('condition', '')
            # Check if condition is in local vars
            if cond_var in local_vars:
                lines.append(f"        return {cond_var}")
            else:
                lines.append(f"        return {get_operand(cond_var, local_vars)}")
        else:
            lines.append("        return False")

        # Generate body function
        lines.append("")
        lines.append("    def _loop_body(_loop_state):")
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        # Recompute header instructions (needed for body)
        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_eval_loop_instruction(inst, local_vars, defined_vars)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Generate body instructions
        for inst in body_insts:
            result = inst.get('result', '')
            expr = self._translate_eval_loop_instruction(inst, local_vars, defined_vars)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Return new state (the loop_val from each PHI)
        new_state_parts = []
        for _, _, loop_val in loop_carried:
            if loop_val in local_vars:
                new_state_parts.append(loop_val)
            else:
                new_state_parts.append(get_operand(loop_val, local_vars))

        lines.append(f"        return ({', '.join(new_state_parts)},)")

        # Call while_loop
        lines.append("")
        lines.append("    _loop_result = lax.while_loop(_loop_cond, _loop_body, _loop_state_init)")

        # Unpack results to variables
        for i, (result, _, _) in enumerate(loop_carried):
            lines.append(f"    {result} = _loop_result[{i}]")
            defined_vars.add(result)

        return lines

    def _translate_eval_loop_instruction(self, inst: dict, local_vars: Set[str],
                                         defined_vars: Set[str]) -> Optional[str]:
        """Translate an instruction for use inside an eval loop

        Args:
            inst: The instruction to translate
            local_vars: Variables defined locally in the loop (no prefix)
            defined_vars: Variables defined in eval scope
        """
        def get_operand(op: str) -> str:
            # Local vars first
            if op in local_vars:
                return op
            if op in defined_vars:
                return op
            if op in self.constants:
                return repr(self.constants[op])
            return op

        return self._translate_instruction_impl(inst, get_operand)

    def _translate_loop_instruction(self, inst: dict, local_vars: Set[str],
                                    init_defined: Set[str]) -> Optional[str]:
        """Translate an instruction for use inside a loop

        Args:
            inst: The instruction to translate
            local_vars: Variables defined locally in the loop (no prefix)
            init_defined: Variables defined in init scope (with prefix)
        """
        def get_operand(op: str) -> str:
            # Local vars first (no prefix)
            if op in local_vars:
                return op
            prefixed = f"init_{op}"
            if prefixed in init_defined:
                return prefixed
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return _jax_bool_repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            return prefixed

        return self._translate_instruction_impl(inst, get_operand)

    def _group_init_instructions_by_block(self) -> Dict[str, List[dict]]:
        """Group init instructions by their block"""
        by_block: Dict[str, List[dict]] = {}
        for inst in self.init_mir_data.get('instructions', []):
            block = inst.get('block', 'block0')
            if block not in by_block:
                by_block[block] = []
            by_block[block].append(inst)
        return by_block

    def _group_eval_instructions_by_block(self) -> Dict[str, List[dict]]:
        """Group eval instructions by their block"""
        by_block: Dict[str, List[dict]] = {}
        for inst in self.mir_data.get('instructions', []):
            block = inst.get('block', 'block0')
            if block not in by_block:
                by_block[block] = []
            by_block[block].append(inst)
        return by_block

    def _build_branch_conditions(self) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Build a map of (block -> successor -> (condition, polarity)) for eval function.

        Results are cached for performance (avoids rebuilding for every PHI node).
        """
        if not hasattr(self, '_cached_branch_conditions'):
            self._cached_branch_conditions = self._build_branch_conditions_impl(
                self.mir_data.get('instructions', [])
            )
        return self._cached_branch_conditions

    def _build_init_branch_conditions(self) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Build a map of (block -> successor -> (condition, polarity)) for init function.

        Results are cached for performance (avoids rebuilding for every PHI node).
        """
        if not hasattr(self, '_cached_init_branch_conditions'):
            self._cached_init_branch_conditions = self._build_branch_conditions_impl(
                self.init_mir_data.get('instructions', [])
            )
        return self._cached_init_branch_conditions

    def _build_branch_conditions_impl(self, instructions: List) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Build a map of (block -> successor -> (condition, polarity))

        For each block with 2 successors, find the condition that determines the branch.
        Returns a dict mapping: block -> {successor: (condition_var, is_true_branch)}
        """
        conditions = {}

        # Find branch instructions with explicit conditions
        for inst in instructions:
            op = inst.get('opcode', '').lower()
            if op == 'br' and 'condition' in inst:
                block = inst.get('block', '')
                cond = inst['condition']
                true_block = inst.get('true_block', '')
                false_block = inst.get('false_block', '')
                if block and cond and true_block and false_block:
                    conditions[block] = {
                        true_block: (cond, True),
                        false_block: (cond, False),
                    }

        return conditions

    def _get_phi_condition(self, phi_block: str, pred_blocks: List[str]) -> Optional[Tuple[str, str, str]]:
        """Get the condition for a PHI node in eval function

        Returns (condition_var, true_value_block, false_value_block) or None
        """
        blocks = self.mir_data.get('blocks', {})
        branch_conds = self._build_branch_conditions()
        return self._get_phi_condition_impl(
            phi_block, pred_blocks, blocks, branch_conds,
            prefix='', succ_pair_map=self._get_eval_succ_pair_map()
        )

    def _get_init_phi_condition(self, phi_block: str, pred_blocks: List[str]) -> Optional[Tuple[str, str, str]]:
        """Get the condition for a PHI node in init function

        Returns (condition_var, true_value_block, false_value_block) or None
        """
        blocks = self.init_mir_data.get('blocks', {})
        branch_conds = self._build_init_branch_conditions()
        return self._get_phi_condition_impl(
            phi_block, pred_blocks, blocks, branch_conds,
            prefix='init_', succ_pair_map=self._get_init_succ_pair_map()
        )

    def _build_succ_pair_map(self, blocks: Dict) -> Dict[frozenset, List[str]]:
        """Build a map from successor pairs to block names.

        This is an optimization to avoid O(n_blocks) iteration in _get_phi_condition_impl.
        Returns a dict mapping frozenset(successors) -> [block_names].
        """
        succ_to_blocks: Dict[frozenset, List[str]] = {}
        for block_name, block_data in blocks.items():
            succs = block_data.get('successors', [])
            if len(succs) == 2:  # Only care about binary branches
                key = frozenset(succs)
                if key not in succ_to_blocks:
                    succ_to_blocks[key] = []
                succ_to_blocks[key].append(block_name)
        return succ_to_blocks

    def _get_phi_condition_impl(self, phi_block: str, pred_blocks: List[str],
                                 blocks: Dict, branch_conds: Dict,
                                 prefix: str = '',
                                 succ_pair_map: Optional[Dict[frozenset, List[str]]] = None) -> Optional[Tuple[str, str, str]]:
        """Implementation of PHI condition finding

        Returns (condition_var, true_value_block, false_value_block) or None
        The condition_var is prefixed with prefix if provided.

        Optimization: Pass succ_pair_map to avoid O(n_blocks) iteration.
        """
        if len(pred_blocks) != 2:
            return None

        pred0, pred1 = pred_blocks

        # Check if either predecessor is the branching block
        if pred0 in branch_conds:
            # pred0 branches to phi_block and pred1
            cond_info = branch_conds[pred0].get(phi_block)
            if cond_info:
                cond_var, is_true = cond_info
                cond_var = f"{prefix}{cond_var}" if prefix else cond_var
                if is_true:
                    return (cond_var, pred0, pred1)
                else:
                    return (cond_var, pred1, pred0)

        if pred1 in branch_conds:
            cond_info = branch_conds[pred1].get(phi_block)
            if cond_info:
                cond_var, is_true = cond_info
                cond_var = f"{prefix}{cond_var}" if prefix else cond_var
                if is_true:
                    return (cond_var, pred1, pred0)
                else:
                    return (cond_var, pred0, pred1)

        # Check if there's a common dominator that branches
        # Look for a block that is predecessor to both pred0 and pred1
        # Optimization: use precomputed succ_pair_map instead of iterating all blocks
        pred_key = frozenset(pred_blocks)
        if succ_pair_map is not None:
            candidate_blocks = succ_pair_map.get(pred_key, [])
        else:
            # Fallback to iteration if map not provided
            candidate_blocks = [
                block_name for block_name, block_data in blocks.items()
                if set(block_data.get('successors', [])) == set(pred_blocks)
            ]

        for block_name in candidate_blocks:
            if block_name in branch_conds:
                cond_info = branch_conds[block_name]
                if pred0 in cond_info and pred1 in cond_info:
                    cond_var, is_true0 = cond_info[pred0]
                    cond_var = f"{prefix}{cond_var}" if prefix else cond_var
                    if is_true0:
                        return (cond_var, pred0, pred1)
                    else:
                        return (cond_var, pred1, pred0)

        return None

    def _build_multi_way_phi(self, phi_block: str, phi_ops: List[dict],
                             val_by_block: Dict[str, str],
                             get_operand: Callable[[str], str]) -> str:
        """Build nested jnp.where for PHI nodes with >2 predecessors

        For MOSFET-style if-elseif-else (cutoff/linear/saturation), we need:
        jnp.where(cutoff_cond, cutoff_val,
                  jnp.where(linear_cond, linear_val, sat_val))

        The structure in the MIR is:
        - block4 branches on cutoff condition (v44) to block5 (cutoff) or block6
        - block6 branches on linear condition (v52) to block8 (linear) or block9 (saturation)
        - block7 has PHI with values from block5, block8, block9
        """
        blocks = self.mir_data.get('blocks', {})
        branch_conds = self._build_branch_conditions()

        # Get predecessor blocks from PHI operands
        pred_blocks = [op['block'] for op in phi_ops]

        # Find the dominating branch structure
        # Look for blocks that branch to our predecessors
        cond_chain = []  # List of (condition_var, true_block, remaining_blocks)

        # Find the first branch point - a block whose successors include one of our preds
        # and another block that eventually leads to other preds
        remaining_preds = set(pred_blocks)

        for block_name, block_data in blocks.items():
            if block_name in branch_conds:
                succs = block_data.get('successors', [])
                # Check if one successor is one of our predecessors
                for succ in succs:
                    if succ in remaining_preds:
                        cond_var, is_true = branch_conds[block_name][succ]
                        other_succs = [s for s in succs if s != succ]
                        if other_succs:
                            # Found a branch where one path leads directly to a PHI pred
                            cond_chain.append((cond_var, succ, other_succs[0], is_true))

        # Build the nested where expression
        # For MOSFET: v44 determines cutoff (block5) vs not-cutoff (block6)
        #            v52 determines linear (block8) vs saturation (block9)

        # Sort cond_chain by the block numbers to get consistent ordering
        # We want outer conditions first (lower block numbers typically)

        if not cond_chain:
            # Fallback: couldn't figure out the condition structure
            # Just return the first value
            return val_by_block.get(pred_blocks[0], '_ZERO')

        # Build the expression from the condition chain
        # For a 3-way PHI (cutoff/linear/sat):
        # cond_chain might be: [(v44, block5, block6, True), (v52, block8, block9, True)]
        # We want: where(v44, val_block5, where(v52, val_block8, val_block9))

        # Find which pred each condition leads to directly
        # and build the nested where accordingly
        pred_to_val = val_by_block

        # Trace through the condition chain to build the nested where
        # Start with the blocks that are direct successors of branch points
        result = self._build_nested_where_from_blocks(
            phi_block, pred_blocks, pred_to_val, blocks, branch_conds
        )

        return result if result else val_by_block.get(pred_blocks[0], '_ZERO')

    def _build_nested_where_from_blocks(self, phi_block: str, pred_blocks: List[str],
                                         pred_to_val: Dict[str, str], blocks: Dict,
                                         branch_conds: Dict) -> Optional[str]:
        """Recursively build nested jnp.where from CFG structure

        This traces backwards from the PHI block to find the conditions.
        """
        if len(pred_blocks) == 1:
            return pred_to_val.get(pred_blocks[0], '_ZERO')

        # Optimization: If some predecessors have the same value, group them
        # This handles cases like PSP103's current paths where multiple blocks
        # contribute 0 and only one block contributes the actual current.
        val_to_preds: Dict[str, List[str]] = {}
        for pred in pred_blocks:
            val = pred_to_val.get(pred, '_ZERO')
            if val not in val_to_preds:
                val_to_preds[val] = []
            val_to_preds[val].append(pred)

        # If we have exactly 2 unique values, simplify the problem
        unique_vals = list(val_to_preds.keys())
        if len(unique_vals) == 2:
            # Find which conditions lead to each value
            val_a, val_b = unique_vals
            preds_a = val_to_preds[val_a]
            preds_b = val_to_preds[val_b]

            # Try to find a condition that separates preds_a from preds_b
            # by checking if there's a common branching point
            cond_expr = self._find_condition_for_pred_groups(
                phi_block, preds_a, preds_b, blocks, branch_conds
            )
            if cond_expr:
                # cond_expr is True when we reach preds_a
                return f"jnp.where({cond_expr}, {val_a}, {val_b})"

        if len(pred_blocks) == 2:
            # Base case: 2 predecessors, find the branching condition
            pred0, pred1 = pred_blocks
            cond_info = self._get_phi_condition(phi_block, pred_blocks)
            if cond_info:
                cond_var, true_block, false_block = cond_info
                true_val = pred_to_val.get(true_block, '_ZERO')
                false_val = pred_to_val.get(false_block, '_ZERO')
                return f"jnp.where({cond_var}, {true_val}, {false_val})"
            else:
                return pred_to_val.get(pred_blocks[0], '_ZERO')

        # For >2 predecessors, find a block that branches to one of them
        # and another block that leads to the rest
        for block_name in blocks:
            if block_name not in branch_conds:
                continue

            succs = blocks[block_name].get('successors', [])
            if len(succs) != 2:
                continue

            # Check if one successor is in our pred_blocks
            direct_pred = None
            indirect_succ = None
            for succ in succs:
                if succ in pred_blocks:
                    direct_pred = succ
                else:
                    # Check if this successor eventually leads to other preds
                    # (it could be an intermediate block that branches further)
                    indirect_succ = succ

            if direct_pred and indirect_succ:
                # Found a branch: one path goes directly to a PHI pred,
                # other path goes to an intermediate block
                cond_var, is_true = branch_conds[block_name][direct_pred]

                # Get the value for the direct path
                direct_val = pred_to_val.get(direct_pred, '_ZERO')

                # Skip branches where direct_val is a constant zero
                # (see comment in _build_init_nested_where_from_blocks for details)
                is_const_zero = (
                    direct_val == '_ZERO' or
                    direct_val == '0.0' or
                    direct_val == '0'
                )
                # Also check if it's a variable pointing to a constant 0
                if not is_const_zero:
                    if direct_val in self.constants:
                        is_const_zero = (self.constants[direct_val] == 0.0)
                    elif direct_val in self.int_constants:
                        is_const_zero = (self.int_constants[direct_val] == 0)

                if is_const_zero:
                    continue  # Skip this branch, look for a better one

                # Find remaining preds (those not reached by direct path)
                remaining_preds = [p for p in pred_blocks if p != direct_pred]

                # Recursively build where for remaining preds
                # The indirect_succ should eventually lead to remaining_preds
                remaining_expr = self._build_nested_where_from_blocks(
                    phi_block, remaining_preds, pred_to_val, blocks, branch_conds
                )

                if remaining_expr:
                    if is_true:
                        return f"jnp.where({cond_var}, {direct_val}, {remaining_expr})"
                    else:
                        return f"jnp.where({cond_var}, {remaining_expr}, {direct_val})"

        # Fallback: couldn't build the expression
        return None

    def _find_condition_for_pred_groups(self, phi_block: str, preds_a: List[str],
                                        preds_b: List[str], blocks: Dict,
                                        branch_conds: Dict) -> Optional[str]:
        """Find condition expression that is True when reaching preds_a, False for preds_b.

        This handles complex CFG patterns like PSP103's channel current computation:
        - Multiple paths (preds_a) lead to one value (e.g., 0 for no current)
        - Single path (preds_b) leads to another value (e.g., actual current)
        - The condition is a conjunction of multiple branch conditions

        Returns a condition expression string like "v969558 & v972835" or None if not found.
        """
        # Build reachability sets - which blocks can reach preds_a vs preds_b?
        # A decision point is a block where one successor leads to preds_b
        # and another leads to preds_a (or is on a path to preds_a).

        # Build reachability from each phi predecessor
        def get_all_predecessors(start_blocks: List[str], max_depth: int = 20) -> Set[str]:
            """Get all blocks that can reach the start blocks."""
            result = set(start_blocks)
            frontier = list(start_blocks)
            for _ in range(max_depth):
                if not frontier:
                    break
                new_frontier = []
                for block in frontier:
                    for pred in blocks.get(block, {}).get('predecessors', []):
                        if pred not in result:
                            result.add(pred)
                            new_frontier.append(pred)
                frontier = new_frontier
            return result

        # Blocks that can reach preds_a (the default value paths)
        reaches_a = get_all_predecessors(preds_a)
        # Blocks that can reach preds_b (the special value paths)
        reaches_b = get_all_predecessors(preds_b)

        # Find decision points: blocks that branch with one successor
        # leading to preds_b and another leading to preds_a
        conditions = []

        for block_name in branch_conds:
            succs = blocks.get(block_name, {}).get('successors', [])
            if len(succs) != 2:
                continue

            succ0, succ1 = succs

            # Check if this is a decision point between paths to a vs b
            # We want blocks where one path leads ONLY to preds_b (or its chain)
            # and another path leads to preds_a

            # Check both orderings
            for target_succ, default_succ in [(succ0, succ1), (succ1, succ0)]:
                # Does target_succ lead to preds_b?
                target_reaches_b = (target_succ in reaches_b or target_succ in preds_b)
                # Does default_succ lead to preds_a (but NOT to preds_b)?
                default_reaches_a = (default_succ in reaches_a or default_succ in preds_a)
                default_not_reaches_b = (default_succ not in reaches_b and default_succ not in preds_b)

                if target_reaches_b and default_reaches_a and default_not_reaches_b:
                    # This block is a decision point!
                    cond_info = branch_conds[block_name].get(target_succ)
                    if cond_info:
                        cond_var, is_true = cond_info
                        if is_true:
                            conditions.append(cond_var)
                        else:
                            conditions.append(f"(~{cond_var})")
                    break  # Found the decision for this block

        if conditions:
            # Deduplicate and combine
            unique_conds = list(dict.fromkeys(conditions))  # Preserve order, remove dups
            if len(unique_conds) == 1:
                return unique_conds[0]
            else:
                return f"({' & '.join(unique_conds)})"

        return None

    def _build_init_multi_way_phi(self, phi_block: str, phi_ops: List[dict],
                                   val_by_block: Dict[str, str],
                                   get_operand: Callable[[str], str]) -> str:
        """Build nested jnp.where for PHI nodes with >2 predecessors in init function

        Mirrors _build_multi_way_phi but uses init-specific data.
        """
        blocks = self.init_mir_data.get('blocks', {})
        branch_conds = self._build_init_branch_conditions()

        # Get predecessor blocks from PHI operands
        pred_blocks = [op['block'] for op in phi_ops]
        pred_to_val = val_by_block

        # logger.debug(f"_build_init_multi_way_phi: phi_block={phi_block}, num_preds={len(pred_blocks)}")

        # Trace through the condition chain to build the nested where
        result = self._build_init_nested_where_from_blocks(
            phi_block, pred_blocks, pred_to_val, blocks, branch_conds, depth=0
        )

        # logger.debug(f"_build_init_multi_way_phi: done, result={'ok' if result else 'fallback'}")

        return result if result else val_by_block.get(pred_blocks[0], '_ZERO')

    def _build_init_nested_where_from_blocks(self, phi_block: str, pred_blocks: List[str],
                                              pred_to_val: Dict[str, str], blocks: Dict,
                                              branch_conds: Dict, depth: int = 0) -> Optional[str]:
        """Recursively build nested jnp.where from CFG structure for init function

        Mirrors _build_nested_where_from_blocks but uses init-specific condition finding.

        IMPORTANT: This uses a greedy approach - once we find a valid branch point,
        we commit to it. We do NOT backtrack and try other orderings, as that would
        be O(n!) complexity. For complex PHI nodes where the greedy approach fails,
        we fall back to using the first predecessor's value.
        """
        MAX_DEPTH = 100  # Safety limit to prevent infinite recursion

        if depth > MAX_DEPTH:
            logger.warning(f"_build_init_nested_where: MAX_DEPTH exceeded at depth={depth}")
            return None

        # if depth == 0:
        #     logger.debug(f"_build_init_nested_where: phi_block={phi_block}, num_preds={len(pred_blocks)}")

        if len(pred_blocks) == 1:
            return pred_to_val.get(pred_blocks[0], '_ZERO')

        if len(pred_blocks) == 2:
            # Base case: 2 predecessors, find the branching condition
            cond_info = self._get_init_phi_condition(phi_block, pred_blocks)
            if cond_info:
                cond_var, true_block, false_block = cond_info
                true_val = pred_to_val.get(true_block, '_ZERO')
                false_val = pred_to_val.get(false_block, '_ZERO')
                return f"jnp.where({cond_var}, {true_val}, {false_val})"
            else:
                return pred_to_val.get(pred_blocks[0], '_ZERO')

        # For >2 predecessors, find a block that branches to one of them
        # and another block that leads to the rest
        for block_name in blocks:
            if block_name not in branch_conds:
                continue

            succs = blocks[block_name].get('successors', [])
            if len(succs) != 2:
                continue

            # Check if one successor is in our pred_blocks
            direct_pred = None
            indirect_succ = None
            for succ in succs:
                if succ in pred_blocks:
                    direct_pred = succ
                else:
                    indirect_succ = succ

            if direct_pred and indirect_succ:
                # Found a branch: one path goes directly to a PHI pred
                cond_var, is_true = branch_conds[block_name][direct_pred]
                # Prefix the condition variable for init function
                cond_var = f"init_{cond_var}"

                # Get the value for the direct path
                direct_val = pred_to_val.get(direct_pred, '_ZERO')

                # Skip branches where direct_val is a constant zero (_ZERO or literal 0.0)
                # These are typically "default" paths that don't contribute meaningful values,
                # and we want to find a branch point that captures the actual computation.
                # This fixes the PSP103 TYPE conditional issue where SWGEO_i was incorrectly
                # wrapped in a TYPE check, yielding 0 for NMOS instead of the computed value.
                #
                # Check for constant zero: either '_ZERO', literal '0.0'/'0', or a variable
                # that's defined as init_v<N> where v<N> is a known zero constant.
                is_const_zero = (
                    direct_val == '_ZERO' or
                    direct_val == '0.0' or
                    direct_val == '0'
                )
                # Also check if it's a variable pointing to a constant 0
                if not is_const_zero and direct_val.startswith('init_'):
                    var_name = direct_val[5:]  # Remove 'init_' prefix
                    if var_name in self.init_constants:
                        const_val = self.init_constants[var_name]
                        is_const_zero = (const_val == 0.0)
                    elif var_name in self.init_int_constants:
                        const_val = self.init_int_constants[var_name]
                        is_const_zero = (const_val == 0)

                if is_const_zero:
                    continue  # Skip this branch, look for a better one

                # Find remaining preds (those not reached by direct path)
                remaining_preds = [p for p in pred_blocks if p != direct_pred]

                # logger.debug(f"_build_init_nested_where: depth={depth}, removed {direct_pred}, remaining={len(remaining_preds)}")

                # Recursively build where for remaining preds
                # GREEDY: We commit to this choice and don't backtrack
                remaining_expr = self._build_init_nested_where_from_blocks(
                    phi_block, remaining_preds, pred_to_val, blocks, branch_conds, depth + 1
                )

                # Even if remaining_expr is None, we still return here to avoid
                # O(n!) backtracking. Use the first remaining pred as fallback.
                if remaining_expr is None:
                    remaining_expr = pred_to_val.get(remaining_preds[0], '_ZERO')

                if is_true:
                    return f"jnp.where({cond_var}, {direct_val}, {remaining_expr})"
                else:
                    return f"jnp.where({cond_var}, {remaining_expr}, {direct_val})"

        # Fallback: couldn't find any branch point
        # logger.debug(f"_build_init_nested_where: fallback at depth={depth}, pred_blocks={pred_blocks}")
        return None

    def _translate_init_instruction(self, inst: dict, defined_vars: Set[str]) -> Optional[str]:
        """Translate an init function instruction with prefixed variables"""

        def get_operand(op: str) -> str:
            prefixed = f"init_{op}"
            if prefixed in defined_vars:
                return prefixed
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return _jax_bool_repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            # Fallback to prefixed anyway
            return prefixed

        # Handle PHI nodes specially for init function
        opcode = inst.get('opcode', '').lower()
        if opcode == 'phi':
            phi_ops = inst.get('phi_operands', [])
            phi_block = inst.get('block', '')
            if phi_ops and len(phi_ops) >= 2:
                # Get the predecessor blocks from PHI operands
                pred_blocks = [op['block'] for op in phi_ops]
                val_by_block = {op['block']: get_operand(op['value']) for op in phi_ops}

                # For PHIs with more than 2 predecessors, use multi-way PHI builder
                if len(pred_blocks) > 2:
                    return self._build_init_multi_way_phi(phi_block, phi_ops, val_by_block, get_operand)

                # Try to find the condition that determines the branch (using init version)
                cond_info = self._get_init_phi_condition(phi_block, pred_blocks)
                if cond_info:
                    cond_var, true_block, false_block = cond_info
                    true_val = val_by_block.get(true_block, '_ZERO')
                    false_val = val_by_block.get(false_block, '_ZERO')
                    return f"jnp.where({cond_var}, {true_val}, {false_val})"
                else:
                    # Fallback: just use first value (may be incorrect)
                    val0 = get_operand(phi_ops[0]['value'])
                    return val0
            elif phi_ops:
                return get_operand(phi_ops[0]['value'])
            return '_ZERO'

        return self._translate_instruction_impl(inst, get_operand)

    def _translate_instruction(self, inst: dict, defined_vars: Set[str]) -> Optional[str]:
        """Translate a single instruction to JAX expression"""

        def get_operand(op: str) -> str:
            if op in defined_vars:
                return op
            if op in self.constants:
                return repr(self.constants[op])
            return op

        return self._translate_instruction_impl(inst, get_operand)

    def _translate_instruction_impl(self, inst: dict, get_operand: Callable[[str], str]) -> Optional[str]:
        """Implementation of instruction translation with custom operand resolver.

        Uses class-level lookup tables for fast opcode dispatch (Optimization 1).
        """
        opcode = inst.get('opcode', '').lower()
        operands = inst.get('operands', [])

        # Fast path: binary arithmetic ops (fadd, fsub, fmul, iadd, isub, imul, etc.)
        if opcode in self._BINARY_ARITH_OPS:
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} {self._BINARY_ARITH_OPS[opcode]} {ops[1]})"

        # Fast path: comparison ops (feq, flt, fgt, ieq, ilt, etc.)
        if opcode in self._COMPARE_OPS:
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} {self._COMPARE_OPS[opcode]} {ops[1]})"

        # Safe sqrt: clamp negative inputs to zero to avoid NaN
        if opcode == 'sqrt':
            ops = [get_operand(op) for op in operands]
            return f"jnp.sqrt(jnp.maximum({ops[0]}, _ZERO))"

        # Safe ln (log): clamp non-positive inputs to small positive value to avoid NaN/inf
        if opcode == 'ln':
            ops = [get_operand(op) for op in operands]
            return f"jnp.log(jnp.maximum({ops[0]}, jnp.float64(1e-300)))"

        # Fast path: unary jnp functions with same name (exp, sin, cos, etc.)
        if opcode in self._UNARY_JNP_SAME:
            ops = [get_operand(op) for op in operands]
            return f"jnp.{opcode}({ops[0]})"

        # Fast path: unary jnp functions with different name (ln->log, asin->arcsin, etc.)
        if opcode in self._UNARY_JNP_MAP:
            ops = [get_operand(op) for op in operands]
            return f"jnp.{self._UNARY_JNP_MAP[opcode]}({ops[0]})"

        # Fast path: binary jnp functions (pow->power, atan2->arctan2)
        if opcode in self._BINARY_JNP_MAP:
            ops = [get_operand(op) for op in operands]
            return f"jnp.{self._BINARY_JNP_MAP[opcode]}({ops[0]}, {ops[1]})"

        # Special cases that need custom handling
        if opcode == 'fdiv':
            ops = [get_operand(op) for op in operands]
            # Use safe division to avoid div-by-zero when conditionals are linearized
            return f"jnp.where({ops[1]} == _ZERO, _ZERO, {ops[0]} / jnp.where({ops[1]} == _ZERO, _ONE, {ops[1]}))"

        if opcode == 'fneg' or opcode == 'ineg':
            ops = [get_operand(op) for op in operands]
            return f"(-{ops[0]})" if ops else ('_ZERO' if opcode == 'fneg' else '0')

        if opcode == 'optbarrier':
            # Optimization barrier - just pass through
            ops = [get_operand(op) for op in operands]
            return ops[0] if ops else '_ZERO'

        elif opcode == 'phi':
            # PHI node - select value based on control flow
            phi_ops = inst.get('phi_operands', [])
            phi_block = inst.get('block', '')
            if phi_ops and len(phi_ops) >= 2:
                # Get the predecessor blocks from PHI operands
                pred_blocks = [op['block'] for op in phi_ops]
                val_by_block = {op['block']: get_operand(op['value']) for op in phi_ops}

                # For PHIs with more than 2 predecessors, we need to chain jnp.where calls
                # This happens with if-elseif-else patterns (e.g., cutoff/linear/saturation)
                if len(pred_blocks) > 2:
                    # Need to find ALL branch conditions to build nested where
                    # Build a chain: where(cond1, val1, where(cond2, val2, val3))
                    return self._build_multi_way_phi(phi_block, phi_ops, val_by_block, get_operand)

                # Try to find the condition that determines the branch
                cond_info = self._get_phi_condition(phi_block, pred_blocks)
                if cond_info:
                    cond_var, true_block, false_block = cond_info
                    true_val = val_by_block.get(true_block, '_ZERO')
                    false_val = val_by_block.get(false_block, '_ZERO')
                    return f"jnp.where({cond_var}, {true_val}, {false_val})"
                else:
                    # Fallback: just use first value (may be incorrect)
                    val0 = get_operand(phi_ops[0]['value'])
                    return val0
            elif phi_ops:
                return get_operand(phi_ops[0]['value'])
            elif operands:
                return get_operand(operands[0])
            return '_ZERO'

        elif opcode == 'call':
            # Function call - handle known functions
            func_ref = inst.get('func_ref', '')
            func_decls = self.mir_data.get('function_decls', {})

            if func_ref in func_decls:
                fn_name = func_decls[func_ref].get('name', '')

                if 'simparam' in fn_name.lower():
                    # $simparam("name", default)
                    # Get the parameter name from first operand (string constant)
                    param_name = None
                    if len(operands) >= 1:
                        first_operand = operands[0]
                        # Look up the actual string value from str_constants
                        param_name = self.str_constants.get(first_operand, None)

                    if param_name == "gmin":
                        # gmin is passed as the LAST element in inputs array
                        # Mark that this model uses $simparam("gmin")
                        self.uses_simparam_gmin = True
                        return 'inputs[-1]'
                    else:
                        # Other simparams: return the default value
                        if len(operands) >= 2:
                            return get_operand(operands[1])
                        return '_ZERO'  # Default for unknown simparams

                elif fn_name == 'Analysis':
                    # analysis("type") - returns 1 if current analysis matches
                    # Analysis type is passed as inputs[-2] (before gmin at inputs[-1])
                    # Encoding: 0=dc/static, 1=ac, 2=tran, 3=noise, 4=nodeset
                    self.uses_analysis = True
                    if len(operands) >= 1:
                        first_operand = operands[0]
                        analysis_type_str = self.str_constants.get(first_operand, '')
                        if analysis_type_str:
                            # Get the integer code for this analysis type
                            type_code = self.analysis_type_map.get(analysis_type_str.lower(), -1)
                            if type_code >= 0:
                                # Compare against inputs[-2] (analysis_type parameter)
                                return f'(inputs[-2] == {type_code})'
                    # Unknown analysis type - return False
                    return _jax_bool_repr(False)

                elif 'ddt' in fn_name.lower() or 'TimeDerivative' in fn_name:
                    # Time derivative - return the charge expression Q
                    # The transient solver computes dQ/dt = (Q - Q_prev) / dt
                    if operands:
                        return get_operand(operands[0])
                    return '_ZERO'

                elif 'ddx' in fn_name.lower() or 'NodeDerivative' in fn_name:
                    # Derivative with respect to a variable - return 0 for now
                    return '_ZERO'

                elif 'noise' in fn_name.lower():
                    # Noise functions - return 0 for DC analysis
                    return '_ZERO'

                elif 'collapse' in fn_name.lower():
                    # Node collapsing - side effect only
                    return None

            # Unknown function - return 0
            return '_ZERO'

        elif opcode == 'ifcast':
            # Integer to float cast - just pass through the operand
            # In JAX, integers and floats are often compatible
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"jnp.float64({ops[0]})"
            return '_ZERO'

        elif opcode == 'ibcast':
            # Integer to bool cast - check if non-zero
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"({ops[0]} != 0)"
            return _jax_bool_repr(False)

        elif opcode == 'ficast':
            # Float to integer cast
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"jnp.int32({ops[0]})"
            return '0'

        elif opcode == 'fbcast':
            # Float to bool cast - check if non-zero
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"({ops[0]} != _ZERO)"
            return _jax_bool_repr(False)

        # Note: irem, idiv, ige, igt, ile, ilt, ieq, ine, fne, beq, bne,
        # iand, ior, ixor, iadd, isub, imul are now handled by lookup tables above

        if opcode == 'bnot':
            # Boolean not - use jnp.logical_not for JIT compatibility
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"jnp.logical_not({ops[0]})"
            return _jax_bool_repr(True)

        elif opcode == 'bicast':
            # Bool to int cast
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"jnp.int32({ops[0]})"
            return '0'

        elif opcode in ('br', 'jmp', 'exit'):
            # Control flow - handled at block level
            return None

        return None


    def get_parameter_info(self) -> List[Tuple[str, str, str]]:
        """Get parameter information

        Returns:
            List of (value_name, param_name, param_kind) tuples
        """
        result = []
        for i, param in enumerate(self.params):
            if i < len(self.module.param_names):
                result.append((param, self.module.param_names[i], self.module.param_kinds[i]))
            else:
                result.append((param, f"cached_{i}", "cached"))
        return result

    def get_generated_code(self) -> str:
        """Get the generated JAX code as a string"""
        return '\n'.join(self._generate_code())


def compile_va(va_path: str) -> CompiledDevice:
    """Compile a Verilog-A file to a JAX-compatible device

    Args:
        va_path: Path to the .va file

    Returns:
        CompiledDevice with eval function and metadata
    """
    translator = OpenVAFToJAX.from_file(va_path)
    eval_fn = translator.translate()

    return CompiledDevice(
        name=va_path,
        module_name=translator.module.name,
        nodes=translator.module.nodes,
        param_names=translator.module.param_names,
        param_kinds=translator.module.param_kinds,
        eval_fn=eval_fn,
        num_residuals=translator.module.num_residuals,
        num_jacobian=translator.module.num_jacobian,
    )


if __name__ == "__main__":
    import numpy as np

    # Test with resistor
    print("="*60)
    print("Testing OpenVAF to JAX translator with resistor")
    print("="*60)

    translator = OpenVAFToJAX.from_file(
        "/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF/integration_tests/RESISTOR/resistor.va"
    )

    print("\nParameter mapping:")
    for val_name, param_name, kind in translator.get_parameter_info():
        print(f"  {val_name} -> {param_name} ({kind})")

    print("\nGenerated JAX code:")
    print("-"*60)
    print(translator.get_generated_code())
    print("-"*60)

    # Compile and test
    eval_fn = translator.translate()

    # Build inputs array
    # v16=V(A,B), v17=vres, v18=R, v19=$temp, v20=tnom, v22=zeta, v25=res, v28=mfactor
    V = 1.0
    R = 1000.0
    inputs = [
        V,      # v16 = V(A,B)
        V,      # v17 = vres (hidden state = V)
        R,      # v18 = R
        300.15, # v19 = $temperature
        300.0,  # v20 = tnom
        0.0,    # v22 = zeta
        R,      # v25 = res (hidden state = R)
        1.0,    # v28 = mfactor
    ]

    residuals, jacobian = eval_fn(inputs)

    print(f"\nInputs: V={V}, R={R}")
    print(f"Residuals: {residuals}")
    print(f"Expected: I = V/R = {V/R}")

    # Test with diode
    print("\n" + "="*60)
    print("Testing with diode")
    print("="*60)

    translator = OpenVAFToJAX.from_file(
        "/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF/integration_tests/DIODE/diode.va"
    )

    print(f"\nModule: {translator.module.name}")
    print(f"Parameters: {len(translator.params)}")
    print(f"Constants: {len(translator.constants)}")
    print(f"Blocks: {len(translator.mir_data['blocks'])}")

    eval_fn = translator.translate()

    # For diode, need to set many hidden states
    # This is complex - the translator generates code but hidden states must be pre-computed
    print("\nNote: Diode evaluation requires pre-computed hidden states")
    print("The JAX code is generated but evaluation needs proper setup")
