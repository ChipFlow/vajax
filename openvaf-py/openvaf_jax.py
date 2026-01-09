"""OpenVAF to JAX translator

Translates OpenVAF MIR to JAX-compatible functions for JIT compilation.
Uses the MIR interpreter for init (one-time setup) and generates
traced JAX code for eval (hot path).
"""

import hashlib
import logging
import math
from typing import Any, Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import lax

logger = logging.getLogger("jax_spice.openvaf")

# Module-level caches for function reuse
_vmapped_jit_cache: Dict[Tuple[str, Tuple], Callable] = {}


def get_vmapped_jit(code_hash: str, fn: Callable, in_axes: Tuple) -> Callable:
    """Get a cached vmapped+jit'd version of a function.

    This caches the entire jax.jit(jax.vmap(fn, in_axes=in_axes)) result,
    avoiding repeated JIT compilation for the same function.

    Args:
        code_hash: Hash identifying the function
        fn: The function to vmap and jit
        in_axes: vmap in_axes specification

    Returns:
        vmapped and jit'd function
    """
    cache_key = (code_hash, in_axes)

    if cache_key in _vmapped_jit_cache:
        logger.debug(f"    vmapped_jit: using cached (hash={code_hash[:8]}, in_axes={in_axes})")
        return _vmapped_jit_cache[cache_key]

    vmapped_jit_fn = jax.jit(jax.vmap(fn, in_axes=in_axes))
    _vmapped_jit_cache[cache_key] = vmapped_jit_fn
    logger.debug(f"    vmapped_jit: cached new (hash={code_hash[:8]}, in_axes={in_axes})")
    return vmapped_jit_fn


def clear_cache():
    """Clear all function caches."""
    global _vmapped_jit_cache
    _vmapped_jit_cache.clear()


def cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        'vmapped_jit_count': len(_vmapped_jit_cache),
    }


# JAX-compatible MIR operations
def jax_fadd(a, b):
    return a + b

def jax_fsub(a, b):
    return a - b

def jax_fmul(a, b):
    return a * b

def jax_fdiv(a, b):
    return jnp.where(b != 0, a / b, jnp.where(a > 0, jnp.inf, jnp.where(a < 0, -jnp.inf, jnp.nan)))

def jax_fneg(a):
    return -a

def jax_flt(a, b):
    return a < b

def jax_fle(a, b):
    return a <= b

def jax_fgt(a, b):
    return a > b

def jax_fge(a, b):
    return a >= b

def jax_feq(a, b):
    return a == b

def jax_fne(a, b):
    return a != b

def jax_pow(base, exp):
    return jnp.power(base, exp)

def jax_ln(x):
    return jnp.log(jnp.maximum(x, 1e-300))  # Safe log

def jax_exp(x):
    return jnp.exp(jnp.clip(x, -700, 700))  # Safe exp

def jax_sqrt(x):
    return jnp.sqrt(jnp.maximum(x, 0.0))

def jax_sin(x):
    return jnp.sin(x)

def jax_cos(x):
    return jnp.cos(x)

def jax_tan(x):
    return jnp.tan(x)

def jax_atan(x):
    return jnp.arctan(x)

def jax_atan2(y, x):
    return jnp.arctan2(y, x)

def jax_abs(x):
    return jnp.abs(x)

def jax_min(a, b):
    return jnp.minimum(a, b)

def jax_max(a, b):
    return jnp.maximum(a, b)

def jax_ifcast(x):
    return jnp.float64(x)

def jax_optbarrier(x):
    return x


JAX_OPCODE_MAP = {
    'fadd': jax_fadd,
    'fsub': jax_fsub,
    'fmul': jax_fmul,
    'fdiv': jax_fdiv,
    'fneg': jax_fneg,
    'flt': jax_flt,
    'fle': jax_fle,
    'fgt': jax_fgt,
    'fge': jax_fge,
    'feq': jax_feq,
    'fne': jax_fne,
    'pow': jax_pow,
    'ln': jax_ln,
    'exp': jax_exp,
    'sqrt': jax_sqrt,
    'sin': jax_sin,
    'cos': jax_cos,
    'tan': jax_tan,
    'atan': jax_atan,
    'atan2': jax_atan2,
    'abs': jax_abs,
    'min': jax_min,
    'max': jax_max,
    'ifcast': jax_ifcast,
    'optbarrier': jax_optbarrier,
}


@dataclass
class OpenVAFToJAX:
    """Translator from OpenVAF MIR to JAX functions.

    This class provides:
    - init_fn: Python function to compute cache values (run once at setup)
    - eval_fn: JAX-traced function for model evaluation (JIT-compilable)
    """

    module: Any
    dae_data: Dict = None
    _init_fn: Callable = None
    _eval_fn: Callable = None
    _cache_size: int = 0
    _metadata: Dict = None
    _init_mir: Dict = None

    # Feature tracking (for engine compatibility)
    uses_simparam_gmin: bool = False
    uses_analysis: bool = False
    analysis_type_map: Dict = None

    def __post_init__(self):
        """Initialize data from module."""
        self.dae_data = self.module.get_dae_system()
        self._cache_size = self.module.num_cached_values
        self._metadata = self.module.get_codegen_metadata()
        self._init_mir = self.module.get_init_mir_instructions()

        # Default analysis type map
        if self.analysis_type_map is None:
            self.analysis_type_map = {
                'dc': 0, 'static': 0,
                'ac': 1,
                'tran': 2, 'transient': 2,
                'noise': 3,
                'nodeset': 4,
            }

    def get_dae_metadata(self) -> Dict:
        """Get DAE system metadata.

        Returns:
            Dict with node_names, jacobian_keys, terminals, internal_nodes, etc.
        """
        residuals = self.dae_data.get('residuals', [])
        jacobian = self.dae_data.get('jacobian', [])

        return {
            'node_names': [r.get('node_name', f'node{i}') for i, r in enumerate(residuals)],
            'jacobian_keys': [
                (j.get('row_node_name', ''), j.get('col_node_name', ''))
                for j in jacobian
            ],
            'terminals': self.dae_data.get('terminals', []),
            'internal_nodes': self.dae_data.get('internal_nodes', []),
            'num_terminals': self.dae_data.get('num_terminals', 0),
            'num_internal': self.dae_data.get('num_internal', 0),
        }

    def translate_init_array(self) -> Tuple[Callable, Dict]:
        """Generate a vmappable init function.

        Returns:
            Tuple of (init_fn, metadata) where:
            - init_fn takes an input array and returns (cache_array, collapse_decisions)
            - metadata contains param_names, param_kinds, cache_size, etc.
        """
        self._build_init_fn()

        # Get init param info from module
        init_param_names = list(self.module.init_param_names)
        init_param_kinds = list(self.module.init_param_kinds)
        cache_mapping = list(self._init_mir.get('cache_mapping', []))
        collapsible_pairs = list(self.module.collapsible_pairs)
        collapse_outputs = list(self.module.collapse_decision_outputs)

        # Wrap the Python init_fn to work with arrays
        python_init_fn = self._init_fn
        n_cache = self._cache_size
        n_collapse = max(len(collapse_outputs), 1)

        # Create a pure callback version for vmap compatibility
        def _python_init_impl(inputs_np):
            """Pure Python implementation for callback."""
            import numpy as np

            # Get dtype from input
            out_dtype = inputs_np.dtype

            inputs_list = inputs_np.tolist() if hasattr(inputs_np, 'tolist') else list(inputs_np)

            # Build kwargs from input array
            # Use param_kinds to detect param_given types (add _given suffix)
            kwargs = {}
            for i, (name, kind) in enumerate(zip(init_param_names, init_param_kinds)):
                if i < len(inputs_list):
                    val = inputs_list[i]
                    # Handle param_given kind as booleans with _given suffix
                    if kind == 'param_given':
                        actual_name = f"{name}_given"
                        kwargs[actual_name] = val != 0.0
                    else:
                        kwargs[name] = val

            # Call Python init
            cache_list = python_init_fn(**kwargs)

            # Convert to numpy arrays with matching dtype
            cache = np.array([v if v is not None else 0.0 for v in cache_list], dtype=out_dtype)
            collapse = np.zeros(n_collapse, dtype=out_dtype)

            return cache, collapse

        def array_init_fn(inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Array-based init function (vmap-compatible via pure_callback).

            Args:
                inputs: Array of init parameters

            Returns:
                (cache_array, collapse_decisions_array)
            """
            # Use the same dtype as inputs for output
            out_dtype = inputs.dtype

            # Define result shapes for pure_callback
            result_shape = (
                jax.ShapeDtypeStruct((n_cache,), out_dtype),
                jax.ShapeDtypeStruct((n_collapse,), out_dtype),
            )

            # Use pure_callback to call Python from within JAX tracing
            # vmap_method='sequential' makes vmap call the function once per batch element
            cache, collapse = jax.pure_callback(
                _python_init_impl,
                result_shape,
                inputs,
                vmap_method='sequential',
            )

            return cache, collapse

        metadata = {
            'param_names': init_param_names,
            'param_kinds': init_param_kinds,
            'cache_size': n_cache,
            'cache_mapping': cache_mapping,
            'collapsible_pairs': collapsible_pairs,
            'collapse_decision_outputs': collapse_outputs,
            'param_defaults': {},
        }

        return array_init_fn, metadata

    def translate_init_array_split(
        self,
        shared_indices: List[int],
        varying_indices: List[int],
        init_to_eval: List[int]
    ) -> Tuple[Callable, Dict]:
        """Generate split init function (for GPU optimization).

        Args:
            shared_indices: Indices into eval param array that are shared (constant across devices)
            varying_indices: Indices into eval param array that vary per device
            init_to_eval: Mapping from init param index to eval param index
                         e.g. [0, 4, 0] means init[0]=eval[0], init[1]=eval[4], init[2]=eval[0]
        """
        base_fn, metadata = self.translate_init_array()

        n_eval_params = len(shared_indices) + len(varying_indices)

        # Create JAX arrays for index mapping (for use inside JIT)
        shared_idx_arr = jnp.array(shared_indices, dtype=jnp.int32)
        varying_idx_arr = jnp.array(varying_indices, dtype=jnp.int32)
        init_to_eval_arr = jnp.array(init_to_eval, dtype=jnp.int32)

        def split_init_fn(shared_params: jnp.ndarray, device_params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # 1. Reconstruct full eval params array in original order
            eval_params = jnp.zeros(n_eval_params, dtype=shared_params.dtype)
            eval_params = eval_params.at[shared_idx_arr].set(shared_params)
            eval_params = eval_params.at[varying_idx_arr].set(device_params)

            # 2. Extract init params from eval params using init_to_eval mapping
            init_params = eval_params[init_to_eval_arr]

            # 3. Call base init function with correctly ordered init params
            return base_fn(init_params)

        # Generate a code hash for caching
        code_hash = hashlib.sha256(
            f"{self.module.name}:{shared_indices}:{varying_indices}".encode()
        ).hexdigest()

        metadata['shared_indices'] = shared_indices
        metadata['varying_indices'] = varying_indices
        metadata['code_hash'] = code_hash

        return split_init_fn, metadata

    def release_mir_data(self):
        """Release MIR data after code generation is complete.

        Call this after all translate_*() methods have been called to free memory.
        """
        self._init_mir = None
        # Keep dae_data as it may be needed for metadata queries

    def translate_eval_array_with_cache_split(
        self,
        shared_indices: List[int],
        varying_indices: List[int],
        shared_cache_indices: List[int] = None,
        varying_cache_indices: List[int] = None
    ) -> Tuple[Callable, Dict]:
        """Generate split eval function with cache (for GPU optimization).

        Returns eval function that takes split params and cache arrays.
        """
        self._build_eval_fn()
        eval_fn = self._eval_fn

        residuals_meta = self._metadata['residuals']
        jacobian_meta = self._metadata['jacobian']

        n_residuals = len(residuals_meta)
        n_jacobian = len(jacobian_meta)
        n_params = len(shared_indices) + len(varying_indices)

        # Build reverse mapping: original_idx -> (is_shared, position_in_split_array)
        # shared_indices and varying_indices are original param indices
        shared_set = set(shared_indices)
        varying_set = set(varying_indices)

        # Create JAX arrays for index mapping (for use inside JIT)
        shared_idx_arr = jnp.array(shared_indices, dtype=jnp.int32)
        varying_idx_arr = jnp.array(varying_indices, dtype=jnp.int32)

        use_cache_split = shared_cache_indices is not None and varying_cache_indices is not None

        if use_cache_split:
            shared_cache_idx_arr = jnp.array(shared_cache_indices, dtype=jnp.int32)
            varying_cache_idx_arr = jnp.array(varying_cache_indices, dtype=jnp.int32)
            n_cache = len(shared_cache_indices) + len(varying_cache_indices)

            def split_eval_fn(shared_params, device_params, shared_cache, device_cache):
                # Reconstruct full params array in original order
                full_params = jnp.zeros(n_params, dtype=shared_params.dtype)
                full_params = full_params.at[shared_idx_arr].set(shared_params)
                full_params = full_params.at[varying_idx_arr].set(device_params)

                # Reconstruct full cache array in original order
                full_cache = jnp.zeros(n_cache, dtype=shared_cache.dtype)
                full_cache = full_cache.at[shared_cache_idx_arr].set(shared_cache)
                full_cache = full_cache.at[varying_cache_idx_arr].set(device_cache)

                (res_resist, res_react), (jac_resist, jac_react) = eval_fn(full_params, full_cache)
                lim_rhs_resist = jnp.zeros_like(res_resist)
                lim_rhs_react = jnp.zeros_like(res_react)
                return res_resist, res_react, jac_resist, jac_react, lim_rhs_resist, lim_rhs_react
        else:
            def split_eval_fn(shared_params, device_params, cache):
                # Reconstruct full params array in original order
                full_params = jnp.zeros(n_params, dtype=shared_params.dtype)
                full_params = full_params.at[shared_idx_arr].set(shared_params)
                full_params = full_params.at[varying_idx_arr].set(device_params)

                (res_resist, res_react), (jac_resist, jac_react) = eval_fn(full_params, cache)
                lim_rhs_resist = jnp.zeros_like(res_resist)
                lim_rhs_react = jnp.zeros_like(res_react)
                return res_resist, res_react, jac_resist, jac_react, lim_rhs_resist, lim_rhs_react

        metadata = {
            'node_names': [r.get('node_name', f'node{i}') for i, r in enumerate(residuals_meta)],
            'jacobian_keys': [(j['row'], j['col']) for j in jacobian_meta],
            'num_residuals': n_residuals,
            'num_jacobian': n_jacobian,
            'use_cache_split': use_cache_split,
        }

        return split_eval_fn, metadata

    def translate(self) -> Callable:
        """Generate the eval function.

        Returns:
            A callable that takes input array and returns (residuals, jacobian)
        """
        self._build_init_fn()
        self._build_eval_fn()
        return self._create_wrapped_eval()

    def _build_init_fn(self):
        """Build the init function from MIR."""
        # Import our Python MIR interpreter for init
        # (init runs once at setup, doesn't need JAX)
        import sys
        from pathlib import Path

        # Add scripts to path for mir_codegen
        scripts_path = Path(__file__).parent.parent / "scripts"
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))

        from mir_codegen import generate_init_function
        self._init_fn = generate_init_function(self.module)

    def _build_eval_fn(self):
        """Build the eval function from MIR for JAX tracing."""
        eval_mir = self.module.get_mir_instructions()
        metadata = self.module.get_codegen_metadata()

        constants = eval_mir['constants']
        params = eval_mir.get('params', [])
        instructions = eval_mir['instructions']
        blocks = eval_mir['blocks']

        eval_param_mapping = metadata['eval_param_mapping']
        cache_info = metadata['cache_info']
        residuals_meta = metadata['residuals']
        jacobian_meta = metadata['jacobian']

        # Build cache-to-param index mapping
        cache_to_param_idx = {}
        for ci in cache_info:
            eval_param = ci['eval_param']
            if eval_param.startswith('v'):
                param_idx = int(eval_param[1:])
                cache_to_param_idx[ci['cache_idx']] = param_idx

        # Build param name to index mapping
        param_name_to_idx = {}
        for name, mir_var in eval_param_mapping.items():
            for i, p in enumerate(params):
                if p == mir_var:
                    param_name_to_idx[name] = i
                    break

        # For simple straight-line code (no complex control flow),
        # we can build a JAX function directly
        # Group instructions by block
        block_instrs = {}
        for instr in instructions:
            block = instr.get('block', 'default')
            if block not in block_instrs:
                block_instrs[block] = []
            block_instrs[block].append(instr)

        def eval_fn(input_array: jnp.ndarray, cache: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Evaluate the device model.

            Args:
                input_array: Array of input parameter values (voltages, params)
                cache: Array of cache values from init

            Returns:
                (residuals, jacobian) arrays
            """
            # Initialize values dict with constants
            values = {k: jnp.float64(v) for k, v in constants.items()}

            # Set input params
            for i, var in enumerate(params):
                if i < len(input_array):
                    values[var] = input_array[i]

            # Set cache values
            for cache_idx, param_idx in cache_to_param_idx.items():
                if param_idx < len(params) and cache_idx < len(cache):
                    values[params[param_idx]] = cache[cache_idx]

            # Execute instructions (simple case: single block, no control flow)
            # For complex models with control flow, we'd need lax.cond/lax.switch
            for block_name in sorted(block_instrs.keys()):
                for instr in block_instrs[block_name]:
                    opcode = instr['opcode']
                    result_var = instr.get('result')
                    operands = instr.get('operands', [])

                    if opcode in ('jmp', 'br', 'phi'):
                        # Skip control flow for now
                        continue

                    if opcode in JAX_OPCODE_MAP:
                        func = JAX_OPCODE_MAP[opcode]
                        op_values = [values.get(op, jnp.float64(0.0)) for op in operands]
                        if result_var:
                            values[result_var] = func(*op_values)
                    elif opcode == 'call':
                        # Handle special function calls
                        if result_var:
                            values[result_var] = jnp.float64(0.0)

            # Extract residuals
            n_residuals = len(residuals_meta)
            resist_residuals = jnp.zeros(n_residuals)
            react_residuals = jnp.zeros(n_residuals)

            for i, r in enumerate(residuals_meta):
                resist_residuals = resist_residuals.at[i].set(
                    values.get(r['resist_var'], jnp.float64(0.0))
                )
                react_residuals = react_residuals.at[i].set(
                    values.get(r['react_var'], jnp.float64(0.0))
                )

            # Extract jacobian
            n_jacobian = len(jacobian_meta)
            resist_jacobian = jnp.zeros(n_jacobian)
            react_jacobian = jnp.zeros(n_jacobian)

            for i, j in enumerate(jacobian_meta):
                resist_jacobian = resist_jacobian.at[i].set(
                    values.get(j['resist_var'], jnp.float64(0.0))
                )
                react_jacobian = react_jacobian.at[i].set(
                    values.get(j['react_var'], jnp.float64(0.0))
                )

            return (resist_residuals, react_residuals), (resist_jacobian, react_jacobian)

        self._eval_fn = eval_fn

    def _create_wrapped_eval(self) -> Callable:
        """Create wrapped eval that handles dict I/O."""
        metadata = self.module.get_codegen_metadata()
        eval_param_mapping = metadata['eval_param_mapping']
        residuals_meta = metadata['residuals']
        jacobian_meta = metadata['jacobian']

        mir_params = self.module.get_mir_instructions().get('params', [])
        param_name_to_idx = {}
        for name, mir_var in eval_param_mapping.items():
            for i, p in enumerate(mir_params):
                if p == mir_var:
                    param_name_to_idx[name] = i
                    break

        init_fn = self._init_fn
        eval_fn = self._eval_fn
        n_params = len(mir_params)
        cache_size = self._cache_size

        def wrapped_eval(inputs: List[float]) -> Tuple[Dict, Dict]:
            """Evaluate the device.

            Args:
                inputs: List of input values in param order

            Returns:
                (residuals_dict, jacobian_dict)
            """
            # Convert to arrays
            input_array = jnp.array(inputs, dtype=jnp.float64)

            # For now, compute cache on each call
            # In a real implementation, cache would be computed once at init
            # and stored in the device instance

            # Build init params from eval inputs
            # Map semantic names to values from eval inputs
            init_params = {}
            init_param_mapping = metadata.get('init_param_mapping', {})

            # First, build a mapping from eval semantic names to input values
            eval_values = {}
            for name, mir_var in eval_param_mapping.items():
                if name in param_name_to_idx:
                    idx = param_name_to_idx[name]
                    if idx < len(inputs):
                        eval_values[name] = inputs[idx]

            # Now map to init params
            for name, mir_var in init_param_mapping.items():
                if name in eval_values:
                    # Direct mapping from eval param
                    init_params[name] = eval_values[name]
                elif name.endswith('_given'):
                    # Check if base param is in eval_values (non-zero means given)
                    base_name = name[:-6]  # Remove '_given' suffix
                    if base_name in eval_values and eval_values[base_name] != 0:
                        init_params[name] = True
                    else:
                        init_params[name] = False
                elif name == 'mfactor' and 'mfactor' in eval_values:
                    init_params[name] = eval_values['mfactor']
                else:
                    init_params[name] = 0.0

            # Compute cache
            cache_list = init_fn(**init_params)
            cache = jnp.array([v if v is not None else 0.0 for v in cache_list], dtype=jnp.float64)

            # Run eval
            (resist_res, react_res), (resist_jac, react_jac) = eval_fn(input_array, cache)

            # Build output dicts
            residuals = {}
            for i, r in enumerate(residuals_meta):
                node_idx = r['residual_idx']
                residuals[node_idx] = {
                    'resist': float(resist_res[i]),
                    'react': float(react_res[i]),
                }

            jacobian = {}
            for i, j in enumerate(jacobian_meta):
                key = (j['row'], j['col'])
                jacobian[key] = {
                    'resist': float(resist_jac[i]),
                    'react': float(react_jac[i]),
                }

            return residuals, jacobian

        return wrapped_eval

    def get_jit_eval(self) -> Callable:
        """Get a JIT-compiled version of the eval function.

        Returns:
            JIT-compiled eval function taking (input_array, cache) arrays
        """
        return jax.jit(self._eval_fn)
