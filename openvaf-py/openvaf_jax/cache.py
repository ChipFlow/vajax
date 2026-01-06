"""Function caching utilities for OpenVAF to JAX translation.

This module provides caching for:
- Compiled Python functions (by code hash)
- vmapped+jit'd functions (by code hash + in_axes)
"""

import hashlib
import logging
from typing import Callable, Dict, Tuple, Union

logger = logging.getLogger("jax_spice.openvaf")

# Module-level cache for exec'd functions (keyed by code hash)
# This allows JAX to reuse JIT-compiled functions across translator instances
_exec_fn_cache: Dict[str, Callable] = {}

# Module-level cache for vmapped+jit'd functions (keyed by (code_hash, in_axes))
# This avoids repeated JIT compilation for the same function with same vmap axes
_vmapped_jit_cache: Dict[Tuple[str, Tuple], Callable] = {}


def exec_with_cache(code: str, fn_name: str,
                    return_hash: bool = False) -> Union[Callable, Tuple[Callable, str]]:
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

    Args:
        code_hash: Hash of the function's source code
        fn: The function to vmap and jit
        in_axes: vmap in_axes specification

    Returns:
        vmapped and jit'd function
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


def clear_cache():
    """Clear all function caches (useful for testing or memory management)."""
    global _exec_fn_cache, _vmapped_jit_cache
    _exec_fn_cache.clear()
    _vmapped_jit_cache.clear()


def cache_stats() -> Dict[str, int]:
    """Get cache statistics.

    Returns:
        Dict with 'exec_fn_count' and 'vmapped_jit_count'
    """
    return {
        'exec_fn_count': len(_exec_fn_cache),
        'vmapped_jit_count': len(_vmapped_jit_cache),
    }
