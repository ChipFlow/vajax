"""OpenVAF model compilation and setup for VA-JAX.

This module handles:
- Compiling Verilog-A models via OpenVAF
- Caching compiled models across sessions
- Computing device parameter splits (shared vs varying)
- Computing node collapse decisions
- Generating split eval functions for efficient batched evaluation

The compiled_models dict structure expected by mna_builder:
    compiled_models[model_type] = {
        # Required for mna_builder:
        "device_params": Array,            # (n_devices, n_varying_params)
        "shared_params": Array,            # (n_shared_params,)
        "vmapped_split_eval": Callable,    # Batched eval function
        "voltage_positions_in_varying": Array,  # Where voltages are in device_params
        "shared_cache": Array,             # (n_shared_cache,)
        "device_cache": Array,             # (n_devices, n_varying_cache)
        "default_simparams": Array,        # [analysis_type, mfactor, gmin]
        "use_device_limiting": bool,
        "uses_analysis": bool,
        "uses_simparam_gmin": bool,
        "num_limit_states": int,

        # Additional metadata:
        "translator": OpenVAFToJAX,
        "param_names": List[str],
        "param_kinds": List[str],
        "nodes": List[str],
        "dae_metadata": Dict,
        "collapsible_pairs": List[Tuple[int, int]],
        ...
    }
"""

import logging
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from vajax import configure_xla_cache, get_float_dtype

logger = logging.getLogger(__name__)

# Try to import OpenVAF support
_project_root = Path(__file__).parent.parent.parent
_openvaf_jax_path = _project_root / "openvaf_jax"
_openvaf_py_path = _openvaf_jax_path / "openvaf_py"
if _openvaf_jax_path.exists() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if _openvaf_py_path.exists() and str(_openvaf_py_path) not in sys.path:
    sys.path.insert(0, str(_openvaf_py_path))

try:
    import openvaf_py

    import openvaf_jax

    HAS_OPENVAF = True
except ImportError:
    HAS_OPENVAF = False
    openvaf_py = None
    openvaf_jax = None


# Module-level cache of compiled OpenVAF models
# Keyed by model_type, contains translator, functions, metadata
COMPILED_MODEL_CACHE: Dict[str, Any] = {}

# OpenVAF model sources: model_type -> (base_path_key, relative_path)
MODEL_PATHS = {
    "psp103": ("integration_tests", "PSP103/psp103.va"),
    "resistor": ("vacask", "resistor.va"),
    "capacitor": ("vacask", "capacitor.va"),
    "diode": ("vacask", "diode.va"),
    "sp_diode": ("vacask", "spice/sn/diode.va"),
}


# Base paths for VA model sources
def _get_base_paths() -> Dict[str, Path]:
    """Get base paths for VA model sources."""
    project_root = Path(__file__).parent.parent.parent
    return {
        "integration_tests": project_root / "vendor" / "OpenVAF" / "integration_tests",
        "vacask": project_root / "vendor" / "VACASK" / "devices",
    }


def warmup_models(
    model_types: Optional[List[str]] = None,
    trigger_xla: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Pre-compile OpenVAF models and optionally trigger XLA compilation.

    This function compiles device models ahead of time and caches them,
    reducing startup time for subsequent simulations. When trigger_xla=True,
    it also runs dummy evaluations to trigger XLA compilation.

    Args:
        model_types: List of model types to compile (e.g., ['psp103', 'resistor']).
            If None, compiles all available models.
        trigger_xla: If True, run dummy evaluations to trigger XLA compilation.
        log_fn: Optional logging function for progress output.

    Returns:
        Dict mapping model_type to compiled model info dict.
    """
    global COMPILED_MODEL_CACHE

    if not HAS_OPENVAF:
        raise ImportError("OpenVAF support required but openvaf_py not available")

    configure_xla_cache()

    def log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    if model_types is None:
        model_types = list(MODEL_PATHS.keys())

    base_paths = _get_base_paths()
    log(f"Warming up models: {model_types}")
    results = {}

    for model_type in model_types:
        # Check if already cached
        if model_type in COMPILED_MODEL_CACHE:
            cached = COMPILED_MODEL_CACHE[model_type]
            log(f"  {model_type}: already cached ({len(cached['param_names'])} params)")
            results[model_type] = cached
            continue

        model_info = MODEL_PATHS.get(model_type)
        if not model_info:
            log(f"  {model_type}: unknown model type, skipping")
            continue

        base_key, va_path = model_info
        base_path = base_paths.get(base_key)
        if not base_path:
            log(f"  {model_type}: unknown base path key {base_key}, skipping")
            continue

        full_path = base_path / va_path
        if not full_path.exists():
            log(f"  {model_type}: VA file not found at {full_path}, skipping")
            continue

        t0 = time.perf_counter()

        from openvaf_jax.cache import compute_va_hash, get_model_cache_path

        # Try to load from persistent MIR cache
        va_hash = compute_va_hash(full_path)
        cache_path = get_model_cache_path(model_type, va_hash)
        mir_cache_file = cache_path / "mir_data.pkl"

        translator = None

        if mir_cache_file.exists():
            try:
                log(f"  {model_type}: loading from MIR cache...")
                with open(mir_cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                translator = openvaf_jax.OpenVAFToJAX.from_cache(cached_data)
                t1 = time.perf_counter()
                log(f"  {model_type}: loaded from cache in {t1 - t0:.1f}s")
            except Exception as e:
                log(f"  {model_type}: cache load failed ({e}), recompiling...")
                translator = None

        if translator is None:
            log(f"  {model_type}: compiling VA...")
            modules = openvaf_py.compile_va(str(full_path))
            t1 = time.perf_counter()
            log(f"  {model_type}: VA compiled in {t1 - t0:.1f}s")

            if not modules:
                log(f"  {model_type}: compilation failed, skipping")
                continue

            module = modules[0]
            translator = openvaf_jax.OpenVAFToJAX(module)

            # Save to MIR cache
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_data = translator.to_cache()
                with open(mir_cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                log(f"  {model_type}: saved MIR cache")
            except Exception as e:
                log(f"  {model_type}: failed to save cache: {e}")

        # Generate init function
        t2 = time.perf_counter()
        init_fn, init_meta = translator.translate_init_array()
        t3 = time.perf_counter()
        log(f"  {model_type}: init_fn generated in {t3 - t2:.1f}s")

        # Generate eval function with simple split
        n_eval_params = len(translator.params)
        eval_param_kinds = list(translator.module.param_kinds)
        shared_indices = list(range(n_eval_params))
        varying_indices = [i for i, kind in enumerate(eval_param_kinds) if kind == "voltage"]

        eval_fn, eval_meta = translator.translate_eval_array_with_cache_split(
            shared_indices, varying_indices
        )
        t4 = time.perf_counter()
        log(f"  {model_type}: eval_fn generated in {t4 - t3:.1f}s")

        # Cache the compiled model
        compiled = {
            "translator": translator,
            "init_fn": init_fn,
            "init_meta": init_meta,
            "eval_fn": eval_fn,
            "eval_meta": eval_meta,
            "param_names": init_meta["param_names"],
            "nodes": translator.module.nodes,
        }
        COMPILED_MODEL_CACHE[model_type] = compiled
        results[model_type] = compiled

        # Trigger XLA compilation with dummy data
        if trigger_xla:
            log(f"  {model_type}: triggering XLA compilation...")
            try:
                n_devices = 1
                n_init_params = len(init_meta["param_names"])

                dummy_init_inputs = jnp.zeros((n_devices, n_init_params))
                vmapped_init = jax.jit(jax.vmap(init_fn))
                _ = vmapped_init(dummy_init_inputs)

                t5 = time.perf_counter()
                log(f"  {model_type}: XLA warmup done in {t5 - t4:.1f}s")
            except Exception as e:
                log(f"  {model_type}: XLA warmup failed: {e}")

        t_total = time.perf_counter() - t0
        log(f"  {model_type}: total warmup time {t_total:.1f}s")

    return results


def compile_openvaf_models(
    devices: List[Dict],
    compiled_models: Dict[str, Any],
    model_paths: Dict[str, Tuple[str, str]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Compile OpenVAF models needed by the circuit.

    Uses module-level cache to reuse jitted functions across instances.

    Args:
        devices: List of device dicts with 'model' and 'is_openvaf' keys
        compiled_models: Existing compiled models dict (updated in place)
        model_paths: Model path mappings (model_type -> (base_key, va_path))
        log_fn: Optional logging function

    Returns:
        Updated compiled_models dict
    """
    global COMPILED_MODEL_CACHE

    if not HAS_OPENVAF:
        raise ImportError("OpenVAF support required but openvaf_py not available")

    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            logger.info(msg)

    # Find unique OpenVAF model types
    openvaf_types = set()
    for dev in devices:
        if dev.get("is_openvaf"):
            openvaf_types.add(dev["model"])

    log(f"Compiling OpenVAF models: {openvaf_types}")
    base_paths = _get_base_paths()

    for model_type in openvaf_types:
        # Check instance cache first
        if model_type in compiled_models:
            continue

        # Check module-level cache
        if model_type in COMPILED_MODEL_CACHE:
            cached = COMPILED_MODEL_CACHE[model_type]
            log(
                f"  {model_type}: reusing cached ({len(cached['param_names'])} params, {len(cached['nodes'])} nodes)"
            )
            compiled_models[model_type] = cached
            continue

        model_info = model_paths.get(model_type)
        if not model_info:
            raise ValueError(f"Unknown OpenVAF model type: {model_type}")

        base_key, va_path = model_info
        base_path = base_paths.get(base_key)
        if not base_path:
            raise ValueError(f"Unknown base path key: {base_key}")

        full_path = base_path / va_path
        if not full_path.exists():
            raise FileNotFoundError(f"VA model not found: {full_path}")

        from openvaf_jax.cache import compute_va_hash, get_model_cache_path

        t0 = time.perf_counter()

        # Try to load from persistent cache
        va_hash = compute_va_hash(full_path)
        cache_path = get_model_cache_path(model_type, va_hash)
        mir_cache_file = cache_path / "mir_data.pkl"

        translator = None
        module = None

        if mir_cache_file.exists():
            try:
                log(f"  {model_type}: loading from persistent cache...")
                with open(mir_cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                translator = openvaf_jax.OpenVAFToJAX.from_cache(cached_data)
                t1 = time.perf_counter()
                log(f"  {model_type}: loaded from cache in {t1 - t0:.1f}s")
            except Exception as e:
                log(f"  {model_type}: cache load failed ({e}), recompiling...")
                translator = None

        if translator is None:
            log(f"  {model_type}: compiling VA...")
            modules = openvaf_py.compile_va(str(full_path))
            t1 = time.perf_counter()
            log(f"  {model_type}: VA compiled in {t1 - t0:.1f}s")
            if not modules:
                raise ValueError(f"Failed to compile {va_path}")

            log(f"  {model_type}: creating translator...")
            module = modules[0]
            translator = openvaf_jax.OpenVAFToJAX(module)
            t2 = time.perf_counter()
            log(f"  {model_type}: translator created in {t2 - t1:.1f}s")

            # Save to persistent cache
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_data = translator.get_cache_data()
                with open(mir_cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                log(f"  {model_type}: saved to persistent cache")
            except Exception as e:
                log(f"  {model_type}: failed to save cache: {e}")

        # Get model metadata
        if module is not None:
            param_names = list(module.param_names)
            param_kinds = list(module.param_kinds)
            nodes = list(module.nodes)
            collapsible_pairs = list(module.collapsible_pairs)
            num_collapsible = module.num_collapsible
        else:
            param_names = cached_data["param_names"]
            param_kinds = cached_data["param_kinds"]
            nodes = cached_data["nodes"]
            collapsible_pairs = cached_data["collapsible_pairs"]
            num_collapsible = cached_data["num_collapsible"]

        # Get DAE metadata
        t2 = time.perf_counter()
        dae_metadata = translator.get_dae_metadata()
        t3 = time.perf_counter()
        log(f"  {model_type}: DAE metadata extracted in {t3 - t2:.3f}s")

        # Generate init function
        log(f"  {model_type}: generating init function...")
        init_fn, init_meta = translator.translate_init_array()
        vmapped_init = jax.jit(jax.vmap(init_fn))
        t4 = time.perf_counter()
        log(
            f"  {model_type}: init function done in {t4 - t3:.1f}s (cache_size={init_meta['cache_size']})"
        )

        # Build init->eval index mapping
        eval_name_to_idx = {n.lower(): i for i, n in enumerate(param_names)}
        init_to_eval_indices = []
        for name in init_meta["param_names"]:
            eval_idx = eval_name_to_idx.get(name.lower(), -1)
            init_to_eval_indices.append(eval_idx)
        init_to_eval_indices = jnp.array(init_to_eval_indices, dtype=jnp.int32)

        compiled = {
            "module": module,
            "translator": translator,
            "dae_metadata": dae_metadata,
            "param_names": param_names,
            "param_kinds": param_kinds,
            "nodes": nodes,
            "collapsible_pairs": collapsible_pairs,
            "num_collapsible": num_collapsible,
            "init_fn": init_fn,
            "vmapped_init": vmapped_init,
            "init_param_names": list(init_meta["param_names"]),
            "init_param_kinds": list(init_meta["param_kinds"]),
            "cache_size": init_meta["cache_size"],
            "cache_mapping": init_meta["cache_mapping"],
            "init_param_defaults": init_meta.get("param_defaults", {}),
            "init_to_eval_indices": init_to_eval_indices,
            "uses_simparam_gmin": translator.uses_simparam_gmin,
            "uses_analysis": translator.uses_analysis,
            "analysis_type_map": translator.analysis_type_map,
        }

        compiled_models[model_type] = compiled
        COMPILED_MODEL_CACHE[model_type] = compiled
        log(f"  {model_type}: done ({len(param_names)} params, {len(nodes)} nodes)")

    return compiled_models


def compute_early_collapse_decisions(
    devices: List[Dict],
    compiled_models: Dict[str, Any],
) -> Dict[str, List[Tuple[int, int]]]:
    """Compute collapse decisions for all devices using OpenVAF vmapped_init.

    OPTIMIZATION: Groups devices by unique parameter combinations and computes
    collapse decisions once per unique combo.

    Args:
        devices: List of device dicts
        compiled_models: Dict of compiled model info

    Returns:
        Dict[device_name, List[Tuple[int, int]]] - collapse pairs per device
    """
    device_collapse_decisions: Dict[str, List[Tuple[int, int]]] = {}

    # Group devices by model type
    devices_by_type: Dict[str, List[Dict]] = {}
    for dev in devices:
        if dev.get("is_openvaf"):
            model_type = dev["model"]
            devices_by_type.setdefault(model_type, []).append(dev)

    for model_type, devs in devices_by_type.items():
        compiled = compiled_models.get(model_type)
        if not compiled:
            continue

        init_fn = compiled.get("init_fn")

        if init_fn is None:
            collapsible_pairs = compiled.get("collapsible_pairs", [])
            for dev in devs:
                device_collapse_decisions[dev["name"]] = list(collapsible_pairs)
            continue

        init_param_names = compiled.get("init_param_names", [])
        init_param_defaults = compiled.get("init_param_defaults", {})
        n_init_params = len(init_param_names)
        collapsible_pairs = compiled.get("collapsible_pairs", [])
        n_collapsible = len(collapsible_pairs)

        if n_init_params == 0 or n_collapsible == 0:
            if init_fn is not None and n_collapsible > 0:
                try:
                    cpu_device = jax.devices("cpu")[0]
                    with jax.default_device(cpu_device):
                        _, collapse_decisions = init_fn(jnp.array([]))
                    pairs = []
                    for i, (n1, n2) in enumerate(collapsible_pairs):
                        if i < len(collapse_decisions) and float(collapse_decisions[i]) > 0.5:
                            pairs.append((n1, n2))
                    for dev in devs:
                        device_collapse_decisions[dev["name"]] = pairs
                    continue
                except Exception as e:
                    logger.warning(f"Error computing collapse decisions for {model_type}: {e}")
            for dev in devs:
                device_collapse_decisions[dev["name"]] = list(collapsible_pairs)
            continue

        # Group devices by unique parameter combinations
        def get_param_key(dev: Dict) -> Tuple:
            device_params = dev.get("params", {})
            values = []
            for pname in init_param_names:
                pname_lower = pname.lower()
                if pname_lower in device_params:
                    values.append(float(device_params[pname_lower]))
                elif pname_lower in init_param_defaults:
                    values.append(float(init_param_defaults[pname_lower]))
                else:
                    values.append(0.0)
            return tuple(values)

        unique_params: Dict[Tuple, List[Dict]] = {}
        for dev in devs:
            key = get_param_key(dev)
            unique_params.setdefault(key, []).append(dev)

        n_unique = len(unique_params)
        logger.info(
            f"Computing collapse decisions for {model_type}: {len(devs)} devices, {n_unique} unique param combos"
        )

        cpu_device = jax.devices("cpu")[0]
        for param_key, param_devs in unique_params.items():
            try:
                with jax.default_device(cpu_device):
                    init_inputs = jnp.array(param_key, dtype=get_float_dtype())
                    _, collapse_decisions = init_fn(init_inputs)

                pairs = []
                collapse_np = np.asarray(collapse_decisions)
                for i, (n1, n2) in enumerate(collapsible_pairs):
                    if i < len(collapse_np) and collapse_np[i] > 0.5:
                        pairs.append((n1, n2))

                for dev in param_devs:
                    device_collapse_decisions[dev["name"]] = pairs

            except Exception as e:
                logger.warning(f"Error computing collapse for {model_type}: {e}")
                for dev in param_devs:
                    device_collapse_decisions[dev["name"]] = list(collapsible_pairs)

    logger.debug(f"Computed collapse decisions for {len(device_collapse_decisions)} devices")
    return device_collapse_decisions


def prepare_static_inputs(
    model_type: str,
    openvaf_devices: List[Dict],
    device_internal_nodes: Dict[str, Dict[str, int]],
    compiled_models: Dict[str, Any],
    simulation_temperature: float,
    use_device_limiting: bool,
    parse_voltage_param_fn: Callable,
    ground: int = 0,
) -> Tuple[List[int], List[Dict], Array, Array]:
    """Prepare device inputs and generate split eval function.

    This analyzes parameter constancy across devices and generates optimized
    split eval functions that separate constant (shared) params from varying
    (per-device) params.

    Args:
        model_type: OpenVAF model type
        openvaf_devices: List of device dicts for this model type
        device_internal_nodes: Map of device_name -> {node_name: circuit_node_idx}
        compiled_models: Dict of compiled model info (updated in place)
        simulation_temperature: Simulation temperature in Kelvin
        use_device_limiting: Whether to use device limiting functions
        parse_voltage_param_fn: Function to parse voltage parameter names
        ground: Ground node index

    Returns:
        (voltage_indices, device_contexts, cache, collapse_decisions)
    """
    compiled = compiled_models.get(model_type)
    if not compiled:
        raise ValueError(f"OpenVAF model {model_type} not compiled")

    param_names = compiled["param_names"]
    param_kinds = compiled["param_kinds"]
    model_nodes = compiled["nodes"]

    # Find voltage parameter indices
    voltage_indices = []
    voltage_set = set()
    for i, kind in enumerate(param_kinds):
        if kind == "voltage":
            voltage_indices.append(i)
            voltage_set.add(i)

    n_devices = len(openvaf_devices)
    n_params = len(param_names)
    device_contexts = []

    # Build col_values: col_idx -> scalar (shared) or array (varying)
    col_values: Dict[int, Any] = {}
    varying_cols_set = set(voltage_set)

    if n_devices > 0:
        all_dev_params = [dev["params"] for dev in openvaf_devices]
        model_defaults = compiled.get("init_param_defaults", {})

        # Build param_name -> column index mapping
        param_to_cols = {}
        param_given_to_cols = {}
        limit_param_map: Dict[int, Tuple[str, str]] = {}

        for idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
            name_lower = name.lower()
            if kind == "param":
                param_to_cols.setdefault(name_lower, []).append(idx)
            elif kind == "param_given":
                param_given_to_cols.setdefault(name_lower, []).append(idx)
            elif kind == "temperature":
                col_values[idx] = simulation_temperature
            elif kind == "sysfun" and name_lower == "mfactor":
                col_values[idx] = 1.0
            elif kind in ("prev_state", "enable_lim", "new_state", "enable_integration"):
                limit_param_map[idx] = (kind, name)

        # Get unique params from devices
        all_unique = set()
        for p in all_dev_params:
            all_unique.update(k.lower() for k in p.keys())

        # Fill param values
        for pname in all_unique:
            if pname in param_to_cols:
                vals = np.array(
                    [float(p.get(pname, p.get(pname.upper(), 0.0))) for p in all_dev_params]
                )
                if np.all(vals == vals[0]):
                    for col in param_to_cols[pname]:
                        col_values[col] = float(vals[0])
                else:
                    for col in param_to_cols[pname]:
                        col_values[col] = vals
                        varying_cols_set.add(col)
            if pname in param_given_to_cols:
                for col in param_given_to_cols[pname]:
                    col_values[col] = 1.0

        # Defaults for params not in any device
        for pname, cols in param_to_cols.items():
            if pname not in all_unique:
                if pname in ("tnom", "tref", "tr"):
                    default = 27.0
                elif pname in ("nf", "mult", "ns", "nd"):
                    default = 1.0
                else:
                    default = model_defaults.get(pname, 0.0)
                for col in cols:
                    col_values[col] = default

    # Build device contexts
    for dev_idx, dev in enumerate(openvaf_devices):
        ext_nodes = dev["nodes"]
        internal_nodes = device_internal_nodes.get(dev["name"], {})

        # Build node map
        node_map = {}
        n_ext_terminals = len(ext_nodes)
        for i in range(n_ext_terminals):
            model_node = model_nodes[i]
            node_map[model_node] = ext_nodes[i]

        for model_node, global_idx in internal_nodes.items():
            node_map[model_node] = global_idx

        # Map VA node names
        metadata = compiled.get("dae_metadata", {})
        va_terminals = metadata.get("terminals", [])
        va_internal = metadata.get("internal_nodes", [])

        for i, va_name in enumerate(va_terminals):
            if i < len(ext_nodes):
                node_map[va_name] = ext_nodes[i]

        num_terminals = len(va_terminals)
        for i, va_name in enumerate(va_internal):
            internal_key = f"node{num_terminals + i}"
            if internal_key in internal_nodes:
                node_map[va_name] = internal_nodes[internal_key]

        # Pre-compute voltage node pairs
        voltage_node_pairs = []
        for idx in voltage_indices:
            name = param_names[idx]
            node_pair = parse_voltage_param_fn(name, node_map, model_nodes, ground)
            voltage_node_pairs.append(node_pair)

        device_contexts.append(
            {
                "name": dev["name"],
                "node_map": node_map,
                "ext_nodes": ext_nodes,
                "voltage_node_pairs": voltage_node_pairs,
            }
        )

    # Add analysis_type and gmin if needed
    uses_analysis = compiled.get("uses_analysis", False)
    uses_simparam_gmin = compiled.get("uses_simparam_gmin", False)
    n_params_total = n_params

    if uses_analysis and n_devices > 0:
        col_values[n_params] = 0.0
        col_values[n_params + 1] = 1e-12
        n_params_total = n_params + 2
    elif uses_simparam_gmin and n_devices > 0:
        col_values[n_params] = 1e-12
        n_params_total = n_params + 1

    # Build shared_params and device_params
    if n_devices >= 1 and n_params_total > 0:
        limit_param_indices = set(limit_param_map.keys())
        shared_indices = []
        varying_indices_list = []

        for col in range(n_params_total):
            if col in limit_param_indices:
                continue
            elif col in varying_cols_set:
                varying_indices_list.append(col)
            else:
                shared_indices.append(col)

        n_const = len(shared_indices)
        n_varying = len(varying_indices_list)
        logger.info(
            f"{model_type}: {n_const}/{n_params_total} constant, {n_varying} varying across {n_devices} devices"
        )

        # Build shared_params
        shared_params_list = []
        for col in shared_indices:
            val = col_values.get(col, 0.0)
            if isinstance(val, np.ndarray):
                shared_params_list.append(float(val[0]))
            else:
                shared_params_list.append(float(val))
        shared_params = jnp.array(shared_params_list, dtype=get_float_dtype())

        # Build device_params
        if n_varying > 0:
            device_params_cols = []
            for col in varying_indices_list:
                val = col_values.get(col)
                if val is None:
                    device_params_cols.append(np.zeros(n_devices, dtype=get_float_dtype()))
                elif isinstance(val, np.ndarray):
                    device_params_cols.append(val)
                else:
                    device_params_cols.append(
                        np.full(n_devices, float(val), dtype=get_float_dtype())
                    )
            device_params = jnp.array(np.column_stack(device_params_cols), dtype=get_float_dtype())
        else:
            device_params = jnp.empty((n_devices, 0), dtype=get_float_dtype())

        del col_values

        # Generate split functions
        translator = compiled.get("translator")
        if translator is None or translator.dae_data is None:
            raise RuntimeError(f"{model_type}: translator.dae_data not available")

        # Compute init cache
        init_to_eval = compiled.get("init_to_eval_indices")
        if init_to_eval is not None:
            logger.info(f"{model_type}: generating split init function...")
            init_to_eval_list = [int(x) for x in init_to_eval]
            split_init_fn, init_split_meta = translator.translate_init_array_split(
                shared_indices, varying_indices_list, init_to_eval_list
            )
            code_hash = init_split_meta.get("code_hash", "")
            vmapped_split_init = openvaf_jax.get_vmapped_jit(
                code_hash, split_init_fn, in_axes=(None, 0)
            )

            logger.info(f"Computing init cache for {model_type} ({n_devices} devices)...")
            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                cache, collapse_decisions = vmapped_split_init(shared_params, device_params)
            logger.info(f"Init cache computed: shape={cache.shape}")

            # Analyze cache constancy
            n_cache_cols = cache.shape[1] if cache.ndim > 1 else 0
            if n_cache_cols > 0 and n_devices > 1:
                cache_np = np.asarray(cache)
                const_mask = np.all(cache_np == cache_np[0:1, :], axis=0)
                shared_cache_indices = [int(i) for i in np.where(const_mask)[0]]
                varying_cache_indices = [int(i) for i in np.where(~const_mask)[0]]
                logger.info(
                    f"{model_type}: cache: {len(shared_cache_indices)}/{n_cache_cols} shared"
                )
            else:
                shared_cache_indices = []
                varying_cache_indices = list(range(n_cache_cols))
        else:
            cache = jnp.empty((n_devices, 0), dtype=get_float_dtype())
            collapse_decisions = jnp.empty((n_devices, 0), dtype=jnp.float32)
            shared_cache_indices = []
            varying_cache_indices = []

        # Generate eval function with cache split
        from vajax.analysis.limiting import fetlim, pnjlim

        limit_funcs = {"pnjlim": pnjlim, "fetlim": fetlim}

        logger.info(
            f"{model_type}: generating split eval function (limit={use_device_limiting})..."
        )
        split_fn, split_meta = translator.translate_eval_array_with_cache_split(
            shared_indices,
            varying_indices_list,
            shared_cache_indices,
            varying_cache_indices,
            use_limit_functions=use_device_limiting,
            limit_param_map=limit_param_map,
        )
        # Safety check: if limiting is enabled but lim_rhs could not be computed
        # (model uses inline limiting without $limit/BuiltinLimit calls), disable
        # limiting for this model to avoid NR inconsistency.
        if use_device_limiting and split_meta.get("limit_metadata"):
            raw_operands = split_meta["limit_metadata"].get("raw_operands", {})
            limit_count = split_meta["limit_metadata"].get("limit_count", 0)
            if limit_count > 0 and len(raw_operands) == 0:
                logger.warning(
                    f"{model_type}: {limit_count} limit state(s) but no BuiltinLimit calls "
                    "traced (inline limiting?) - disabling limiting for this model"
                )
                use_device_limiting = False
                split_fn, split_meta = translator.translate_eval_array_with_cache_split(
                    shared_indices,
                    varying_indices_list,
                    shared_cache_indices,
                    varying_cache_indices,
                    use_limit_functions=False,
                    limit_param_map=limit_param_map,
                )

        split_fn = partial(split_fn, limit_funcs=limit_funcs)
        vmapped_split_fn = jax.jit(jax.vmap(split_fn, in_axes=(None, 0, None, 0, None, 0)))

        # Split cache arrays
        shared_cache = cache[0, shared_cache_indices]
        device_cache = cache[:, varying_cache_indices]

        # Build default simparams from model metadata
        simparams_used = split_meta.get("simparams_used", ["$analysis_type", "$mfactor", "gmin"])
        simparam_count = split_meta.get("simparam_count", len(simparams_used))
        # Import defaults from vajax
        from vajax import SIMPARAM_DEFAULTS

        default_simparams_list = []
        for name in simparams_used:
            if name in SIMPARAM_DEFAULTS:
                default_simparams_list.append(float(SIMPARAM_DEFAULTS[name]))
            else:
                # Unknown simparam - use 0.0 and log warning
                logger.warning(f"{model_type}: unknown simparam '{name}', using 0.0")
                default_simparams_list.append(0.0)
        default_simparams = jnp.array(default_simparams_list, dtype=get_float_dtype())
        logger.info(f"{model_type}: simparams_used={simparams_used}, count={simparam_count}")

        # Voltage positions in device_params
        varying_idx_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(varying_indices_list)}
        voltage_positions = [
            varying_idx_to_pos[v] for v in voltage_indices if v in varying_idx_to_pos
        ]
        voltage_positions = jnp.array(voltage_positions, dtype=jnp.int32)

        # Store in compiled dict
        compiled["split_eval_fn"] = split_fn
        compiled["vmapped_split_eval"] = vmapped_split_fn
        compiled["shared_indices"] = shared_indices
        compiled["varying_indices"] = varying_indices_list
        compiled["shared_params"] = shared_params
        compiled["device_params"] = device_params
        compiled["voltage_positions_in_varying"] = voltage_positions
        compiled["shared_cache"] = shared_cache
        compiled["device_cache"] = device_cache
        compiled["shared_cache_indices"] = shared_cache_indices
        compiled["varying_cache_indices"] = varying_cache_indices
        compiled["default_simparams"] = default_simparams
        compiled["simparams_used"] = simparams_used
        compiled["simparam_indices"] = split_meta.get("simparam_indices", {})
        compiled["use_device_limiting"] = use_device_limiting
        compiled["limit_param_map"] = limit_param_map

        if use_device_limiting and split_meta.get("limit_metadata"):
            compiled["limit_metadata"] = split_meta["limit_metadata"]
            compiled["num_limit_states"] = split_meta["limit_metadata"].get("limit_count", 0)
        else:
            compiled["limit_metadata"] = None
            compiled["num_limit_states"] = 0

        logger.info(f"{model_type}: split eval ready")
    else:
        raise AssertionError(f"Cannot prepare inputs for {model_type}")

    return voltage_indices, device_contexts, cache, collapse_decisions


def warmup_device_models(
    compiled_models: Dict[str, Any],
    static_inputs_cache: Dict[str, Tuple],
) -> None:
    """Trigger XLA compilation of vmapped device functions.

    Args:
        compiled_models: Dict of compiled model info
        static_inputs_cache: Dict mapping model_type to cached static inputs
    """
    for model_type, compiled in compiled_models.items():
        if model_type not in static_inputs_cache:
            continue

        t0 = time.perf_counter()
        logger.info(f"Warming up device model: {model_type}...")

        vmapped_split_eval = compiled["vmapped_split_eval"]
        shared_params = compiled["shared_params"]
        device_params = compiled["device_params"]
        shared_cache = compiled["shared_cache"]
        device_cache = compiled["device_cache"]
        default_simparams = compiled.get("default_simparams", jnp.array([0.0, 1.0, 1e-12]))
        num_limit_states = compiled.get("num_limit_states", 0)

        n_devices = device_params.shape[0]
        n_lim = max(1, num_limit_states)
        limit_state_in = jnp.zeros((n_devices, n_lim), dtype=get_float_dtype())

        try:
            _ = vmapped_split_eval(
                shared_params,
                device_params,
                shared_cache,
                device_cache,
                default_simparams,
                limit_state_in,
            )
            jax.block_until_ready(_)

            t1 = time.perf_counter()
            logger.info(f"  {model_type}: XLA compilation complete ({t1 - t0:.2f}s)")
        except Exception as e:
            logger.warning(f"  {model_type}: warmup failed: {e}")
