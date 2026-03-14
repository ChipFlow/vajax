#!/usr/bin/env python3
"""Benchmark JAX vs MLX for device model evaluation.

Compiles Verilog-A models via openvaf_jax, then runs the generated
eval/init functions on both JAX and MLX backends.  Compares correctness
(float32 tolerance) and throughput.

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/benchmark_mlx.py
    JAX_PLATFORMS=cpu uv run python scripts/benchmark_mlx.py --models resistor,diode
    JAX_PLATFORMS=cpu uv run python scripts/benchmark_mlx.py --batched --batch-sizes 1,100,1000
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code capture: intercept generated code from openvaf_jax
# ---------------------------------------------------------------------------

_captured_code: Dict[str, str] = {}


def _install_code_capture():
    """Monkey-patch exec_with_cache to capture generated Python source."""
    import openvaf_jax.cache as cache_mod

    original = cache_mod.exec_with_cache

    def capturing(code, fn_name, return_hash=False):
        _captured_code[fn_name] = code
        return original(code, fn_name, return_hash)

    cache_mod.exec_with_cache = capturing
    # Also patch the re-export in the package
    import openvaf_jax
    openvaf_jax.exec_with_cache = capturing


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "resistor": "vendor/VACASK/devices/resistor.va",
    "capacitor": "vendor/VACASK/devices/capacitor.va",
    "diode": "vendor/VACASK/devices/diode.va",
}

# Default parameters per model (empty = use model defaults)
MODEL_PARAMS = {
    "resistor": {"R": 1000.0},
    "capacitor": {"C": 1e-12},
    "diode": {},
}

# Test voltages per model: list of device_params arrays
MODEL_VOLTAGES = {
    "resistor": [0.5],       # V(A,B) = 0.5V
    "capacitor": [1.0],      # V(A,B) = 1.0V
    "diode": [0.7, 0.0],     # V(A,C) = 0.7V forward, V(C,K) = 0V
}


@dataclass
class CompiledModel:
    """A compiled model with JAX and MLX eval functions."""
    name: str
    jax_init_fn: Callable
    jax_eval_fn: Callable
    mlx_init_fn: Callable
    mlx_eval_fn: Callable
    init_meta: Dict[str, Any]
    eval_meta: Dict[str, Any]
    init_code: str
    eval_code: str
    mlx_init_code: str
    mlx_eval_code: str


def compile_model(name: str) -> CompiledModel:
    """Compile a model and produce both JAX and MLX eval functions."""
    from openvaf_jax import OpenVAFToJAX
    from scripts.jax_to_mlx import get_mlx_exec_namespace, jax_to_mlx

    va_path = MODEL_PATHS[name]
    params = MODEL_PARAMS.get(name, {})

    _captured_code.clear()

    translator = OpenVAFToJAX.from_file(va_path)
    init_fn, init_meta = translator.translate_init(
        params, temperature=300.0, mfactor=1.0, debug=False
    )
    eval_fn, eval_meta = translator.translate_eval(
        params, temperature=300.0, mfactor=1.0, debug=False
    )

    init_code = _captured_code.get("init_fn", "")
    eval_code = _captured_code.get("eval_fn", "")
    assert init_code and eval_code, f"Failed to capture code for {name}"

    # Transform to MLX
    mlx_init_code = jax_to_mlx(init_code)
    mlx_eval_code = jax_to_mlx(eval_code)

    # Compile MLX functions
    ns = get_mlx_exec_namespace()
    exec(mlx_init_code, ns)
    exec(mlx_eval_code, ns)

    return CompiledModel(
        name=name,
        jax_init_fn=init_fn,
        jax_eval_fn=eval_fn,
        mlx_init_fn=ns["init_fn"],
        mlx_eval_fn=ns["eval_fn"],
        init_meta=init_meta,
        eval_meta=eval_meta,
        init_code=init_code,
        eval_code=eval_code,
        mlx_init_code=mlx_init_code,
        mlx_eval_code=mlx_eval_code,
    )


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model: str
    backend: str
    # Correctness
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    # Timing (seconds)
    compile_time: float = 0.0
    first_eval_time: float = 0.0
    eval_mean_us: float = 0.0
    eval_std_us: float = 0.0
    n_evals: int = 0
    # Batched
    batch_size: int = 1
    batched_eval_mean_us: float = 0.0
    batched_eval_std_us: float = 0.0
    # Code stats
    code_lines: int = 0


def _time_fn(fn, *args, n_warmup=3, n_iter=100, eval_fn=None):
    """Time a function, returning (mean_us, std_us).

    eval_fn: optional callable to force evaluation (e.g., mx.eval for MLX).
    """
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args)
        if eval_fn:
            eval_fn(result)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn(*args)
        if eval_fn:
            eval_fn(result)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    return np.mean(times), np.std(times)


def run_single_benchmark(model: CompiledModel, n_iter: int = 1000) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Run JAX vs MLX benchmark for a single model.

    Returns (jax_result, mlx_result).
    """
    import jax.numpy as jnp
    import mlx.core as mx

    voltages = MODEL_VOLTAGES[model.name]

    # ---- JAX ----
    jax_result = BenchmarkResult(model=model.name, backend="JAX")
    jax_result.code_lines = len(model.eval_code.splitlines())

    # Compile + init
    t0 = time.perf_counter()
    jax_cache = model.jax_init_fn(jnp.array(model.init_meta["init_inputs"]))[0]
    jax_result.compile_time = time.perf_counter() - t0

    # Build eval args
    jax_shared = jnp.array(model.eval_meta["shared_inputs"])
    jax_device = jnp.array(voltages)
    jax_shared_cache = jnp.array([])
    jax_simparams = jnp.array([0.0, 1.0, 1e-12])
    jax_limit = jnp.zeros(1)
    jax_args = (jax_shared, jax_device, jax_shared_cache, jax_cache, jax_simparams, jax_limit, None)

    # First eval (includes JIT)
    t0 = time.perf_counter()
    jax_out = model.jax_eval_fn(*jax_args)
    jax_result.first_eval_time = time.perf_counter() - t0

    # Throughput
    mean, std = _time_fn(model.jax_eval_fn, *jax_args, n_iter=n_iter)
    jax_result.eval_mean_us = mean
    jax_result.eval_std_us = std
    jax_result.n_evals = n_iter

    # ---- MLX ----
    mlx_result = BenchmarkResult(model=model.name, backend="MLX")
    mlx_result.code_lines = len(model.mlx_eval_code.splitlines())

    # Compile + init
    t0 = time.perf_counter()
    mlx_cache = model.mlx_init_fn(mx.array(model.init_meta["init_inputs"], dtype=mx.float32))[0]
    mx.eval(mlx_cache)
    mlx_result.compile_time = time.perf_counter() - t0

    # Build eval args
    mlx_shared = mx.array(model.eval_meta["shared_inputs"], dtype=mx.float32)
    mlx_device = mx.array(voltages, dtype=mx.float32)
    mlx_shared_cache = mx.array([], dtype=mx.float32)
    mlx_simparams = mx.array([0.0, 1.0, 1e-12], dtype=mx.float32)
    mlx_limit = mx.zeros(1)
    mlx_args = (mlx_shared, mlx_device, mlx_shared_cache, mlx_cache, mlx_simparams, mlx_limit, None)

    # First eval
    t0 = time.perf_counter()
    mlx_out = model.mlx_eval_fn(*mlx_args)
    mx.eval(mlx_out)
    mlx_result.first_eval_time = time.perf_counter() - t0

    # Throughput
    mean, std = _time_fn(model.mlx_eval_fn, *mlx_args, n_iter=n_iter, eval_fn=mx.eval)
    mlx_result.eval_mean_us = mean
    mlx_result.eval_std_us = std
    mlx_result.n_evals = n_iter

    # ---- Correctness comparison ----
    jax_flat = np.concatenate([np.array(r) for r in jax_out[:4]])
    mlx_flat = np.concatenate([np.array(r) for r in mlx_out[:4]])
    abs_diff = np.abs(jax_flat - mlx_flat)
    mlx_result.max_abs_diff = float(np.max(abs_diff))
    nonzero = np.abs(jax_flat) > 1e-30
    if np.any(nonzero):
        mlx_result.max_rel_diff = float(np.max(abs_diff[nonzero] / np.abs(jax_flat[nonzero])))

    return jax_result, mlx_result


def run_batched_benchmark(
    model: CompiledModel,
    batch_sizes: List[int],
    n_iter: int = 100,
) -> List[Tuple[BenchmarkResult, BenchmarkResult]]:
    """Run batched (vmap-equivalent) benchmark.

    JAX uses jax.vmap; MLX uses a Python loop (no vmap equivalent for
    arbitrary functions, though mx.vmap exists for some ops).
    """
    import jax
    import jax.numpy as jnp
    import mlx.core as mx

    voltages = MODEL_VOLTAGES[model.name]
    results = []

    for batch_size in batch_sizes:
        # JAX: vmap over device_params
        jax_cache_single = model.jax_init_fn(jnp.array(model.init_meta["init_inputs"]))[0]
        jax_shared = jnp.array(model.eval_meta["shared_inputs"])
        jax_shared_cache = jnp.array([])
        jax_simparams = jnp.array([0.0, 1.0, 1e-12])
        jax_limit = jnp.zeros(1)

        # Batch: replicate device_params and cache
        jax_device_batch = jnp.tile(jnp.array(voltages), (batch_size, 1))
        jax_cache_batch = jnp.tile(jax_cache_single, (batch_size, 1))

        # vmap over (device_params, device_cache), broadcast rest
        vmapped = jax.vmap(
            model.jax_eval_fn,
            in_axes=(None, 0, None, 0, None, None, None),
        )
        jax_args = (jax_shared, jax_device_batch, jax_shared_cache, jax_cache_batch, jax_simparams, jax_limit, None)

        jax_res = BenchmarkResult(model=model.name, backend="JAX", batch_size=batch_size)
        mean, std = _time_fn(vmapped, *jax_args, n_iter=n_iter)
        jax_res.batched_eval_mean_us = mean
        jax_res.batched_eval_std_us = std

        # MLX: loop (or try mx.vmap if available)
        mlx_cache_single = model.mlx_init_fn(mx.array(model.init_meta["init_inputs"], dtype=mx.float32))[0]
        mx.eval(mlx_cache_single)
        mlx_shared = mx.array(model.eval_meta["shared_inputs"], dtype=mx.float32)
        mlx_shared_cache = mx.array([], dtype=mx.float32)
        mlx_simparams = mx.array([0.0, 1.0, 1e-12], dtype=mx.float32)
        mlx_limit = mx.zeros(1)

        mlx_device_batch = [mx.array(voltages, dtype=mx.float32) for _ in range(batch_size)]
        mlx_cache_batch = [mlx_cache_single for _ in range(batch_size)]

        def mlx_batched_eval():
            results = []
            for i in range(batch_size):
                r = model.mlx_eval_fn(
                    mlx_shared, mlx_device_batch[i], mlx_shared_cache,
                    mlx_cache_batch[i], mlx_simparams, mlx_limit, None,
                )
                results.append(r)
            return results

        mlx_res = BenchmarkResult(model=model.name, backend="MLX", batch_size=batch_size)
        mean, std = _time_fn(mlx_batched_eval, n_iter=n_iter, eval_fn=lambda r: mx.eval([x for t in r for x in t]))
        mlx_res.batched_eval_mean_us = mean
        mlx_res.batched_eval_std_us = std

        results.append((jax_res, mlx_res))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_single_results(results: List[Tuple[BenchmarkResult, BenchmarkResult]]):
    print("\n" + "=" * 80)
    print("SINGLE-DEVICE EVALUATION BENCHMARK")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<12} {'Backend':<6} {'Lines':>6} {'1st eval':>10} "
          f"{'Mean (us)':>10} {'Std (us)':>10} {'Max |diff|':>12} {'Max rel':>10}")
    print("-" * 80)

    for jax_r, mlx_r in results:
        print(f"{jax_r.model:<12} {'JAX':<6} {jax_r.code_lines:>6} "
              f"{jax_r.first_eval_time*1e3:>9.2f}ms "
              f"{jax_r.eval_mean_us:>10.1f} {jax_r.eval_std_us:>10.1f} "
              f"{'(ref)':>12} {'(ref)':>10}")
        print(f"{'':12} {'MLX':<6} {mlx_r.code_lines:>6} "
              f"{mlx_r.first_eval_time*1e3:>9.2f}ms "
              f"{mlx_r.eval_mean_us:>10.1f} {mlx_r.eval_std_us:>10.1f} "
              f"{mlx_r.max_abs_diff:>12.2e} {mlx_r.max_rel_diff:>10.2e}")

        speedup = jax_r.eval_mean_us / mlx_r.eval_mean_us if mlx_r.eval_mean_us > 0 else 0
        winner = "MLX" if speedup > 1 else "JAX"
        print(f"{'':12} {'':6} {'':6} {'':10} "
              f"  -> {winner} {max(speedup, 1/speedup):.1f}x faster")
        print()


def print_batched_results(results: List[Tuple[BenchmarkResult, BenchmarkResult]]):
    if not results:
        return

    print("\n" + "=" * 80)
    print("BATCHED EVALUATION BENCHMARK (JAX vmap vs MLX loop)")
    print("=" * 80)

    print(f"\n{'Model':<12} {'Batch':>6} {'JAX vmap (us)':>14} {'MLX loop (us)':>14} {'Speedup':>10}")
    print("-" * 60)

    for jax_r, mlx_r in results:
        speedup = mlx_r.batched_eval_mean_us / jax_r.batched_eval_mean_us if jax_r.batched_eval_mean_us > 0 else 0
        winner = "JAX" if speedup > 1 else "MLX"
        print(f"{jax_r.model:<12} {jax_r.batch_size:>6} "
              f"{jax_r.batched_eval_mean_us:>13.1f} {mlx_r.batched_eval_mean_us:>13.1f} "
              f"  {winner} {max(speedup, 1/speedup):.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX vs MLX device model evaluation")
    parser.add_argument(
        "--models", default="resistor,diode",
        help="Comma-separated model names (default: resistor,diode)",
    )
    parser.add_argument("--n-iter", type=int, default=1000, help="Iterations for timing")
    parser.add_argument("--batched", action="store_true", help="Run batched benchmark")
    parser.add_argument(
        "--batch-sizes", default="1,10,100,1000",
        help="Batch sizes for batched benchmark",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    # Check MLX availability
    try:
        import mlx.core as mx
        print(f"MLX {mx.__version__}, device: {mx.default_device()}")
    except ImportError:
        print("ERROR: mlx not installed. Run: uv add --optional mlx mlx")
        return 1

    import jax
    print(f"JAX {jax.__version__}, platform: {jax.default_backend()}")
    print()

    # Install code capture
    _install_code_capture()

    models = args.models.split(",")
    single_results = []

    for name in models:
        name = name.strip()
        if name not in MODEL_PATHS:
            print(f"Unknown model: {name} (available: {', '.join(MODEL_PATHS)})")
            continue

        print(f"Compiling {name}...")
        model = compile_model(name)
        print(f"  eval: {len(model.eval_code.splitlines())} lines JAX -> "
              f"{len(model.mlx_eval_code.splitlines())} lines MLX")

        print(f"Benchmarking {name} (n={args.n_iter})...")
        jax_r, mlx_r = run_single_benchmark(model, n_iter=args.n_iter)
        single_results.append((jax_r, mlx_r))

    print_single_results(single_results)

    if args.batched:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        batched_results = []
        for name in models:
            name = name.strip()
            if name not in MODEL_PATHS:
                continue
            print(f"\nBatched benchmark for {name} (sizes: {batch_sizes})...")
            model = compile_model(name)
            batch_res = run_batched_benchmark(model, batch_sizes, n_iter=min(args.n_iter, 100))
            batched_results.extend(batch_res)

        print_batched_results(batched_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
