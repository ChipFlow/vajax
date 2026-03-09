"""Analyze config group specialization effectiveness.

Compares HLO op counts between unified and per-config-group eval functions.
Run after prepare_static_inputs to see SCCP branch elimination impact.

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/analyze_config_groups.py [ring|rc|graetz]
"""

import logging
import os
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy loggers during setup
for name in ["vajax", "openvaf_jax", "jax", "absl"]:
    logging.getLogger(name).setLevel(logging.WARNING)

import jax
import jax.numpy as jnp

import vajax
from vajax.benchmarks.runner import BenchmarkRunner


def count_hlo_ops(fn, *args):
    """Count HLO ops in a traced function."""
    lowered = jax.jit(fn).lower(*args)
    hlo_text = lowered.as_text()

    ops = {}
    for line in hlo_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("{") or line.startswith("}"):
            continue
        m = re.match(r"%\S+\s*=\s*(\S+)", line)
        if m:
            op = m.group(1)
            ops[op] = ops.get(op, 0) + 1

    total = sum(ops.values())
    return total, ops


# Benchmark configs (matching compare_vacask.py)
BENCHMARK_CONFIGS = {
    "rc": {"sim": "vendor/VACASK/simulations/rc/rc.sim", "t_stop": 1e-3, "dt": 5e-8},
    "graetz": {"sim": "vendor/VACASK/simulations/graetz/graetz.sim", "t_stop": 10e-3, "dt": 5e-8},
    "ring": {"sim": "vendor/VACASK/simulations/ring/ring.sim", "t_stop": 1e-6, "dt": 5e-11},
}


def analyze_benchmark(benchmark_name: str):
    """Analyze config groups for a benchmark."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing config groups for: {benchmark_name}")
    logger.info(f"{'='*70}")

    config = BENCHMARK_CONFIGS.get(benchmark_name)
    if not config:
        logger.error(f"Unknown benchmark: {benchmark_name}")
        return

    from pathlib import Path

    sim_path = Path(config["sim"])

    # Use BenchmarkRunner to set up the circuit (it calls run_transient internally)
    # But we need the compiled_models dict. Let's parse and prepare directly.
    from vajax.analysis.dc_operating_point import analyze_dc_operating_point
    from vajax.parser import parse_sim_file

    circuit = parse_sim_file(sim_path)

    # Run a quick transient to populate compiled_models
    runner = BenchmarkRunner()
    result = runner.run_benchmark(sim_path, t_stop=config["t_stop"], dt=config["dt"])

    # Access compiled_models from the transient analyzer
    analyzer = runner._analyzer
    if analyzer is None:
        logger.error("No analyzer available")
        return

    compiled_models = analyzer._compiled_models

    for model_type, compiled in compiled_models.items():
        config_groups = compiled.get("config_groups")
        vmapped_unified = compiled.get("vmapped_split_eval")

        if vmapped_unified is None:
            continue

        shared_params = compiled["shared_params"]
        device_params = compiled["device_params"]
        shared_cache = compiled["shared_cache"]
        device_cache = compiled["device_cache"]
        default_simparams = compiled["default_simparams"]
        n_devices = device_params.shape[0]
        num_limit_states = compiled.get("num_limit_states", 0)
        limit_state = jnp.zeros((n_devices, max(num_limit_states, 1)), dtype=device_params.dtype)

        logger.info(f"\n--- {model_type} ({n_devices} devices) ---")
        total_unified, ops_unified = count_hlo_ops(
            vmapped_unified,
            shared_params,
            device_params,
            shared_cache,
            device_cache,
            default_simparams,
            limit_state,
        )
        compare_unified = ops_unified.get("compare", 0)
        select_unified = ops_unified.get("select", 0)
        logger.info(
            f"  Unified eval:  {total_unified:>6} total ops, "
            f"{compare_unified:>4} compare, {select_unified:>4} select"
        )

        if config_groups is None:
            logger.info("  No config groups (single config)")
            continue

        total_specialized = 0
        compare_specialized = 0
        select_specialized = 0
        for group in config_groups:
            indices = group["device_indices"]
            group_dp = device_params[indices]
            group_dc = device_cache[indices]
            group_ls = limit_state[indices]
            group_eval = group["vmapped_eval"]

            total_g, ops_g = count_hlo_ops(
                group_eval,
                shared_params,
                group_dp,
                shared_cache,
                group_dc,
                default_simparams,
                group_ls,
            )
            compare_g = ops_g.get("compare", 0)
            select_g = ops_g.get("select", 0)
            total_specialized += total_g
            compare_specialized += compare_g
            select_specialized += select_g
            logger.info(
                f"  Config {group['group_id']} ({group['n_devices']} devs): "
                f"{total_g:>6} total ops, {compare_g:>4} compare, {select_g:>4} select  "
                f"static={group['static_config']}"
            )

        logger.info(f"\n  Summary:")
        logger.info(
            f"    Unified:      {total_unified:>6} ops, "
            f"{compare_unified:>4} compare, {select_unified:>4} select"
        )
        avg_specialized = total_specialized // len(config_groups)
        avg_compare = compare_specialized // len(config_groups)
        avg_select = select_specialized // len(config_groups)
        logger.info(
            f"    Specialized:  {avg_specialized:>6} ops/group avg ({len(config_groups)} groups), "
            f"{avg_compare:>4} compare, {avg_select:>4} select"
        )
        delta_ops = total_unified - avg_specialized
        delta_cmp = compare_unified - avg_compare
        delta_sel = select_unified - avg_select
        if total_unified > 0:
            logger.info(
                f"    Per-group reduction: {delta_ops} ops ({100*delta_ops/total_unified:.1f}%), "
                f"{delta_cmp} compare, {delta_sel} select"
            )

    runner.clear()


if __name__ == "__main__":
    benchmarks = sys.argv[1:] if len(sys.argv) > 1 else ["ring"]
    for b in benchmarks:
        analyze_benchmark(b)
