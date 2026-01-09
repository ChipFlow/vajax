#!/usr/bin/env python3
"""Debug first NR iteration from V=0 to understand why it fails."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine

def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("Debug First NR Iteration from V=0")
    print("="*80)
    print("\nVACSK iteration 1:")
    print("  Input:  V(1-9) = 0V, V(vdd) = 1.2V")
    print("  Output: V(1-9) rise (PMOSes charge nodes)")
    print("  Expected: Positive ΔV for all internal nodes")
    print()

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run minimal transient to setup solver
    result = engine.run_transient(t_stop=0.0, dt=0.1e-9)

    # Get the cached solver
    nr_solve = engine._cached_nr_solve
    device_arrays = engine._device_arrays
    setup = engine._transient_setup_cache
    n_nodes = engine.num_nodes
    n_unknowns = n_nodes - 1

    # Get build_system for manual inspection
    build_system, _ = engine._make_gpu_resident_build_system_fn(
        source_device_data=setup['source_device_data'],
        vmapped_fns=setup.get('vmapped_fns', {}),
        static_inputs_cache=setup.get('static_inputs_cache', {}),
        n_unknowns=n_unknowns,
        use_dense=True,
    )

    # Initial voltage: V=0 everywhere except VDD
    V = jnp.zeros(n_nodes, dtype=jnp.float64)
    vdd_idx = engine.node_names['vdd']
    V = V.at[vdd_idx].set(1.2)

    print(f"Initial voltage vector:")
    print(f"  V[0] (gnd) = {float(V[0]):.3f}V")
    print(f"  V[1] = {float(V[1]):.3f}V")
    print(f"  V[vdd={vdd_idx}] = {float(V[vdd_idx]):.3f}V")
    print()

    # Get DC source values
    vsource_dc_vals = jnp.array([1.2])
    isource_dc_vals = jnp.array([0.0])
    Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

    # Build system at V=0
    print("Building Jacobian and residual at V=0...")
    J, f, Q = build_system(V, vsource_dc_vals, isource_dc_vals, Q_prev, 0.0, device_arrays)

    print(f"\nJacobian diagonal (first 5 nodes):")
    for i in range(min(5, n_unknowns)):
        print(f"  J[{i},{i}] = {float(J[i,i]):.6e} S")

    print(f"\nResidual (first 5 nodes):")
    for i in range(min(5, n_unknowns)):
        node_name = next((n for n, idx in engine.node_names.items() if idx == i+1), str(i+1))
        print(f"  f[{i}] (node {node_name}) = {float(f[i]):.6e} A")

    print(f"\nExpected behavior:")
    print("  - Residual should be NEGATIVE (current flowing OUT from PMOS)")
    print("  - ΔV = -J^-1 * f should be POSITIVE (nodes rise toward VDD)")
    print()

    # Run one NR iteration
    print("Running first NR iteration...")
    V_new, iters, converged, max_f, Q_new = nr_solve(
        V, vsource_dc_vals, isource_dc_vals, Q_prev, 0.0, device_arrays
    )

    print(f"\nAfter iteration {int(iters)}:")
    print(f"  V[1] = {float(V_new[1]):.6f}V (change: {float(V_new[1] - V[1]):+.6f}V)")
    print(f"  Converged: {bool(converged)}")
    print(f"  Max residual: {float(max_f):.6e}A")
    print()

    if float(V_new[1]) < 0:
        print("⚠️  ERROR: V[1] went NEGATIVE!")
        print("   This indicates:")
        print("     - Wrong current direction in device models, OR")
        print("     - Wrong Jacobian sign, OR")
        print("     - Wrong residual sign")
    elif float(V_new[1]) > 0.1:
        print("✓ GOOD: V[1] increased (nodes charging up)")
        print("   This is expected behavior matching VACASK")
    else:
        print("⚠️  UNEXPECTED: V[1] barely changed")

    return 0

if __name__ == "__main__":
    sys.exit(main())
