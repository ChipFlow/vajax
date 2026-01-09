#!/usr/bin/env python3
"""Test that isource actually affects the residual vector.

This directly builds the MNA system with and without the pulse current
to verify that the isource is being stamped correctly.
"""

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
    print("Testing isource Stamping")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run a minimal transient to get setup
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    # Get setup from cache
    setup = engine._transient_setup_cache
    if setup is None:
        print("No transient setup cache found!")
        return 1

    device_arrays = engine._device_arrays
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

    # Get build_system from cached NR solver
    # (we need to extract this from the nr_solve function)
    print(f"\nCircuit info:")
    print(f"  External nodes: {n_external}")
    print(f"  Unknowns: {n_unknowns}")
    print(f"  Node 1 index: {engine.node_names.get('1', 'NOT FOUND')}")

    # Get DC voltage
    V_dc = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltages in result.voltages.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V_dc = V_dc.at[idx].set(float(voltages[0]))

    print(f"\nDC voltage at node 1: {float(V_dc[1]):.6f}V")

    # Build vsource values (VDD = 1.2V)
    vsource_vals = jnp.zeros(1, dtype=jnp.float64)  # One vsource (vdd)
    vsource_vals = vsource_vals.at[0].set(1.2)

    # Build system with NO isource (0µA)
    isource_0 = jnp.zeros(1, dtype=jnp.float64)
    Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

    print("\n" + "="*80)
    print("Building system with isource = 0µA")
    print("="*80)

    try:
        # We need to call the actual build_system function
        # It's embedded in the NR solver, so let's try to access it via the solver factory
        from jax_spice.analysis.solver_factories import make_dense_solver

        # Get source device data from setup
        source_device_data = setup['source_device_data']
        device_internal_nodes = setup.get('device_internal_nodes', {})
        vmapped_fns = setup.get('vmapped_fns', {})
        static_inputs_cache = setup.get('static_inputs_cache', {})

        #  Build build_system function
        build_system_fn, _ = engine._make_gpu_resident_build_system_fn(
            source_device_data=source_device_data,
            vmapped_fns=vmapped_fns,
            static_inputs_cache=static_inputs_cache,
            n_unknowns=n_unknowns,
            use_dense=True,
        )

        J_0, f_0, Q_0 = build_system_fn(V_dc, vsource_vals, isource_0, Q_prev, 0.0, device_arrays)

        print(f"Residual f with isource=0 (f excludes ground!):")
        print(f"  f[0] (node 1 residual): {float(f_0[0]):.6e} A")
        print(f"  f[1] (node 2 residual): {float(f_0[1]):.6e} A")
        print(f"  f[2] (node 3 residual): {float(f_0[2]):.6e} A")
        print(f"  Max |f|: {float(jnp.max(jnp.abs(f_0))):.6e} A")

        # Build system WITH isource (10µA)
        print("\n" + "="*80)
        print("Building system with isource = 10µA")
        print("="*80)

        isource_10 = jnp.array([10e-6], dtype=jnp.float64)
        J_10, f_10, Q_10 = build_system_fn(V_dc, vsource_vals, isource_10, Q_prev, 0.0, device_arrays)

        print(f"Residual f with isource=10µA (f excludes ground!):")
        print(f"  f[0] (node 1 residual): {float(f_10[0]):.6e} A")
        print(f"  f[1] (node 2 residual): {float(f_10[1]):.6e} A")
        print(f"  f[2] (node 3 residual): {float(f_10[2]):.6e} A")
        print(f"  Max |f|: {float(jnp.max(jnp.abs(f_10))):.6e} A")

        # Compare residuals
        print("\n" + "="*80)
        print("Residual difference (10µA - 0µA)")
        print("="*80)

        df = f_10 - f_0
        print(f"  df[0] (node 1 residual): {float(df[0]):.6e} A")
        print(f"  df[1] (node 2 residual): {float(df[1]):.6e} A")
        print(f"  df[2] (node 3 residual): {float(df[2]):.6e} A")

        # The isource is connected (0 1), so current flows from ground (0) to node 1
        # Since ground is not in the residual vector, we only see:
        # f[0] (node 1 residual) += 10µA (current enters node 1)
        expected_df0 = 10e-6  # f[0] is node 1's residual

        actual_df0 = float(df[0])
        if abs(actual_df0 - expected_df0) < 1e-9:
            print(f"\n✓ isource is being stamped correctly!")
            print(f"  Expected df[0] (node 1 residual) = {expected_df0:.6e} A")
            print(f"  Actual df[0]   (node 1 residual) = {actual_df0:.6e} A")
        else:
            print(f"\n✗ isource stamping is WRONG!")
            print(f"  Expected df[0] (node 1 residual) = {expected_df0:.6e} A")
            print(f"  Actual df[0]   (node 1 residual) = {actual_df0:.6e} A")
            print(f"  Error: {abs(actual_df0 - expected_df0):.6e} A")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
