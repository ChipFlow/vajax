#!/usr/bin/env python3
"""Compare PSP103 device evaluation at DC operating point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import json

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"
    vacask_ref = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "debug_traced_data.json"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("PSP103 Device Evaluation at DC Operating Point")
    print("="*80)

    # Load VACASK reference
    with open(vacask_ref) as f:
        vacask_data = json.load(f)

    # Get VACASK Jacobian diagonal at DC
    vacask_jacobian = vacask_data['jacobian_dc']['diagonal_entries']
    vacask_g1 = vacask_jacobian['node_1']
    print(f"\nVACASK Jacobian diagonal at V(1)=0.661V:")
    print(f"  G(1,1) = {vacask_g1:.6e} S")

    # Run JAX-SPICE and check Jacobian
    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run minimal transient to get setup
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    setup = engine._transient_setup_cache
    device_arrays = engine._device_arrays
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

    # Get DC voltage
    V_dc = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltages in result.voltages.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V_dc = V_dc.at[idx].set(float(voltages[0]))

    print(f"\nJAX-SPICE DC operating point: V(1) = {float(V_dc[1]):.6f}V")

    # Build system and get Jacobian
    build_system_fn, _ = engine._make_gpu_resident_build_system_fn(
        source_device_data=setup['source_device_data'],
        vmapped_fns=setup.get('vmapped_fns', {}),
        static_inputs_cache=setup.get('static_inputs_cache', {}),
        n_unknowns=n_unknowns,
        use_dense=True,
    )

    vsource_vals = jnp.array([1.2])
    isource_vals = jnp.array([0.0])
    Q_prev = jnp.zeros(n_unknowns)

    J_dc, f_dc, Q_dc = build_system_fn(V_dc, vsource_vals, isource_vals, Q_prev, 0.0, device_arrays)

    jax_g1 = float(J_dc[0, 0])
    print(f"\nJAX-SPICE Jacobian diagonal at V(1)={float(V_dc[1]):.6f}V:")
    print(f"  G(1,1) = {jax_g1:.6e} S")

    # Compare
    ratio = jax_g1 / vacask_g1
    print(f"\nRatio: {ratio:.4f}x")

    if abs(ratio - 1.0) > 0.05:
        print(f"\n⚠️  WARNING: Jacobian differs by {abs(ratio - 1.0) * 100:.1f}%!")
        print(f"   This suggests PSP103 models produce different results.")

        # Check residual at DC
        print(f"\nResidual at DC:")
        print(f"  Max |f| = {float(jnp.max(jnp.abs(f_dc))):.6e} A")
        print(f"  Should be < 1e-12 A for converged DC")

        # Try evaluating at VACASK's DC point (0.661V)
        print(f"\n" + "="*80)
        print("Forcing JAX-SPICE to use VACASK's DC voltage (0.661V)")
        print("="*80)

        V_vacask = jnp.full(n_external, 0.660597, dtype=jnp.float64)
        V_vacask = V_vacask.at[0].set(0.0)  # Ground
        V_vacask = V_vacask.at[engine.node_names['vdd']].set(1.2)  # VDD

        J_vacask, f_vacask, Q_vacask = build_system_fn(V_vacask, vsource_vals, isource_vals, Q_prev, 0.0, device_arrays)

        print(f"  Residual at V=0.661V:")
        print(f"    Max |f| = {float(jnp.max(jnp.abs(f_vacask))):.6e} A")
        print(f"    f[0] (node 1) = {float(f_vacask[0]):.6e} A")

        print(f"  Jacobian at V=0.661V:")
        print(f"    G(1,1) = {float(J_vacask[0, 0]):.6e} S")

        # If residual is small, then 0.661V is also a valid DC point for JAX-SPICE
        if abs(float(f_vacask[0])) < 1e-9:
            print(f"\n⚠️  MULTIPLE DC SOLUTIONS: V=0.661V is ALSO a valid DC point!")
            print(f"   JAX-SPICE converged to different solution (V=0.600V)")
            print(f"   This suggests bistability or different convergence path")

    return 0


if __name__ == "__main__":
    sys.exit(main())
