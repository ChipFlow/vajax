#!/usr/bin/env python3
"""Compare DC operating point: JAX-SPICE vs VACASK."""

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
    print("DC Operating Point Comparison: JAX-SPICE vs VACASK")
    print("="*80)

    # Load VACASK reference data
    with open(vacask_ref) as f:
        vacask_data = json.load(f)

    # Get VACASK DC voltage from transient data (first timestep)
    vacask_v1_array = vacask_data['transient_v1']['voltages_v']
    vacask_dc_v1 = vacask_v1_array[0]
    print(f"\nVACASK DC operating point:")
    print(f"  V(1) = {vacask_dc_v1:.6f}V")

    # Run JAX-SPICE DC analysis
    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run a minimal transient to trigger DC computation
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    jax_dc_v1 = float(result.voltages['1'][0])
    print(f"\nJAX-SPICE DC operating point:")
    print(f"  V(1) = {jax_dc_v1:.6f}V")

    # Compare
    diff = jax_dc_v1 - vacask_dc_v1
    pct_diff = (diff / vacask_dc_v1) * 100

    print(f"\nDifference:")
    print(f"  ΔV(1) = {diff:.6f}V ({pct_diff:+.2f}%)")

    if abs(pct_diff) > 1.0:
        print(f"\n⚠️  WARNING: DC operating points differ by {abs(pct_diff):.1f}%!")
        print(f"   This could affect transient analysis significantly.")

        # Check all nodes
        print(f"\nChecking all nodes:")
        for i in range(1, 10):
            node_name = str(i)
            if node_name in result.voltages:
                jax_v = float(result.voltages[node_name][0])
                print(f"  V({i}) = {jax_v:.6f}V")

        # Also check VDD
        if 'vdd' in result.voltages:
            jax_vdd = float(result.voltages['vdd'][0])
            print(f"  V(vdd) = {jax_vdd:.6f}V")
    else:
        print(f"\n✓ DC operating points match within 1%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
