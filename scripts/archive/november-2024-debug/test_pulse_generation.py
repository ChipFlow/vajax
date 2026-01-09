#!/usr/bin/env python3
"""Test that pulse source function generates correct values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("Testing Pulse Source Generation")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Find the pulse isource
    pulse_dev = None
    for dev in engine.devices:
        if dev['model'] == 'isource' and dev['params'].get('type') == 'pulse':
            pulse_dev = dev
            break

    if pulse_dev is None:
        print("No pulse isource found!")
        return 1

    print(f"\nPulse parameters:")
    print(f"  val0: {pulse_dev['params']['val0']}")
    print(f"  val1: {pulse_dev['params']['val1']}")
    print(f"  delay: {pulse_dev['params']['delay']}")
    print(f"  rise: {pulse_dev['params']['rise']}")
    print(f"  fall: {pulse_dev['params']['fall']}")
    print(f"  width: {pulse_dev['params']['width']}")

    # Get the source function
    source_fn = engine._get_source_fn_for_device(pulse_dev)

    if source_fn is None:
        print("\nERROR: No source function generated!")
        return 1

    # Test the function at various times
    print("\nPulse values at different times:")
    print(f"{'Time (ns)':>10} {'Current (µA)':>15} {'Expected':>15}")
    print("-"*42)

    test_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
    for t_ns in test_times:
        t = t_ns * 1e-9
        current = source_fn(t)

        # Expected value
        delay = 1e-9
        width = 1e-9
        val0 = 0.0
        val1 = 10e-6

        if t < delay:
            expected = val0
        elif t < delay + width:
            expected = val1
        else:
            expected = val0

        current_ua = float(current) * 1e6
        expected_ua = float(expected) * 1e6

        match = "✓" if abs(current - expected) < 1e-9 else "✗"
        print(f"{t_ns:>10.1f} {current_ua:>15.6f} {expected_ua:>15.6f}  {match}")

    # Test that function works with JAX arrays
    print("\n" + "="*80)
    print("Testing with JAX array input")
    print("="*80)

    times_array = jnp.linspace(0, 10e-9, 101)
    try:
        # Try to vmap the function
        vmapped_fn = jax.vmap(source_fn)
        currents = vmapped_fn(times_array)

        print(f"✓ Function is vmappable")
        print(f"  Shape: {currents.shape}")
        print(f"  Min: {float(jnp.min(currents))*1e6:.6f} µA")
        print(f"  Max: {float(jnp.max(currents))*1e6:.6f} µA")
        print(f"  Non-zero values: {jnp.sum(currents != 0.0)}/{len(currents)}")

    except Exception as e:
        print(f"✗ Function is NOT vmappable: {e}")

        # Try individual calls
        print("\nTrying individual calls:")
        currents = []
        for t in times_array:
            try:
                i = source_fn(float(t))
                currents.append(i)
            except Exception as e2:
                print(f"  Error at t={float(t)*1e9:.2f}ns: {e2}")
                currents.append(0.0)

        currents = jnp.array(currents)
        print(f"  Collected {len(currents)} values")
        print(f"  Min: {float(jnp.min(currents))*1e6:.6f} µA")
        print(f"  Max: {float(jnp.max(currents))*1e6:.6f} µA")

    return 0


if __name__ == "__main__":
    sys.exit(main())
