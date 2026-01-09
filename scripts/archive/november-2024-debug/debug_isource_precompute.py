#!/usr/bin/env python3
"""Debug: Check if isource values are pre-computed correctly during transient."""

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
    print("Debugging isource Pre-computation")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run a minimal transient to trigger setup
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    # Get setup from cache
    setup = engine._transient_setup_cache
    if setup is None:
        print("No transient setup cache found!")
        return 1

    source_device_data = setup['source_device_data']

    # Manually pre-compute isource values like scan.py does
    t_stop = 10e-9
    dt = 0.1e-9
    num_timesteps = int(t_stop / dt) + 1
    times = jnp.linspace(0.0, t_stop, num_timesteps)

    print(f"\nSimulation parameters:")
    print(f"  t_stop: {t_stop*1e9:.1f}ns")
    print(f"  dt: {dt*1e9:.3f}ns")
    print(f"  num_timesteps: {num_timesteps}")

    # Pre-compute isource values
    if 'isource' not in source_device_data:
        print("\nERROR: No isource in source_device_data!")
        return 1

    source_names = source_device_data['isource']['names']
    print(f"\nisource names: {source_names}")

    all_values = []
    for name in source_names:
        dev = next((d for d in engine.devices if d['name'] == name), None)
        if dev:
            print(f"\nProcessing isource: {name}")
            print(f"  Params: {dev['params']}")

            src_fn = engine._get_source_fn_for_device(dev)
            if src_fn is not None:
                # Time-varying source - evaluate at all times
                print(f"  Source function found - evaluating at {len(times)} timesteps")
                vals = jax.vmap(src_fn)(times)
                all_values.append(vals)

                # Print statistics
                print(f"\n  Pre-computed values statistics:")
                print(f"    Min: {float(jnp.min(vals))*1e6:.6f} µA")
                print(f"    Max: {float(jnp.max(vals))*1e6:.6f} µA")
                print(f"    Mean: {float(jnp.mean(vals))*1e6:.6f} µA")

                # Check when pulse is active
                active_mask = vals > 5e-6  # More than 5µA
                active_times = times[active_mask] * 1e9
                if len(active_times) > 0:
                    print(f"    Pulse active: {float(active_times[0]):.3f}ns to {float(active_times[-1]):.3f}ns")
                    print(f"    Active timesteps: {jnp.sum(active_mask)}/{len(vals)}")
                else:
                    print(f"    WARNING: Pulse NEVER active (no values > 5µA)!")

                # Print first few values
                print(f"\n  First 30 timesteps:")
                print(f"    {'Time (ns)':>10} {'Current (µA)':>15}")
                for i in range(min(30, len(times))):
                    print(f"    {float(times[i])*1e9:>10.3f} {float(vals[i])*1e6:>15.6f}")

            else:
                # DC source
                dc_val = dev['params'].get('dc', 0.0)
                print(f"  DC source: {dc_val}")
                vals = jnp.full(num_timesteps, float(dc_val))
                all_values.append(vals)

    if all_values:
        all_isource_vals = jnp.stack(all_values, axis=1)
        print(f"\nFinal all_isource_vals shape: {all_isource_vals.shape}")
        print(f"Expected shape: [{num_timesteps}, {len(source_names)}]")

        # Verify pulse is in the pre-computed array
        pulse_vals = all_isource_vals[:, 0]  # Assuming first isource is the pulse
        active_count = jnp.sum(pulse_vals > 5e-6)
        print(f"\nVerification:")
        print(f"  Timesteps with I > 5µA: {int(active_count)}/{num_timesteps}")
        if active_count == 0:
            print(f"  ⚠️  WARNING: No active pulse found in pre-computed array!")
        else:
            print(f"  ✓ Pulse is in pre-computed array")
    else:
        print("\nERROR: No isource values pre-computed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
