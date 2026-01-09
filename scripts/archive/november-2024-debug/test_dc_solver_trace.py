#!/usr/bin/env python3
"""Test DC solver with debug tracing to see if it's actually called."""

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
    print("DC Solver Trace Test")
    print("="*80)
    print("\nThis will run a minimal transient which triggers DC computation.")
    print("Look for DEBUG: messages showing DC solver execution.\n")

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run minimal transient to trigger DC
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"V(1) from transient result: {float(result.voltages['1'][0]):.6f}V")
    print(f"Expected (VACASK): 0.660597V")
    print(f"Error: {(float(result.voltages['1'][0]) - 0.660597) / 0.660597 * 100:.2f}%")

    return 0

if __name__ == "__main__":
    sys.exit(main())
