#!/usr/bin/env python3
"""Test DC solver with V=0 initial guess (VACASK style) and debug NR iterations."""

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
    print("Test V=0 Initial Guess (VACASK Style)")
    print("="*80)
    print("\nVACSK starts from V=0 and converges in 4 iterations.")
    print("Let's see what happens with JAX-SPICE...\n")

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Modify solver to add debug output
    # We need to trace NR iterations from V=0

    # For now, just run and see result
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    v1 = float(result.voltages['1'][0])
    print(f"V(1) from DC: {v1:.6f}V")
    print(f"Expected (VACASK): 0.660597V")
    print(f"Error: {(v1 - 0.660597) / 0.660597 * 100:.2f}%")

    if abs(v1 - 0.660597) > 0.1:
        print("\n⚠️  FAILED: DC did not converge to correct solution from V=0")
        print("VACASK converges from V=0 in 4 iterations, but JAX-SPICE cannot.")
        print("This indicates a difference in:")
        print("  - Device model evaluation")
        print("  - Jacobian computation")
        print("  - Stamping logic")
        print("  - Or convergence behavior")
    else:
        print("\n✓ SUCCESS: Converged correctly from V=0 like VACASK!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
