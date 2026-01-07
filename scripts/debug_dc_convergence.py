#!/usr/bin/env python3
"""Debug DC convergence - check if direct NR succeeds or fails."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import logging

# Enable INFO logging for jax_spice only
logging.basicConfig(level=logging.WARNING)  # Suppress JAX debug spam
logging.getLogger("jax_spice").setLevel(logging.INFO)
logging.getLogger("jax_spice").addHandler(logging.StreamHandler())

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("DC Convergence Debug (watch for homotopy messages)")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run transient - this will compute DC first
    print("\nRunning transient (watch for DC solver messages)...")
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    print(f"\nFinal DC: V(1) = {float(result.voltages['1'][0]):.6f}V")
    print(f"Expected: V(1) = 0.660597V (VACASK)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
