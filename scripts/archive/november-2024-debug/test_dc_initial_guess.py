#!/usr/bin/env python3
"""Test DC solver with different initial guesses."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine

# Test different initial guess fractions
test_cases = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

print("="*80)
print("DC Solver with Different Initial Guesses (VDD = 1.2V)")
print("="*80)
print(f"{'Initial (V)':>12} {'DC Result (V)':>15} {'Error vs 0.66V':>18}")
print("-"*80)

for frac in test_cases:
    # Modify the engine's DC initialization
    from jax_spice import analysis
    orig_compute_dc = analysis.engine.CircuitEngine._compute_dc_operating_point

    def patched_compute_dc(self, *args, **kwargs):
        # Temporarily override mid_rail calculation
        vdd_value = self._get_vdd_value()
        test_voltage = vdd_value * frac

        # Call original but we'll inject our voltage
        # This is a hack - better to modify the source
        result = orig_compute_dc(self, *args, **kwargs)
        return result

    # Don't patch, just modify source and test
    # Instead, edit engine.py before each run
    pass

# Actually, let's just run the comparison script which reads the current value from engine.py

sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

for frac in test_cases:
    # Read engine.py
    engine_path = Path(__file__).parent.parent / "jax_spice" / "analysis" / "engine.py"
    with open(engine_path) as f:
        content = f.read()

    # Replace the mid_rail line
    old_line = content.split("mid_rail = vdd_value *")[1].split("#")[0].strip().split()[0]
    new_content = content.replace(
        f"mid_rail = vdd_value * {old_line}",
        f"mid_rail = vdd_value * {frac}"
    )

    with open(engine_path, "w") as f:
        f.write(new_content)

    # Run simulation
    engine = CircuitEngine(sim_path)
    engine.parse()
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    v1 = float(result.voltages['1'][0])
    initial_v = 1.2 * frac
    error = v1 - 0.660597

    print(f"{initial_v:>12.3f} {v1:>15.6f} {error:>18.6f}")

# Restore original
with open(engine_path) as f:
    content = f.read()
new_content = content.replace(
    f"mid_rail = vdd_value * {test_cases[-1]}",
    f"mid_rail = vdd_value * 0.5"
)
with open(engine_path, "w") as f:
    f.write(new_content)

print("="*80)
print("Expected: 0.660597V (VACASK)")
