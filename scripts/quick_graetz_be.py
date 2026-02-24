#!/usr/bin/env python3
"""Quick graetz test with Backward Euler to compare against Gear2."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

import numpy as np

from vajax.analysis.engine import CircuitEngine
from vajax.analysis.integration import IntegrationMethod

sim_path = Path("vendor/VACASK/benchmark/graetz/vacask/runme.sim")
engine = CircuitEngine(sim_path)
engine.parse()

# Override integration method to BE via analysis_params (not just engine.options)
# This ensures the transient solver actually uses BE
engine.analysis_params["tran_method"] = IntegrationMethod.BACKWARD_EULER
print(f"Integration method in analysis_params: {engine.analysis_params['tran_method']}")

t_stop = 600e-6
dt = 1e-6

print(f"Running graetz with BE: t_stop={t_stop}, dt={dt}")
engine.prepare(t_stop=t_stop, dt=dt)
result = engine.run_transient()

print(f"Result: {result.num_steps} steps")
times = np.array(result.times)

for name in result.node_names[:5]:
    v = np.array(result.voltage(name))
    has_nan = np.any(np.isnan(v))
    has_inf = np.any(np.isinf(v))
    print(f"  Node {name}: min={v.min():.4f} max={v.max():.4f} nan={has_nan} inf={has_inf}")

# Check voltage at specific times
node0 = result.node_names[0]
v0 = np.array(result.voltage(node0))
print(f"\nVoltage at node '{node0}' over time:")
for frac in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    idx = min(int(frac * len(times)) - 1, len(times) - 1)
    if idx >= 0:
        print(f"  t={times[idx]:.3e} ({frac * 100:.0f}%): V={v0[idx]:.6f}")

# Find first "unreasonable" voltage (> 50V)
for i in range(len(v0)):
    if abs(v0[i]) > 50:
        print(f"\n  First unreasonable voltage at step {i}, t={times[i]:.3e}: V={v0[i]:.2e}")
        if i > 0:
            print(f"  Previous step t={times[i - 1]:.3e}: V={v0[i - 1]:.6f}")
        break
else:
    print("\n  All voltages reasonable (< 50V)")
