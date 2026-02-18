#!/usr/bin/env python3
"""Quick diagnostic: test graetz NR convergence with current settings."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_SPICE_NO_PROGRESS"] = "1"

import time

from jax_spice.benchmarks.registry import get_benchmark

info = get_benchmark("graetz")
sim_path = info.sim_path

from jax_spice.analysis.engine import CircuitEngine

engine = CircuitEngine(sim_path)
engine.parse()

# Print current options
opts = engine.options
print(f"abstol={opts.abstol}, vntol={opts.vntol}, reltol={opts.reltol}")
print(f"tran_itl={opts.tran_itl}, op_itl={opts.op_itl}")
print(f"gmin={opts.gmin}, tran_gshunt={opts.tran_gshunt}")
print(f"tran_method={opts.tran_method}")

# Run transient for 100ms (5 periods of 50Hz)

t0 = time.time()
try:
    engine.prepare(
        t_stop=100e-3,
        temperature=300.15,
    )
    result = engine.run_transient()
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    times = result.times
    n_steps = len(times)
    print(f"Steps: {n_steps}")
    print(f"Time range: {float(times[0]):.6e} to {float(times[-1]):.6e}")

    # Print voltage ranges for key nodes
    for node_name in ["inp", "inn", "outp", "outn"]:
        if node_name in result.voltages:
            v = result.voltages[node_name]
            print(f"  {node_name}: [{float(v.min()):.4f}, {float(v.max()):.4f}]")

    # Check for common-mode drift
    if "inp" in result.voltages and "inn" in result.voltages:
        cm = (result.voltages["inp"] + result.voltages["inn"]) / 2
        print(f"  CM range: [{float(cm.min()):.4f}, {float(cm.max()):.4f}]")

        # Check differential (should be V_source)
        diff = result.voltages["inp"] - result.voltages["inn"]
        print(f"  Diff (V_source): [{float(diff.min()):.4f}, {float(diff.max()):.4f}]")

        # Output voltage (rectified)
        if "outp" in result.voltages and "outn" in result.voltages:
            vout = result.voltages["outp"] - result.voltages["outn"]
            print(f"  Vout (outp-outn): [{float(vout.min()):.4f}, {float(vout.max()):.4f}]")

    # Stats
    if hasattr(result, "stats") and result.stats:
        stats = result.stats
        for key in [
            "rejected_steps",
            "nr_failures",
            "convergence_rate",
            "total_nr_iterations",
            "max_nr_iterations",
        ]:
            if key in stats:
                print(f"  {key}: {stats[key]}")

except Exception as e:
    elapsed = time.time() - t0
    print(f"\nFailed after {elapsed:.1f}s: {e}")
    import traceback

    traceback.print_exc()
