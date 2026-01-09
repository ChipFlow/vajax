#!/usr/bin/env python3
"""Debug PSP103 init function to understand parameter flow."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "openvaf-py"))

import openvaf_py
import json

def main():
    # Compile PSP103
    psp103_va = Path(__file__).parent.parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
    modules = openvaf_py.compile_va(str(psp103_va))
    psp103 = modules[0]

    print("=" * 80)
    print("PSP103 Init Function Debug")
    print("=" * 80)
    print()

    # Load model parameters
    params_file = Path(__file__).parent / 'psp103_model_params.json'
    with open(params_file) as f:
        all_params = json.load(f)
    pmos_model_params = all_params['pmos']

    # Build minimal init params (no voltages - init doesn't need them)
    init_params = {}
    for name in psp103.init_param_names:
        if name == '$temperature':
            init_params[name] = 300.0
        elif name.upper() == 'TYPE':
            init_params[name] = -1.0
        elif name.upper() == 'W':
            init_params[name] = 20e-6
        elif name.upper() == 'L':
            init_params[name] = 1e-6
        elif name in pmos_model_params:
            init_params[name] = pmos_model_params[name]
        elif name.lower() == 'mfactor':
            init_params[name] = 1.0
        else:
            init_params[name] = 0.0

    print(f"Init params: {len(init_params)} parameters")
    print(f"Expected: {len(psp103.init_param_names)} parameters")
    print()

    # Check if module has an init function
    if hasattr(psp103, 'run_init'):
        print("Module has run_init() method")
        try:
            cache = psp103.run_init(init_params)
            print(f"✓ Init succeeded!")
            print(f"  Cache values: {len(cache)} entries")
            print(f"  Expected: {psp103.num_cached_values} entries")
            print()

            # Check for inf/nan in cache
            inf_count = sum(1 for v in cache if v == float('inf') or v == float('-inf'))
            nan_count = sum(1 for v in cache if v != v)  # NaN != NaN
            print(f"Cache diagnostics:")
            print(f"  Inf values: {inf_count}")
            print(f"  NaN values: {nan_count}")

            if inf_count > 0:
                print()
                print("Inf value indices:")
                for i, v in enumerate(cache):
                    if v == float('inf') or v == float('-inf'):
                        print(f"  cache[{i}] = {v}")
                        if i < 10:  # Only show first 10
                            pass
        except Exception as e:
            print(f"✗ Init failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Module does NOT have run_init() method")
        print(f"Available methods: {[m for m in dir(psp103) if not m.startswith('_')]}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
