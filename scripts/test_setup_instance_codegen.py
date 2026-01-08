#!/usr/bin/env python3
"""Test setup_instance() code generation for capacitor."""

import sys
import math
sys.path.insert(0, '..')

def test_capacitor_setup_instance():
    """Test setup_instance generation for capacitor model."""
    print("="*80)
    print("Testing setup_instance() Code Generation - Capacitor")
    print("="*80)

    try:
        import openvaf_py
        from jax_spice.codegen.mir_parser import parse_mir_dict
        from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return 1

    # Compile capacitor model
    print("\n1. Compiling capacitor.va...")
    modules = openvaf_py.compile_va("../vendor/VACASK/devices/capacitor.va")
    cap = modules[0]
    print(f"   ✓ Compiled: {cap.name}")
    print(f"   ✓ Init params: {cap.init_num_params}")
    print(f"   ✓ Num cached values: {cap.num_cached_values}")

    # Get init MIR
    print("\n2. Getting init MIR...")
    init_mir_dict = cap.get_init_mir_instructions()
    print(f"   ✓ Got init MIR ({len(init_mir_dict['instructions'])} instructions)")

    # Parse MIR to get structured representation
    print("\n3. Parsing init MIR...")
    mir_func = parse_mir_dict(init_mir_dict)
    print(f"   ✓ Parsed MIR function")
    print(f"   ✓ Params: {[p.name for p in mir_func.params]}")
    print(f"   ✓ Blocks: {len(mir_func.blocks)}")

    # Get cache mapping
    cache_mapping = init_mir_dict['cache_mapping']
    cache_tuples = [(entry['init_value'], entry['eval_param']) for entry in cache_mapping]
    print(f"   ✓ Cache mapping: {cache_tuples}")

    # Create parameter name mapping
    print("\n4. Creating parameter mapping...")
    param_map = {
        # Parameters
        'v18': 'c',          # capacitance
        'v20': 'mfactor',    # system function
        'v32': 'c_given',    # given flag

        # Constants
        'v1': 'FALSE',
        'v2': 'TRUE',
        'v3': 'ZERO',
        'v4': 'ZERO_INT',
        'v5': 'ONE_INT',
        'v6': 'ONE',
        'v7': 'NEG_ONE',
        'v15': 'INF',
        'v38': 'DEFAULT_C',  # 1e-12

        # Intermediate values
        'v33': 'c_validated',
        'v17': 'neg_c',
        'v19': 'cache_0_val',  # mfactor * c
        'v22': 'cache_1_val',  # mfactor * (-c)

        # Cache slots
        'v27': 'cache_0',  # optbarrier(v19)
        'v28': 'cache_1',  # optbarrier(v22)
    }

    # Generate setup_instance code
    print("\n5. Generating setup_instance() code...")
    setup_code = generate_setup_instance_from_mir(
        mir_func,
        param_map,
        cache_tuples,
        "capacitor"
    )

    print("\n" + "="*80)
    print("Generated Code:")
    print("="*80)
    print(setup_code)
    print("="*80)

    # Execute and test the generated code
    print("\n6. Testing generated setup_instance()...")

    # Add math module to namespace
    namespace = {'math': math}

    # Execute the generated code
    try:
        exec(setup_code, namespace)
        setup_instance_capacitor = namespace['setup_instance_capacitor']
        print("   ✓ Code executed successfully")
    except Exception as e:
        print(f"   ✗ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test cases
    test_cases = [
        {
            'name': 'Explicit c=1e-9, mfactor=1',
            'params': {'c': 1e-9, 'c_given': 1.0, 'mfactor': 1.0},
            'expected_cache': [1e-9, -1e-9],  # [mfactor*c, -mfactor*c]
        },
        {
            'name': 'Default c (not given)',
            'params': {'c': 999.0, 'c_given': 0.0, 'mfactor': 1.0},
            'expected_cache': [1e-12, -1e-12],  # [mfactor*default, -mfactor*default]
        },
        {
            'name': 'c=1e-6, mfactor=2.5',
            'params': {'c': 1e-6, 'c_given': 1.0, 'mfactor': 2.5},
            'expected_cache': [2.5e-6, -2.5e-6],  # [2.5*1e-6, -2.5*1e-6]
        },
    ]

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test['name']}")
        print(f"   Input: {test['params']}")

        try:
            cache = setup_instance_capacitor(**test['params'])
            print(f"   Result: {cache}")
            print(f"   Expected: {test['expected_cache']}")

            # Check cache values
            if len(cache) != len(test['expected_cache']):
                print(f"   ✗ Cache size mismatch: got {len(cache)}, expected {len(test['expected_cache'])}")
                all_passed = False
            else:
                for j, (got, expected) in enumerate(zip(cache, test['expected_cache'])):
                    if abs(got - expected) > 1e-15:
                        print(f"   ✗ cache[{j}]: {got} != {expected}")
                        all_passed = False
                    else:
                        print(f"   ✓ cache[{j}]: {got} ≈ {expected}")

                if len(cache) == len(test['expected_cache']):
                    if all(abs(g - e) <= 1e-15 for g, e in zip(cache, test['expected_cache'])):
                        print(f"   ✓ PASS")
                    else:
                        print(f"   ✗ FAIL")

        except Exception as e:
            print(f"   ✗ Execution error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("SUCCESS! setup_instance() code generation working!")
        print("="*80)
        return 0
    else:
        print("FAILURE - Some tests did not pass")
        print("="*80)
        return 1

if __name__ == '__main__':
    sys.exit(test_capacitor_setup_instance())
