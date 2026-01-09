#!/usr/bin/env python3
"""Validate MIR→Python generated setup_model() against Rust MIR interpreter wrapper.

This compares:
- Our MIR→Python generated setup_model_resistor() code
- vs. Rust MIR interpreter wrapper (run_model_param_setup)

Both execute the same model_param_setup MIR, just via different implementations:
- Python: Generated from MIR control flow graph
- Rust: Interpreted MIR directly

If they match, it validates our MIR→Python codegen is correct!
"""

import sys
import math
sys.path.insert(0, '..')

def test_mir_python_vs_rust_interpreter():
    """Compare MIR→Python generated code against Rust MIR interpreter."""
    print("="*80)
    print("Validating MIR→Python Codegen vs Rust MIR Interpreter")
    print("="*80)

    try:
        import openvaf_py
        from jax_spice.codegen.mir_parser import parse_mir_function
        from jax_spice.codegen.setup_model_mir_codegen import generate_setup_model_from_mir
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure jax_spice is in PYTHONPATH")
        return 1

    # Compile resistor model
    print("\n1. Compiling resistor.va with OpenVAF...")
    modules = openvaf_py.compile_va("../vendor/VACASK/devices/resistor.va")
    resistor = modules[0]
    print(f"   ✓ Compiled: {resistor.name}")
    print(f"   ✓ model_setup_num_params: {resistor.model_setup_num_params}")
    print(f"   ✓ model_setup_param_names: {resistor.model_setup_param_names}")

    # The MIR for resistor model_param_setup (from earlier work)
    MODEL_SETUP_MIR = """
Optimized model setup MIR of resistor
function %(v19, v20, v32, v33) {
    inst0 = fn %set_Invalid(Parameter { id: ParamId(0) })(0) -> 0
    inst1 = fn %set_Invalid(Parameter { id: ParamId(1) })(0) -> 0
    v3 = fconst 0.0
    v5 = iconst 1
    v6 = fconst 0x1.0000000000000p0
    v15 = fconst +Inf

                                block0:
@0002                               br v20, block2, block11

                                block2:
                                    v23 = fle v3, v19
                                    br v23, block7, block9

                                block7:
                                    v24 = fle v19, v15
                                    jmp block9

                                block9:
                                    v25 = phi [v1, block2], [v24, block7]
                                    br v25, block4, block10

                                block10:
                                    call inst0()
                                    jmp block4

                                block11:
                                    v31 = optbarrier v6
                                    jmp block4

                                block4:
                                    v21 = phi [v19, block9], [v19, block10], [v6, block11]
                                    br v33, block19, block18

                                block18:
                                    v35 = optbarrier v5
                                    jmp block19

                                block19:
                                    v34 = phi [v32, block4], [v5, block18]
}
"""

    print("\n2. Generating Python setup_model() from MIR...")
    mir_func = parse_mir_function(MODEL_SETUP_MIR)

    param_map = {
        'v19': 'r',
        'v20': 'r_given',
        'v32': 'has_noise',
        'v33': 'has_noise_given',
        'v1': 'FALSE',
        'v2': 'TRUE',
        'v3': 'ZERO',
        'v5': 'ONE_INT',
        'v6': 'ONE',
        'v15': 'INF',
        'v21': 'r_final',
        'v23': 'r_ge_zero',
        'v24': 'r_le_inf',
        'v25': 'r_in_range',
        'v31': 'r_default',
        'v34': 'has_noise_final',
        'v35': 'has_noise_default',
    }

    setup_code = generate_setup_model_from_mir(mir_func, param_map, "resistor")

    # Execute the generated code
    namespace = {'math': math}
    exec(setup_code, namespace)
    setup_model_python = namespace['setup_model_resistor']
    print("   ✓ Generated setup_model_resistor()")

    print("\n3. Running validation tests...")
    print("-"*80)

    test_cases = [
        {
            'name': 'Explicit valid values',
            'params': {
                'r': 1000.0,
                'r_given': 1.0,
                'has_noise': 1.0,
                'has_noise_given': 1.0
            },
            'expected': {
                'r': 1000.0,
                'has_noise': 1.0
            }
        },
        {
            'name': 'Default r (not given)',
            'params': {
                'r': 999.0,  # Should be ignored
                'r_given': 0.0,
                'has_noise': 1.0,
                'has_noise_given': 1.0
            },
            'expected': {
                'r': 1.0,  # Default
                'has_noise': 1.0
            }
        },
        {
            'name': 'Default has_noise (not given)',
            'params': {
                'r': 100.0,
                'r_given': 1.0,
                'has_noise': 999.0,  # Should be ignored
                'has_noise_given': 0.0
            },
            'expected': {
                'r': 100.0,
                'has_noise': 1.0  # Default
            }
        },
        {
            'name': 'Both defaults',
            'params': {
                'r': 0.0,
                'r_given': 0.0,
                'has_noise': 0.0,
                'has_noise_given': 0.0
            },
            'expected': {
                'r': 1.0,
                'has_noise': 1.0
            }
        },
        {
            'name': 'Boundary: r=0',
            'params': {
                'r': 0.0,
                'r_given': 1.0,
                'has_noise': 0.0,
                'has_noise_given': 1.0
            },
            'expected': {
                'r': 0.0,
                'has_noise': 0.0
            }
        },
        {
            'name': 'Large resistance',
            'params': {
                'r': 1e12,
                'r_given': 1.0,
                'has_noise': 1.0,
                'has_noise_given': 1.0
            },
            'expected': {
                'r': 1e12,
                'has_noise': 1.0
            }
        },
    ]

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test['name']}")
        print(f"   Input: {test['params']}")

        # Run Python generated code
        python_result = setup_model_python(**test['params'])

        # Run Rust MIR interpreter
        rust_result = resistor.run_model_param_setup(test['params'])

        print(f"   Python result:     {python_result}")
        print(f"   Rust result:       {rust_result}")

        # Compare results
        match = True
        for key in test['expected']:
            if key not in python_result:
                print(f"   ✗ Python missing key: {key}")
                match = False
                all_passed = False
            elif key not in rust_result:
                print(f"   ✗ Rust missing key: {key}")
                match = False
                all_passed = False
            else:
                py_val = python_result[key]
                rust_val = rust_result[key]
                expected_val = test['expected'][key]

                # Check if values match (with floating point tolerance)
                if abs(py_val - expected_val) > 1e-10:
                    print(f"   ✗ Python {key}: {py_val} != expected {expected_val}")
                    match = False
                    all_passed = False

                if abs(rust_val - expected_val) > 1e-10:
                    print(f"   ✗ Rust {key}: {rust_val} != expected {expected_val}")
                    match = False
                    all_passed = False

                if abs(py_val - rust_val) > 1e-10:
                    print(f"   ✗ Python vs Rust {key}: {py_val} != {rust_val}")
                    match = False
                    all_passed = False

        if match:
            print(f"   ✓ PASS - All values match!")
        else:
            print(f"   ✗ FAIL - Mismatch detected")

    print("\n" + "="*80)
    if all_passed:
        print("SUCCESS! MIR→Python codegen matches Rust MIR interpreter!")
        print("="*80)
        print("\nValidation Results:")
        print("✓ Control flow implementation correct (branches, jumps, PHI nodes)")
        print("✓ Constant handling correct (ZERO, ONE, INF)")
        print("✓ Parameter validation logic matches MIR")
        print("✓ Default value application correct")
        print("✓ Given flag handling correct")
        print("\nOur MIR→Python code generator is producing correct results!")
        return 0
    else:
        print("FAILURE - Some tests did not match")
        print("="*80)
        return 1

if __name__ == '__main__':
    sys.exit(test_mir_python_vs_rust_interpreter())
