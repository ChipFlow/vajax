#!/usr/bin/env python3
"""Validate generated Python code against openvaf-py native evaluation.

This compares:
1. openvaf-py's MIR interpreter (reference)
2. Our generated Python code from MIR

Both use the same MIR source, so they should produce identical results.
"""

import sys
import math
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def extract_outputs(result_dict, metadata):
    """Extract residuals and Jacobian from generated code result dictionary.

    Args:
        result_dict: Dict returned by generated eval function (MIR var names → values)
        metadata: Metadata dict from model.get_codegen_metadata()

    Returns:
        (residuals, jacobian) tuple matching format of run_init_eval()
        - residuals: List[(resist, react)]
        - jacobian: List[(row, col, resist, react)]
    """
    # Extract residuals
    residuals = []
    for res_info in metadata['residuals']:
        resist_var = res_info['resist_var']
        react_var = res_info['react_var']
        resist = result_dict[resist_var]
        react = result_dict[react_var]
        residuals.append((resist, react))

    # Extract Jacobian
    jacobian = []
    for jac_info in metadata['jacobian']:
        row = jac_info['row']
        col = jac_info['col']
        resist_var = jac_info['resist_var']
        react_var = jac_info['react_var']
        resist = result_dict[resist_var]
        react = result_dict[react_var]
        jacobian.append((row, col, resist, react))

    return residuals, jacobian


def validate_model(model_name, model_path, test_cases):
    """Validate generated code for a single model.

    Args:
        model_name: Name of model (e.g., "capacitor", "psp103")
        model_path: Path to .va file
        test_cases: List of parameter dicts to test

    Returns:
        True if all tests pass, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Validating {model_name}")
    print(f"{'='*80}\n")

    try:
        import openvaf_py
        from jax_spice.codegen.mir_parser import parse_mir_dict
        from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir
        from jax_spice.codegen.eval_mir_codegen import generate_eval_from_mir
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    # Compile model
    print(f"1. Compiling {model_path}...")
    try:
        modules = openvaf_py.compile_va(model_path)
        model = modules[0]
        print(f"   ✓ Compiled: {model.name}")
        print(f"   ✓ Cache slots: {model.num_cached_values}")
        print(f"   ✓ Residuals: {model.num_residuals}")
        print(f"   ✓ Jacobian entries: {model.num_jacobian}")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}")
        return False

    # Get MIR
    print(f"\n2. Loading MIR...")
    try:
        init_mir_dict = model.get_init_mir_instructions()
        eval_mir_dict = model.get_mir_instructions()
        cache_mapping = init_mir_dict.get('cache_mapping', [])

        print(f"   ✓ Init: {len(init_mir_dict['instructions'])} instructions")
        print(f"   ✓ Eval: {len(eval_mir_dict['instructions'])} instructions")
        print(f"   ✓ Cache: {len(cache_mapping)} slots")
    except Exception as e:
        print(f"   ✗ Failed to get MIR: {e}")
        return False

    # Parse MIR
    print(f"\n3. Parsing MIR...")
    try:
        init_mir = parse_mir_dict(init_mir_dict)
        eval_mir = parse_mir_dict(eval_mir_dict)
        print(f"   ✓ Parsed init MIR")
        print(f"   ✓ Parsed eval MIR")
    except Exception as e:
        print(f"   ✗ MIR parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create parameter mappings (use MIR names directly)
    print(f"\n4. Creating parameter mappings...")
    init_param_map = {p.name: p.name for p in init_mir.params}
    eval_param_map = {p.name: p.name for p in eval_mir.params}

    for const_name in init_mir.constants.keys():
        init_param_map[const_name] = const_name
    for const_name in eval_mir.constants.keys():
        eval_param_map[const_name] = const_name

    cache_tuples = [(entry['init_value'], entry['eval_param']) for entry in cache_mapping]
    cache_param_indices = [entry['eval_param'] for entry in cache_mapping]

    # Generate Python code
    print(f"\n5. Generating Python code...")
    try:
        setup_instance_code = generate_setup_instance_from_mir(
            init_mir,
            init_param_map,
            cache_tuples,
            model_name.lower()
        )

        eval_code = generate_eval_from_mir(
            eval_mir,
            eval_param_map,
            cache_param_indices,
            model_name.lower()
        )

        print(f"   ✓ Generated setup_instance_{model_name.lower()}()")
        print(f"   ✓ Generated eval_{model_name.lower()}()")
    except Exception as e:
        print(f"   ✗ Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Execute generated code
    print(f"\n6. Loading generated functions...")
    namespace = {'math': math}
    try:
        exec(setup_instance_code, namespace)
        exec(eval_code, namespace)

        setup_instance_fn = namespace[f'setup_instance_{model_name.lower()}']
        eval_fn = namespace[f'eval_{model_name.lower()}']

        print(f"   ✓ Loaded generated functions")

        # Debug: inspect the actual function signatures
        import inspect
        setup_sig = inspect.signature(setup_instance_fn)
        eval_sig = inspect.signature(eval_fn)
        print(f"   ℹ setup_instance signature: {setup_sig}")
        print(f"   ℹ eval signature: {eval_sig}")

    except Exception as e:
        print(f"   ✗ Failed to execute generated code: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get comprehensive metadata for code generation and validation
    print(f"\n7. Getting code generation metadata...")

    try:
        metadata = model.get_codegen_metadata()
        print(f"   ✓ Got metadata with {len(metadata)} fields")
        print(f"   ✓ Init params: {len(metadata['init_param_mapping'])} parameters")
        print(f"   ✓ Eval params: {len(metadata['eval_param_mapping'])} parameters")
        print(f"   ✓ Cache slots: {metadata['num_cache_slots']}")
        print(f"   ✓ Residuals: {metadata['num_residuals']}")
        print(f"   ✓ Jacobian entries: {metadata['num_jacobian']}")
    except Exception as e:
        print(f"   ✗ Failed to get metadata: {e}")
        return False

    # Run validation tests
    print(f"\n8. Running validation tests...")
    print(f"{'='*80}")

    all_passed = True
    for i, test_params in enumerate(test_cases):
        print(f"\nTest {i+1}/{len(test_cases)}: {test_params}")
        print("-" * 80)

        try:
            # Get reference results from openvaf-py native evaluation
            ref_residuals, ref_jacobian = model.run_init_eval(test_params)

            print(f"\n  Reference (openvaf-py native):")
            print(f"    Residuals: {ref_residuals}")
            if len(ref_jacobian) <= 5:
                print(f"    Jacobian: {ref_jacobian}")
            else:
                print(f"    Jacobian entries: {len(ref_jacobian)} (showing first 3)")
                print(f"              {ref_jacobian[:3]}")

            # Call generated Python code using metadata
            print(f"\n  Generated (Python code):")

            # Step 1: Call setup_instance with init params
            init_kwargs = {}
            for semantic_name in metadata['init_param_mapping'].keys():
                if semantic_name in test_params:
                    init_kwargs[semantic_name] = test_params[semantic_name]
                else:
                    init_kwargs[semantic_name] = 0.0

            cache = setup_instance_fn(**init_kwargs)
            print(f"    Cache: {cache[:3]}..." if len(cache) > 3 else f"    Cache: {cache}")

            # Step 2: Call eval with eval params + cache
            # Build positional args in MIR variable order (sorted by value index)
            # NOTE: Generated function includes ALL MIR params (even hidden_state)
            eval_params_ordered = []
            for name, value_idx in zip(model.param_names, model.param_value_indices):
                eval_params_ordered.append((name, value_idx))

            # Sort by value index to match generated function signature
            eval_params_ordered.sort(key=lambda x: x[1])

            eval_args = []
            for semantic_name, _ in eval_params_ordered:
                if semantic_name in test_params:
                    eval_args.append(test_params[semantic_name])
                else:
                    eval_args.append(0.0)

            gen_result_dict = eval_fn(*eval_args, cache=cache)

            # Debug: Check what keys are in the result
            if i == 0:  # Only print for first test
                print(f"    Result dict keys (first 10): {list(gen_result_dict.keys())[:10]}")
                print(f"    Checking Jacobian variables from metadata:")
                for jac_info in metadata['jacobian'][:2]:  # Check first 2
                    react_var = jac_info['react_var']
                    value = gen_result_dict.get(react_var, 'NOT FOUND')
                    print(f"      {react_var} = {value}")

            # Step 3: Extract residuals and Jacobian using metadata
            gen_residuals, gen_jacobian = extract_outputs(gen_result_dict, metadata)

            print(f"    Residuals: {gen_residuals}")
            if len(gen_jacobian) <= 5:
                print(f"    Jacobian: {gen_jacobian}")
            else:
                print(f"    Jacobian entries: {len(gen_jacobian)} (showing first 3)")
                print(f"              {gen_jacobian[:3]}")

            # Step 4: Numerical comparison
            print(f"\n  Comparison:")

            # Compare residuals
            res_matches = []
            for i, (ref, gen) in enumerate(zip(ref_residuals, gen_residuals)):
                ref_resist, ref_react = ref
                gen_resist, gen_react = gen
                resist_diff = abs(ref_resist - gen_resist)
                react_diff = abs(ref_react - gen_react)
                match = resist_diff < 1e-12 and react_diff < 1e-12
                res_matches.append(match)
                if not match:
                    print(f"    Residual {i}: resist_diff={resist_diff:.2e}, react_diff={react_diff:.2e}")

            # Compare Jacobian
            jac_matches = []
            for i, (ref, gen) in enumerate(zip(ref_jacobian, gen_jacobian)):
                ref_row, ref_col, ref_resist, ref_react = ref
                gen_row, gen_col, gen_resist, gen_react = gen
                resist_diff = abs(ref_resist - gen_resist)
                react_diff = abs(ref_react - gen_react)
                match = resist_diff < 1e-12 and react_diff < 1e-12
                jac_matches.append(match)
                if not match and i < 3:  # Show first few mismatches
                    print(f"    Jacobian [{ref_row},{ref_col}]: resist_diff={resist_diff:.2e}, react_diff={react_diff:.2e}")

            all_match = all(res_matches) and all(jac_matches)
            if all_match:
                print(f"    ✓ Perfect match! All values within 1e-12 tolerance")
                print(f"\n  ✓ Test PASSED - Generated code produces identical results!")
            else:
                res_pass_rate = sum(res_matches) / len(res_matches) * 100 if res_matches else 0
                jac_pass_rate = sum(jac_matches) / len(jac_matches) * 100 if jac_matches else 0
                print(f"    ⚠️  Residuals: {res_pass_rate:.1f}% match")
                print(f"    ⚠️  Jacobian: {jac_pass_rate:.1f}% match")
                print(f"\n  ⚠️  Test PARTIAL - Some numerical differences detected")
                all_passed = False

        except Exception as e:
            print(f"\n  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print(f"✓ {model_name} validation PASSED")
    else:
        print(f"✗ {model_name} validation FAILED")
    print(f"{'='*80}")

    return all_passed


def main():
    """Run validation tests on multiple models."""
    print("="*80)
    print("Code Generation Validation Suite")
    print("="*80)
    print("\nComparing generated Python code vs openvaf-py native evaluation")
    print("Both use the same MIR source, so results should be identical.\n")

    results = {}

    # Test 1: Capacitor (simple, 2 cache slots)
    print("\n" + "="*80)
    print("TEST 1: CAPACITOR")
    print("="*80)

    capacitor_tests = [
        # Test with explicit capacitance
        {'c': 1e-9, 'c_given': True, 'V_A_B': 0.0, 'mfactor': 1.0},
        {'c': 1e-9, 'c_given': True, 'V_A_B': 1.0, 'mfactor': 1.0},
        {'c': 1e-9, 'c_given': True, 'V_A_B': 5.0, 'mfactor': 1.0},
        # Test with default capacitance
        {'c_given': False, 'V_A_B': 1.0, 'mfactor': 1.0},
        # Test with scaled mfactor
        {'c': 1e-6, 'c_given': True, 'V_A_B': 2.5, 'mfactor': 2.5},
    ]

    results['capacitor'] = validate_model(
        'capacitor',
        str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'),
        capacitor_tests
    )

    # Test 2: Diode (more complex, ~16 cache slots)
    print("\n" + "="*80)
    print("TEST 2: DIODE")
    print("="*80)

    diode_tests = [
        # Forward bias - add default diode parameters
        {
            'V_A_B': 0.7,
            'mfactor': 1.0,
            'TEMP': 300.0,
            'AREA': 1.0,
            'IS': 1e-14,     # Saturation current
            'N': 1.0,         # Ideality factor
            'CJ0': 1e-15,     # Zero-bias junction capacitance
            'VJ': 0.7,        # Built-in voltage
            'M': 0.5,         # Grading coefficient
        },
        # Reverse bias
        {
            'V_A_B': -5.0,
            'mfactor': 1.0,
            'TEMP': 300.0,
            'AREA': 1.0,
            'IS': 1e-14,
            'N': 1.0,
            'CJ0': 1e-15,
            'VJ': 0.7,
            'M': 0.5,
        },
        # Zero bias
        {
            'V_A_B': 0.0,
            'mfactor': 1.0,
            'TEMP': 300.0,
            'AREA': 1.0,
            'IS': 1e-14,
            'N': 1.0,
            'CJ0': 1e-15,
            'VJ': 0.7,
            'M': 0.5,
        },
    ]

    results['diode'] = validate_model(
        'diode',
        str(Path(__file__).parent.parent / 'vendor/VACASK/devices/diode.va'),
        diode_tests
    )

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for model, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{model:20} {status}")

    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("SUCCESS! All models validated successfully!")
        print("="*80)
        print("\nGenerated Python code produces identical results to openvaf-py native evaluation.")
        return 0
    else:
        print("PARTIAL SUCCESS - Some models failed validation")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
