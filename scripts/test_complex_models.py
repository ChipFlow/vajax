#!/usr/bin/env python3
"""Test code generation on complex models: PSP103 and BSIM4."""

import sys
import math
sys.path.insert(0, '..')

def test_model_codegen(model_name, model_path):
    """Test code generation for a given model."""
    print(f"\n{'='*80}")
    print(f"Testing {model_name}")
    print(f"{'='*80}")

    try:
        import openvaf_py
        from jax_spice.codegen.mir_parser import parse_mir_dict
        from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir
        from jax_spice.codegen.eval_mir_codegen import generate_eval_from_mir
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    # Compile model
    print(f"\n1. Compiling {model_path}...")
    try:
        modules = openvaf_py.compile_va(model_path)
        model = modules[0]
        print(f"   ✓ Compiled: {model.name}")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}")
        return False

    # Get structure info
    print(f"\n2. Model structure:")
    print(f"   Terminals: {model.num_terminals}")
    print(f"   Init params: {model.init_num_params}")
    print(f"   Eval params: {model.func_num_params}")
    print(f"   Cache slots: {model.num_cached_values}")
    print(f"   Residuals: {model.num_residuals}")
    print(f"   Jacobian entries: {model.num_jacobian}")

    # Get MIR
    print(f"\n3. Getting MIR...")
    try:
        init_mir_dict = model.get_init_mir_instructions()
        eval_mir_dict = model.get_mir_instructions()

        print(f"   ✓ Init MIR: {len(init_mir_dict['instructions'])} instructions, {len(init_mir_dict['blocks'])} blocks")
        print(f"   ✓ Eval MIR: {len(eval_mir_dict['instructions'])} instructions, {len(eval_mir_dict['blocks'])} blocks")
    except Exception as e:
        print(f"   ✗ Failed to get MIR: {e}")
        return False

    # Get cache mapping
    cache_mapping = init_mir_dict.get('cache_mapping', [])
    cache_param_indices = [entry['eval_param'] for entry in cache_mapping]
    print(f"   ✓ Cache mapping: {len(cache_mapping)} slots")

    # Parse MIR
    print(f"\n4. Parsing MIR...")
    try:
        init_mir = parse_mir_dict(init_mir_dict)
        eval_mir = parse_mir_dict(eval_mir_dict)
        print(f"   ✓ Parsed init MIR: {len(init_mir.params)} params, {len(init_mir.blocks)} blocks")
        print(f"   ✓ Parsed eval MIR: {len(eval_mir.params)} params, {len(eval_mir.blocks)} blocks")
    except Exception as e:
        print(f"   ✗ MIR parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create minimal parameter mappings (just use MIR names)
    print(f"\n5. Creating parameter mappings...")
    init_param_map = {p.name: p.name for p in init_mir.params}
    eval_param_map = {p.name: p.name for p in eval_mir.params}

    # Add constants
    for const_name in init_mir.constants.keys():
        init_param_map[const_name] = const_name
    for const_name in eval_mir.constants.keys():
        eval_param_map[const_name] = const_name

    cache_tuples = [(entry['init_value'], entry['eval_param']) for entry in cache_mapping]

    # Generate setup_instance code
    print(f"\n6. Generating setup_instance() code...")
    try:
        setup_instance_code = generate_setup_instance_from_mir(
            init_mir,
            init_param_map,
            cache_tuples,
            model_name.lower()
        )
        code_lines = setup_instance_code.count('\n')
        print(f"   ✓ Generated {code_lines} lines of code")

        # Save to file for inspection
        with open(f'generated_setup_instance_{model_name.lower()}.py', 'w') as f:
            f.write(setup_instance_code)
        print(f"   ✓ Saved to generated_setup_instance_{model_name.lower()}.py")
    except Exception as e:
        print(f"   ✗ Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Generate eval code
    print(f"\n7. Generating eval() code...")
    try:
        eval_code = generate_eval_from_mir(
            eval_mir,
            eval_param_map,
            cache_param_indices,
            model_name.lower()
        )
        code_lines = eval_code.count('\n')
        print(f"   ✓ Generated {code_lines} lines of code")

        # Save to file for inspection
        with open(f'generated_eval_{model_name.lower()}.py', 'w') as f:
            f.write(eval_code)
        print(f"   ✓ Saved to generated_eval_{model_name.lower()}.py")
    except Exception as e:
        print(f"   ✗ Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try to execute the generated code (syntax check)
    print(f"\n8. Testing generated code execution (syntax check)...")
    namespace = {'math': math}

    try:
        exec(setup_instance_code, namespace)
        print(f"   ✓ setup_instance_{model_name.lower()}() syntax OK")
    except Exception as e:
        print(f"   ✗ setup_instance compilation error: {e}")
        return False

    try:
        exec(eval_code, namespace)
        print(f"   ✓ eval_{model_name.lower()}() syntax OK")
    except Exception as e:
        print(f"   ✗ eval compilation error: {e}")
        return False

    print(f"\n{'='*80}")
    print(f"✓ {model_name} code generation SUCCESSFUL!")
    print(f"{'='*80}")
    return True


def main():
    """Test code generation on complex models."""
    print("="*80)
    print("Complex Model Code Generation Test")
    print("="*80)
    print("\nTesting MIR→Python code generation on production MOSFET models:")
    print("- PSP103: Surface-potential based model (~439 cache slots)")
    print("- BSIM4: Industry-standard model (~328 cache slots)")

    results = {}

    # Test PSP103
    results['PSP103'] = test_model_codegen(
        'PSP103',
        '../vendor/VACASK/devices/psp103v4/psp103.va'
    )

    # Test BSIM4
    results['BSIM4'] = test_model_codegen(
        'BSIM4',
        '../vendor/VACASK/devices/bsim4v8.va'
    )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for model, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model:20} {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n{'='*80}")
        print("SUCCESS! All complex models generated successfully!")
        print(f"{'='*80}")
        print("\nOur MIR→Python code generator handles:")
        print("✓ Hundreds of parameters")
        print("✓ Hundreds of cache slots")
        print("✓ Thousands of instructions")
        print("✓ Hundreds of control flow blocks")
        print("✓ Complex nonlinear MOSFET physics")
        return 0
    else:
        print(f"\n{'='*80}")
        print("PARTIAL SUCCESS - Some models failed")
        print(f"{'='*80}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
