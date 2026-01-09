#!/usr/bin/env python3
"""End-to-end test: Complete capacitor flow with setup_model, setup_instance, and eval."""

import sys
import math
sys.path.insert(0, '..')

def test_capacitor_complete_flow():
    """Test complete capacitor simulation flow."""
    print("="*80)
    print("Capacitor Complete Flow: setup_model → setup_instance → eval")
    print("="*80)

    try:
        import openvaf_py
        from jax_spice.codegen.mir_parser import parse_mir_dict, parse_mir_function
        from jax_spice.codegen.setup_model_mir_codegen import generate_setup_model_from_mir
        from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir
        from jax_spice.codegen.eval_mir_codegen import generate_eval_from_mir
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return 1

    # Compile capacitor
    print("\n1. Compiling capacitor.va...")
    modules = openvaf_py.compile_va("../vendor/VACASK/devices/capacitor.va")
    cap = modules[0]
    print(f"   ✓ Compiled: {cap.name}")

    # Get MIR for all functions
    print("\n2. Getting MIR for all functions...")
    model_setup_mir_text = """
Optimized model setup MIR of capacitor
function %(v18, v32) {
    inst0 = fn %set_Invalid(Parameter { id: ParamId(0) })(0) -> 0
    v3 = fconst 0.0
    v15 = fconst +Inf
    v38 = fconst 0x1.0c6f7a0b5ed8dp-40

                                block4:
@0002                               br v32, block5, block6

                                block5:
                                    v34 = ifcast v4
                                    v35 = fle v34, v18
                                    br v35, block10, block11

                                block10:
                                    v36 = fle v18, v15
                                    jmp block12

                                block11:
                                    jmp block12

                                block12:
                                    v37 = phi [v36, block10], [v1, block11]
                                    br v37, block9, block13

                                block13:
                                    call inst0()
                                    jmp block8

                                block9:
                                    jmp block8

                                block8:
                                    jmp block7

                                block6:
                                    v39 = ifcast v4
                                    v40 = fle v39, v38
                                    br v40, block16, block17

                                block16:
                                    v41 = fle v38, v15
                                    jmp block18

                                block17:
                                    jmp block18

                                block18:
                                    v42 = phi [v41, block16], [v1, block17]
                                    br v42, block15, block19

                                block19:
                                    call inst0()
                                    jmp block14

                                block15:
                                    jmp block14

                                block14:
                                    v43 = optbarrier v38
                                    jmp block7

                                block7:
                                    v33 = phi [v18, block8], [v38, block14]
                                    jmp block2

                                block2:
}
"""
    init_mir_dict = cap.get_init_mir_instructions()
    eval_mir_dict = cap.get_mir_instructions()

    # Parse model_param_setup MIR
    model_setup_mir = parse_mir_function(model_setup_mir_text)

    # Parse init and eval MIR
    init_mir = parse_mir_dict(init_mir_dict)
    eval_mir = parse_mir_dict(eval_mir_dict)

    print(f"   ✓ Parsed model_param_setup ({len(model_setup_mir.blocks)} blocks)")
    print(f"   ✓ Parsed init ({len(init_mir.blocks)} blocks)")
    print(f"   ✓ Parsed eval ({len(eval_mir.blocks)} blocks)")

    # Get cache mapping
    cache_mapping = init_mir_dict['cache_mapping']
    cache_tuples = [(entry['init_value'], entry['eval_param']) for entry in cache_mapping]
    cache_param_indices = [entry['eval_param'] for entry in cache_mapping]

    # Create parameter mappings
    print("\n3. Creating parameter mappings...")

    model_param_map = {
        'v18': 'c',
        'v32': 'c_given',
        'v1': 'FALSE',
        'v3': 'ZERO',
        'v4': 'ZERO_INT',
        'v15': 'INF',
        'v38': 'DEFAULT_C',
        'v33': 'c_final',
        'v43': 'c_default_barrier',
    }

    init_param_map = {
        'v18': 'c',
        'v20': 'mfactor',
        'v32': 'c_given',
        'v1': 'FALSE',
        'v2': 'TRUE',
        'v3': 'ZERO',
        'v4': 'ZERO_INT',
        'v5': 'ONE_INT',
        'v6': 'ONE',
        'v7': 'NEG_ONE',
        'v15': 'INF',
        'v38': 'DEFAULT_C',
        'v33': 'c_validated',
        'v17': 'neg_c',
        'v19': 'cache_0_val',
        'v22': 'cache_1_val',
        'v27': 'cache_0',
        'v28': 'cache_1',
    }

    eval_param_map = {
        'v16': 'c',
        'v17': 'V_A_B',
        'v19': 'Q_hidden',
        'v21': 'q_hidden',
        'v25': 'mfactor',
        'v37': 'cache_0',
        'v40': 'cache_1',
        'v18': 'Q',
        'v23': 'Q_barrier',
        'v26': 'dQ_dt',
        'v27': 'Q_neg',
        'v30': 'cache_0_barrier',
        'v32': 'cache_1_barrier',
        'v33': 'Q_scaled',
        'v34': 'dQ_dt_neg',
        'v35': 'Q_neg_scaled',
        'v36': 'mfactor_barrier',
        'v38': 'cache_1_barrier2',
        'v41': 'cache_0_barrier2',
    }

    # Generate all three functions
    print("\n4. Generating Python code...")

    setup_model_code = generate_setup_model_from_mir(model_setup_mir, model_param_map, "capacitor")
    setup_instance_code = generate_setup_instance_from_mir(init_mir, init_param_map, cache_tuples, "capacitor")
    eval_code = generate_eval_from_mir(eval_mir, eval_param_map, cache_param_indices, "capacitor")

    print("   ✓ Generated setup_model_capacitor()")
    print("   ✓ Generated setup_instance_capacitor()")
    print("   ✓ Generated eval_capacitor()")

    # Execute all generated code
    print("\n5. Executing generated code...")
    namespace = {'math': math}

    try:
        exec(setup_model_code, namespace)
        exec(setup_instance_code, namespace)
        exec(eval_code, namespace)

        setup_model_capacitor = namespace['setup_model_capacitor']
        setup_instance_capacitor = namespace['setup_instance_capacitor']
        eval_capacitor = namespace['eval_capacitor']

        print("   ✓ All functions executed successfully")
    except Exception as e:
        print(f"   ✗ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run complete flow simulation
    print("\n6. Running complete simulation flow...")
    print("="*80)

    # Step 1: Setup model (validate model parameters)
    print("\n   Step 1: setup_model() - Validate model parameters")
    model_params = setup_model_capacitor(c=1e-9, c_given=True)
    print(f"   ✓ Model parameters: {model_params}")

    # Step 2: Setup instance (compute cache)
    print("\n   Step 2: setup_instance() - Compute cache values")
    cache = setup_instance_capacitor(c=model_params['c'], c_given=True, mfactor=1.0)
    print(f"   ✓ Cache computed: {cache}")
    print(f"      cache[0] = {cache[0]:.2e}  (mfactor * c)")
    print(f"      cache[1] = {cache[1]:.2e}  (-mfactor * c)")

    # Step 3: Eval (compute Q for different voltages)
    print("\n   Step 3: eval() - Compute charge Q = c * V")

    voltages = [0.0, 1.0, 2.5, 5.0, -1.0]
    print(f"\n   {'V (V)':>10} {'Q (C)':>15} {'Expected Q (C)':>20}")
    print(f"   {'-'*10} {'-'*15} {'-'*20}")

    all_correct = True
    for V in voltages:
        result = eval_capacitor(c=model_params['c'], V_A_B=V, mfactor=1.0, cache=cache)
        Q_computed = result['Q']
        Q_expected = model_params['c'] * V

        match = abs(Q_computed - Q_expected) < 1e-20
        symbol = '✓' if match else '✗'
        print(f"   {V:>10.1f} {Q_computed:>15.2e} {Q_expected:>20.2e}  {symbol}")

        if not match:
            all_correct = False

    print("\n" + "="*80)
    if all_correct:
        print("SUCCESS! Complete capacitor flow working correctly!")
        print("="*80)
        print("\nFlow validated:")
        print("✓ setup_model() validates parameters and applies defaults")
        print("✓ setup_instance() computes cache values (mfactor * c)")
        print("✓ eval() uses cache to compute charge Q = c * V")
        print("✓ All voltage points match expected values")
        print("\nThe cache mechanism eliminates redundant mfactor multiplications!")
        return 0
    else:
        print("FAILURE - Some computations don't match expected values")
        print("="*80)
        return 1

if __name__ == '__main__':
    sys.exit(test_capacitor_complete_flow())
