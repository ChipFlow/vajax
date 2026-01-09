"""Test compilation of PDK Verilog-A models to JAX"""

import openvaf_py
import openvaf_jax
import time
from pathlib import Path

# PDK model paths
GF130_PATH = Path("/Users/roberttaylor/Code/ChipFlow/Backend/pdk-gf130")
IHP_PATH = Path("/Users/roberttaylor/Code/ChipFlow/PDK/IHP-Open-PDK")
SKYWATER_PATH = Path("/Users/roberttaylor/Code/ChipFlow/PDK/skywater-pdk")

def find_va_files(base_path):
    """Find all .va files under a path"""
    return list(base_path.glob("**/*.va"))

def test_model(va_path, allow_analog_in_cond=False):
    """Test compiling a single model"""
    try:
        start = time.time()
        modules = openvaf_py.compile_va(str(va_path), allow_analog_in_cond=allow_analog_in_cond)
        compile_time = time.time() - start

        if not modules:
            return None, "No modules returned"

        m = modules[0]

        # Try to create JAX function
        try:
            start = time.time()
            translator = openvaf_jax.OpenVAFToJAX(m)
            jax_fn = translator.translate()
            jax_time = time.time() - start
        except Exception as e:
            return None, f"JAX translation failed: {e}"

        return {
            'module': m.name,
            'nodes': len(m.nodes),
            'params': sum(1 for k in m.param_kinds if k == 'param'),
            'hidden': sum(1 for k in m.param_kinds if k == 'hidden_state'),
            'jacobian': m.num_jacobian,
            'compile_time': compile_time,
            'jax_time': jax_time,
        }, None

    except Exception as e:
        return None, str(e)


def test_pdk(name, base_path, allow_analog_in_cond=False):
    """Test all models in a PDK"""
    print(f"\n{'='*80}")
    print(f"Testing {name}")
    print(f"{'='*80}")

    va_files = find_va_files(base_path)
    if not va_files:
        print(f"No .va files found in {base_path}")
        return []

    results = []
    passed = 0
    failed = 0

    for va_path in sorted(va_files):
        rel_path = va_path.relative_to(base_path)
        result, error = test_model(va_path, allow_analog_in_cond)

        if result:
            passed += 1
            print(f"OK   {str(rel_path)[:50]:<50s} "
                  f"nodes={result['nodes']:2d} params={result['params']:4d} "
                  f"hidden={result['hidden']:4d} jac={result['jacobian']:3d} "
                  f"compile={result['compile_time']:.2f}s jax={result['jax_time']:.2f}s")
            results.append({
                'path': str(rel_path),
                **result
            })
        else:
            failed += 1
            short_error = error[:60] if error else "Unknown error"
            print(f"FAIL {str(rel_path)[:50]:<50s} {short_error}")

    print(f"\n{name} Summary: {passed} passed, {failed} failed out of {len(va_files)} models")
    return results


if __name__ == '__main__':
    all_results = []

    # Test GF130 (needs allow_analog_in_cond)
    if GF130_PATH.exists():
        results = test_pdk("GF130 PDK", GF130_PATH, allow_analog_in_cond=True)
        all_results.extend(results)

    # Test IHP
    if IHP_PATH.exists():
        results = test_pdk("IHP-Open-PDK", IHP_PATH)
        all_results.extend(results)

    # Test Skywater
    if SKYWATER_PATH.exists():
        results = test_pdk("Skywater PDK", SKYWATER_PATH)
        all_results.extend(results)

    # Overall summary
    print(f"\n{'='*80}")
    print("Overall Summary")
    print(f"{'='*80}")
    print(f"Total models compiled to JAX: {len(all_results)}")

    if all_results:
        # Show complexity distribution
        print("\nTop 10 by complexity (hidden states):")
        for r in sorted(all_results, key=lambda x: -x['hidden'])[:10]:
            print(f"  {r['path'][:40]:<40s} hidden={r['hidden']:5d} params={r['params']:4d}")
