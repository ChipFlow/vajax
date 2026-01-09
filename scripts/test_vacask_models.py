#!/usr/bin/env -S uv run --script
"""Test compilation and code generation for VACASK device models.

Tests all .va files in vendor/VACASK/devices/
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import traceback

# Add scripts to path for mir_codegen
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mir_codegen import generate_init_function, generate_eval_function

VACASK_DEVICES = REPO_ROOT / "vendor" / "VACASK" / "devices"


@dataclass
class TestResult:
    model_name: str
    va_path: Path
    compile_ok: bool
    init_gen_ok: bool
    eval_gen_ok: bool
    init_run_ok: bool
    eval_run_ok: bool
    error: Optional[str] = None
    num_params: int = 0
    num_nodes: int = 0
    num_cache: int = 0


def test_model(va_path: Path) -> TestResult:
    """Test a single Verilog-A model."""
    import openvaf_py

    # Get relative path from VACASK devices dir
    try:
        rel_path = va_path.relative_to(VACASK_DEVICES)
        model_name = str(rel_path).replace(".va", "")
    except ValueError:
        model_name = va_path.stem

    result = TestResult(
        model_name=model_name,
        va_path=va_path,
        compile_ok=False,
        init_gen_ok=False,
        eval_gen_ok=False,
        init_run_ok=False,
        eval_run_ok=False,
    )

    # Step 1: Compile
    try:
        modules = openvaf_py.compile_va(str(va_path))
        if not modules:
            result.error = "No modules returned from compilation"
            return result
        module = modules[0]
        result.compile_ok = True
        result.num_params = len(module.param_names)
        result.num_nodes = len(module.nodes)
    except Exception as e:
        result.error = f"Compilation failed: {e}"
        return result

    # Step 2: Generate init function
    try:
        init_fn = generate_init_function(module)
        result.init_gen_ok = True
    except Exception as e:
        result.error = f"Init generation failed: {e}"
        return result

    # Step 3: Generate eval function
    try:
        eval_fn = generate_eval_function(module)
        result.eval_gen_ok = True
    except Exception as e:
        result.error = f"Eval generation failed: {e}"
        return result

    # Step 4: Run init with default params
    try:
        metadata = module.get_codegen_metadata()
        init_param_mapping = metadata.get('init_param_mapping', {})

        init_params = {}
        for name in init_param_mapping.keys():
            if name.endswith('_given'):
                init_params[name] = False
            elif name == '$temperature':
                init_params[name] = 300.15
            elif name == 'Tnom':
                init_params[name] = 300.15
            elif name == 'mfactor':
                init_params[name] = 1.0
            else:
                init_params[name] = 0.0

        cache = init_fn(**init_params)
        result.num_cache = len(cache) if cache else 0
        result.init_run_ok = True
    except Exception as e:
        result.error = f"Init execution failed: {e}"
        return result

    # Step 5: Run eval with default params
    try:
        eval_param_mapping = metadata.get('eval_param_mapping', {})

        eval_params = {}
        for name in eval_param_mapping.keys():
            if name.startswith('V('):
                eval_params[name] = 0.0
            elif name == '$temperature':
                eval_params[name] = 300.15
            elif name == 'mfactor':
                eval_params[name] = 1.0
            else:
                eval_params[name] = 0.0

        eval_result = eval_fn(cache, **eval_params)
        result.eval_run_ok = True

        assert 'residuals_resist' in eval_result
        assert 'residuals_react' in eval_result

    except Exception as e:
        result.error = f"Eval execution failed: {e}"
        return result

    return result


def main():
    if not VACASK_DEVICES.exists():
        print(f"VACASK devices not found at {VACASK_DEVICES}")
        print("Clone with: git clone https://github.com/koenenwmn/VACASK vendor/VACASK")
        return False

    # Find all .va files
    va_files = sorted(VACASK_DEVICES.glob("**/*.va"))
    print(f"Found {len(va_files)} Verilog-A models in VACASK/devices/\n")

    results = []
    passed = 0
    failed = 0

    for va_path in va_files:
        result = test_model(va_path)
        results.append(result)

        if result.eval_run_ok:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        stages = []
        if result.compile_ok:
            stages.append("compile")
        if result.init_gen_ok:
            stages.append("init_gen")
        if result.eval_gen_ok:
            stages.append("eval_gen")
        if result.init_run_ok:
            stages.append("init_run")
        if result.eval_run_ok:
            stages.append("eval_run")

        stage_str = ", ".join(stages) if stages else "none"
        print(f"{status} {result.model_name:45} [{stage_str}]")
        if result.error and not result.eval_run_ok:
            print(f"       Error: {result.error[:80]}...")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(results)} models")
    print("=" * 70)

    if failed > 0:
        print("\nFailed models:")
        for r in results:
            if not r.eval_run_ok:
                print(f"  {r.model_name}")
                if r.error:
                    print(f"    Error: {r.error}")

    return failed == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
