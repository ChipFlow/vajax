# MIR-to-Python Code Generation: Restart Prompt

## Context

We are implementing a JAX-based SPICE simulator. The previous approach (November 2024) attempted to directly implement OpenVAF's MIR interpreter in Python, but discovered our PSP103 model was broken. We are now **reimplementing from scratch** by comparing each stage against VACASK's OSDI implementation as ground truth.

## Goal

Generate correct Python code from OpenVAF's MIR (Mid-level Intermediate Representation) that produces **numerically identical** results to VACASK's OSDI (Open Source Device Interface) compiled libraries.

The approach is:
1. Use VACASK OSDI as the **reference implementation** (loaded via ctypes)
2. Build Python code generators stage-by-stage from OpenVAF MIR
3. Validate each stage against OSDI before proceeding

## Previous Work to IGNORE

**CRITICAL**: The following code from November 2024 is **BROKEN** and should be ignored:
- `openvaf-py/src/lib.rs::run_init_eval()` - Our old broken MIR interpreter
- Most of the existing `openvaf-py` validation code
- Any references to "our implementation working"

These were implemented without proper validation against OSDI and are known to be incorrect.

## Where We Started (Before Getting Derailed)

The correct starting point was:
- **Script**: `scripts/test_vacask_osdi_psp103.py`
- **Goal**: Load VACASK's compiled OSDI library via ctypes and call it directly
- **Purpose**: Establish OSDI as ground truth before building anything

## What We Discovered (Key Learnings)

### 1. PHI Nodes Are Critical

**Document**: `docs/PHI_NODE_BUG.md`

- PHI nodes are SSA (Static Single Assignment) merge points in control flow
- They select values based on which predecessor block was executed
- Example from capacitor init:
  ```
  v33 = phi {
    v18 (from block8)   // User-provided capacitance
    v38 (from block14)  // Default capacitance (1e-12)
  }
  ```
- Our code generators (`control_flow_codegen.py`) **DO** handle phi nodes correctly
- Generated code uses `if prev_block == "blockX": v33 = v18` pattern

### 2. Parameter Mapping Was Wrong

**Discovery**: `scripts/check_init_param_arrays.py`

OpenVAF's init function has parameters with duplicate names but different kinds:
```python
init_param_names = ['c', 'mfactor', 'c']
init_param_kinds = ['param', 'sysfun', 'param_given']
init_param_value_indices = [18, 20, 32]
```

This means:
- v18 = c (the value)
- v20 = mfactor
- v32 = c_given (the boolean flag)

When building a dict mapping `name → variable`, the second 'c' overwrites the first!

**Fix Applied**: `openvaf-py/src/lib.rs:1091-1096`
```rust
// Append "_given" suffix for param_given to avoid name collision
let map_name = if kind.contains("param_given") {
    format!("{}_given", name)
} else {
    name.to_string()
};
```

### 3. Generated Code Works When Mapping Is Correct

**Test**: `scripts/fix_and_test_init.py`

With manual correct mapping:
```python
manual_param_map = {
    'v18': 'c',           # Capacitance value
    'v20': 'mfactor',     # Multiplication factor
    'v32': 'c_given',     # Given flag
}
```

Results were **perfect**:
```
Test: c=1e-9, c_given=True
  Generated cache: [1e-09, -1e-09] ✓

Test: c=1e-9, c_given=False
  Generated cache: [1e-12, -1e-12] ✓

Test: c=2e-9, c_given=True
  Generated cache: [4e-09, -4e-09] ✓
```

This proves:
- PHI node code generation works
- Control flow handling works
- We just need correct parameter metadata

## What Needs to Happen Next

### Step 1: Complete openvaf-py Rebuild

The fix at `openvaf-py/src/lib.rs:1091-1096` needs to be compiled:
```bash
cd openvaf-py
maturin develop
cd ..
```

### Step 2: Return to OSDI Validation

Go back to the **original plan**:

1. **Load OSDI library** via ctypes (`scripts/test_vacask_osdi_psp103.py`)
   - Define OSDI API structures from spec
   - Call `osdi_descriptor()` to get model metadata
   - Call `osdi_init_instance()` and `osdi_eval()` functions
   - This is the **ground truth reference**

2. **Compare against our generated code**
   - Use corrected metadata from rebuilt openvaf-py
   - Generate init/eval Python functions from MIR
   - Call with same parameters as OSDI
   - Validate outputs match within tolerance

3. **Start with simple models**
   - Capacitor (2 cache slots, simple phi)
   - Diode (16 cache slots, junction conditions)
   - PSP103 (462 cache slots, complex control flow)

### Step 3: Validation Framework

Once OSDI reference works:
```python
# Pseudocode
osdi_lib = ctypes.CDLL("capacitor.osdi")
descriptor = osdi_lib.osdi_descriptor()

# Get OSDI results
osdi_cache = call_osdi_init(params)
osdi_residuals, osdi_jacobian = call_osdi_eval(voltages, osdi_cache)

# Get generated code results
gen_cache = setup_instance_capacitor(**params)
gen_results = eval_capacitor(*voltages, cache=gen_cache)
gen_residuals, gen_jacobian = extract_outputs(gen_results, metadata)

# Compare
assert np.allclose(osdi_cache, gen_cache)
assert np.allclose(osdi_residuals, gen_residuals)
assert np.allclose(osdi_jacobian, gen_jacobian)
```

## Key Files and References

### Documentation Created
- `docs/PHI_NODE_BUG.md` - Why PHI nodes are critical, how they work
- `docs/VALIDATION_STATUS.md` - Status before we got derailed (partially obsolete)
- `docs/CODEGEN_METADATA_API.md` - Metadata API reference
- `docs/PARAMETER_MAPPING_SOLUTION.md` - Parameter mapping details
- `/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/openvaf-py/vendor/OpenVAF/docs/JACOBIAN_METADATA.md` - OpenVAF's internal docs on derivatives vs values

### Code Generators (These Work!)
- `jax_spice/codegen/mir_parser.py` - Parses MIR dict to structured objects
- `jax_spice/codegen/control_flow_codegen.py` - Generates Python with control flow, handles PHI
- `jax_spice/codegen/setup_instance_mir_codegen.py` - Wraps control flow for init function
- `jax_spice/codegen/eval_mir_codegen.py` - Wraps control flow for eval function

### Investigation Scripts
- `scripts/test_vacask_osdi_psp103.py` - **START HERE** - OSDI loading
- `scripts/fix_and_test_init.py` - Proves code gen works with correct mapping
- `scripts/check_init_param_arrays.py` - Shows the duplicate name problem
- `scripts/check_generated_phi_code.py` - Shows PHI code generation works

### What to Delete/Ignore
- `scripts/validate_codegen_vs_native.py` - Compares against broken `run_init_eval`
- `scripts/debug_*.py` - Investigation scripts for wrong problem
- `scripts/investigate_*.py` - More dead-end investigations

## Success Criteria

We'll know we're done when:

1. ✅ OSDI library loads via ctypes
2. ✅ OSDI metadata is readable
3. ✅ OSDI init/eval calls succeed
4. ✅ Generated Python code for capacitor matches OSDI within 1e-12
5. ✅ Generated Python code for diode matches OSDI within 1e-12
6. ✅ Generated Python code for PSP103 matches OSDI within 1e-12

Then we can move to JAX code generation.

## Summary

**We got derailed** by trying to validate against our own broken implementation (`run_init_eval`).

**The correct approach**:
1. Use VACASK OSDI (compiled C library) as ground truth
2. Load it via ctypes
3. Compare our generated Python against it
4. Fix any differences

**We already fixed the main bugs**:
- PHI nodes work ✓
- Control flow works ✓
- Parameter mapping fix is ready (needs rebuild) ✓

**Next action**: Rebuild openvaf-py with the metadata fix, then return to `scripts/test_vacask_osdi_psp103.py` and implement OSDI ctypes interface.
