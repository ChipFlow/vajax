# OSDI Binding Design Options

## ⚠️ CRITICAL UPDATE: MIR Interpreter Not Reliable

**Discovery**: The MIR interpreter has known bugs (PSP103 returns zero residuals).
See `docs/MIR_INTERPRETER_RELIABILITY.md` for details.

**Impact**: We CANNOT use the MIR interpreter as a validation reference.

## Revised Validation Strategy

### For setup_model() Validation

**Option A: VACASK Comparison** (RECOMMENDED) ✅

Compare our generated setup_model() against VACASK operating point:

```python
# 1. Our generated code
setup_result = setup_model_resistor(r=1000.0, r_given=True, ...)

# 2. Run VACASK with same parameters
# Parse operating point from .prn file

# 3. Compare
assert setup_result['r'] == vacask_op_point['r']
```

**Pros**:
- VACASK is known-good (production simulator)
- External validation (not circular)
- Real-world test

**Cons**:
- Requires VACASK integration
- Need .prn file parser
- Slower (external process)

**Complexity**: Medium (already have VACASK framework)

**Option B: Analytical Physics** (Simple models only)

```python
# Resistor: Should accept valid values
result = setup_model(r=1000.0, r_given=True)
assert result['r'] == 1000.0

# Should use default if not given
result = setup_model(r=999, r_given=False)
assert result['r'] == 1.0  # VA default

# Should reject invalid (once callbacks implemented)
with pytest.raises(ValueError):
    setup_model(r=-100.0, r_given=True)  # Negative resistance
```

**Pros**:
- Fast
- No external dependencies
- Physics-based ground truth

**Cons**:
- Only works for simple models
- Limited coverage
- Doesn't test complex validation logic

**Complexity**: Low (already partially done)

**Option C: Manual MIR Review**

Read MIR text dump and verify logic by hand:

```bash
openvaf --dump-mir resistor.va > resistor_setup.mir
# Manually trace control flow
# Verify parameter validation logic
# Check defaults match VA source
```

**Pros**:
- Absolutely trustworthy
- Deep understanding of codegen
- No bugs in tooling

**Cons**:
- Extremely tedious
- Not scalable
- Requires MIR expertise

**Complexity**: High (labor-intensive)

## The OSDI C Function Challenge

The actual OSDI `setup_model()` has a complex signature:
```c
void setup_model(void* handle, void* model, void* simparam, osdi_init_info* result);
```

**Problems**:
1. `model` is opaque struct with model-specific layout
2. Parameters at unknown offsets (need OSDI descriptor)
3. Complex FFI marshalling required
4. Fragile (breaks if OSDI changes)

**Conclusion**: Not worth implementing for validation purposes.

## Recommended Approach

### Phase 1: Analytical + Physics-Based ✅

**For simple models** (resistor, capacitor, diode):
- Test parameter pass-through
- Test default application
- Test boundary values
- Compare against physical expectations

**Implementation**: `test_setup_model_comparison.py` (already done)

**Status**: ✅ Working

### Phase 2: VACASK Integration (Future)

**When needed**: For complex models or production validation

**Approach**:
1. Run VACASK DC analysis with test parameters
2. Parse `.prn` output for operating point
3. Compare parameter values

**Files to create**:
- `scripts/vacask_setup_model_compare.py`
- Parser for VACASK model cards
- Operating point extractor

**Status**: Not started (not urgent)

### Phase 3: OSDI C Binding (If Ever Needed)

**When**: Only if we need to test actual OSDI compiled output

**Options**:
1. Add Python-friendly wrapper to OpenVAF codegen
2. Complex FFI marshalling from Python
3. Use LLVM IR inspection instead

**Status**: Deferred indefinitely

## What We Have Now

**test_setup_model_comparison.py**:
- ✅ Tests parameter defaults (r=1.0, has_noise=1)
- ✅ Tests parameter pass-through (r=1000 → 1000)
- ✅ Tests control flow paths (given vs not given)
- ✅ Tests boundary values (r=0, r=1e12)
- ⚠️ Doesn't test against external reference yet

**Validation level**: **Good enough for development, not production**

## Decision

**Skip complex OSDI bindings**. Use:
1. Analytical/physics validation (current)
2. VACASK comparison when needed (future)
3. Manual MIR review for critical paths

This avoids:
- ❌ Buggy MIR interpreter
- ❌ Complex OSDI FFI
- ❌ Circular validation

Gives us:
- ✅ Physics-based ground truth
- ✅ External validation path (VACASK)
- ✅ Simple implementation
- ✅ Trustworthy results
