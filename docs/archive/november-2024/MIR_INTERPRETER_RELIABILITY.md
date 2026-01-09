# MIR Interpreter Reliability Analysis

## TL;DR

**The MIR interpreter is NOT fully reliable** - it has known bugs with complex models like PSP103. We should NOT use it as the authoritative reference for validation.

## Evidence of Issues

### PSP103 Zero Residuals Bug

**Commit**: `58b9ae7` - "WIP: Ring oscillator investigation - PSP103 returning zero residuals"

**Problem**:
- PSP103 transistors return 0A current when they shouldn't
- Ring oscillator stuck (doesn't oscillate)
- `run_init_eval()` returns all-zero residuals

**Impact**: The MIR interpreter cannot be trusted for complex models with:
- Large parameter sets (~1700 params like PSP103)
- Hidden state computations
- Complex control flow

### What Works

**Simple models appear OK**:
- Resistor: ✅ Follows Ohm's law (I=V/R validated in tests)
- Diode: ⚠️ Unknown - needs verification
- Capacitor: ⚠️ Unknown - needs verification

**But**: We only have our own tests validating these, not external reference.

## What CAN We Trust?

### 1. VACASK Simulation Results ✅

**Status**: Known good - industry simulator

**Evidence**:
- Ring oscillator works correctly in VACASK
- PSP103 models produce correct behavior
- Used in production IC design

**How to use**:
- Run VACASK simulations
- Compare waveforms, operating points
- Parse `.prn` output files

**Files**:
- `scripts/compare_vacask.py` - Comparison framework
- `vendor/VACASK/` - Simulator binaries

### 2. Analytical Solutions ✅

**Status**: Physics-based ground truth

**Examples**:
- Resistor: I = V/R (exact)
- Diode: Shockley equation at equilibrium
- RC circuit: τ = RC time constant

**Limitation**: Only works for simple circuits

### 3. Physical Expectations ✅

**Status**: Sanity checks

**Examples**:
- Current should be finite
- Voltage should be continuous
- Energy should be conserved
- Currents should sum to zero (KCL)

## What We CANNOT Trust

### ❌ MIR Interpreter as Reference

**Problems**:
1. PSP103 zero residuals bug
2. Possible hidden state initialization issues
3. Cache value computation bugs
4. No external validation against VACASK

**Why it seemed trustworthy**:
- We wrote tests that said "MIR interpreter is authoritative"
- But we didn't validate the interpreter itself!
- Circular reasoning

### ❌ Our Own Tests Without External Reference

**Problem**: We compared:
- Our generated code → MIR interpreter
- Both could be wrong!

**Analogy**: Comparing two watches that both show wrong time

## Recommended Validation Strategy

### For setup_model() Validation

**Option 1: VACASK Operating Point** (BEST)
```python
# 1. Generate setup_model() from MIR
setup_result = our_setup_model(r=1000.0, ...)

# 2. Run VACASK DC analysis with same parameters
# Parse .prn file for operating point

# 3. Compare parameter values used in simulation
assert vacask_params['r'] == setup_result['r']
```

**Option 2: Analytical Physics** (Simple models only)
```python
# For resistor
result = setup_model(r=1000.0, r_given=True)
assert result['r'] == 1000.0  # Should pass through valid param

result = setup_model(r=-100.0, r_given=True)
# Should either reject or clamp to valid range
# (our current impl passes through - WRONG!)
```

**Option 3: Direct MIR Analysis** (Manual verification)
- Read the MIR text dump
- Trace control flow by hand
- Verify logic matches Verilog-A semantics
- Labor-intensive but trustworthy

### For eval() Validation

**Must use VACASK**:
1. Same circuit, same parameters
2. Same operating point (DC)
3. Compare residuals & Jacobian
4. Match within numerical tolerance

**Already working for**:
- Simple resistor networks (rc benchmark)
- Diode bridge (graetz benchmark)

**Still broken for**:
- Ring oscillator (PSP103 issue)

## Revised Assessment

### What Our Work Validated

**Phase 1-2b (MIR→Python setup_model)**:
- ✅ Generates syntactically correct Python
- ✅ Implements control flow state machine correctly
- ✅ PHI nodes work (predecessor tracking)
- ✅ Constants properly initialized
- ⚠️ NOT yet validated against external reference

**What we still need**:
1. VACASK comparison for setup_model() output
2. Test with known-good parameters
3. Verify parameter validation logic
4. Test with multiple models (not just resistor)

## Action Items

1. **Don't use MIR interpreter for validation** until PSP103 bug is fixed
2. **Add VACASK comparison** for setup_model() output
3. **Test simple models first** (resistor, capacitor, diode)
4. **Document limitations** clearly in code comments
5. **Fix PSP103 bug** before using interpreter as reference

## Bottom Line

The MIR interpreter is a **debugging tool**, not a validation reference.

For validation, use:
- VACASK (gold standard)
- Analytical physics (simple cases)
- Manual MIR review (tedious but reliable)
