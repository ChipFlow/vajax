# Phase 3a Complete: setup_instance() Code Generation

## Summary

Successfully implemented MIR→Python code generation for the `setup_instance()` function (OpenVAF's init function), which computes cached values from model/instance parameters.

## What Was Built

### 1. setup_instance Code Generator

**File**: `jax_spice/codegen/setup_instance_mir_codegen.py`

Generates Python code from init MIR that:
- Takes model/instance parameters (e.g., c, c_given, mfactor)
- Implements parameter validation and default logic
- Computes cached values (e.g., mfactor * c)
- Returns cache array for use by eval function

**Key Features**:
- Reuses control flow codegen infrastructure (branches, jumps, PHI nodes)
- Handles param_given flags properly
- Maps cache values to eval parameter indices

### 2. MIR Dict Parser

**File**: `jax_spice/codegen/mir_parser.py` - Added `parse_mir_dict()`

Converts dict-format MIR (from openvaf_py Rust wrapper) to structured MIRFunction:
- Parses instructions with proper opcode handling
- Handles branches, jumps, PHI nodes, calls
- Merges constants from multiple dicts (float, int, bool)

### 3. Control Flow Improvements

**File**: `jax_spice/codegen/control_flow_codegen.py`

Fixed critical issues:
1. **Infinity Constants**: Now generates `math.inf` instead of raw `inf`
2. **Type Cast Operations**: Added support for:
   - `ifcast` - integer to float cast
   - `ficast` - float to integer cast
   - `optbarrier` - optimization barrier (pass through)

## Validation

Created comprehensive test: `scripts/test_setup_instance_codegen.py`

**Test Results** (Capacitor):
```
✅ Test 1: Explicit c=1e-9, mfactor=1
   Input: {c: 1e-9, c_given: 1, mfactor: 1}
   Cache: [1e-9, -1e-9]  ← mfactor*c, -mfactor*c

✅ Test 2: Default c (not given)
   Input: {c: 999, c_given: 0, mfactor: 1}
   Cache: [1e-12, -1e-12]  ← default 1e-12

✅ Test 3: c=1e-6, mfactor=2.5
   Input: {c: 1e-6, c_given: 1, mfactor: 2.5}
   Cache: [2.5e-6, -2.5e-6]
```

All tests pass! ✅

## Generated Code Example

From capacitor init MIR:

```python
def setup_instance_capacitor(**params):
    """Compute cached values for capacitor instance."""

    # Extract parameters
    c = params.get("c", 0.0)
    mfactor = params.get("mfactor", 1.0)
    c_given = params.get("c_given", False)

    # Initialize constants
    FALSE = False
    TRUE = True
    INF = math.inf
    DEFAULT_C = 1e-12
    # ... more constants

    # Control flow state machine
    current_block = "block4"
    prev_block = None

    while current_block is not None:
        if current_block == "block4":
            if c_given:
                prev_block, current_block = current_block, "block5"
            else:
                prev_block, current_block = current_block, "block6"

        elif current_block == "block2":
            # Compute cache values
            neg_c = -c_validated
            cache_0_val = mfactor * c_validated
            cache_0 = cache_0_val
            cache_1_val = mfactor * neg_c
            cache_1 = cache_1_val
            # ...

        # ... more blocks

    # Return cache array
    cache = [
        cache_0,  # cache[0] -> eval param[5]
        cache_1,  # cache[1] -> eval param[6]
    ]
    return cache
```

## Cache Mapping

For capacitor:
- **cache[0]**: mfactor * c → passed as eval Param[5]
- **cache[1]**: -mfactor * c → passed as eval Param[6]

These pre-computed values avoid recomputing `mfactor * c` on every eval call.

## Architecture

```
Model Compilation Flow:
┌──────────────────────────────────────────────────────────────┐
│ Verilog-A Source                                              │
│   parameter real c = 1e-12;                                   │
│   Q = c * V(br);                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │ OpenVAF compilation
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ OpenVAF MIR Functions                                         │
│                                                               │
│  ┌────────────────────────────────────────────┐             │
│  │ model_param_setup (Phase 2b)               │             │
│  │  - Validate model parameters                │             │
│  │  - Apply defaults                           │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  ┌────────────────────────────────────────────┐             │
│  │ init / setup_instance (Phase 3a) ← NEW     │             │
│  │  - Validate instance parameters             │             │
│  │  - Compute cached values                    │             │
│  │  - Return cache array                       │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  ┌────────────────────────────────────────────┐             │
│  │ eval (Phase 1 + 3b next)                    │             │
│  │  - Takes voltages + cache                   │             │
│  │  - Computes residuals & Jacobian            │             │
│  └────────────────────────────────────────────┘             │
└──────────────────────┬───────────────────────────────────────┘
                       │ MIR→Python codegen
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Generated Python Functions                                     │
│                                                                │
│  setup_model_capacitor(**params) -> dict                       │
│  setup_instance_capacitor(**params) -> cache_array            │
│  eval_capacitor(voltages, cache) -> (residuals, jacobian)     │
└────────────────────────────────────────────────────────────────┘
```

## Next Steps: Phase 3b

Modify eval() code generation to:
1. Accept `cache` parameter
2. Map cache indices (5, 6, ...) to cache array access
3. Replace hidden_state params with cache lookups
4. Validate against VACASK reference

## Files Created/Modified

### Created:
- `jax_spice/codegen/setup_instance_mir_codegen.py` - Main generator
- `scripts/test_setup_instance_codegen.py` - Validation tests
- `docs/CACHE_SLOTS_ANALYSIS.md` - Architecture documentation
- `docs/PHASE3A_SUMMARY.md` - This summary

### Modified:
- `jax_spice/codegen/mir_parser.py` - Added `parse_mir_dict()`
- `jax_spice/codegen/control_flow_codegen.py` - Fixed inf, added ifcast/optbarrier

## Lessons Learned

1. **Infinity Constant Handling**: Python `inf` objects from Rust need `math.inf` in generated code
2. **Type Casts Matter**: MIR has explicit type conversions (ifcast) that must be handled
3. **Optimization Barriers**: `optbarrier` instructions are present even in optimized MIR
4. **Cache is Key**: Pre-computing parameter expressions dramatically reduces eval() cost

## Metrics

- **Lines of code**: ~200 (generator + parser extension)
- **Test coverage**: 3 test cases, 100% pass rate
- **Capacitor cache slots**: 2 (mfactor*c, -mfactor*c)
- **Performance benefit**: Eliminates mfactor multiplication on every eval call

Phase 3a: **COMPLETE** ✅
