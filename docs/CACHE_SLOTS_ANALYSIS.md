# Cache Slots Analysis

## Overview

OpenVAF's compilation pipeline separates device evaluation into two functions:
1. **init** (setup_instance in OSDI): Computes cached intermediate values from model/instance parameters
2. **eval** (func in OSDI): Uses cached values + voltages to compute residuals and Jacobian

## Capacitor Example

### Model Structure

Verilog-A code:
```verilog
Q = c * V(br);
I(br) <+ ddt(Q);
```

### Init Function (setup_instance)

**Inputs:**
- v18 = c (capacitance parameter)
- v20 = mfactor (system function)
- v32 = c_given (boolean flag)

**Computation:**
1. Validate c parameter (check range [0, inf], apply default 1e-12 if not given)
2. v33 = validated c value (PHI node)
3. v19 = mfactor * c
4. v22 = mfactor * (-c) = -mfactor * c
5. v27 = optbarrier(v19) → **cache[0]**
6. v28 = optbarrier(v22) → **cache[1]**

**Cache mapping:**
- cache[0]: mfactor * c → passed as eval Param[5]
- cache[1]: -mfactor * c → passed as eval Param[6]

### Eval Function

**Inputs:**
- v16 (Param[0]) = c (parameter)
- v17 (Param[1]) = V(A,B) (voltage)
- v19 (Param[2]) = Q (hidden_state - unused, inlined)
- v21 (Param[3]) = q (hidden_state - unused, inlined)
- v25 (Param[4]) = mfactor (system function)
- **v37 (Param[5]) = cache[0]** (pre-computed mfactor * c)
- **v40 (Param[6]) = cache[1]** (pre-computed -mfactor * c)

**Usage:**
- Uses v37 and v40 (cached values) directly instead of recomputing mfactor * c
- Computes residuals and Jacobian using cached values

## Key Insights

1. **Cache Purpose**: Pre-compute expensive parameter expressions
   - Reduces computation in eval (called many times per timestep)
   - init is called once per device instance

2. **Hidden State Parameters**:
   - Q and q appear as hidden_state params
   - But they're never actually read by eval (inlined by optimizer)
   - See `docs/MIR_INTERPRETER_RELIABILITY.md` for analysis

3. **optbarrier**:
   - MIR optimization barrier to prevent over-optimization
   - Ensures values are computed and stored in cache

## Implementation Strategy

### Phase 3a: Generate setup_instance() from init MIR

Create a Python code generator similar to setup_model():
1. Parse init MIR structure
2. Generate Python function that:
   - Takes parameters (c, c_given, mfactor)
   - Implements control flow (branches, PHI nodes)
   - Returns dict of cache values
3. Use same control flow state machine as setup_model()

### Phase 3b: Integrate cache into eval()

Modify eval() generation to:
1. Accept cache parameter: `def eval_capacitor(c, voltage, mfactor, cache):`
2. Map cache indices to variable names
3. Use cached values instead of recomputing

Example:
```python
def eval_capacitor(c, V_A_B, mfactor, cache):
    # Instead of: mfactor_times_c = mfactor * c
    # Use: mfactor_times_c = cache[0]
    Q = cache[0] * V_A_B  # mfactor * c * V
    # ...
```

### Testing

1. Compare against Rust MIR interpreter:
   - Run init with params → get cache values
   - Run eval with cache → get residuals/Jacobian
2. Validate against VACASK reference

## File Structure

```
jax_spice/codegen/
├── mir_parser.py              # Existing - parse MIR text
├── setup_model_mir_codegen.py # Existing - model_param_setup codegen
├── setup_instance_mir_codegen.py  # NEW - init (cache) codegen
└── eval_mir_codegen.py        # MODIFIED - add cache parameter support
```
