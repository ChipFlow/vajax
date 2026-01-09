# PHI Node Bug: Root Cause of Zero Jacobian Values

## Summary

The validation failure (Jacobian = 0 instead of capacitance) is caused by **missing phi node support** in our MIR→Python code generators.

## Investigation Trail

1. **Symptom**: Jacobian validation fails - reactive entries are 0 when they should be 1e-9
2. **Initial hypothesis**: Wrong MIR variables in metadata
3. **Investigation**: Traced Jacobian variables through MIR
4. **Discovery**: Jacobian variables wrap cache values v37/v40
5. **Problem**: Cache is `[0.0, -0.0]` instead of `[1e-9, -1e-9]`
6. **Root cause**: Cache computation depends on `v33 = phi []` which evaluates to 0

## The PHI Node

Capacitor init MIR has:

```
v33 = phi {
  value: v18  from: block8
  value: v38  from: block14
}
```

This should select:
- `v18` if execution came from block8
- `v38` if execution came from block14

One of these should be the capacitance value `c`.

## Cache Computation

```
cache[0] = mfactor * v33   // Should be: 1.0 * 1e-9 = 1e-9
cache[1] = mfactor * (-v33) // Should be: 1.0 * (-1e-9) = -1e-9
```

But since v33 = 0 (undefined phi), both cache values are 0.

## Why Code Generators Don't Handle PHI

Looking at:
- `jax_spice/codegen/setup_instance_mir_codegen.py` - NO phi handling
- `jax_spice/codegen/eval_mir_codegen.py` - NO phi handling

Both generators have:
- Opcodes: fadd, fsub, fmul, fdiv, fneg, fabs, etc.
- Control flow: br, jmp
- **Missing**: phi

## MIR Parser Status

The MIR parser (`mir_parser.py:310-320`) CORRECTLY parses phi nodes:

```python
elif opcode == 'phi':
    phi_operands = inst['phi_operands']
    phi_args = [(op['value'], op['block']) for op in phi_operands]
    mir_inst = MIRInstruction(
        result=result,
        opcode='phi',
        args=[],
        phi_args=phi_args
    )
```

So the data is available, just not used by code generators.

## Impact

**All models with control flow** (conditionals, loops) will have phi nodes.
Without phi support:
- ✗ Capacitor (has phi for parameter validation)
- ✗ Diode (likely has phi for junction conditions)
- ✗ PSP103, BSIM4 (definitely have phi for complex conditionals)

The only models that might work without phi are trivial resistor-like devices with no conditionals.

## Solution

Implement phi node code generation. Two approaches:

### Option 1: Direct Translation (Naive)

Generate Python code that explicitly tracks which block we came from:

```python
# At start of function
prev_block = None

# At each block entry
if prev_block == 'block8':
    v33 = v18
elif prev_block == 'block14':
    v33 = v38

# At block exit
prev_block = 'block8'  # or whichever block we're leaving
```

**Issues**:
- Verbose
- Requires block execution tracking
- Complex for multiple phis

### Option 2: Dataflow Analysis (Better)

Analyze the control flow graph to determine which phi operand is actually used:

1. Trace forward from function entry
2. For each phi, determine which predecessor is reachable
3. Replace phi with the selected value
4. This can often be done statically if conditions are parameter-dependent

For the capacitor case:
- Analyze blocks 4→8 and 4→14 paths
- Determine v33 should be v18 (or v38) based on parameter conditions
- Generate code without runtime phi selection

### Option 3: SSA Deconstruction (Best)

Convert out of SSA form before code generation:

1. Insert copy operations at phi predecessors
2. Merge phi results into single variables
3. Generate sequential code without phis

This is the standard compiler approach.

## Immediate Fix

For now, we can:

1. **Verify the hypothesis**: Check if v18 or v38 equals the capacitance
2. **Implement basic phi support**: Add phi opcode handling to both generators
3. **Test**: Verify cache becomes [1e-9, -1e-9] and Jacobian validation passes

## File References

- MIR Parser: `jax_spice/codegen/mir_parser.py:310-320`
- Setup Instance Generator: `jax_spice/codegen/setup_instance_mir_codegen.py`
- Eval Generator: `jax_spice/codegen/eval_mir_codegen.py`
- Investigation Scripts: `scripts/inspect_phi_nodes.py`, `scripts/debug_init_mir.py`

## Next Steps

1. Identify what v18 and v38 actually are in the capacitor init MIR
2. Implement phi node code generation (choose approach based on complexity)
3. Test on capacitor to verify cache values
4. Run full validation suite
5. Consider if we need full SSA deconstruction for complex models
