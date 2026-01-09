# OSDI Bindings Feature

## Overview

The `osdi-bindings` feature flag enables direct OSDI function bindings for validating our MIR→Python code generator against the actual OSDI compiled output.

## Building with OSDI Support

### Standard Build (Default)
```bash
cd openvaf-py
cargo build --release
maturin develop --release
```

### With OSDI Bindings
```bash
cd openvaf-py
cargo build --release --features osdi-bindings
maturin develop --release --features osdi-bindings
```

## Why Behind a Feature Flag?

1. **Default Build Simplicity**: Most users don't need OSDI bindings - they just use the MIR interpreter
2. **Optional Dependency**: OSDI crate adds compilation complexity
3. **Testing-Only**: This is primarily for validation during development

## Available Methods (when enabled)

### `VaModule.call_osdi_setup_model(params: dict) -> dict`

**Status**: Stub implementation (raises NotImplementedError)

Intended to call the actual OSDI `setup_model()` function for direct comparison:

```python
import openvaf_py

# Compile module with OSDI bindings enabled
module = openvaf_py.compile_va("resistor.va")[0]

# Our generated setup_model() (from MIR→Python codegen)
from jax_spice.codegen.setup_model_mir_codegen import generate_setup_model_from_mir
# ... generate setup_model_resistor() ...

# OSDI reference (when implemented)
osdi_result = module.call_osdi_setup_model({
    'r': 1000.0,
    'r_given': True,
    'has_noise': 1,
    'has_noise_given': True
})

# Our generated code
generated_result = setup_model_resistor(
    r=1000.0,
    r_given=True,
    has_noise=1,
    has_noise_given=True
)

# Should match!
assert osdi_result == generated_result
```

## Implementation Status

| Function | Status | Notes |
|----------|--------|-------|
| `call_osdi_setup_model()` | Stub | Returns error - needs OSDI FFI integration |
| `call_osdi_setup_instance()` | Not started | For Phase 3 (cache slots) |
| `call_osdi_eval()` | Not needed | Already available via `run_init_eval()` |

## Why Not Implemented Yet?

The OSDI `setup_model()` function is compiled as a native function that:
1. Takes raw pointers to C structures (handle, model, simparam)
2. Modifies the model structure in-place
3. Uses OSDI-specific calling conventions

To call it from Python requires:
1. Allocating OSDI C structures
2. Marshaling Python dict → OSDI model structure
3. Calling the function pointer
4. Marshaling OSDI model structure → Python dict

This is non-trivial FFI work. For now, we validate against:
- Physical correctness (Ohm's law, expected defaults)
- MIR interpreter output (same as OSDI for eval)
- Expected OSDI behavior from documentation

## Alternative: Use MIR Interpreter

For validation, you can use the existing MIR interpreter which executes the same
MIR that OSDI would compile:

```python
# MIR interpreter gives same results as OSDI (for eval)
params = {'V(A,B)': 1.0, 'R': 1000.0, ...}
residuals, jacobian = module.run_init_eval(params)
```

The MIR interpreter executes `model_param_setup`, `setup_instance`, and `eval`
MIR, giving the same numerical results as OSDI would produce.

## Future Work

When we need to actually implement `call_osdi_setup_model()`:

1. Study existing OSDI bindings in VACASK/ngspice
2. Create C structs matching OSDI 0.4 spec
3. Use `pyo3-ffi` or `bindgen` for FFI
4. Marshal between Python dicts and OSDI structures
5. Call the function pointer safely

For now, the stub serves as documentation and prevents feature-flag issues.
