# Code Generation Metadata API

## Overview

Added comprehensive `get_codegen_metadata()` method to openvaf-py that returns all metadata needed for MIR→Python code generation and validation.

## Implementation

### Location

`openvaf-py/src/lib.rs:1049-1151` - New method on `VaModule` class

### Signature

```rust
fn get_codegen_metadata(&self) -> PyResult<HashMap<String, PyObject>>
```

### Python Usage

```python
import openvaf_py

modules = openvaf_py.compile_va("capacitor.va")
model = modules[0]

# Get all code generation metadata
metadata = model.get_codegen_metadata()
```

## Returned Metadata

### 1. Parameter Mappings

Maps semantic parameter names to MIR variable names, filtering out `hidden_state` parameters that are inlined by the optimizer.

**Eval parameters:**
```python
metadata['eval_param_mapping']
# {'c': 'v16', 'V(A,B)': 'v17', 'mfactor': 'v25'}
```

**Init parameters:**
```python
metadata['init_param_mapping']
# {'c': 'v32', 'mfactor': 'v20'}
```

### 2. Cache Information

Describes each cache slot with its source (init function) and destination (eval function).

```python
metadata['cache_info']
# [
#   {'cache_idx': 0, 'init_value': 'v27', 'eval_param': 'v5'},
#   {'cache_idx': 1, 'init_value': 'v28', 'eval_param': 'v6'}
# ]
```

### 3. Residual Metadata

Identifies which MIR values correspond to residual components.

```python
metadata['residuals']
# [
#   {'residual_idx': 0, 'resist_var': 'v3', 'react_var': 'v26'},
#   {'residual_idx': 1, 'resist_var': 'v3', 'react_var': 'v34'}
# ]
```

### 4. Jacobian Metadata

Identifies which MIR values correspond to Jacobian entries.

```python
metadata['jacobian']
# [
#   {'jacobian_idx': 0, 'row': 0, 'col': 0, 'resist_var': 'v3', 'react_var': 'v30'},
#   {'jacobian_idx': 1, 'row': 0, 'col': 1, 'resist_var': 'v3', 'react_var': 'v38'},
#   ...
# ]
```

### 5. Model Info

Basic model information for convenience.

```python
metadata['model_name']        # 'capacitor'
metadata['num_terminals']     # 2
metadata['num_residuals']     # 2
metadata['num_jacobian']      # 4
metadata['num_cache_slots']   # 2
```

## Use Cases

### 1. Calling Generated Functions

Use parameter mappings to convert test parameters to MIR variable names:

```python
# Test params use semantic names
test_params = {'c': 1e-9, 'V_A_B': 1.0, 'mfactor': 1.0}

# Get mapping
param_map = metadata['eval_param_mapping']  # {'c': 'v16', ...}

# Build args for generated function: eval_capacitor(v16, v17, v25, cache)
eval_args = []
for semantic_name, mir_name in param_map.items():
    eval_args.append(test_params.get(semantic_name, 0.0))
```

### 2. Extracting Outputs

Use residual/Jacobian metadata to identify output values:

```python
# Generated function returns dict of all MIR values
result = eval_capacitor(v16, v17, v25, cache)

# Extract residuals
residuals = []
for res_info in metadata['residuals']:
    resist = result[res_info['resist_var']]
    react = result[res_info['react_var']]
    residuals.append((resist, react))

# Extract Jacobian
jacobian = []
for jac_info in metadata['jacobian']:
    row = jac_info['row']
    col = jac_info['col']
    resist = result[jac_info['resist_var']]
    react = result[jac_info['react_var']]
    jacobian.append((row, col, resist, react))
```

### 3. Numerical Validation

Compare generated code against reference implementation:

```python
# Reference (native MIR interpreter)
ref_residuals, ref_jacobian = model.run_init_eval(params)

# Generated code
cache = setup_instance_capacitor(**params)
result = eval_capacitor(*eval_args, cache=cache)

# Extract outputs using metadata
gen_residuals = extract_residuals(result, metadata['residuals'])
gen_jacobian = extract_jacobian(result, metadata['jacobian'])

# Compare
assert np.allclose(ref_residuals, gen_residuals)
assert np.allclose(ref_jacobian, gen_jacobian)
```

## Benefits

### Single API Call

Instead of multiple method calls:
```python
# OLD: Multiple calls
param_names = model.param_names
param_indices = model.param_value_indices
param_kinds = model.param_kinds
# ... manually filter and process
```

```python
# NEW: One call, everything needed
metadata = model.get_codegen_metadata()
```

### Complete Information

Provides everything needed for:
- ✅ Parameter mapping (semantic → MIR)
- ✅ Function calling (ordered arguments)
- ✅ Output extraction (residuals, Jacobian)
- ✅ Numerical comparison (validation)
- ✅ Cache management (init → eval data flow)

### Pre-filtered Data

Automatically filters `hidden_state` parameters that are inlined by the optimizer, saving validation code from having to do this filtering.

## Test Results

### Capacitor Model

```
✓ Eval param mapping: 3 parameters (c, V(A,B), mfactor)
✓ Init param mapping: 2 parameters (c, mfactor)
✓ Cache info: 2 slots
✓ Residuals: 2 equations
✓ Jacobian: 4 entries
```

All metadata correctly extracted and usable for validation!

## Implementation Notes

### Hidden State Filtering

```rust
for ((name, value_idx), kind) in self.param_names.iter()
    .zip(&self.param_value_indices)
    .zip(&self.param_kinds)
{
    if !kind.contains("hidden_state") {
        eval_param_map.set_item(name, format!("v{}", value_idx))?;
    }
}
```

Hidden state parameters are computational intermediates that OpenVAF's optimizer inlines. They appear in parameter lists but are never actually used in evaluation.

### Python Type Conversion

Uses `.into_py(py)` for Rust primitives (String, usize) and `.into()` for Python objects (PyDict, PyList):

```rust
metadata.insert("model_name".to_string(), self.name.clone().into_py(py));
metadata.insert("eval_param_mapping".to_string(), eval_param_map.into());
```

## Next Steps

1. **Update Validation Framework**: Use new API in `validate_codegen_vs_native.py`
2. **Extract Output Values**: Implement residual/Jacobian extraction from generated code results
3. **Numerical Comparison**: Compare reference vs generated values numerically
4. **Documentation**: Update code generation docs with metadata API usage

## Status

**✅ COMPLETE AND TESTED**

- Implementation: openvaf-py/src/lib.rs:1049-1151
- Test: scripts/test_codegen_metadata.py
- All metadata fields working correctly
- Ready for integration into validation framework

## Summary

The `get_codegen_metadata()` API provides a clean, comprehensive interface for all code generation and validation needs. Instead of manually assembling metadata from multiple sources, a single call returns everything needed to:

1. Map semantic names to MIR variables
2. Call generated functions with correct arguments
3. Extract and identify output values
4. Validate numerical correctness

This completes the metadata infrastructure needed for full numerical validation of generated code!
