# MIR-to-Python Code Generation: Complete Implementation Plan

## Overview

**Goal**: Generate Python code from OpenVAF MIR that produces numerically identical results to VACASK OSDI libraries.

**Strategy**: Build and validate incrementally, using VACASK OSDI as ground truth at each stage.

**Models**: Start simple, increase complexity
1. Capacitor (2 cache, linear)
2. Diode (16 cache, nonlinear with conditionals)
3. PSP103 (462 cache, complex control flow)

## Phase 1: OSDI Reference Implementation (CRITICAL FOUNDATION)

**Goal**: Establish VACASK OSDI as our reference implementation we can call from Python.

### Task 1.1: Define OSDI API Structures

**File**: `scripts/osdi_interface.py` (create new)

Define ctypes structures from OSDI 0.4 spec:
- `OsdiDescriptor` - Model metadata
- `OsdiInitInfo` - Init function interface
- `OsdiEvalInfo` - Eval function interface
- Parameter/node structures

**Reference**: Look at existing OSDI wrapper code if available, or define from spec.

### Task 1.2: Load OSDI Library

**File**: `scripts/osdi_interface.py`

```python
class OsdiModel:
    def __init__(self, osdi_path: str):
        self.lib = ctypes.CDLL(osdi_path)
        self.descriptor = self.lib.osdi_descriptor()
        # Parse descriptor to get:
        # - Number of terminals/nodes
        # - Number of cache values
        # - Number of parameters
        # - Init/eval function pointers
```

### Task 1.3: Call OSDI Init

**File**: `scripts/osdi_interface.py`

```python
def osdi_init(self, params: dict) -> np.ndarray:
    """Call OSDI init function, return cache array."""
    # Allocate cache array
    # Call init function via ctypes
    # Return cache values
```

### Task 1.4: Call OSDI Eval

**File**: `scripts/osdi_interface.py`

```python
def osdi_eval(self, voltages: np.ndarray, cache: np.ndarray) -> dict:
    """Call OSDI eval function, return residuals and Jacobian."""
    # Allocate output arrays
    # Call eval function via ctypes
    # Parse outputs
    # Return {residuals, jacobian}
```

### Task 1.5: Test OSDI Interface

**File**: `scripts/test_osdi_capacitor.py` (create new)

```python
# Load capacitor.osdi
model = OsdiModel("path/to/capacitor.osdi")

# Test init
cache = model.osdi_init({'c': 1e-9, 'mfactor': 1.0})
print(f"Cache: {cache}")
assert len(cache) == 2
assert abs(cache[0] - 1e-9) < 1e-15

# Test eval
voltages = np.array([1.0, 0.0])  # V(A), V(B)
result = model.osdi_eval(voltages, cache)
print(f"Residuals: {result['residuals']}")
print(f"Jacobian: {result['jacobian']}")
```

**Acceptance Criteria**:
- ✅ OSDI library loads without errors
- ✅ Init returns cache array of correct size
- ✅ Eval returns residuals and Jacobian
- ✅ Values are numerically sensible (not NaN, not all zero)

## Phase 2: Rebuild openvaf-py with Metadata Fix

**Goal**: Get corrected parameter metadata from OpenVAF.

### Task 2.1: Compile Metadata Fix

**Already done**: `openvaf-py/src/lib.rs:1091-1096`

```bash
cd openvaf-py
maturin develop
cd ..
```

### Task 2.2: Verify Metadata Correction

**File**: `scripts/test_metadata_fix.py` (create new)

```python
import openvaf_py

modules = openvaf_py.compile_va('vendor/VACASK/devices/capacitor.va')
cap = modules[0]
metadata = cap.get_codegen_metadata()

print("Init param mapping:")
for name, var in metadata['init_param_mapping'].items():
    print(f"  {name} → {var}")

# Should show:
#   c → v18
#   c_given → v32
#   mfactor → v20
assert metadata['init_param_mapping']['c'] == 'v18'
assert metadata['init_param_mapping']['c_given'] == 'v32'
assert metadata['init_param_mapping']['mfactor'] == 'v20'
```

**Acceptance Criteria**:
- ✅ No duplicate keys in init_param_mapping
- ✅ param_given types have "_given" suffix
- ✅ All three parameters mapped correctly

## Phase 3: Code Generation with Correct Metadata

**Goal**: Generate Python code using corrected metadata.

### Task 3.1: Generate Init Function

**File**: `scripts/generate_capacitor_init.py` (create new)

```python
from jax_spice.codegen.mir_parser import parse_mir_dict
from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir

# Get MIR and metadata
init_mir_dict = cap.get_init_mir_instructions()
metadata = cap.get_codegen_metadata()

# Parse MIR
init_mir = parse_mir_dict(init_mir_dict)

# Build parameter mapping from metadata
param_map = metadata['init_param_mapping'].copy()
for const_name in init_mir.constants.keys():
    param_map[const_name] = const_name

# Get cache mapping
cache_tuples = [(entry['init_value'], entry['eval_param'])
                for entry in metadata['cache_info']]

# Generate code
setup_code = generate_setup_instance_from_mir(
    init_mir,
    param_map,
    cache_tuples,
    'capacitor'
)

# Save and execute
Path('generated_capacitor_init.py').write_text(setup_code)
```

### Task 3.2: Generate Eval Function

**File**: `scripts/generate_capacitor_eval.py` (create new)

Similar to 3.1 but for eval function using `generate_eval_from_mir`.

### Task 3.3: Test Generated Init

**File**: `scripts/test_generated_init.py` (create new)

```python
# Import generated function
from generated_capacitor_init import setup_instance_capacitor

# Test cases
test_cases = [
    {'c': 1e-9, 'c_given': True, 'mfactor': 1.0},
    {'c': 1e-9, 'c_given': False, 'mfactor': 1.0},
    {'c': 2e-9, 'c_given': True, 'mfactor': 2.0},
]

for params in test_cases:
    cache = setup_instance_capacitor(**params)
    print(f"Params: {params}")
    print(f"Cache: {cache}")
```

**Acceptance Criteria**:
- ✅ Function executes without errors
- ✅ Returns cache array of correct length
- ✅ c_given=True: cache ≈ [c*mfactor, -c*mfactor]
- ✅ c_given=False: cache ≈ [1e-12*mfactor, -1e-12*mfactor]

## Phase 4: Validate Against OSDI (Capacitor)

**Goal**: Prove generated code matches OSDI exactly.

### Task 4.1: Validate Init

**File**: `scripts/validate_capacitor_init.py` (create new)

```python
# Load OSDI reference
osdi_model = OsdiModel('path/to/capacitor.osdi')

# Import generated code
from generated_capacitor_init import setup_instance_capacitor

# Test multiple operating points
test_cases = [...]

for params in test_cases:
    # OSDI
    osdi_cache = osdi_model.osdi_init(params)

    # Generated
    gen_cache = setup_instance_capacitor(**params)

    # Compare
    diff = np.abs(osdi_cache - gen_cache)
    max_diff = np.max(diff)

    print(f"Params: {params}")
    print(f"  OSDI cache:     {osdi_cache}")
    print(f"  Generated cache: {gen_cache}")
    print(f"  Max diff:        {max_diff}")

    assert max_diff < 1e-12, f"Init validation failed: {max_diff}"
```

### Task 4.2: Validate Eval

**File**: `scripts/validate_capacitor_eval.py` (create new)

Similar to 4.1 but for eval function.

```python
# For each voltage operating point:
voltages = [...]
cache = [...]

# OSDI
osdi_result = osdi_model.osdi_eval(voltages, cache)

# Generated
gen_result_dict = eval_capacitor(*voltages, cache=cache)
gen_residuals, gen_jacobian = extract_outputs(gen_result_dict, metadata)

# Compare
assert np.allclose(osdi_result['residuals'], gen_residuals, atol=1e-12)
assert np.allclose(osdi_result['jacobian'], gen_jacobian, atol=1e-12)
```

**Acceptance Criteria**:
- ✅ Init matches OSDI within 1e-12 for all test cases
- ✅ Eval residuals match OSDI within 1e-12
- ✅ Eval Jacobian (resist) matches OSDI within 1e-12
- ✅ Eval Jacobian (react) matches OSDI within 1e-12

## Phase 5: Diode Model

**Goal**: Validate more complex model with nonlinear equations.

Repeat Phases 3-4 for diode:
- Diode has ~16 cache slots
- Has junction capacitance (nonlinear charge)
- Has exponential IV curves
- Has temperature dependence

### Task 5.1-5.4: Same as Phase 3-4 but for diode

**Files**: Create `scripts/*_diode.py` versions of capacitor scripts.

**Acceptance Criteria**: Same as Phase 4 but for diode.

## Phase 6: PSP103 Model

**Goal**: Validate most complex model.

Repeat Phases 3-4 for PSP103:
- PSP103 has ~462 cache slots
- Complex control flow with many PHI nodes
- Temperature models
- Multiple operating regions
- Noise models

### Task 6.1-6.4: Same as Phase 3-4 but for PSP103

**Files**: Create `scripts/*_psp103.py` versions.

**Acceptance Criteria**: Same as Phase 4 but for PSP103.

## Phase 7: Automated Validation Framework

**Goal**: Generalize validation to any model.

### Task 7.1: Create Unified Validator

**File**: `scripts/validate_model.py` (create new)

```python
class ModelValidator:
    def __init__(self, va_path: str, osdi_path: str):
        self.va_path = va_path
        self.osdi_path = osdi_path
        self.osdi_model = OsdiModel(osdi_path)
        self.openvaf_model = openvaf_py.compile_va(va_path)[0]

    def generate_code(self):
        """Generate init and eval functions."""
        ...

    def validate_init(self, test_cases: list) -> bool:
        """Validate init across test cases."""
        ...

    def validate_eval(self, test_cases: list) -> bool:
        """Validate eval across test cases."""
        ...

    def run_full_validation(self) -> dict:
        """Run complete validation suite."""
        ...
```

### Task 7.2: Test Suite

**File**: `tests/test_all_models.py` (create new)

```python
def test_capacitor():
    validator = ModelValidator(
        'vendor/VACASK/devices/capacitor.va',
        'path/to/capacitor.osdi'
    )
    result = validator.run_full_validation()
    assert result['init_pass']
    assert result['eval_pass']

def test_diode():
    ...

def test_psp103():
    ...
```

**Acceptance Criteria**:
- ✅ Validator works for all three models
- ✅ All tests pass
- ✅ Can add new models easily

## Phase 8: JAX Code Generation (Future)

Once Python code generation is validated, proceed to JAX:

1. Replace Python operators with JAX equivalents
2. Handle control flow with jax.lax.cond/switch
3. Ensure JIT-compilability
4. Validate numerical equivalence
5. Benchmark performance

This is a separate effort after Python validation is complete.

## Dependencies

```
Phase 1 (OSDI Interface) → Phase 2 (Metadata Fix) → Phase 3 (Code Gen)
                                                   ↓
                                                Phase 4 (Validate Capacitor)
                                                   ↓
                                                Phase 5 (Validate Diode)
                                                   ↓
                                                Phase 6 (Validate PSP103)
                                                   ↓
                                                Phase 7 (Unified Framework)
```

Phases 1-2 must complete before Phase 3.
Phases 3-4 must complete successfully before Phase 5.
Phase 7 requires Phases 4-6 complete.

## Risk Mitigation

### Risk 1: OSDI Interface Complexity

**Mitigation**:
- Start with Python-based OSDI wrapper if available
- Reference existing ctypes OSDI interfaces
- Focus on minimal API surface (init, eval only)

### Risk 2: Numerical Differences

**Mitigation**:
- Use tight tolerance (1e-12)
- Test at multiple operating points
- Check derivatives numerically if needed
- Compare intermediate values, not just final outputs

### Risk 3: Model Complexity

**Mitigation**:
- Start with simplest model (capacitor)
- Only proceed to next model after current validates
- Document each model's specific challenges

## Success Metrics

### Phase 4 Complete
- Capacitor: 100% of test cases pass validation
- Max numerical error < 1e-12

### Phase 5 Complete
- Diode: 100% of test cases pass validation
- Handles nonlinear equations correctly

### Phase 6 Complete
- PSP103: 100% of test cases pass validation
- Handles complex control flow

### Phase 7 Complete
- Framework validates any model automatically
- Test suite covers edge cases
- Documentation complete

## Timeline Estimate

Assuming no major issues:

- Phase 1: 1-2 days (OSDI interface)
- Phase 2: 1 hour (rebuild)
- Phase 3: 1 day (code generation)
- Phase 4: 1-2 days (capacitor validation)
- Phase 5: 2-3 days (diode validation)
- Phase 6: 3-5 days (PSP103 validation)
- Phase 7: 2-3 days (framework)

**Total**: 2-3 weeks

Critical path: Phase 1 → Phase 4 (establishing OSDI interface and first validation)

## Next Action

Start Phase 1, Task 1.1: Define OSDI API structures in `scripts/osdi_interface.py`.

Reference OSDI specification and look for existing Python OSDI wrappers to inform the interface design.
