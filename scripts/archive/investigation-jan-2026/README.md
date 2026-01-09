# January 2026 Investigation Scripts

Scripts created during the investigation that discovered the parameter mapping bug and proved PHI nodes work correctly.

## Key Discoveries

### Parameter Mapping Bug
- `check_init_param_arrays.py` - Found duplicate 'c' name collision
- `check_init_param_mapping.py` - Analyzed the bug
- `identify_v18_v38.py` - Identified what phi operands represent
- `fix_and_test_init.py` - Proved PHI nodes work with correct mapping! ⭐

### PHI Node Investigation
- `check_generated_phi_code.py` - Showed PHI code generation works
- `inspect_phi_nodes.py` - Analyzed phi node structure
- `investigate_optbarrier.py` - Traced through optbarrier operations

### Cache Investigation
- `check_cache_mapping.py` - Analyzed cache system
- `trace_v37_v40_origin.py` - Traced cache variable origins
- `simple_check_cache.py` - Simple cache checks

### MIR Structure
- `check_mir_structure.py` - Understood MIR dict format
- `investigate_mir_derivatives.py` - Looked for derivative calls

## Outcome

These investigations proved:
1. ✅ PHI node code generation works correctly
2. ✅ Control flow handling works correctly
3. ✅ The bug was just parameter mapping (duplicate names)
4. ✅ Fixed in `openvaf-py/src/lib.rs:1091-1096`

## Current Status

Fix applied, ready to:
1. Rebuild openvaf-py (Phase 2)
2. Validate against OSDI (Phases 3-7)

These scripts can be archived once Phase 1 (OSDI interface) is complete.
