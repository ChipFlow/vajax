# November 2024 Debug Scripts Archive

These scripts were used to debug our November 2024 MIR-to-Python implementation that turned out to be fundamentally broken.

## Why Archived

All these scripts:
1. Debug or test our broken `run_init_eval` MIR interpreter
2. Validate against our own buggy code instead of OSDI
3. Are no longer needed since we're restarting with OSDI validation

## Contents

**Debug scripts (debug_*.py):**
- Investigated cache computation, Jacobian values, device charges, etc.
- All debugging symptoms of the broken implementation

**Test scripts (test_*.py):**
- Tested code generation, complete flows, complex models
- Compared against broken reference (run_init_eval)

**Validation:**
- `validate_codegen_vs_native.py` - Validated against run_init_eval (broken!)

## What We Learned

Despite being broken, these investigations led to discoveries documented in:
- `../../docs/PHI_NODE_BUG.md` - PHI nodes work correctly
- `../../docs/reference/osdi-vacask/CACHE_SLOTS_ANALYSIS.md` - Cache system understanding

## New Approach (Jan 2026)

See:
- `../../docs/RESTART_PROMPT.md` - Why we're restarting
- `../../docs/IMPLEMENTATION_PLAN.md` - New OSDI-based plan
- `../test_vacask_osdi_psp103.py` - Phase 1 starting point (OSDI interface)
