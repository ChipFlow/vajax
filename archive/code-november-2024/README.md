# November 2024 Code Archive

**BROKEN IMPLEMENTATION - DO NOT USE**

This directory contains the broken November 2024 implementation of openvaf-py.

---

## Why Archived

This code was implemented without proper validation against OSDI (the actual reference implementation). Instead, it validated against its own broken code, leading to false positives.

**The fundamental problem**: We built MIR interpreters in Python/Rust and compared generated code against those interpreters, not against VACASK OSDI. When the interpreter was wrong, everything looked consistent but was actually broken.

---

## What's in openvaf-py/

### ❌ BROKEN - Do Not Use

**Interpreter functions in `src/lib.rs`:**
- `run_init_eval()` (line ~917) - Broken init MIR interpreter
- `debug_init_cache()` (line ~1006) - Uses broken interpreter
- `evaluate()` (line ~1179) - Broken eval MIR interpreter
- `evaluate_full()` (line ~1236) - Broken eval interpreter
- `run_model_param_setup()` (line ~1337) - Model parameter setup interpreter

**Why broken**: These interpret MIR directly in Rust but have bugs in:
- Control flow handling
- Parameter mapping (duplicate 'c' name collision)
- Cache computation
- Voltage extraction

**Impact**: Any code using these functions produces incorrect results.

### ⚠️ PARTIALLY USEFUL - Review Before Use

**Metadata extraction in `src/lib.rs`:**
- `get_codegen_metadata()` (line ~1064) - Extracts MIR metadata ⭐
- `get_init_mir_instructions()` (line ~566) - Init MIR extraction
- `get_mir_instructions()` (line ~385) - Eval MIR extraction
- `get_osdi_descriptor()` (line ~283) - OSDI descriptor
- `get_dae_system()` (line ~771) - DAE system info
- `get_cache_mapping()` (line ~226) - Cache slot mapping

**Status**: These extract MIR structure and metadata. Most should work, but need validation against OSDI behavior.

**Key fix needed**: `get_codegen_metadata()` has parameter mapping fix at lines 1091-1096 (append "_given" suffix for param_given kinds to avoid duplicate 'c' collision).

### ✅ POTENTIALLY OK - Pure Data Extraction

**Info functions that don't interpret:**
- `get_param_defaults()` (line ~233) - Parameter default values
- `get_str_constants()` (line ~239) - String constants
- `debug_model_setup_outputs()` (line ~245) - Model setup outputs
- `debug_model_setup_phi_nodes()` (line ~259) - PHI node info

**Status**: These just extract data from MIR, should be safe.

---

## What We Learned

Despite being broken, this implementation led to important discoveries:

### 1. Parameter Mapping Bug

**File**: `src/lib.rs:1091-1096`

Found that init parameters have duplicate names with different kinds:
```python
init_param_names = ['c', 'mfactor', 'c']
init_param_kinds = ['param', 'sysfun', 'param_given']
```

Second 'c' overwrites first in dict! **Fix**: Append "_given" suffix for param_given kinds.

**Documented in**: `../../docs/PHI_NODE_BUG.md`

### 2. PHI Nodes Work Correctly

**Proof**: `../../scripts/archive/investigation-jan-2026/fix_and_test_init.py`

When parameter mapping is correct, generated PHI node code produces correct results. The code generation approach is sound.

**Documented in**: `../../docs/PHI_NODE_BUG.md`

### 3. Cache System Understanding

**Analysis**: `../../docs/reference/osdi-vacask/CACHE_SLOTS_ANALYSIS.md`

Figured out how OpenVAF's init/eval cache system works:
- Init computes derivatives and caches them
- Eval uses cached values
- Massive optimization (PSP103: 1705 hidden_state params all inlined!)

---

## What to Use Instead

### Phase 1: OSDI Interface (Current)
**Don't use this code at all**. Load VACASK OSDI via ctypes:
- Start: `../../scripts/test_vacask_osdi_psp103.py`
- Reference: `../../docs/reference/osdi-vacask/SIMULATOR_INTERNALS.md`

### Phase 2: Rebuild openvaf-py (Next)
Build **new** openvaf-py with:
1. Only metadata extraction (no interpreters!)
2. Parameter mapping fix applied
3. Validate metadata against OSDI behavior

**What to salvage**:
- `get_codegen_metadata()` with parameter fix
- MIR extraction functions (review first)
- Structure/approach for metadata extraction

**What to discard**:
- All interpreter functions (run_init_eval, evaluate, etc.)
- Any validation code that compares against interpreters

### Phase 3-7: Code Generation
Generate Python code from MIR, validate each step against OSDI ctypes from Phase 1.

---

## Directory Structure

```
archive/code-november-2024/
└── openvaf-py/              # Broken November implementation
    ├── src/
    │   └── lib.rs           # Broken interpreters + metadata extraction
    ├── tests/               # Tests against broken code (unreliable)
    ├── Cargo.toml
    └── pyproject.toml
```

---

## Important Notes

**Do NOT**:
- Use any interpreter functions
- Trust test results from this code
- Compare new implementations against this code

**DO**:
- Review metadata extraction functions before reusing
- Apply parameter mapping fix in new code
- Use learnings (PHI nodes, cache system, parameter bug)
- Validate everything against OSDI, not this code

---

## Impact of Archiving

**Broken imports**: The following files in `jax_spice/` currently import `openvaf_py`:
- `jax_spice/analysis/engine.py`
- `jax_spice/devices/verilog_a.py`
- `jax_spice/codegen/mir_parser.py`

These imports will fail until Phase 2 (rebuild openvaf-py) is complete. This is intentional - we don't want to use the broken code.

**Expected behavior**:
- Imports will raise `ModuleNotFoundError: No module named 'openvaf_py'`
- Code using OpenVAF device models will not work
- This is CORRECT - forces us to rebuild properly before using

**When fixed**: Phase 2 will create new openvaf-py with only metadata extraction (no broken interpreters).

---

## Timeline

- **November 2024**: Implemented this code
- **December 2024**: Discovered PSP103 model broken
- **January 2026**: Investigated, found root causes, archived this code
- **Phase 2 (future)**: Rebuild openvaf-py correctly

See `../../planning/IMPLEMENTATION_PLAN.md` for the path forward.
