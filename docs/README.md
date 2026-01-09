# JAX-SPICE Documentation

**Start here for context on the current implementation effort.**

---

## 🎯 Starting Fresh (January 2026)

Our November 2024 MIR-to-Python implementation was fundamentally broken - it validated against its own buggy code instead of OSDI ground truth. We're restarting with a clean approach.

### Essential Reading (in order)

1. **[RESTART_PROMPT.md](RESTART_PROMPT.md)** ⭐ **START HERE**
   - Why we're restarting the code generation effort
   - What to ignore (run_init_eval = our old broken code)
   - What we learned (PHI nodes work correctly!)
   - Where to start (OSDI ctypes interface)

2. **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** ⭐ **THE PLAN**
   - Detailed 7-phase implementation plan
   - Phase 1: OSDI ctypes interface (ground truth)
   - Phases 2-7: Incremental validation (capacitor → diode → PSP103)
   - Acceptance criteria for each phase

3. **[PHI_NODE_BUG.md](PHI_NODE_BUG.md)** ⭐ **TECHNICAL REFERENCE**
   - PHI node implementation deep-dive
   - Why code generation works correctly
   - Parameter mapping bug explanation and fix
   - Read when implementing code generation (Phase 3+)

---

## 📚 Reference Documentation

External reference material for OpenVAF, OSDI, and VACASK.

### OSDI/VACASK Reference
**Location**: `reference/osdi-vacask/`

Essential for Phase 1 (OSDI interface):
- **SIMULATOR_INTERNALS.md** (30KB) - OSDI API, descriptors, function signatures
- **osdi_parameter_architecture.md** - Parameter handling in OSDI
- **vacask_osdi_inputs.md** - Input structure for init/eval functions

Understanding OpenVAF compilation:
- **CACHE_SLOTS_ANALYSIS.md** - Init/eval cache system
- **VACASK_MIR_TO_OSDI_PIPELINE.md** - VA → MIR → OSDI compilation

Analysis and formats:
- **ANALYSIS.md** - Ring benchmark analysis
- **VACASK-ANALYSIS.md** - Circuit analysis examples
- **vacask_sim_format.md** - VACASK .sim netlist format

See [reference/osdi-vacask/README.md](reference/osdi-vacask/README.md) for details.

### OpenVAF Internals
**Location**: `reference/openvaf/`

- **OPENVAF_MIR_CONSTANTS.md** - Pre-allocated MIR constants (v0-v7)

See [reference/openvaf/README.md](reference/openvaf/README.md) for details.

---

## 📖 General Documentation

### Code Generation & Metadata
- **CODEGEN_METADATA_API.md** - Metadata extraction API (needs update for param_given fix)
- **PARAMETER_MAPPING_SOLUTION.md** - Parameter mapping solution (needs update)

### Architecture & Development
- **ARCHITECTURE.md** - Overall system architecture
- **DEVELOPMENT.md** - Development workflow and conventions
- **GPU_OPTIMIZATION.md** - GPU optimization strategies
- **SPARSE_SOLVER_ANALYSIS.md** - Sparse matrix solver details
- **CONTRIBUTING.md** - Contributing guidelines
- **TODO.md** - General TODOs

---

## 🗄️ Archives

**Location**: `archive/`

### November 2024 Implementation
**Location**: `archive/november-2024/`

Documentation from the broken November implementation:
- PHASE3_COMPLETE_SUMMARY.md
- COMPLEX_MODELS_SUCCESS.md (false positives!)
- And 5 other docs

See [archive/november-2024/README.md](archive/november-2024/README.md) for why these were archived.

---

## 🚀 Quick Start

### For Fresh Context (New Session)
```bash
# 1. Read the restart prompt
cat docs/RESTART_PROMPT.md

# 2. Read the implementation plan
cat docs/IMPLEMENTATION_PLAN.md

# 3. Check current phase
cat docs/IMPLEMENTATION_PLAN.md | grep -A 10 "Current Phase"
```

### For Continuing Work
```bash
# Check which phase we're on
cat docs/IMPLEMENTATION_PLAN.md | grep "Status:" | head -7

# Read relevant reference docs based on phase
# Phase 1: docs/reference/osdi-vacask/SIMULATOR_INTERNALS.md
# Phase 2+: docs/PHI_NODE_BUG.md
```

---

## 📋 Reading Order by Phase

### Phase 1: OSDI Interface
Must read:
1. RESTART_PROMPT.md (context)
2. IMPLEMENTATION_PLAN.md → Phase 1 section
3. reference/osdi-vacask/SIMULATOR_INTERNALS.md (OSDI API)
4. reference/osdi-vacask/osdi_parameter_architecture.md

### Phase 2: Rebuild openvaf-py
Must read:
1. PHI_NODE_BUG.md (parameter mapping fix)
2. PARAMETER_MAPPING_SOLUTION.md

### Phase 3-7: Code Generation & Validation
Must read:
1. PHI_NODE_BUG.md (how PHI nodes work)
2. reference/osdi-vacask/CACHE_SLOTS_ANALYSIS.md (cache system)
3. CODEGEN_METADATA_API.md

---

## 🎯 The Mission

**Use VACASK OSDI as ground truth, not our own code.**

Build incrementally:
- Simple (capacitor) before complex (PSP103)
- Validate every step against OSDI
- No false positives, no confusion

**Starting point**: `scripts/test_vacask_osdi_psp103.py`

See IMPLEMENTATION_PLAN.md for the complete roadmap.
