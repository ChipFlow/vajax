# OSDI/VACASK Reference Documentation

## OSDI (Open Source Device Interface)

OSDI is the standard API for compact device models in circuit simulators.

### Core Documents
- **osdi_parameter_architecture.md** - How OSDI handles parameters
- **vacask_osdi_inputs.md** - OSDI input structure
- **CACHE_SLOTS_ANALYSIS.md** - OpenVAF's init/eval cache system

## VACASK Simulator

VACASK is our reference implementation using OSDI models.

### Core Documents
- **SIMULATOR_INTERNALS.md** (30KB) - Detailed VACASK internals
  - OSDI descriptor structure
  - Newton-Raphson solver
  - Timestep control
  - Convergence criteria
- **ANALYSIS.md** (15KB) - Ring benchmark analysis
- **VACASK-ANALYSIS.md** - Circuit analysis

## OpenVAF → OSDI Pipeline

- **VACASK_MIR_TO_OSDI_PIPELINE.md** - How OpenVAF compiles VA to OSDI
  - MIR structure
  - Optimization passes
  - OSDI code generation

## File Formats

- **vacask_sim_format.md** - VACASK .sim netlist format

## Usage for Our Work

These documents are the **reference** for Phase 1 (OSDI Interface):
1. Read SIMULATOR_INTERNALS.md to understand OSDI API
2. Use osdi_parameter_architecture.md for parameter handling
3. Reference CACHE_SLOTS_ANALYSIS.md for init/eval separation
4. Use VACASK_MIR_TO_OSDI_PIPELINE.md to understand what OSDI does

All validation in Phases 3-7 compares against VACASK/OSDI as ground truth.
