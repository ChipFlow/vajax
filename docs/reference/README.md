# Reference Documentation

External reference material for understanding OpenVAF, OSDI, and VACASK.

## Directories

### osdi-vacask/
Documentation about OSDI (Open Source Device Interface) and VACASK simulator:
- OSDI API specification and usage
- VACASK simulator internals
- How OpenVAF compiles to OSDI
- Benchmark analysis

### openvaf/
Documentation about OpenVAF internals:
- MIR (Mid-level IR) format
- Pre-allocated constants
- Compilation pipeline

## Why Separate?

This reference material documents **external systems** (OpenVAF, OSDI, VACASK), not our implementation. Keeping it separate makes it clear what's reference vs. our code.

## Our Implementation Docs

See `../` for documentation of our jax-spice implementation:
- `../RESTART_PROMPT.md` - Starting point for code generation work
- `../IMPLEMENTATION_PLAN.md` - 7-phase OSDI validation plan
- `../PHI_NODE_BUG.md` - PHI node implementation details
