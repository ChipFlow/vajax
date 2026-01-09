# OpenVAF Reference Documentation

Documentation of OpenVAF's internal structures and behavior.

## Documents

- **OPENVAF_MIR_CONSTANTS.md** - Pre-allocated MIR constants (v0-v7)
  - v1 = FALSE
  - v2 = TRUE
  - v3 = F_ZERO (0.0)
  - v6 = F_ONE (1.0)
  - etc.

## Additional References

See also:
- `../osdi-vacask/VACASK_MIR_TO_OSDI_PIPELINE.md` - Full compilation pipeline
- `../osdi-vacask/CACHE_SLOTS_ANALYSIS.md` - Cache system details
- OpenVAF source: `openvaf-py/vendor/OpenVAF/`

## Usage

These documents help understand OpenVAF's MIR format when:
- Parsing MIR dumps
- Understanding why certain constants appear
- Debugging MIR→Python code generation
