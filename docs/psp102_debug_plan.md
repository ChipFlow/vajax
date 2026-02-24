# PSP102 Debug Plan

## Current Status

PSP102 model comparison tests show **false positives** - tests pass due to absolute tolerance masking a 5 order of magnitude error.

### Quantified Problem

| Metric | OSDI | JAX | Issue |
|--------|------|-----|-------|
| Ids (drain current) | 4.79e-11 A | 5e-16 A | ~100,000x too small |
| Jacobian max | 4.76e+11 | 1.0 | Almost all zeros |
| Non-zero Jacobian entries | 11 | 1 | Missing entries |

The test uses `max_diff < 1e-6` absolute tolerance. Since `4.79e-11 < 1e-6`, the test passes despite JAX returning essentially zero current.

### What Works

- **Cache computation (init)**: 364 values, 242 non-zero, no inf/nan
- **Voltage mapping**: Correct node names (D, G, S, B, NOI, NOI2, GP, BP, BI, BS, BD)
- **Parameter setting**: TYPE=1, W=1e-6, L=1e-7 passed correctly to init

### What Doesn't Work

- **Eval function**: Returns near-zero residuals and Jacobian
- **NMOS/PMOS branching**: PSP102 uses TYPE parameter for device polarity
- **PHI node resolution**: 313 blocks with PHIs in eval function

## Root Cause Hypothesis

PSP102 has complex NMOS/PMOS conditional branching controlled by the TYPE parameter. The JAX translator likely has issues with:

1. **PHI node resolution** in the NMOS/PMOS branch selection
2. **TYPE parameter flow** from init cache to eval computation
3. **Control flow merging** where NMOS and PMOS paths rejoin

Since the current is ~1e-15 (numerical noise level), the entire current computation is likely being skipped or zeroed out due to incorrect branch selection.

## Debugging Strategy

### Phase 1: Trace TYPE Parameter Flow

1. Find where TYPE is stored in the cache
2. Verify TYPE value propagates correctly to eval
3. Check if TYPE-dependent branches are executing

### Phase 2: Identify Divergence Point

Compare OSDI vs JAX at intermediate computation points:

1. **Input arrays**: Verify voltages and cache values match
2. **Early computations**: Find first value that diverges
3. **PHI resolutions**: Check if PHI nodes select correct predecessor

### Phase 3: MIR Analysis

Use the debug tools we developed:

```bash
# Find PHI nodes in PSP102 eval
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --find-phis

# Trace control flow to specific blocks
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --branches
```

### Phase 4: Compare with Working Models

EKV and MVSG work correctly. Compare their:
- PHI node count and complexity
- TYPE parameter handling (EKV also has TYPE for NMOS/PMOS)
- Control flow structure

## Tools Available

| Tool | Purpose |
|------|---------|
| `scripts/analyze_mir_cfg.py` | CFG analysis, PHI node finding, path tracing |
| `vajax/debug/mir_tracer.py` | Value flow tracing through MIR |
| `vajax/debug/param_analyzer.py` | Parameter kind analysis |
| `vajax/debug/jacobian.py` | Format-aware Jacobian comparison |

## Next Steps

1. [ ] Generate detailed MIR dump for PSP102 eval function
2. [ ] Find TYPE-dependent branch points in the CFG
3. [ ] Trace which path JAX takes vs which path it should take
4. [ ] Compare with EKV (working) to identify pattern differences
5. [ ] Fix PHI resolution or branch selection issue
6. [ ] Update test tolerances to catch this class of error

## Test Improvement

After fixing PSP102, update tests to use relative tolerance:

```python
# Current (masks errors):
assert max_diff < 1e-6

# Better (catches magnitude errors):
assert np.allclose(osdi_ids, jax_ids, rtol=1e-4, atol=1e-20)
```
