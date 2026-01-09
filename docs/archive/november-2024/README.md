# November 2024 Archive

This directory contains documentation from the November 2024 implementation attempt.

## Why Archived

These documents reference our initial MIR-to-Python implementation that:
1. Used a custom MIR interpreter (`run_init_eval`)
2. Never validated against OSDI (the actual reference)
3. Claimed success (COMPLEX_MODELS_SUCCESS.md) that was actually false positives

The implementation had fundamental bugs including:
- Wrong parameter mapping (duplicate 'c' name collision)
- No validation against VACASK OSDI ground truth
- Comparing against our own broken code

## What We Learned

Despite the broken implementation, we learned valuable things documented in:
- `../PHI_NODE_BUG.md` - How PHI nodes work (code gen is correct!)
- `../CACHE_SLOTS_ANALYSIS.md` - How OpenVAF's cache system works
- `../VACASK_MIR_TO_OSDI_PIPELINE.md` - OpenVAF compilation pipeline

## Current Approach (Jan 2026)

See:
- `../RESTART_PROMPT.md` - Why and how we're restarting
- `../IMPLEMENTATION_PLAN.md` - New 7-phase plan using OSDI validation

The key difference: **Use VACASK OSDI as ground truth, not our own code.**
