# Plan: Generic Runtime Collapse Pattern from OpenVAF (Approach 2)

**Status**: Saved for later implementation
**Date**: 2025-12-23
**Context**: After node count was fixed (47 nodes), DC operating point still differs. This plan addresses the PSP103-specific hack with a generic solution.

---

## Problem Summary
- VA-JAX currently has PSP103-specific collapse logic hardcoded in runner.py
- This is fragile and won't work for other models
- OpenVAF's init function already computes collapse decisions at runtime
- Need to expose this capability through openvaf-py

## Current State
- Node count now matches VACASK (47) with PSP103-specific hack
- DC operating point still differs (0.5085V vs 0.661V)
- PSP103-specific code in `_get_psp103_collapse_pairs()` needs to be replaced

## How Collapse Works in OpenVAF

### Static Information (already exposed)
`compiled.node_collapse.pairs()` returns potential collapse pairs:
- Each pair has a `CollapsePair` index
- Currently exposed as `collapsible_pairs: Vec<(u32, u32)>`

### Runtime Information (needs exposure)
There are TWO sources of collapse:

1. **CollapseHint callbacks** (from `V(N1,N2) <+ 0` statements):
   - Called at runtime when the collapse code path is executed
   - In VA: `if (R > 0) I<+G*V else V<+0` → CollapseHint called in else branch
   - The callback signals "this pair should collapse"

2. **CollapseImplicitEquation outputs** (from implicit equations):
   - Boolean values in `init.intern.outputs`
   - Less common than CollapseHint

### Data Flow for PSP103's CollapsableR
```
VA Code: `CollapsableR macro
    ↓
if (R > 0) I(N1,N2) <+ G * V(N1,N2)
else V(N1,N2) <+ 0   → CollapseHint(N1, N2) callback
    ↓
During init: CollapseHint callbacks track which pairs collapse
    ↓
collapse_pattern: [bool; num_collapsible_pairs]
```

### Key Insight
- VACASK's `setup_instance()` clears collapse pattern, then callbacks write `true`
- openvaf-py can do the same: hook CollapseHint callbacks during init interpretation

## Implementation Plan

### Step 1: Modify openvaf-py/src/lib.rs

**1a. Add callback tracking info to VaModule struct**

```rust
// In VaModule struct, add:
collapse_callback_indices: Vec<(usize, u32, u32)>,  // (callback_idx, hi_node, lo_node_or_max)
```

**1b. Store collapse callback indices during compilation (lines ~960-968)**

After extracting `collapsible_pairs`, also map which callbacks correspond to which pairs:

```rust
// Build mapping from callback index to collapse pair
// compiled.init.intern.callbacks contains CallBackKind entries
let mut collapse_callback_indices: Vec<(usize, u32, u32)> = Vec::new();
for (cb_idx, kind) in compiled.init.intern.callbacks.iter().enumerate() {
    if let CallBackKind::CollapseHint(hi, lo) = *kind {
        let hi_idx: u32 = /* convert Node to index */;
        let lo_idx: u32 = lo.map(|n| /* convert */).unwrap_or(u32::MAX);
        collapse_callback_indices.push((cb_idx, hi_idx, lo_idx));
    }
}
```

**1c. Add `get_collapse_pattern()` method with callback hooks**

```rust
/// Query which collapsible pairs should actually collapse based on parameters
/// Returns: Vec<bool> where true = this pair collapses
fn get_collapse_pattern(&self, params: std::collections::HashMap<String, f64>) -> PyResult<Vec<bool>> {
    // Track which pairs collapse via callback invocation
    let collapse_pattern = std::cell::RefCell::new(vec![false; self.num_collapsible]);
    let callback_mapping = &self.collapse_callback_indices;
    let all_pairs = &self.collapsible_pairs;

    // Collapse callback - marks pair as collapsed when invoked
    let collapse_callback = |state: &mut InterpreterState, args: &[Value], rets: &[Value], data: *mut c_void| {
        // Get callback index from data pointer
        let cb_idx = data as usize;

        // Find the pair this callback corresponds to
        for (idx, hi, lo) in callback_mapping {
            if *idx == cb_idx {
                // Find pair index in collapsible_pairs
                for (pair_idx, &(p_hi, p_lo)) in all_pairs.iter().enumerate() {
                    if p_hi == *hi && p_lo == *lo {
                        collapse_pattern.borrow_mut()[pair_idx] = true;
                    }
                }
            }
        }
    };

    // Build callbacks array with collapse_callback for CollapseHint, stub for others
    let callbacks = /* build callbacks with collapse_callback hooked */;

    // Run init function
    let mut init_args = /* build from params */;
    let mut init_interp = Interpreter::new(&self.init_func, &callbacks, &init_args);
    init_interp.run();

    Ok(collapse_pattern.into_inner())
}
```

### Step 2: Modify vajax/benchmarks/runner.py

**2a. Update `_get_model_collapse_pairs()`**

```python
def _get_model_collapse_pairs(self, model_type: str, device_params: Dict[str, float]) -> List[Tuple[int, int]]:
    """Get collapse pairs based on runtime parameter evaluation."""
    compiled = self._compiled_models.get(model_type)
    if not compiled:
        return []

    module = compiled.get('module')
    if module is None:
        return []

    all_pairs = module.collapsible_pairs

    # Query runtime collapse pattern
    try:
        collapse_pattern = module.get_collapse_pattern(device_params)
        return [pair for pair, should_collapse in zip(all_pairs, collapse_pattern) if should_collapse]
    except Exception:
        # Fallback: no collapse if query fails
        return []
```

**2b. Remove PSP103-specific code**

Delete `_get_psp103_collapse_pairs()` method entirely.

### Step 3: Update Tests

- Remove any PSP103-specific test expectations
- Verify generic collapse works for all benchmark models

## Files to Modify

1. **`openvaf-py/src/lib.rs`** (~60 lines)
   - Add `collapse_callback_indices: Vec<(usize, u32, u32)>` to VaModule struct
   - Build callback → pair mapping during compilation
   - Add `get_collapse_pattern()` method with callback hooks

2. **`vajax/benchmarks/runner.py`** (~30 lines)
   - Update `_get_model_collapse_pairs()` to use generic query
   - Remove `_get_psp103_collapse_pairs()` (~50 lines deleted)

3. **`tests/test_vacask_benchmarks.py`**
   - Remove PSP103-specific test expectations

## Verification

1. Ring benchmark: node count matches VACASK (47)
2. Ring benchmark: DC operating point matches VACASK (~0.661V)
3. All VACASK benchmark tests pass
4. Other models (resistor, capacitor, diode) still work correctly

## Risk/Complexity Assessment

**Low risk:**
- Adding new field to VaModule struct
- Adding new Python method to runner.py

**Medium risk:**
- Callback hooking in Rust requires careful lifetime management
- The RefCell pattern for capturing collapse state in callback closures
- Node index conversion from HIR Node to our node indices

**Mitigation:**
- Can test callback hooking in isolation first
- The existing `run_init_eval` code already handles callback setup
- Start with PSP103/Ring test case to validate before generalizing

## Alternative Approach (Simpler but Less General)

If callback hooking proves complex, could instead:
1. Parse `collapsible_pairs` node names to get node indices
2. Pass these to Python and let Python match against `collapse_callback_indices`
3. Run init with a Python callback that tracks invocations

This moves complexity to Python but may be easier to debug.
