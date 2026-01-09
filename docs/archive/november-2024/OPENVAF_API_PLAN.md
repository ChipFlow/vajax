# OpenVAF API Clarity Plan

## Executive Summary

The current data flow from Verilog-A source through `openvaf_py` to JAX-SPICE has several naming convention issues that make the code difficult to understand and debug. This document proposes a clear, well-documented data flow with unambiguous naming conventions.

## Current Issues

### 1. `sim_node{}` Synthetic Naming

In `openvaf_py/src/lib.rs`, the `get_dae_system()` function uses synthetic `sim_node{i}` names:

```rust
// Current problematic code (lib.rs:692-717)
for (i, node) in self.nodes.iter().enumerate() {
    unknowns.set_item(format!("sim_node{}", i), node.clone()).unwrap();
}
// ...
for i in 0..self.num_residuals {
    residuals.set_item(format!("sim_node{}", i), res).unwrap();
}
```

**Problems:**
- Creates synthetic names that must be decoded (`sim_node3` → what actual node?)
- Assumes 1:1 mapping between unknowns/residuals/jacobians by index
- Consumer code in `engine.py` must strip `sim_` prefix and do lookups
- Different representations for same concept (node name vs `sim_node{i}`)

### 2. `v{}` Variable Name Clashes

MIR variables are named with `v{}` prefix (e.g., `v123`, `v456`):

```rust
constants.set_item(format!("v{}", u32::from(val)), float_val).unwrap();
// ...
residuals["resist"] = "v{index}"  // e.g., "v1234"
```

**Problems:**
- `v{}` could clash with voltage-related naming conventions
- Not immediately clear if `v123` refers to MIR value 123 or something else
- Same prefix used for constants, params, residuals, jacobians

### 3. Multiple Representations of Same Data

The DAE system has:
- `unknowns`: Maps `sim_node{i}` → actual node name (e.g., `KirchoffLaw(d)`)
- `residuals`: Maps `sim_node{i}` → `{resist: v{idx}, react: v{idx}}`
- `jacobian`: List of `{row: sim_node{i}, col: sim_node{j}, resist: v{idx}, react: v{idx}}`
- `nodes`: List of raw node names from OpenVAF (e.g., `KirchoffLaw(NodeId(0))`)

Consumer code must constantly translate between these representations.

## Proposed Solution

### Phase 1: Clarify Data Structure (Non-Breaking)

Add a new `get_dae_system_v2()` method with clearer naming:

```python
{
    # Node information (indexed by node_idx 0..N-1)
    "nodes": [
        {"idx": 0, "name": "d", "kind": "KirchoffLaw", "is_internal": False},
        {"idx": 1, "name": "g", "kind": "KirchoffLaw", "is_internal": False},
        ...
    ],

    # Residual equations (indexed by equation_idx)
    "residuals": [
        {
            "equation_idx": 0,
            "node_idx": 0,  # Which node this residual stamps into
            "node_name": "d",
            "resist_var": "mir_1234",  # MIR variable for resistive part
            "react_var": "mir_5678",   # MIR variable for reactive part
        },
        ...
    ],

    # Jacobian entries
    "jacobian": [
        {
            "entry_idx": 0,
            "row_node_idx": 0,
            "row_node_name": "d",
            "col_node_idx": 1,
            "col_node_name": "g",
            "resist_var": "mir_2345",
            "react_var": "mir_6789",
            "has_resist": True,
            "has_react": True,
        },
        ...
    ],

    # Terminal nodes (external ports)
    "terminals": ["d", "g", "s", "b"],
    "num_terminals": 4,

    # Internal nodes
    "internal_nodes": ["di", "gi", "si", "bi", "noi"],
    "num_internal": 5,
}
```

### Phase 2: Rename MIR Variables (Non-Breaking)

Change `v{}` prefix to `mir_{}` to clearly indicate these are MIR intermediate values:

```python
# Before: "v1234"
# After:  "mir_1234"
```

This makes it unambiguous that these refer to OpenVAF MIR SSA values.

### Phase 3: Document Data Flow

Create clear documentation showing data flow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VERILOG-A SOURCE                                  │
│  module psp103(d, g, s, b);                                             │
│    inout d, g, s, b;                                                    │
│    electrical d, g, s, b, di, gi, si, bi;                               │
│    I(d, di) <+ gds * V(d, di);                                          │
│    ...                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OPENVAF COMPILER                                 │
│  Parses VA → HIR → MIR (SSA form)                                       │
│  - Parameters: param_names[], param_kinds[]                             │
│  - Nodes: node_names[] with terminal + internal                         │
│  - DAE System: residuals[], jacobian[]                                  │
│  - Init function: computes cache values                                 │
│  - Eval function: computes I, dI/dV at operating point                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OPENVAF-PY (Rust)                                │
│  lib.rs: VaModule struct with:                                          │
│  - param_names: ["TYPE", "W", "L", ...]                                 │
│  - param_kinds: ["param", "param", "param", ...]                        │
│  - nodes: ["KirchoffLaw(d)", "KirchoffLaw(g)", ...]                     │
│  - residual_resist_indices: [mir_idx for each residual]                 │
│  - jacobian_rows/cols: [node_idx for each jac entry]                    │
│                                                                          │
│  get_dae_system() → Python dict with:                                   │
│    unknowns: {sim_node0: "d", sim_node1: "g", ...}  ← CONFUSING         │
│    residuals: {sim_node0: {resist: v123, react: v456}, ...}             │
│    jacobian: [{row: sim_node0, col: sim_node1, resist: v789}, ...]      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       OPENVAF_JAX.PY (Python)                            │
│  OpenVAFToJAX class:                                                     │
│  - Translates MIR instructions → JAX code                               │
│  - Uses dae_data to map outputs:                                        │
│      for node, res in dae_data['residuals'].items():                    │
│          # node is "sim_node{i}" - must decode!                         │
│          resist_val = res['resist']  # "v{idx}" - MIR variable          │
│                                                                          │
│  Generated functions:                                                    │
│  - init_fn(params) → cache[N_cache]                                     │
│  - eval_fn(params, cache) → (residuals, jacobian)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ENGINE.PY (JAX-SPICE)                              │
│  CircuitEngine._prepare_openvaf_batched_inputs():                        │
│  - Builds node_map: {model_node_name → circuit_node_idx}                │
│  - ALSO adds sim_node{i} entries for DAE residual mapping               │
│                                                                          │
│  _build_stamp_indices():                                                 │
│  - Uses metadata['node_names'] which are "sim_node{i}" strings          │
│  - Must strip "sim_" prefix to get actual model node                    │
│                                                                          │
│  _stamp_batched_results():                                               │
│  - Maps "sim_node{i}" → model_node → circuit_node_idx                   │
│  - Stamps into global f[], J[][] matrices                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Refactor Engine.py Node Mapping

Current code in `engine.py:987-990`:
```python
# Also map sim_node{i} names (DAE residuals use sequential indices)
for i, model_node in enumerate(model_nodes):
    node_map[f'sim_node{i}'] = node_map.get(model_node, ground)
```

Proposed: Use the new v2 API to get direct node indices without synthetic names:

```python
# With v2 API - no synthetic names needed
for residual in dae_v2['residuals']:
    node_name = residual['node_name']  # Direct node name: "d", "g", etc.
    node_idx = residual['node_idx']    # Index within model
    global_idx = circuit_node_map.get(node_name, ground)
```

## Implementation Plan

### Step 1: Add `get_dae_system_v2()` in lib.rs

Add new method that returns cleaner data structure alongside existing method (backward compatible).

### Step 2: Update `openvaf_jax.py`

Modify to use v2 API internally while maintaining external interface.

### Step 3: Update `engine.py`

Remove `sim_node{}` workarounds once v2 API is used.

### Step 4: Deprecate `get_dae_system()`

Mark original method as deprecated, remove in future version.

## Naming Convention Reference

| Concept | Old Convention | New Convention |
|---------|---------------|----------------|
| MIR SSA value | `v123` | `mir_123` |
| Node reference in DAE | `sim_node{i}` | `node_idx: i, node_name: "d"` |
| Residual equation | `sim_node{i}` key | `equation_idx: i, node_idx: j` |
| Jacobian entry | `row: sim_node{i}` | `row_node_idx: i, row_node_name: "d"` |

## Testing Strategy

1. Add unit tests for `get_dae_system_v2()` output format
2. Add integration tests comparing v1 and v2 outputs for known models
3. Test with ring oscillator benchmark to verify currents are non-zero
4. Compare JAX-SPICE vs VACASK results

## Files to Modify

1. `openvaf-py/src/lib.rs` - Add `get_dae_system_v2()` method
2. `openvaf-py/openvaf_jax.py` - Use v2 API, update code generation
3. `jax_spice/analysis/engine.py` - Remove `sim_node{}` workarounds
4. Tests for all above

## Success Criteria

1. No more `sim_node{}` synthetic names in API
2. MIR variables clearly identified with `mir_` prefix
3. Data flow is documented and follows naming conventions
4. Ring oscillator produces non-zero currents and oscillates correctly
5. VACASK benchmark alignment improves

## Timeline

This is a medium-sized refactoring effort. The key insight is that the current confusion around `sim_node{}` naming is making it very hard to debug why currents are zero - we can't easily trace which model node maps to which circuit node.
