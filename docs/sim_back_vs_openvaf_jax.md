# sim_back vs openvaf_jax: Compilation Pipeline Comparison

This document compares the OpenVAF `sim_back` crate's compilation pipeline with our `openvaf_jax` implementation. Understanding the differences helps identify where our implementation may be missing features or taking shortcuts.

## Overview

| Aspect | sim_back (OpenVAF) | openvaf_jax |
|--------|-------------------|-------------|
| **Target** | OSDI (LLVM IR) | JAX Python code |
| **Approach** | Full compilation with optimization | Interpret MIR, generate Python |
| **Optimization** | Multi-pass (GVN, DCE, SCCP) | None (relies on JAX JIT) |
| **Auto-diff** | Internal MIR-based autodiff | Relies on JAX autodiff (partial) |

---

## Pipeline Stages Comparison

### sim_back Pipeline

```
HIR → MIR → Optimize → Topology → DAE System → Optimize → Init Extraction → OSDI Codegen
```

### openvaf_jax Pipeline

```
MIR (from openvaf_py) → Parse → Generate Python → exec() → JAX JIT
```

---

## Stage-by-Stage Comparison

### 1. Input Representation

| sim_back | openvaf_jax |
|----------|-------------|
| Builds MIR from HIR using `MirBuilder` | Receives pre-computed MIR from `openvaf_py` |
| Has full access to HIR for symbol resolution | Has serialized MIR data + metadata |
| Can modify/rebuild MIR during compilation | Read-only interpretation of MIR |

**What we receive from openvaf_py:**
- `get_mir_instructions()` - Eval function MIR
- `get_init_mir_instructions()` - Init function MIR
- `get_dae_system()` - Pre-computed DAE structure
- Metadata: params, nodes, cache_mapping

### 2. Optimization

| sim_back | openvaf_jax |
|----------|-------------|
| **Initial**: DCE, SCCP, inst_combine, CFG simplify, GVN | None |
| **PostDerivative**: SCCP, inst_combine, CFG simplify, GVN | None |
| **Final**: Aggressive DCE | None |

**Gap**: We skip all IR-level optimization, relying on:
- OpenVAF's optimization (runs before MIR export)
- JAX's JIT compilation (XLA)

### 3. Topology Construction

| sim_back | openvaf_jax |
|----------|-------------|
| **Branch extraction** from MIR | Pre-computed in `dae_data['residuals']` |
| **Linearization** of `ddt()`, noise | Pre-computed (result embedded in MIR) |
| **Small-signal detection** | Not implemented |
| **Implicit equation creation** | Pre-computed |

**sim_back's linearization logic:**
```
For each ddt()/noise:
  If only used in linear chain (fadd, fsub, fneg, fmul with non-OP-dep) →
    Extract as separate "dimension" (react coefficient)
  Else →
    Create implicit equation (new unknown)
```

**Our approach**: OpenVAF has already done linearization. The MIR we receive has:
- Resistive contributions (`resist`)
- Reactive contributions (`react`)
- These are separate outputs in `dae_data`

### 4. DAE System Construction

| sim_back | openvaf_jax |
|----------|-------------|
| Builds residuals from branch contributions | Receives `dae_data['residuals']` |
| Runs autodiff on MIR to get Jacobian | Receives `dae_data['jacobian']` |
| Computes `lim_rhs` correction terms | Receives `lim_rhs_resist`, `lim_rhs_react` |
| Handles mfactor scaling | Pre-computed |
| Extracts noise sources | Pre-computed (not used in JAX path) |

**sim_back's autodiff:**
```rust
let derivatives = auto_diff(ctx, residuals, unknowns);
// Returns: (residual_value, unknown) → derivative_value
```

**Our approach**: We receive pre-differentiated MIR. The Jacobian entries in `dae_data['jacobian']` reference MIR values that are already the partial derivatives.

### 5. OP-Dependence Analysis

| sim_back | openvaf_jax |
|----------|-------------|
| Tracks which values depend on unknowns | Not explicit |
| Uses dominance frontiers for control deps | CFG analysis for loops only |
| Two-phase: `init_op_dependent_insts`, `refresh_op_dependent_insts` | Implicit via init/eval split |

**sim_back's OP-dependent roots:**
- `ParamKind::Voltage` - Node voltages
- `ParamKind::Current` - Branch currents
- `ParamKind::ImplicitUnknown` - Internal unknowns
- System functions: `$abstime`, `$temperature`

**Our approach**: OpenVAF already split the code:
- `init` function: OP-independent (cached values)
- `eval` function: OP-dependent (per Newton iteration)

### 6. Initialization Extraction

| sim_back | openvaf_jax |
|----------|-------------|
| `Initialization::new()` splits MIR | Pre-split by OpenVAF |
| GVN deduplication of cache slots | Pre-computed cache_mapping |
| Creates separate `init` and `eval` MIR functions | Receives separate MIR for each |
| Cache slot assignment | `cache_mapping[i]['init_value']` → `cache_mapping[i]['eval_param']` |

**sim_back cache slot assignment:**
```rust
fn ensure_cache_slot(inst, res, ty) -> CacheSlot {
    let class = gvn.inst_class(inst);  // GVN equivalence
    cache_slots.insert((class, res), ty)
}
```

**Our approach**: `cache_mapping` from OpenVAF maps init values to eval params.

### 7. Node Collapse

| sim_back | openvaf_jax |
|----------|-------------|
| `NodeCollapse::new()` detects collapsible pairs | `collapsible_pairs` from module |
| Runtime collapse decisions | `collapse_decision_outputs` from init |

**Both handle**: Nodes that can be shorted under certain parameter conditions.

### 8. Code Generation

| sim_back | openvaf_jax |
|----------|-------------|
| LLVM IR via `osdi` crate | Python AST → source code |
| Type-safe register allocation | Dynamic Python typing |
| Inlined operations | JAX operations |
| Memory layout control | JAX arrays |

---

## Feature Coverage

### Fully Supported

| Feature | sim_back | openvaf_jax |
|---------|----------|-------------|
| Resistive contributions | ✅ | ✅ |
| Reactive contributions (ddt) | ✅ | ✅ |
| Sparse Jacobian | ✅ | ✅ |
| Node collapse | ✅ | ✅ |
| Parameter caching (init/eval split) | ✅ | ✅ |
| Limiting (lim_rhs correction) | ✅ | ✅ |
| mfactor scaling | ✅ | ✅ (pre-computed) |

### Partially Supported

| Feature | sim_back | openvaf_jax | Gap |
|---------|----------|-------------|-----|
| Implicit equations | ✅ | ⚠️ Partial | We handle pre-resolved implicit eqns |
| Switch branches | ✅ | ⚠️ Partial | Not fully tested |
| Temperature dependence | ✅ | ⚠️ | Need to verify param mapping |

### Not Supported

| Feature | sim_back | openvaf_jax | Reason |
|---------|----------|-------------|--------|
| Small-signal analysis | ✅ | ❌ | No small_signal_vals handling |
| AC analysis | ✅ | ❌ | Not in scope |
| Noise analysis | ✅ | ❌ | Not in scope |
| Runtime branch switching | ✅ | ❌ | Complex control flow |

---

## Data Structure Mapping

### Residuals

```
sim_back:                          openvaf_jax:
Residual {                         dae_data['residuals'][i] = {
  resist: Value,                     'resist': 'mir_XX',
  react: Value,                      'react': 'mir_YY',
  resist_small_signal: Value,        (not exposed)
  react_small_signal: Value,         (not exposed)
  resist_lim_rhs: Value,             'resist_lim_rhs': 'mir_ZZ',
  react_lim_rhs: Value,              'react_lim_rhs': 'mir_WW',
  nature_kind: Flow|Potential|Switch 'node_name': 'A'
}                                  }
```

### Jacobian

```
sim_back:                          openvaf_jax:
MatrixEntry {                      dae_data['jacobian'][i] = {
  row: SimUnknown,                   'row_node_name': 'A',
  col: SimUnknown,                   'col_node_name': 'B',
  resist: Value,                     'resist': 'mir_XX',
  react: Value,                      'react': 'mir_YY',
}                                  }
```

### Cache

```
sim_back:                          openvaf_jax:
Initialization {                   init_mir_data['cache_mapping'][i] = {
  cached_vals: {                     'init_value': 'mir_XX',
    eval_value → CacheSlot           'eval_param': 42
  },                               }
  cache_slots: {
    (gvn_class, res_idx) → Type
  }
}
```

---

## Performance Implications

### sim_back Advantages
1. **Multi-pass optimization** reduces instruction count before codegen
2. **GVN deduplication** minimizes cache size
3. **Aggressive DCE** removes truly dead code
4. **LLVM backend** can do further optimization

### openvaf_jax Advantages
1. **JAX JIT** provides XLA optimization at runtime
2. **vmap** enables efficient batched device evaluation
3. **GPU acceleration** via JAX backends
4. **Python ecosystem** for debugging/analysis

### Potential Improvements for openvaf_jax

1. **Dead code elimination** - Some MIR values may be unused
2. **Constant folding** - Could pre-compute constant expressions
3. **Cache optimization** - Verify no redundant cache entries
4. **Small-signal support** - For AC analysis in future

---

## Recommendations

### Short-term
1. Verify all `dae_data` fields are correctly used
2. Add validation for cache_mapping completeness
3. Test limiting (lim_rhs) paths more thoroughly

### Medium-term
1. Consider MIR-level dead code elimination
2. Profile cache usage to identify redundancies
3. Implement small_signal handling if AC needed

### Long-term
1. Evaluate whether MIR-level optimization adds value
2. Consider autodiff at Python level for flexibility
3. Noise analysis support if needed

---

## References

- `vendor/OpenVAF/docs/sim_back/sim_back.md` - Main overview
- `vendor/OpenVAF/docs/sim_back/init.md` - Initialization extraction
- `vendor/OpenVAF/docs/sim_back/dae.md` - DAE system construction
- `vendor/OpenVAF/docs/sim_back/topology.md` - Branch/linearization
- `vendor/OpenVAF/docs/sim_back/context.md` - Compilation context
