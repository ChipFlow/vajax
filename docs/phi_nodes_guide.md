# Guide to PHI Nodes in OpenVAF MIR

This guide explains PHI nodes: what they are, how they work in SSA (Static Single Assignment) form, how they're represented in OpenVAF's MIR, and how they translate to JAX's functional programming model.

## Table of Contents

1. [What Are PHI Nodes?](#what-are-phi-nodes)
2. [How PHI Nodes Work](#how-phi-nodes-work)
3. [PHI Nodes in OpenVAF MIR](#phi-nodes-in-openvaf-mir)
4. [Converting PHI Nodes to JAX](#converting-phi-nodes-to-jax)
5. [Debugging PHI Node Issues](#debugging-phi-node-issues)

---

## What Are PHI Nodes?

PHI (φ) nodes are a fundamental construct in **Static Single Assignment (SSA)** form, an intermediate representation used by compilers. They solve a key problem: how do you track which value a variable has when control flow merges from multiple paths?

### The Problem PHI Nodes Solve

Consider this Verilog-A snippet:

```verilog
if (TYPE == 1)
    ids = ids_nmos;  // NMOS branch
else
    ids = ids_pmos;  // PMOS branch
// What is `ids` here?
```

After the if-else, `ids` could be either `ids_nmos` or `ids_pmos` depending on which branch executed. In SSA form, every variable can only be assigned **once**, so we can't write:

```
ids = ids_nmos    // definition 1
ids = ids_pmos    // definition 2 - ERROR! SSA violation
```

PHI nodes resolve this by explicitly representing the merge:

```
block_merge:
    ids = phi [ids_nmos, block_nmos], [ids_pmos, block_pmos]
```

The PHI node says: "The value of `ids` depends on which predecessor block we came from."

### Why SSA Form?

SSA form is used by most high-quality optimizing compilers (LLVM, GCC, OpenVAF) because it makes optimizations easier:

- **Dead code elimination**: Unused definitions are obvious
- **Constant propagation**: Each value has one definition to trace
- **Register allocation**: No need to track "live ranges" of variables

---

## How PHI Nodes Work

### Semantics

A PHI node:
1. **Is placed at the start of a basic block** (before any other instructions)
2. **Selects a value based on control flow** — which predecessor we came from
3. **Executes instantaneously** — conceptually happens "on the edge" between blocks

```
phi [v1, block_a], [v2, block_b], [v3, block_c]
```

Means:
- If we came from `block_a`, use `v1`
- If we came from `block_b`, use `v2`
- If we came from `block_c`, use `v3`

### Example: Loop Counter

```
block_entry:
    v0 = 0
    br block_loop

block_loop:
    v1 = phi [v0, block_entry], [v2, block_loop]  // i = 0 or i+1
    v2 = add v1, 1
    cond = lt v2, 10
    br_if cond, block_loop, block_exit
```

The PHI node tracks the loop counter: on first iteration it's `v0` (0), on subsequent iterations it's `v2` (the incremented value).

### Example: Conditional Assignment

```verilog
real x;
if (a > 0)
    x = sqrt(a);
else
    x = 0;
y = x * 2;
```

Becomes:

```
block_entry:
    cond = gt a, 0
    br_if cond, block_true, block_false

block_true:
    v1 = sqrt a
    br block_merge

block_false:
    v2 = 0.0
    br block_merge

block_merge:
    v3 = phi [v1, block_true], [v2, block_false]
    v4 = mul v3, 2.0
```

---

## PHI Nodes in OpenVAF MIR

### Data Structure

From `openvaf/mir/src/instructions.rs`:

```rust
pub struct PhiNode {
    pub args: ValueList,   // Pool-allocated list of SSA values
    pub blocks: PhiMap,    // Maps predecessor Block → index into args
}
```

Key design choices:
- **Memory efficient**: Values stored in external pool, not inline
- **B-forest mapping**: O(log n) lookup for predecessor values
- **Sorted iteration**: Deterministic block ordering

### Supporting Types

```rust
pub type PhiForest = bforest::MapForest<Block, u32>;
pub type PhiMap = bforest::Map<Block, u32>;
```

The `PhiMap` uses a B-forest (balanced tree forest) data structure for efficient predecessor lookups.

### Text Representation

When you see MIR output (via `openvaf-viz` or debug prints):

```
v17 = phi [v16, block0], [v18, block1], [v19, block2]
```

This means:
- `v17` is the result of the PHI node
- Coming from `block0` → use `v16`
- Coming from `block1` → use `v18`
- Coming from `block2` → use `v19`

### PHI Node Creation (Braun Algorithm)

OpenVAF uses the Braun et al. (2013) algorithm for efficient SSA construction. PHI nodes are created when a variable is used in a block that has multiple predecessor blocks with different definitions.

From `openvaf/mir_build/src/ssa.rs`:

```rust
ZeroOneOrMore::More => {
    // Predecessors disagree on value - create PHI node
    let mut args = ValueList::new();
    let mut blocks = Map::new();

    for (pred_block, pred_val) in predecessors {
        let i = args.push(pred_val, &mut pool);
        blocks.insert(pred_block, i);
    }

    // Insert PHI at block entry point
    cursor.at_first_insertion_point(dest_block)
        .build(PhiNode { blocks, args });
}
```

### PHI Node Optimization

PHI nodes can be simplified when all inputs are identical:

```rust
// From openvaf/mir_opt/src/simplify.rs
pub fn simplify_phi(&mut self, phi: PhiNode) -> Option<Value> {
    let mut iter = self.func.dfg.phi_edges(&phi);
    if let Some((_, all_eq_val)) = iter.next() {
        // If ALL edges have the same value, eliminate the PHI
        if iter.all(|(_, val)| self.map_val(val) == all_eq_val) {
            return Some(all_eq_val);  // Replace PHI with single value
        }
    }
    None
}
```

---

## Converting PHI Nodes to JAX

### The Equivalence: SSA ≈ Functional Programming

There's a well-known correspondence between SSA form and functional programming, established by Appel's seminal paper "SSA is Functional Programming":

| SSA Concept | Functional Equivalent |
|-------------|----------------------|
| Basic block | Named function |
| PHI node | Function parameter |
| Jump to block | Tail call to function |
| PHI edge value | Argument in tail call |

In functional terms, each basic block becomes a function, and PHI nodes become the parameters to those functions. The call sites determine which values reach the PHI.

### If-Conversion: PHI to Select

For JAX specifically, we use **if-conversion** (also called **predication**) to transform control flow with PHI nodes into data-parallel select operations. This is necessary because:

1. JAX traces programs into a dataflow graph (jaxpr)
2. `jnp.where` evaluates both branches (no lazy short-circuiting)
3. GPU execution benefits from branchless code

The transformation converts:

```
block_merge:
    v3 = phi [v1, block_true], [v2, block_false]
```

Into:

```python
v3 = jnp.where(condition, v1, v2)
```

### The Algorithm

The general if-conversion algorithm for PHI nodes:

1. **Identify the condition** that determines which predecessor was taken
2. **Compute both branch values** (even if one won't be used)
3. **Use select to merge** based on the condition

For a 2-way PHI (if-else):
```python
# MIR: v_result = phi [v_true, block_true], [v_false, block_false]
# JAX: v_result = jnp.where(cond, v_true, v_false)
```

For an N-way PHI (switch/case):
```python
# MIR: v_result = phi [v0, block0], [v1, block1], [v2, block2]
# JAX: v_result = jnp.select([cond0, cond1, cond2], [v0, v1, v2], default)
# Or:  v_result = lax.switch(index, [lambda: v0, lambda: v1, lambda: v2])
```

### JAX Control Flow Primitives

| Primitive | Use Case | Evaluation |
|-----------|----------|------------|
| `jnp.where(cond, t, f)` | 2-way select | Both branches evaluated |
| `jnp.select(conds, vals)` | N-way select | All branches evaluated |
| `lax.cond(cond, t_fn, f_fn)` | Scalar condition | Lazy (one branch) |
| `lax.switch(idx, fns)` | N-way scalar | Lazy (one branch) |

For openvaf_jax, we typically use `jnp.where` because:
- Device evaluation is batched (conditions are arrays)
- Both branches contain useful gradient information
- GPUs prefer branchless code

### Nested PHI Nodes

Complex control flow creates nested PHI nodes:

```verilog
if (a > 0) {
    if (b > 0)
        x = 1;
    else
        x = 2;
} else {
    x = 3;
}
```

This creates PHI nodes at two levels, which translate to nested `jnp.where`:

```python
x_inner = jnp.where(b > 0, 1, 2)
x = jnp.where(a > 0, x_inner, 3)
```

### Performance Considerations

**Eager evaluation**: `jnp.where` always computes both branches. This is usually fine for numerical code but can be wasteful for:
- Expensive computations only needed in one branch
- Branches with different numerical stability requirements

**Solution for expensive branches**: Use `lax.cond` when:
- Condition is a scalar (not batched)
- One branch is significantly more expensive
- You need lazy evaluation for correctness

---

## Debugging PHI Node Issues

### Why PHI Nodes Matter for openvaf_jax

When OpenVAF compiles Verilog-A to JAX Python code, PHI nodes become `jnp.where()` calls. If the condition is wrong, or if one branch returns 0.0 when it shouldn't, the PHI node propagates that error through the entire computation.

### Common PHI Node Bugs

1. **Zero from wrong branch**: A PHI has `[v3, block_unused]` where `v3 = 0.0`. If control flow incorrectly takes this path, output goes to zero.

2. **TYPE parameter issues**: NMOS/PMOS selection depends on a PHI that chooses based on `TYPE == 1`. If TYPE isn't propagated correctly, wrong branch executes.

3. **Constant folding artifacts**: Optimizer inlines a constant into one PHI edge but not others, creating asymmetry.

4. **Dead branch pollution**: A branch that "shouldn't" execute still contributes values that get selected by `jnp.where`.

### Debugging Tools

#### MIR Analysis Script

```bash
# Find all PHI nodes
uv run scripts/analyze_mir_cfg.py model.va --func eval --find-phis

# Trace what value feeds into a specific PHI
uv run scripts/analyze_mir_cfg.py model.va --func eval --trace-value v12345

# See which block defines a suspicious value
uv run scripts/analyze_mir_cfg.py model.va --func eval --analyze-block block4654

# Find branch points (conditionals)
uv run scripts/analyze_mir_cfg.py model.va --func eval --branches
```

#### MIRInspector Class

```python
from vajax.debug import MIRInspector

inspector = MIRInspector(va_path)
inspector.print_phi_summary('eval')  # Show all PHI nodes

# Find PHIs that have v3 (often 0.0) as an operand
inspector.find_phi_nodes_with_value('v3')
```

### Typical Debugging Workflow

1. **Identify symptom**: JAX returns wrong current or zeros
2. **Compare outputs**: Use `ModelComparator` to find discrepancy
3. **Check cache**: Look for inf/nan or wrong temperature values
4. **Inspect PHIs**: Find PHI nodes with suspicious zero operands
5. **Trace control flow**: Identify which branch condition is wrong
6. **Fix translation**: Usually a parameter mapping or type issue

### Example: Zero Current Bug

**Symptom**: OSDI returns 1mA drain current, JAX returns 1e-15A

**Investigation**:
```python
inspector = MIRInspector(va_path)
zero_phis = inspector.find_phi_nodes_with_value('v3')  # v3 = 0.0
# Found: v789 = phi [v788, block_nmos], [v3, block_pmos]
```

**Root cause**: TYPE parameter wasn't passed correctly, so JAX took the PMOS branch (which returns 0 for an NMOS device).

**Fix**: Ensure `TYPE` is included in the parameter array passed to the compiled function.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Purpose** | Merge values from different control flow paths in SSA form |
| **Syntax** | `v_result = phi [v1, block1], [v2, block2], ...` |
| **Semantics** | "Select value based on which predecessor we came from" |
| **Placement** | Always at the start of a basic block |
| **In JAX** | Becomes `jnp.where(condition, val_true, val_false)` |
| **Algorithm** | If-conversion / predication transforms branches to selects |
| **Common bug** | One PHI edge has 0.0 and wrong branch is taken |

---

## References

- Appel, A. W. (1998). [SSA is Functional Programming](https://www.cs.princeton.edu/~appel/papers/ssafun.pdf). ACM SIGPLAN Notices.
- Braun, M. et al. (2013). Simple and Efficient Construction of Static Single Assignment Form. CC 2013.
- Chuang, W. et al. (2003). [Phi-Predication for Light-Weight If-Conversion](https://cseweb.ucsd.edu/~wchuang/CGO-03-PHI.pdf). CGO 2003.
- Kelsey, R. A. (1995). [A Correspondence between Continuation Passing Style and Static Single Assignment Form](https://wingolog.org/archives/2011/07/12/static-single-assignment-for-functional-programmers). IR'95.
- JAX Documentation: [Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow)
