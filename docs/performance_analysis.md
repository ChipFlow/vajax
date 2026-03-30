# Performance Analysis

This document analyzes VAJAX performance characteristics, explaining the
overhead profile relative to VACASK (C++ reference simulator) and the GPU
acceleration crossover point.

## Benchmark Results

All measurements from GitHub Actions CI runners (CPU: ubuntu-latest, GPU: nvidia-runner-1).
VACASK numbers on CPU use live execution; GPU comparisons use reference values.

### CPU: VAJAX vs VACASK

| Benchmark | Nodes | Steps | JAX (ms/step) | VACASK (ms/step) | Ratio |
|-----------|------:|------:|--------------:|-----------------:|------:|
| rc        |     4 |    1M |         0.023 |            0.002 | 12.2x |
| graetz    |     6 |    1M |         0.033 |            0.004 |  8.6x |
| mul       |     8 |  500k |         0.040 |            0.004 | 10.4x |
| ring      |    47 |   20k |         0.516 |            0.108 |  4.8x |
| c6288     | ~5000 |   20  |       163.684 |          288.800 |  0.57x|
| mul64     | ~133k |   15  |      8324.949 |          timeout |   —   |

> **c6288**: VAJAX is **1.8x faster** than VACASK using UMFPACK sparse solver on CPU.
> The crossover where VAJAX exceeds VACASK performance occurs around ~5000 nodes.
>
> **mul64** (64x64 array multiplier): ~266k MOSFETs, ~666k unknowns (133k nodes +
> 533k internal). VACASK times out on CI (>5 min per step). VAJAX completes at
> ~8.3s/step using UMFPACK sparse solver. On GPU with float32 cuDSS factorization
> and iterative refinement, mul64 runs at ~648 ms/step on Tesla T4 (16GB) —
> a **12.8x GPU speedup**.

### GPU: VAJAX Acceleration

| Benchmark | Nodes | GPU (ms/step) | CPU (ms/step) | GPU Speedup | vs VACASK CPU |
|-----------|------:|--------------:|--------------:|------------:|--------------:|
| mul64     | ~133k |        648.00 |      8324.95  |       12.8x |  N/A (timeout)|
| c6288     | ~5000 |         19.81 |       163.68  |        8.3x | 0.07x (faster)|
| ring      |    47 |          1.49 |          0.52 |        0.3x | 14x (slower)  |
| graetz    |     6 |          0.30 |          0.03 |       0.1x  | 75x (slower)  |
| rc        |     4 |          0.24 |          0.02 |       0.08x | 120x (slower) |

> **mul64** uses float32 cuDSS factorization with iterative refinement, reducing
> VRAM usage from >16GB (f64 OOM) to ~10GB on Tesla T4 (16GB). The f32 factorization
> provides near-f64 accuracy via a single refinement step: solve in f32, compute
> residual r=f+J@x in f64 (SpMV only), solve correction in f32, return x+d.

GPU results for circuits below ~500 nodes reflect GPU kernel overhead on tiny
workloads, not simulation inefficiency. The auto-threshold (`gpu_threshold=500`)
prevents this in normal usage; the benchmark uses `--force-gpu` to measure all
circuits for tracking purposes.

### Accuracy (vs VACASK)

| Benchmark | Dense RMS | Sparse RMS | Threshold |
|-----------|----------:|-----------:|----------:|
| rc        |     0.00% |      0.00% |        5% |
| graetz    |     0.00% |      0.00% |       15% |
| mul       |     0.00% |      0.00% |        2% |
| c6288     |         - |      2.01% |       10% |

---

## Per-Step Overhead Analysis

The overhead ratio decreases as circuit size increases, confirming a **fixed
per-step overhead** in the JAX path that VACASK doesn't have:

```
Overhead ratio vs circuit size (CPU):

    rc (4 nodes)       ████████████████████████████████████  12.2x
    mul (8 nodes)      ██████████████████████████████████    10.4x
    graetz (6 nodes)   ██████████████████████████            8.6x
    ring (47 nodes)    ███████████████                       4.8x
    c6288 (5k nodes)   ██                                    0.57x (VAJAX faster)
    mul64 (133k nodes) █                                     N/A (VACASK timeout)
```

### Overhead Breakdown

Each timestep in `full_mna.py` body_fn executes ~11 major operations beyond the
core Newton-Raphson solve:

| Operation | Estimated Cost | VACASK Equivalent | Notes |
|-----------|---------------|-------------------|-------|
| LTE estimation | 2-3 us | Similar | Per-node tolerance checking, error coefficients |
| Voltage prediction | 1-2 us | Simpler predictor | Lagrange polynomial from multi-point history |
| BDF2 coefficients | 0.5-1 us | Pre-computed | Variable-step formula recomputed every step |
| History management | 1-2 us | In-place mutation | `jnp.roll` on V_history, dt_history arrays |
| `jnp.where` branching | 2-3 us | Runtime `if` | Evaluates both branches unconditionally |
| Vmap device eval | 1-2 us/NR iter | Sequential loop | Overhead dominates for <10 devices |
| COO assembly | 1-2 us/NR iter | Direct stamping | Concatenate + sum duplicates |
| Simparams `.at[].set()` | 0.5-1 us/NR iter | Struct writes | Creates intermediate arrays |
| **Total fixed overhead** | **~10-15 us/step** | **~0 us** | |

### Scaling Model

The per-step cost can be modeled as:

```
T_jax(n) = T_fixed + T_compute(n)
T_vacask(n) = T_compute_vacask(n)

where:
  T_fixed ~ 10-15 us (JAX overhead, independent of circuit size)
  T_compute(n) ~ T_compute_vacask(n) for large n (same algorithm)
```

For large circuits (c6288, mul64), VAJAX's UMFPACK sparse solver is actually
**faster** than VACASK — the fixed overhead is negligible and the solver
implementation is competitive:

| Circuit | T_fixed | T_compute | Overhead % | Overall Ratio |
|---------|--------:|----------:|-----------:|--------------:|
| rc      |   10 us |      2 us |        83% |         12.2x |
| graetz  |   10 us |      4 us |        71% |          8.6x |
| ring    |   10 us |    100 us |         9% |          4.8x |
| c6288   |   10 us |163,000 us |     <0.01% |         0.57x |

---

## Why JAX Has Per-Step Overhead

### 1. Functional Array Updates

JAX arrays are immutable. Inside `lax.while_loop`, conditional updates use
`jnp.where` which evaluates **both branches** and selects elementwise:

```python
# JAX: evaluates both new_val and old_val, then selects
state = jnp.where(accept_step, new_val, old_val)
```

```c
// VACASK (C++): skips the else branch entirely
if (accept_step) state = new_val;
```

The body_fn has ~15 such conditional updates per step for history management,
step acceptance, and NR failure handling.

### 2. Vmap Batching for Small Counts

Device evaluation uses `jax.vmap` to batch all instances of each device type.
This is essential for GPU parallelism on large circuits but adds overhead for
small batch sizes:

```python
# 2 resistors: vmap adds vectorization overhead > sequential evaluation
vmapped_eval = jax.vmap(device_eval)(batched_params)
```

For rc (2 resistors), the vmap overhead exceeds the actual computation.
For c6288 (~86,000 transistors), vmap amortizes perfectly.

### 3. COO Matrix Assembly

VAJAX builds the Jacobian from COO (coordinate) format:
1. Each device type produces (row, col, value) triplets
2. Triplets are concatenated across device types
3. Duplicate indices are summed to form the final matrix

VACASK stamps directly into a pre-allocated matrix with known positions.
The COO approach enables JAX tracing and GPU parallelism but adds indirection.

### 4. Integration Coefficient Recomputation

Variable-step BDF2 requires recomputing integration coefficients each step
based on the step-size ratio. VACASK may use lookup tables or simplified
formulas for common step ratios.

---

## GPU Acceleration: Why Large Circuits Win

GPU acceleration becomes beneficial when the per-step compute time exceeds
the kernel launch and memory overhead (~100-500 us per step). This happens
around 500+ nodes.

### mul64 (133k nodes): GPU Stress Test

- ~266k PSP103 MOSFET evaluations per NR iteration
- Sparse Jacobian: 666,409 unknowns (133k nodes + 533k internal + 130 currents)
- 3,259,918 CSR entries (from 30.5M COO triplets)
- Dense solve impossible (~3.3TB memory); sparse solver required
- **Float32 factorization**: cuDSS factorizes J in f32 (halves VRAM), iterative
  refinement recovers near-f64 accuracy. Reduces peak VRAM from >16GB to ~10GB.
- Tesla T4 (16GB): **648 ms/step** — 12.8x faster than CPU (8325 ms/step)
- CPU uses UMFPACK sparse solver at ~8.3s/step; VACASK times out (>5 min/step)

### c6288 (5000 nodes): GPU Advantage

- Jacobian: ~5000x5000 sparse matrix (~86k transistors, ~5k nodes)
- Uses cuDSS sparse solver on GPU, UMFPACK on CPU
- GPU: 19.81 ms/step, CPU: 163.68 ms/step — **8.3x GPU speedup**
- vs VACASK CPU (288.8 ms/step): **14.6x faster on GPU**

### rc (4 nodes): GPU Disadvantage

- Jacobian: 4x4 matrix operations
- Dense solve: 64 multiply-adds per NR iteration
- GPU overhead: kernel launch > actual compute
- Result: 20x slower than CPU (expected and correct)

The `gpu_threshold` parameter (default: 500 nodes) automatically routes small
circuits to CPU, avoiding this overhead in normal usage.

---

## Potential Optimizations

These are known areas where the per-step overhead could be reduced:

1. **Fused operations**: Combine BDF2 coefficient computation with history
   updates into a single kernel to reduce launch overhead.

2. **Direct stamping**: For CPU path, stamp directly into pre-allocated
   Jacobian instead of building COO triplets.

3. **Sequential device eval on CPU**: Use a simple loop instead of vmap
   when running on CPU with few device instances.

4. **Pre-computed coefficients**: Cache BDF2 coefficients for common
   step-size ratios.

5. **Reduced history depth**: Currently maintains multi-step history for
   BDF2; simpler methods (trapezoidal) would reduce overhead.

These optimizations would primarily benefit small-circuit CPU performance
without affecting the large-circuit GPU path where VAJAX already
outperforms VACASK.

## Model Eval Optimization: SCCP and Branchless Ops

### SCCP Constant Propagation (Implemented)

Sparse Conditional Constant Propagation seeds known parameter values
(including init-computed cache/hidden_state) into the codegen. This
eliminates dead blocks at code generation time:

| Model | Dead Blocks | Code Reduction | jnp.where Reduction |
|-------|-------------|----------------|---------------------|
| PSP103 | 89% (695/954) | 42% fewer lines | 67% fewer (2247→743) |
| BSIM4 | 49% | - | - |
| BSIM3 | 47% | - | - |

Runtime improvement on c6288: **19% faster** (58→47ms/step).

### Branchless `lexp` Operator (TODO)

BSIM models define a safe-exp function `lexp(x)` that guards against overflow:
```verilog
if (x > 80)      lexp = MAX_EXPL * (1 + x - 80);   // linear extrapolation
else if (x < -80) lexp = MIN_EXPL;                   // clamp
else               lexp = exp(x);
```

OpenVAF inlines this as 2 MIR blocks + 1 PHI per call → 2 `jnp.where` in
generated code. BSIMBULK has ~155 of these, BSIMCMG ~136.

**Proposed fix**: Add `lexp` as a JAX custom op with `@jax.custom_jvp`:
```python
@jax.custom_jvp
def lexp(x):
    safe_x = jnp.clip(x, -80.0, 80.0)
    base = jnp.exp(safe_x)
    return jnp.where(x > 80.0, MAX_EXPL * (1.0 + x - 80.0),
           jnp.where(x < -80.0, MIN_EXPL, base))
```

This is architecture-independent (CPU SIMD, GPU, Metal all use branchless
select+exp+clip). Integration options:
1. **OpenVAF compiler**: Emit `CallbackKind::Lexp` for `lexp`/`limexp` analog
   functions instead of inlining the body. Then `openvaf_jax` translates the
   callback to the JAX op (like `WhiteNoise`, `TimeDerivative`).
2. **Codegen pattern match**: Detect the lexp MIR pattern (±80 compare, branch,
   exp in one arm) and replace with a single `lexp()` call. More fragile.

Impact: eliminates ~150+ branches per BSIM model eval, reduces `jnp.where`
count, and avoids computing `exp(x)` when x overflows (single `exp(clip(x))`).
