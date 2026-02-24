# Performance Analysis

This document analyzes VA-JAX performance characteristics, explaining the
overhead profile relative to VACASK (C++ reference simulator) and the GPU
acceleration crossover point.

## Benchmark Results

All measurements from GitHub Actions CI runners (CPU: ubuntu-latest, GPU: nvidia-runner-1).
VACASK numbers on CPU use live execution; GPU comparisons use reference values.

### CPU: VA-JAX vs VACASK

| Benchmark | Nodes | Steps | JAX (ms/step) | VACASK (ms/step) | Ratio |
|-----------|------:|------:|--------------:|-----------------:|------:|
| rc        |     4 |    1M |         0.012 |            0.002 |  6.6x |
| graetz    |     6 |    1M |         0.020 |            0.004 |  5.4x |
| mul       |     8 |  500k |         0.041 |            0.004 | 10.9x |
| ring      |    47 |   20k |         0.511 |            0.109 |  4.7x |
| c6288     | ~5000 |    1k |        88.060 |           76.390 |  1.2x |

### GPU: VA-JAX Acceleration

| Benchmark | Nodes | GPU (ms/step) | CPU (ms/step) | GPU Speedup | vs VACASK CPU |
|-----------|------:|--------------:|--------------:|------------:|--------------:|
| c6288     | ~5000 |         19.81 |         88.06 |        4.4x | 0.35x (faster)|
| ring      |    47 |          1.49 |          0.51 |        0.3x | 33x (slower)  |
| graetz    |     6 |          0.30 |          0.02 |       0.07x | 161x (slower) |
| rc        |     4 |          0.24 |          0.01 |       0.05x | 257x (slower) |

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
Overhead ratio vs circuit size:

    mul (8 nodes)     ████████████████████████████████  10.9x
    rc (4 nodes)      ████████████████████             6.6x
    graetz (6 nodes)  ████████████████                 5.4x
    ring (47 nodes)   ██████████████                   4.7x
    c6288 (5k nodes)  ████                             1.2x
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

This explains the convergence to 1.2x at ~5000 nodes:

| Circuit | T_fixed | T_compute | Overhead % | Overall Ratio |
|---------|--------:|----------:|-----------:|--------------:|
| rc      |   10 us |      2 us |        83% |          6.6x |
| graetz  |   10 us |      4 us |        71% |          5.4x |
| ring    |   10 us |    100 us |         9% |          4.7x |
| c6288   |   10 us | 76,000 us |      0.01% |          1.2x |

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

VA-JAX builds the Jacobian from COO (coordinate) format:
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

### c6288 (5000 nodes): GPU Advantage

- Jacobian: 5000x5000 matrix operations
- Dense solve: O(n^3) = O(125 billion) operations per NR iteration
- GPU parallelism: thousands of concurrent threads
- Result: 4.4x speedup over CPU, **2.9x faster than VACASK**

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
without affecting the large-circuit GPU path where VA-JAX already
outperforms VACASK.
