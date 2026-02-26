# Parallelism Architecture: c6288 Case Study

This document traces how VAJAX exploits parallelism at every stage of circuit
simulation, using the c6288 16x16 combinational multiplier as a concrete example.

## Circuit Overview

The c6288 circuit is a 16-bit Wallace-tree multiplier:

- **256 AND gates** (6 MOSFETs each: 3 PMOS + 3 NMOS)
- **2,128 NOR gates** (4 MOSFETs each: 2 PMOS + 2 NMOS)
- **10,048 PSP103 MOSFETs** total (single model, TYPE parameter distinguishes NMOS/PMOS)
- **32 resistors** (in input drivers)
- **34 voltage sources** (32 driver + VDD + VSS)
- **~5,089 external circuit nodes**

After node collapse (PSP103 has 8 internal nodes, 6 collapse), the system has:
- 5,089 external nodes + 20,096 internal nodes (2 per MOSFET) + 34 vsource branch currents
- **~25,219 unknowns** in the augmented MNA system

## End-to-End Pipeline

The entire simulation -- all timesteps, all Newton-Raphson iterations, all device
evaluations -- compiles into a **single XLA program** via `jax.jit`. After the
one-time JIT warmup, zero Python interpreter overhead remains.

```mermaid
flowchart TB
    subgraph once ["One-time setup (Python)"]
        direction TB
        A[Parse .sim netlist] --> B[Compile PSP103.va via OpenVAF]
        B --> C[Group 10,048 MOSFETs by model type]
        C --> D["Split params: shared vs per-device"]
        D --> E[Pre-compute COO stamp indices]
        E --> F[Trial eval → discover sparsity pattern]
        F --> G[Pre-compute COO→CSR permutation]
    end

    subgraph jit ["JIT-compiled XLA program (GPU/CPU)"]
        direction TB
        H["Transient loop (lax.while_loop)"]
        H --> I[Compute integration coefficients]
        I --> J[Evaluate source waveforms]
        J --> K[Predict voltages from history]
        K --> L["NR loop (lax.while_loop)"]
        L --> M{Converged?}
        M -->|No| L
        M -->|Yes| N[LTE estimation]
        N --> O{Accept step?}
        O -->|No, halve dt| H
        O -->|Yes| P[Store solution, advance time]
        P --> H
    end

    once --> jit
```

## Newton-Raphson Iteration Detail

Each NR iteration is the performance-critical inner loop. Here is what happens
inside `build_system_mna` + `linear_solve`:

```mermaid
flowchart TB
    subgraph nr ["Single NR Iteration"]
        direction TB

        subgraph extract ["1. Voltage Extraction (vectorized gather)"]
            V["V: solution vector (25,219)"]
            VE["V[node1] - V[node2]<br/>10,048 × 13 terminal voltages"]
            V --> VE
        end

        subgraph eval ["2. Batched Device Evaluation (jax.vmap)"]
            direction TB
            E1["PSP103 batch: vmap over 10,048 devices<br/>shared_params (broadcast) + device_params (10,048 × ~20)"]
            E2["Resistor batch: vmap over 32 devices"]
            E1 --> R1["10,048 × residuals + 10,048 × Jacobian entries"]
            E2 --> R2["32 × residuals + 32 × Jacobian entries"]
        end

        subgraph stamp ["3. COO Stamping (pre-computed index arrays)"]
            direction TB
            S1["Map local → global indices via stamp_indices"]
            S2["mask_coo_vector: zero out ground-node entries"]
            S1 --> S2
        end

        subgraph asm ["4. Matrix Assembly (segment_sum)"]
            direction TB
            A1["~320K COO triplets (row, col, val)"]
            A2["segment_sum scatter-add → BCOO sparse matrix"]
            A1 --> A2
        end

        subgraph solve ["5. Sparse Linear Solve"]
            direction TB
            L1["COO→CSR via pre-computed permutation + segment_sum"]
            L2["Enforce NOI constraints (zero rows/cols, unit diagonal)"]
            L3["UMFPACK (CPU) or cuDSS (GPU) sparse LU solve"]
            L1 --> L2 --> L3
            L3 --> D1["δV: Newton update (25,219)"]
        end

        extract --> eval --> stamp --> asm --> solve
    end
```

## Where Parallelism Happens

Each stage has a distinct parallelism mechanism:

```mermaid
flowchart LR
    subgraph stages ["Pipeline Stages"]
        direction TB
        S1["Voltage<br/>Extraction"]
        S2["Device<br/>Evaluation"]
        S3["COO<br/>Stamping"]
        S4["Matrix<br/>Assembly"]
        S5["Linear<br/>Solve"]
    end

    subgraph parallel ["Parallelism Mechanism"]
        direction TB
        P1["Vectorized gather<br/>V[indices] - V[indices]"]
        P2["jax.vmap<br/>10,048 parallel threads"]
        P3["Vectorized index<br/>mapping + masking"]
        P4["jax.ops.segment_sum<br/>parallel scatter-add"]
        P5["Sparse LU factorization<br/>(cuDSS on GPU)"]
    end

    subgraph scale ["Scale for c6288"]
        direction TB
        D1["10,048 × 13 lookups"]
        D2["10,048 PSP103 evals<br/>+ 32 resistor evals"]
        D3["~320K COO entries"]
        D4["320K → ~200K unique<br/>in 25K × 25K matrix"]
        D5["25,219 × 25,219<br/>sparse system"]
    end

    S1 --- P1 --- D1
    S2 --- P2 --- D2
    S3 --- P3 --- D3
    S4 --- P4 --- D4
    S5 --- P5 --- D5
```

## Parameter Splitting: Shared vs Per-Device

The key optimization for batched evaluation is separating parameters that are
constant across all 10,048 MOSFETs from those that vary per device.

```mermaid
graph LR
    subgraph shared ["Shared (broadcast, 1D)"]
        SP["~800 model params<br/>(TOX, VFB0, NSUBO, ...)"]
        SC["~400 cache values<br/>(computed by init)"]
        SIM["simparams<br/>(analysis_type, gmin)"]
    end

    subgraph varying ["Per-device (batched, 2D)"]
        VP["device_params<br/>10,048 × ~20<br/>(voltages + W, L, TYPE)"]
        VC["device_cache<br/>10,048 × ~60<br/>(init results that vary)"]
        LS["limit_state<br/>10,048 × n_lim"]
    end

    subgraph vmap_call ["jax.vmap(eval_fn, in_axes=(None, 0, None, 0, None, 0))"]
        EVAL["PSP103 compact<br/>model equations"]
    end

    shared --> vmap_call
    varying --> vmap_call
    vmap_call --> OUT["10,048 × residuals<br/>10,048 × Jacobian entries"]
```

The `in_axes=(None, 0, None, 0, None, 0)` specification tells JAX:
- `None`: broadcast this input to all 10,048 invocations (shared params, shared cache, simparams)
- `0`: slice along the first dimension, one row per device (device params, device cache, limit state)

This means the ~800 shared model parameters are loaded once into registers/cache,
while only the ~20 varying parameters differ per thread.

## COO Stamping and Assembly

Each device produces local residuals and Jacobian entries indexed by local
node numbers (0..12 for PSP103). These must be mapped to global circuit indices.

```mermaid
flowchart TB
    subgraph local ["Per-device local output"]
        direction LR
        LR["residual[0..5]<br/>(6 node contributions)"]
        LJ["jacobian[0..31]<br/>(up to 32 dI/dV entries)"]
    end

    subgraph mapping ["Stamp index mapping"]
        direction LR
        RI["res_indices: (10,048 × 6)<br/>local node → global row"]
        JR["jac_row_indices: (10,048 × 32)"]
        JC["jac_col_indices: (10,048 × 32)"]
    end

    subgraph global ["Global COO arrays"]
        direction LR
        GR["f_resist: ~60K valid entries<br/>→ segment_sum → f[25,219]"]
        GJ["J entries: ~320K triplets<br/>→ segment_sum → J[25,219 × 25,219]"]
    end

    local --> mapping --> global
```

Ground-node entries are mapped to index -1 and masked to zero, so they don't
pollute the system.

## Sparse Solver Path

The c6288 system is too large for dense linear algebra (25K × 25K × 8 bytes = ~5GB).
The sparse path avoids materializing the full matrix:

```mermaid
flowchart TB
    subgraph coo ["BCOO from assembly"]
        C1["~320K COO triplets<br/>(row, col, val)"]
    end

    subgraph convert ["COO → CSR (pre-computed)"]
        C2["sort by (row, col) via permutation"]
        C3["segment_sum to merge duplicates"]
        C4["CSR: ~200K stored elements"]
    end

    subgraph noi ["NOI Constraint Enforcement"]
        N1["Zero out NOI rows/cols in CSR data"]
        N2["Set NOI diagonal to 1.0"]
        N3["Zero NOI entries in RHS"]
    end

    subgraph solve ["Backend-specific solve"]
        direction LR
        GPU["GPU: cuDSS/Spineax<br/>Cached symbolic factorization<br/>Only numerical refactor per NR iter"]
        CPU["CPU: UMFPACK via FFI<br/>Direct sparse LU"]
    end

    coo --> convert --> noi --> solve
```

The COO→CSR conversion is itself parallel: the permutation and segment_sum are
pre-computed during setup, so at runtime it's just a gather + scatter-add.

## Transient Time-Stepping Loop

The outer loop is also JIT-compiled via `lax.while_loop`:

```mermaid
stateDiagram-v2
    [*] --> ComputeCoeffs: t < t_stop

    ComputeCoeffs: Compute integration coefficients
    ComputeCoeffs --> EvalSources: c0, c1 from BDF/Trap

    EvalSources: Evaluate source waveforms
    EvalSources --> Predict: vsource_vals at t+dt

    Predict: Extrapolate V from history
    Predict --> NRSolve: V_pred as initial guess

    NRSolve: Newton-Raphson solve
    NRSolve --> CheckNR

    CheckNR: NR converged?
    CheckNR --> LTE: Yes
    CheckNR --> Reject: No (halve dt)

    LTE: Estimate local truncation error
    LTE --> CheckLTE

    CheckLTE: LTE acceptable?
    CheckLTE --> Accept: Yes
    CheckLTE --> Reject: No (reduce dt)

    Reject: Reduce timestep
    Reject --> ComputeCoeffs

    Accept: Store solution
    Accept --> AdvanceTime

    AdvanceTime: Update history, advance t
    AdvanceTime --> ComputeCoeffs: t < t_stop
    AdvanceTime --> [*]: t >= t_stop
```

## Performance Profile

For c6288 on CPU (CI benchmark results):

| Metric | Value |
|--------|-------|
| Timesteps | ~1,000 |
| NR iterations/step | 5-20 |
| Device evals/NR iter | 10,048 PSP103 + 32 resistors |
| Per-step time (VAJAX) | 90 ms |
| Per-step time (VACASK) | 80 ms |
| Total wall time (VAJAX) | 65.7s (includes JIT) |
| Total wall time (VACASK) | 80.2s |
| Speedup (total) | **1.22x faster** (JIT amortized) |

The per-step overhead (~10ms) comes from adaptive timestep machinery, `jnp.where`
branching, and COO assembly. This overhead is fixed regardless of circuit size,
which is why c6288 (~90ms/step) is competitive while small circuits (rc: 0.014ms
VAJAX vs 0.002ms VACASK) show higher ratios.

On GPU, the vmap'd device evaluation and cuDSS sparse solve provide additional
speedup for large circuits, as the 10,048 parallel PSP103 evaluations map
directly to GPU threads.
