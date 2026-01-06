# Ring Benchmark Analysis

## Circuit Overview

The ring benchmark is a **9-stage ring oscillator** using CMOS inverters built with PSP103 compact transistor models.

## Circuit Topology

```
                  +------------------------------------------------------+
                  |                                                      |
                  v                                                      |
Node 1 -> [INV u1] -> Node 2 -> [INV u2] -> Node 3 -> ... -> [INV u9] -> Node 9
    ^
    |
[ISOURCE i0] - Pulse to trigger oscillation
    |
    v
   GND
```

**Feedback:** Output of u9 (node 9) feeds back to input of u1 (node 1), closing the ring.

## Hierarchy Structure

Each inverter has this hierarchical structure:
```
inverter (subcircuit)
  +-- mp (pmos subcircuit)
  |     +-- m (psp103p OSDI instance) - PMOS transistor
  +-- mn (nmos subcircuit)
        +-- m (psp103n OSDI instance) - NMOS transistor
```

**Total instances:**
- 20 low-level (leaf) instances: 18 PSP transistors + 1 isource + 1 vsource
- 28 subcircuit instances: 9 inverters x 3 (inverter + pmos + nmos)

## Device Parameters

### Inverter Parameters (from netlist)

| Parameter | Value | Description |
|-----------|-------|-------------|
| w | 10u | Transistor width |
| l | 1u | Gate length |
| pfact | 2 | PMOS width factor (PMOS is 2x wider than NMOS) |

### PMOS Transistor (u1:mp:m)

| Parameter | Value |
|-----------|-------|
| w | 20um (10u x pfact) |
| l | 1um |
| as | 10e-12 (source area) |
| ad | 10e-12 (drain area) |
| ps | 41um (source perimeter) |
| pd | 41um (drain perimeter) |
| Terminals | D->node 2, G->node 1, S->vdd, B->vdd |

### NMOS Transistor (u1:mn:m)

| Parameter | Value |
|-----------|-------|
| w | 10um |
| l | 1um |
| as | 5e-12 (source area) |
| ad | 5e-12 (drain area) |
| ps | 21um (source perimeter) |
| pd | 21um (drain perimeter) |
| Terminals | D->node 2, G->node 1, S->GND, B->GND |

### Node Collapsing

Each PSP transistor has internal nodes that are collapsed:
- G -> GP (gate to gate-poly)
- S -> SI (source to internal source)
- D -> DI (drain to internal drain)
- B -> BI, BP -> BI, BS -> BI, BD -> BI (bulk nodes consolidated)

## System Size

| Metric | Value |
|--------|-------|
| Total nodes | 174 |
| Unknowns (after collapsing) | 47 |
| Matrix non-zeros | 220 |
| Sparsity ratio | 0.0996 |

The 47 unknowns include:
- 10 external nodes (0-9, vdd)
- 18 internal noise flow nodes (2 per transistor)
- 18 internal noise nodes (2 per transistor)
- 1 voltage source branch current

## Calculation Order

The evaluation proceeds in this order (see `lib/circuit.cpp:1358` and `lib/osdidevice.cpp:373`):

### 1. Device Iteration

Devices are processed sequentially:
```
__hierarchical__ (skipped - wrapper for subcircuits)
vsource
isource
vccs, vcvs, cccs, ccvs, mutual (builtin, unused)
psp103va (OSDI device - contains all transistors)
```

### 2. Model Iteration

For each device, models are processed:
- For psp103va: psp103n first, then psp103p

### 3. Instance Iteration

For each model, instances are evaluated sequentially:
- Under psp103n: u1:mn:m, u2:mn:m, u3:mn:m, ... u9:mn:m
- Under psp103p: u1:mp:m, u2:mp:m, u3:mp:m, ... u9:mp:m

### 4. Per-Instance Operations

In `evalAndLoad`:
1. `evalCore()` - Evaluate device equations, compute currents and Jacobians
2. `loadCore()` - Load values into system matrix

## Simulation Parameters

```
Transient analysis:
- Method: Trapezoidal (trap)
- Step: 0.05ns
- Stop: 1us
- Max step: 0.05ns
```

## Performance Statistics (from run)

| Metric | Value |
|--------|-------|
| Eval/load calls | 84,686 |
| NR iterations | 84,685 |
| Accepted timepoints | 24,769 |
| Rejected timepoints | 0 |
| Bypass opportunities | 445,824 |
| Bypassed evaluations | 445,824 (100% bypass rate) |

The high bypass rate indicates that the simulator efficiently skips re-evaluating devices when their terminal voltages haven't changed significantly.

## Sparsity Pattern

The Jacobian matrix has entries at the following (row, column) positions. The pattern shows coupling between:
- Adjacent ring nodes (1-2, 2-3, ..., 8-9, 9-1)
- Supply node (vdd = node 12) to all inverter outputs
- Internal noise nodes to their corresponding external nodes

Key sparsity characteristics:
- Diagonal dominance (all diagonal entries present)
- Banded structure for ring connectivity
- Dense column for vdd supply (couples to all stages)

## PSP103 Model Parameters

The PSP103 model uses extensive parameterization for accurate MOSFET modeling. Key parameter categories:

### Geometry Parameters
- `lvaro`, `wvaro` - Length/width variation
- `lap`, `dlq`, `dwq` - Overlap and delta parameters

### Threshold Voltage
- `vfbo` = -1.1V - Flatband voltage
- `toxo` = 1.5nm - Gate oxide thickness
- `nsubo` = 3e23 - Substrate doping

### Mobility
- `uo` = 0.035 - Low-field mobility
- `mueo` = 0.6 - Mobility degradation
- `themuo` = 2.75 - Temperature coefficient

### Saturation
- `thesato` = 1e-6 - Saturation velocity parameter
- `axo` = 20 - CLM parameter

### Junction Parameters
- `cjorbot` = 1e-3 - Bottom junction capacitance
- `vbirbot` = 0.75V - Built-in potential
- `idsatrbot` = 5e-9 - Saturation current

The full model contains 270+ parameters for comprehensive MOSFET behavior modeling.

## Detailed Analysis: evalCore and loadCore

The core evaluation loop for OSDI devices consists of two main functions that are called for each instance during simulation. These functions are defined in `lib/osdiinstance.cpp`.

### evalCore() - Device Evaluation

**Location:** `lib/osdiinstance.cpp:1104-1342`

**Purpose:** Evaluates the device model equations to compute currents, charges, and Jacobian matrices.

#### Input Data Structures

**EvalSetup** (defined in `include/elsetup.h`):
```cpp
struct EvalSetup {
    // State and solution vectors
    VectorRepository<double>* solution;    // Solution history
    VectorRepository<double>* states;      // State variable history
    double* deviceStates;                  // For bypass checking

    // Analysis mode flags
    bool staticAnalysis, dcAnalysis, acAnalysis, tranAnalysis, noiseAnalysis;

    // Limiting control
    bool enableLimiting, initializeLimiting;

    // Evaluation control flags
    bool evaluateResistiveJacobian;
    bool evaluateReactiveJacobian;
    bool evaluateResistiveResidual;
    bool evaluateReactiveResidual;

    // Bypass control
    bool forceBypass, allowBypass, requestHighPrecision;

    // Integration coefficients for transient
    IntegratorCoeffs* integCoeffs;

    // Output: bound step, breakpoints, etc.
    double boundStep, nextBreakPoint, maxFreq;
    double time;
};
```

#### Execution Flow

1. **Bypass Check** (lines 1144-1195):
   ```
   IF device is bypassable AND high precision not requested:
     IF output is converged AND inputs haven't changed significantly:
       SET bypass = true
       SKIP device evaluation
   ```

2. **Core Evaluation** (lines 1197-1250):
   ```
   IF NOT bypass:
     CALL descriptor->eval(handle, instance_core, model_core, simInfo)

     PROCESS eval flags:
       - EVAL_RET_FLAG_LIM: Limiting was applied
       - EVAL_RET_FLAG_FATAL: Fatal error, abort simulation
       - EVAL_RET_FLAG_FINISH: $finish called
       - EVAL_RET_FLAG_STOP: $stop called
   ```

3. **Reactive State Storage** (lines 1256-1286):
   ```
   FOR each nonzero reactive residual node:
     GET reactive residual contribution from instance core
     IF limiting applied:
       SUBTRACT linearized residual component
     STORE in states vector
     IF integCoeffs provided:
       DIFFERENTIATE to get flow (dq/dt)
       STORE flow in states vector
   ```

4. **Bound Step Processing** (lines 1317-1321):
   ```
   IF computeBoundStep enabled:
     GET bound_step from instance core
     UPDATE global bound step (minimum)
   ```

5. **Output Convergence Check** (lines 1323-1339):
   ```
   IF nr_bypass enabled AND device is bypassable:
     CHECK if outputs have converged
     UPDATE convergence flags for next iteration
   ```

#### Key OSDI API Calls

```cpp
// Main device evaluation
evalFlags = descriptor->eval(&handle, core(), model()->core(), &simInfo);
```

The `eval()` function is generated by OpenVAF from the Verilog-A model and:
- Computes all branch currents
- Computes all charges
- Computes Jacobian derivatives (partial derivatives of currents/charges w.r.t. voltages)
- Handles limiting functions for convergence
- Sets internal states for integration

### loadCore() - Matrix Loading

**Location:** `lib/osdiinstance.cpp:1344-1500`

**Purpose:** Loads the computed values from evalCore into the system matrix and RHS vector.

#### Input Data Structures

**LoadSetup** (defined in `include/elsetup.h`):
```cpp
struct LoadSetup {
    VectorRepository<double>* states;

    // Jacobian loading control
    bool loadResistiveJacobian;
    bool loadReactiveJacobian;
    double reactiveJacobianFactor;
    bool loadTransientJacobian;
    IntegratorCoeffs* integCoeffs;
    MatrixEntryIndex jacobianLoadOffset;

    // Residual loading destinations
    double* resistiveResidual;
    double* reactiveResidual;
    double* linearizedResistiveRhsResidual;
    double* linearizedReactiveRhsResidual;
    double* reactiveResidualDerivative;

    // Max contribution tracking
    double* maxResistiveResidualContribution;
    double* maxReactiveResidualContribution;
    double* maxReactiveResidualDerivativeContribution;
};
```

#### Execution Flow

1. **Jacobian Loading** (lines 1376-1398):
   ```
   IF loadResistiveJacobian:
     CALL descriptor->load_jacobian_resist(instance, model)
     // Adds dI/dV contributions to matrix

   IF loadReactiveJacobian:
     CALL descriptor->load_jacobian_react(instance, model, factor)
     // Adds dQ/dV contributions scaled by factor

   IF loadTransientJacobian:
     CALL descriptor->load_jacobian_tran(instance, model, alpha)
     // Adds dI/dV + alpha*dQ/dV where alpha = integCoeffs->leadingCoeff()
   ```

2. **Residual Loading** (lines 1414-1430):
   ```
   IF resistiveResidual provided:
     CALL descriptor->load_residual_resist(instance, model, resistiveResidual)
     // Loads I(V) contributions into RHS vector

   IF reactiveResidual provided:
     CALL descriptor->load_residual_react(instance, model, reactiveResidual)
     // Loads Q(V) contributions into RHS vector

   IF limiting was applied:
     CALL descriptor->load_limit_rhs_resist/react(...)
     // Subtracts linearized residual correction
   ```

3. **Max Contribution Tracking** (lines 1433-1497):
   ```
   FOR each nonzero resistive residual:
     GET contribution from instance core
     IF limiting: SUBTRACT linearized component
     UPDATE max contribution for that unknown

   FOR each nonzero reactive residual:
     GET charge and flow from states vector
     UPDATE max charge contribution
     UPDATE max flow contribution
     IF reactiveResidualDerivative provided:
       ADD flow to RHS
   ```

#### Key OSDI API Calls

```cpp
// Jacobian loading
descriptor->load_jacobian_resist(core(), model()->core());
descriptor->load_jacobian_react(core(), model()->core(), factor);
descriptor->load_jacobian_tran(core(), model()->core(), alpha);

// Residual loading
descriptor->load_residual_resist(core(), model()->core(), rhs);
descriptor->load_residual_react(core(), model()->core(), rhs);

// Limiting corrections
descriptor->load_limit_rhs_resist(core(), model()->core(), rhs);
descriptor->load_limit_rhs_react(core(), model()->core(), rhs);
```

### Data Flow Diagram

```
                    +------------------+
                    |   OsdiDevice::   |
                    |   evalAndLoad()  |
                    +--------+---------+
                             |
             +---------------+---------------+
             |                               |
             v                               v
    +--------+--------+             +--------+--------+
    | FOR each model  |             | FOR each model  |
    +--------+--------+             +--------+--------+
             |                               |
             v                               v
    +--------+--------+             +--------+--------+
    | FOR each inst   |             | FOR each inst   |
    +--------+--------+             +--------+--------+
             |                               |
             v                               v
    +--------+--------+             +--------+--------+
    |    evalCore()   |             |    loadCore()   |
    +--------+--------+             +--------+--------+
             |                               |
             v                               v
    +--------+--------+             +--------+--------+
    | descriptor->    |             | descriptor->    |
    |   eval()        |             | load_jacobian_* |
    +-----------------+             | load_residual_* |
                                    +-----------------+
             |                               |
             v                               v
    +--------+--------+             +--------+--------+
    | Instance Core   |             | System Matrix   |
    | - currents      | ----------> | - Jacobian      |
    | - charges       |             | - RHS vector    |
    | - Jacobians     |             +-----------------+
    +-----------------+
```

### Instance Core Memory Layout

Each OSDI instance has a `core_` pointer to a memory block containing:

```
+---------------------------+
| Node mapping array        |  <- nodeMappingArray()
+---------------------------+
| Collapsed nodes pattern   |  <- collapsedNodesPattern()
+---------------------------+
| Jacobian pointers (resist)|  <- resistiveJacobianPointers()
+---------------------------+
| Jacobian pointers (react) |  <- reactiveJacobianPointer(i)
+---------------------------+
| State index table         |  <- stateIndexTable()
+---------------------------+
| Node residuals (resist)   |  <- nodes[i].resist_residual_off
+---------------------------+
| Node residuals (react)    |  <- nodes[i].react_residual_off
+---------------------------+
| Limiting RHS (resist)     |  <- nodes[i].resist_limit_rhs_off
+---------------------------+
| Limiting RHS (react)      |  <- nodes[i].react_limit_rhs_off
+---------------------------+
| Model-specific data       |  (parameters, internal vars)
+---------------------------+
```

### Bypass Optimization

The bypass mechanism significantly improves performance by skipping device evaluation when:

1. **Input Bypass Check** (`inputBypassCheckCore`):
   - Compares current input voltages to stored values
   - If change < tolerance * bypasstol, bypass is allowed

2. **Output Bypass Check** (`outputBypassCheckCore`):
   - Compares current residuals to stored values
   - If change < tolerance * convtol, outputs are converged
   - Stores history for next iteration

In the ring benchmark, 100% of bypass opportunities were taken (445,824 evaluations skipped), demonstrating the effectiveness of this optimization for steady-state operation.
