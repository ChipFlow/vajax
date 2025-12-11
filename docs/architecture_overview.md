# JAX-SPICE Architecture Overview

This document provides a high-level overview of JAX-SPICE's architecture for developers new to the codebase.

## Design Philosophy

JAX-SPICE is built on three core principles:

1. **Functional Device Models**: Devices are pure functions, not stateful objects
2. **Automatic Differentiation**: Jacobians computed via JAX, no explicit derivatives
3. **Vectorization**: Same-type devices evaluated in parallel

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Code                                  │
│   (Define circuit, run analysis, process results)                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Analysis Layer                               │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │   dc.py     │  │  transient.py    │  │      mna.py           │  │
│  │ DC solver   │  │ Transient solver │  │ Modified Nodal        │  │
│  │ Newton-     │  │ Backward Euler   │  │ Analysis system       │  │
│  │ Raphson     │  │ time integration │  │ Device groups         │  │
│  └─────────────┘  └──────────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Device Layer                                │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────────────┐ │
│  │ Resistor  │ │ Capacitor │ │ MOSFET    │ │ VerilogADevice      │ │
│  │           │ │           │ │ Simple    │ │ (OpenVAF models)    │ │
│  └───────────┘ └───────────┘ └───────────┘ └─────────────────────┘ │
│                          │                                          │
│                          ▼                                          │
│              DeviceStamps (currents, conductances)                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           JAX Runtime                                │
│     ┌──────────────┐  ┌─────────────┐  ┌────────────────────┐      │
│     │     JIT      │  │    vmap     │  │   Autodiff         │      │
│     │ Compilation  │  │ Vectorize   │  │ (Jacobians)        │      │
│     └──────────────┘  └─────────────┘  └────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow: DC Operating Point

```
1. Circuit Definition
   └── MNASystem.add_device(name, device, nodes)

2. System Preparation
   └── MNASystem.build_device_groups()
       └── VectorizedDeviceGroup (per device type)
           ├── node_indices: Array[n_devices, n_terminals]
           └── params: Dict[str, Array[n_devices]]

3. Newton-Raphson Loop
   │
   ├── build_jacobian_and_residual(V)
   │   ├── For each VectorizedDeviceGroup:
   │   │   ├── Extract terminal voltages: V[node_indices]
   │   │   ├── Evaluate devices: vmap(device_fn)(V_terminals, params)
   │   │   └── Stamp into matrix/vector
   │   │
   │   └── Compute Jacobian via JAX autodiff
   │
   ├── Solve: delta_V = J^(-1) * (-residual)
   │   ├── Dense solve (small circuits)
   │   └── Sparse solve (>1000 nodes)
   │
   ├── Update: V = V + delta_V (with limiting)
   │
   └── Check convergence: max(|residual|) < abstol?

4. Return solution V
```

## Data Flow: Transient Analysis

```
1. Initial Conditions
   ├── icmode='op': Run DC operating point
   └── icmode='uic': Use zeros + supply voltages

2. Time Loop (t = 0 to t_stop)
   │
   ├── Build companion models for reactive elements
   │   └── Capacitor: I_eq = C/dt * V_prev, G_eq = C/dt
   │
   ├── Newton-Raphson (same as DC but with companion models)
   │
   ├── Store solution: solutions[t] = V
   │
   └── Advance time: t += dt, V_prev = V

3. Return times, solutions arrays
```

## Key Classes

### MNASystem (`analysis/mna.py`)

The central orchestrator that manages devices and builds circuit equations.

```python
class MNASystem:
    # Core data
    devices: List[DeviceInfo]        # All devices in circuit
    node_map: Dict[str, int]         # Node name → index
    device_groups: Dict[type, VectorizedDeviceGroup]

    # Main methods
    def add_device(name, device, nodes): ...
    def build_device_groups(): ...           # Prepare for vectorized eval
    def build_jacobian_and_residual(V): ...  # Core evaluation
    def get_node_index(name) -> int: ...
```

### VectorizedDeviceGroup (`analysis/mna.py`)

Groups same-type devices for efficient parallel evaluation.

```python
class VectorizedDeviceGroup:
    device_type: type                  # e.g., Resistor, MOSFETSimple
    device_names: List[str]            # ["R1", "R2", ...]
    node_indices: Array                # Shape: (n_devices, n_terminals)
    params: Dict[str, Array]           # {"r": [1000, 2000, ...], ...}

    def evaluate(V, context) -> DeviceStamps:
        # Vectorized evaluation using vmap
        V_terminals = V[self.node_indices]  # Shape: (n_devices, n_terminals)
        return jax.vmap(device_fn)(V_terminals, self.params, context)
```

### DeviceStamps (`devices/base.py`)

The output of device evaluation, used for MNA stamping.

```python
@dataclass
class DeviceStamps:
    currents: Array      # Shape: (n_terminals,) - current into each terminal
    conductances: Array  # Shape: (n_terminals, n_terminals) - dI/dV
    charges: Array       # Optional: for transient analysis
```

### AnalysisContext (`analysis/context.py`)

Runtime context passed to device evaluation.

```python
@dataclass
class AnalysisContext:
    time: float              # Current simulation time
    timestep: float          # dt for transient
    temperature: float       # Circuit temperature (K)
    iteration: int           # Newton-Raphson iteration number
    gmin: float              # GMIN for numerical stability
    integration_coeff: float # Backward Euler coefficient (1/dt)
```

## Device Model Interface

### Pure Function Pattern

Devices are implemented as pure functions for JAX compatibility:

```python
def resistor(
    V: Array,                    # [V_pos, V_neg]
    params: dict,                # {"r": 1000.0, "tc1": 0.0, "tc2": 0.0}
    context: AnalysisContext
) -> DeviceStamps:
    """Temperature-dependent resistor."""
    r = params["r"]
    tc1, tc2 = params.get("tc1", 0.0), params.get("tc2", 0.0)

    # Temperature adjustment
    dT = context.temperature - 300.0
    r_eff = r * (1 + tc1*dT + tc2*dT**2)

    # Current and conductance
    G = 1.0 / r_eff
    V_diff = V[0] - V[1]
    I = G * V_diff

    return DeviceStamps(
        currents=jnp.array([I, -I]),
        conductances=jnp.array([[G, -G], [-G, G]])
    )
```

### Why Pure Functions?

1. **JIT Compilation**: JAX can compile and optimize pure functions
2. **Automatic Differentiation**: Gradients computed correctly
3. **Vectorization**: `vmap` works on pure functions
4. **Parallelization**: GPU execution requires pure functions

## Sparse Matrix Support

For large circuits (>1000 nodes), JAX-SPICE uses sparse matrices:

```python
# Jacobian assembly using scipy.sparse.lil_matrix
def assemble_sparse_jacobian(system, V, context):
    n = len(V)
    J = scipy.sparse.lil_matrix((n, n))

    for group in system.device_groups.values():
        stamps = group.evaluate(V, context)
        for i, device_name in enumerate(group.device_names):
            nodes = group.node_indices[i]
            for row, col in product(range(len(nodes)), repeat=2):
                J[nodes[row], nodes[col]] += stamps.conductances[i, row, col]

    return J.tocsr()  # Convert to CSR for efficient solve

# Linear solve
delta_V = scipy.sparse.linalg.spsolve(J, -residual)
```

### When Sparse is Used

| Circuit Size | Solver | Reason |
|--------------|--------|--------|
| < 1000 nodes | Dense | Lower overhead |
| ≥ 1000 nodes | Sparse | Memory efficiency, faster solve |

The switch is automatic in `dc_operating_point()`.

## OpenVAF Integration

### Compilation Pipeline

```
Verilog-A (.va)
      │
      ▼
OpenVAF Compiler
      │
      ▼
MIR (Mid-level IR)
      │
      ▼
openvaf_jax Translator
      │
      ▼
JAX Function
```

### VerilogADevice Wrapper

```python
class VerilogADevice:
    def __init__(self, compiled_model, params):
        self.eval_fn = openvaf_jax.translate(compiled_model)
        self.params = params
        self.n_internal = compiled_model.n_internal_nodes

    def evaluate(self, V, params, context):
        # Call the JAX-translated function
        outputs = self.eval_fn(V, params)

        # Extract currents and conductances from outputs
        return DeviceStamps(
            currents=outputs.currents,
            conductances=outputs.jacobian
        )
```

## Performance Optimization

### JIT Compilation

```python
@jax.jit
def newton_step(V, system, context):
    residual, J = system.build_jacobian_and_residual(V, context)
    delta_V = jnp.linalg.solve(J, -residual)
    return V + delta_V
```

First call: ~1-5 seconds (compilation)
Subsequent calls: ~1-20 ms

### Vectorized Evaluation

```python
# Without vectorization (slow)
currents = []
for i in range(n_devices):
    I = device_fn(V[nodes[i]], params[i])
    currents.append(I)

# With vectorization (fast)
currents = jax.vmap(device_fn)(V[node_indices], batched_params)
```

Speedup: 10-100x depending on device count

### Batched Parameter Arrays

Pre-computing parameter arrays eliminates Python loops:

```python
# During build_device_groups()
params = {
    "r": jnp.array([d.params["r"] for d in resistors]),
    "tc1": jnp.array([d.params.get("tc1", 0) for d in resistors]),
}
# Now vmap can use these directly
```

## Convergence Strategies

### Source Stepping

Gradually ramps supply voltage from 0 to target:

```
VDD steps: 0.0 → 0.12 → 0.24 → ... → 1.2V

At each step:
1. Solve DC with current VDD
2. Use solution as initial guess for next step
```

Helps with digital circuits that have multiple stable states.

### GMIN Stepping

Adds decreasing conductance to ground at each node:

```
GMIN steps: 1e-3 → 1e-6 → 1e-9 → 1e-12 S

At each step:
1. Add GMIN*V to residual (pulls nodes toward 0)
2. Solve DC
3. Reduce GMIN and repeat
```

Prevents floating nodes and improves conditioning.

## File Reference

| File | LOC | Purpose |
|------|-----|---------|
| `analysis/mna.py` | ~1000 | MNA system, device groups, matrix assembly |
| `analysis/dc.py` | ~800 | DC solvers (Newton-Raphson, source stepping) |
| `analysis/transient.py` | ~600 | Transient solver (Backward Euler) |
| `devices/mosfet_simple.py` | ~300 | Simplified BSIM-like MOSFET |
| `devices/openvaf_device.py` | ~300 | OpenVAF wrapper |
| `netlist/parser.py` | ~400 | VACASK netlist parser |
| `benchmarks/runner.py` | ~500 | Benchmark infrastructure |

## Common Patterns

### Adding Devices to MNA

```python
system = MNASystem()

# Add with node names (string)
system.add_device("R1", Resistor(r=1000), ["vdd", "out"])

# Node 0 is always ground
system.add_device("V1", VoltageSource(dc=5.0), ["vdd", "gnd"])
```

### Running Analysis

```python
# DC
V, info = dc_operating_point(system)
if not info["converged"]:
    V, info = dc_operating_point_source_stepping(system, vdd)

# Transient
times, solutions = transient_analysis_jit(
    system,
    t_stop=1e-6,
    t_step=1e-9,
    icmode='uic'  # Skip DC for digital
)
```

### Extracting Results

```python
# Get node voltage
vout = V[system.get_node_index("out")]

# Get voltage over time
vout_transient = solutions[:, system.get_node_index("out")]
```

## Debugging Entry Points

When investigating issues, start here:

1. **Device evaluation**: Set breakpoint in device's `evaluate()` method
2. **Jacobian assembly**: Check `MNASystem.build_jacobian_and_residual()`
3. **Convergence**: Look at `dc_operating_point()` iteration loop
4. **OpenVAF models**: Verify `openvaf_jax.translate()` output
