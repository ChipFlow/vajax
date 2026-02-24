# Debug Tools Reference

This document describes the debugging utilities in `vajax.debug` for troubleshooting OSDI vs JAX discrepancies.

## Quick Start

```python
from vajax.debug import quick_compare, inspect_model

# Compare OSDI vs JAX at a bias point
result = quick_compare(
    va_path="vendor/OpenVAF/integration_tests/PSP102/psp102.va",
    osdi_path="/tmp/osdi_jax_test_cache/psp102.osdi",
    params={'TYPE': 1, 'W': 1e-6, 'L': 1e-7},
    voltages=[0.5, 0.6, 0.0, 0.0],
)
print(result)

# Inspect MIR structure
inspect_model("vendor/OpenVAF/integration_tests/PSP102/psp102.va")

# Graph-based queries (requires networkx)
from vajax.debug import MIRGraph
graph = MIRGraph.from_va_file("model.va", func='eval')
graph.dae_residual('dt')      # Find residual variable
graph.param_to_value('rth')   # Map param to MIR value
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `model_comparison` | Compare OSDI vs JAX outputs (residuals, Jacobians, cache) |
| `mir_inspector` | Inspect MIR data (params, PHI nodes, constants) |
| `mir_graph` | Graph-based MIR queries (requires networkx) |
| `jacobian` | Format-aware Jacobian comparison (OSDI sparse vs JAX dense) |
| `mir_tracer` | Trace value flow through MIR |
| `param_analyzer` | Analyze parameter kinds and OSDI comparison |
| `mir_analysis` | CFG analysis with networkx (optional dependency) |
| `transient_diagnostics` | Runtime transient step analysis (LTE, NR, step acceptance) |

---

## Model Comparison (`model_comparison.py`)

### ModelComparator

Full-featured comparison between OSDI and JAX implementations.

```python
from vajax.debug import ModelComparator

comparator = ModelComparator(
    va_path="path/to/model.va",
    osdi_path="path/to/model.osdi",
    params={'TYPE': 1, 'W': 1e-6, 'L': 1e-7},
    temperature=300.0,
)

# Compare at a single bias point
result = comparator.compare_at_bias([0.5, 0.6, 0.0, 0.0])
print(result)
print(f"Passed: {result.passed}")
print(f"Issues: {result.issues}")

# Print side-by-side residual table
comparator.print_residual_table([0.5, 0.6, 0.0, 0.0])

# Analyze cache for potential issues
cache_analysis = comparator.analyze_cache()
print(cache_analysis)

# Sweep comparison
results = comparator.sweep_comparison(
    base_voltages=[0.5, 0.0, 0.0, 0.0],
    sweep_index=1,  # Sweep Vgs
    sweep_values=[0.0, 0.3, 0.6, 0.9, 1.2],
)
```

### CacheAnalysis

Detects potential issues in JAX cache values:

- **inf/nan detection**: Catches numerical instabilities
- **Large values**: Flags values > 1e10 that might cause overflow
- **Temperature-related**: Finds VT values (0.025-0.030) and their implied temperatures

```python
cache = comparator.analyze_cache()
print(f"Cache size: {cache.size}")
print(f"Non-zero: {cache.nonzero_count}")
print(f"Has inf: {cache.has_inf}")
print(f"Has nan: {cache.has_nan}")

# Check temperature values
for idx, val, implied_t in cache.temperature_related:
    print(f"cache[{idx}] = {val:.6f} implies T = {implied_t:.1f}K")
```

---

## MIR Inspector (`mir_inspector.py`)

### MIRInspector

Examine MIR (Mid-level IR) structure for debugging translation issues.

```python
from vajax.debug import MIRInspector

inspector = MIRInspector("path/to/model.va")

# Overall statistics
inspector.print_mir_stats()

# Parameter summary
inspector.print_param_summary('eval')  # or 'init'

# PHI node analysis
inspector.print_phi_summary('eval')

# Find TYPE parameter (NMOS/PMOS models)
inspector.print_type_param_info()
```

### Finding Specific Values

```python
# Find constants near a value (e.g., P_CELSIUS0 = 273.15)
constants = inspector.find_constants_near(273.15, tolerance=0.01)
for name, value in constants:
    print(f"{name} = {value}")

# Find PHI nodes with zero operand (indicates conditional branch)
zero_phis = inspector.find_phi_nodes_with_value('v3')  # v3 is typically 0.0
for phi in zero_phis[:5]:
    print(f"PHI {phi.result} in {phi.block}")
    for pred, val in phi.operands:
        print(f"  {pred} -> {val}")
```

---

## MIR Graph (`mir_graph.py`)

Graph-based queries for MIR analysis. Requires `networkx`.

### MIRGraph

Build a queryable graph from a VA model:

```python
from vajax.debug import MIRGraph

graph = MIRGraph.from_va_file("model.va", func='eval', include_dae=True)
```

### Value Tracing

```python
# What instruction defines a value?
graph.who_defines('v273116')
# Returns: {'opcode': 'optbarrier', 'block': 'block1458', ...}

# What instructions use a value?
graph.who_uses('v142825')
# Returns: [{'target': 'value:v142827', 'opcode': 'fgt', ...}, ...]

# Trace dependencies backwards
graph.trace_back('v273116', depth=5)

# Trace usage forwards
graph.trace_forward('v142825', depth=5)
```

### Parameter Mapping

```python
# Find MIR value ID for a parameter
graph.param_to_value('rth')  # Returns 'v142825'

# Reverse lookup
graph.value_to_param('v142825')  # Returns 'rth'
```

### DAE System Queries

```python
# Get resist/react value IDs for a node
graph.dae_residual('dt')
# Returns: {'resist': 'v273116', 'react': 'v273117'}
```

### Control Flow

```python
# Find path from entry to a block
graph.path_to_block('block1451')
# Returns: ['block0', 'block4', ..., 'block1450', 'block1451']

# Get PHI nodes in a block
graph.phi_info('block1453')
# Returns: [{'result': 'v252438', 'operands': [...], ...}, ...]

# Get branch condition for a block
graph.branch_condition('block1450')
# Returns: {'condition': 'v142827', 'true_block': 'block1451', 'false_block': 'block1453'}

# Find all blocks that branch on a value
graph.blocks_with_condition('v142827')
# Returns: ['block1450']
```

### Constants

```python
# Check if a value is a constant
is_const, value = graph.is_constant('v3')
# Returns: (True, 0.0)
```

---

## Jacobian Comparison (`jacobian.py`)

Format-aware comparison between OSDI (sparse, column-major) and JAX (dense, row-major).

```python
from vajax.debug import compare_jacobians, print_jacobian_structure

# Compare Jacobians
result = compare_jacobians(
    osdi_jac, jax_jac, n_nodes, jacobian_keys,
    rtol=1e-4, atol=1e-10
)
print(result.report)
print(f"Passed: {result.passed}")
print(f"Max abs diff: {result.max_abs_diff}")
print(f"Mismatched positions: {result.mismatched_positions}")

# Print structure
print_jacobian_structure(jacobian_keys, n_nodes)
```

---

## CLI Tools

### MIR CFG Analyzer

Analyze control flow graph from command line:

```bash
# Find PHI nodes
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --find-phis

# Find branch points
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --branches

# Trace paths to a block
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --target block123

# Analyze specific block with PHIs
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --analyze-block block4654
```

---

## Transient Diagnostics (`transient_diagnostics.py`)

Runtime analysis of transient simulation steps — LTE behaviour, NR convergence,
step acceptance/rejection, and VACASK comparison.

### Parsing Debug Output

```python
from vajax.debug import parse_debug_output, StepRecord

# Parse debug_steps text captured from a transient run
records: list[StepRecord] = parse_debug_output(debug_text)
for r in records[:5]:
    print(f"Step {r.step}: t={r.t_ps}ps dt={r.dt_ps}ps NR={r.nr_iters} accepted={r.accepted}")
```

### Capturing a Full Step Trace

```python
from vajax.debug import capture_step_trace, print_step_summary

# Run a benchmark with debug_steps=True and get parsed results
records, summary = capture_step_trace("ring", use_sparse=True)
print_step_summary(records, summary)
```

### Convergence Sweep

Run a benchmark at multiple `t_stop` values to find where convergence degrades:

```python
from vajax.debug import convergence_sweep

results = convergence_sweep("graetz", [1e-3, 5e-3, 7e-3, 10e-3])
for r in results:
    print(f"t_stop={r.t_stop:.0e}: steps={r.num_steps}, conv={r.convergence_rate:.1%}, "
          f"rejected={r.rejected_steps}")
```

### VACASK Step Comparison

Parse VACASK `tran_debug=1` output for side-by-side comparison:

```python
from vajax.debug import parse_vacask_debug_output

vacask_records = parse_vacask_debug_output(vacask_stdout)
accepted = [r for r in vacask_records if r.status == "accept"]
rejected = [r for r in vacask_records if r.status == "reject"]
print(f"VACASK: {len(accepted)} accepted, {len(rejected)} rejected")
```

### LTE Solver Comparison CLI

Compare per-step LTE between different solver backends:

```bash
# Capture a trace
JAX_PLATFORMS=cpu uv run python scripts/compare_lte_solvers.py \
    --benchmark ring --output /tmp/lte_trace.json

# Compare two traces
uv run python scripts/compare_lte_solvers.py \
    --compare /tmp/trace_a.json /tmp/trace_b.json

# Run both dense and sparse locally and compare
JAX_PLATFORMS=cpu uv run python scripts/compare_lte_solvers.py \
    --benchmark ring --compare-local
```

---

## Transient Debugging Workflow

### 1. Convergence Sweep

Start by checking if convergence degrades at specific simulation durations:

```python
from vajax.debug import convergence_sweep

results = convergence_sweep("graetz", [1e-3, 5e-3, 7e-3, 10e-3])
# Look for t_stop values where rejected_steps spikes or convergence_rate drops
```

### 2. Capture Step Trace

Zoom in on a problematic duration with full per-step data:

```python
from vajax.debug import capture_step_trace, print_step_summary

records, summary = capture_step_trace("graetz", use_sparse=False)
print_step_summary(records, summary)

# Find rejection clusters
rejected = [r for r in records if not r.accepted]
for r in rejected[:10]:
    print(f"  Step {r.step}: t={r.t_ps}ps LTE={r.lte_norm} NR_fail={r.nr_failed}")
```

### 3. VACASK Comparison

Compare step-by-step behaviour with VACASK reference:

```python
from vajax.debug import parse_debug_output, parse_vacask_debug_output

jax_records = parse_debug_output(jax_debug_text)
vacask_records = parse_vacask_debug_output(vacask_debug_text)

# Compare accepted step counts, dt ranges, rejection patterns
```

---

## Debugging Workflow

### 1. Initial Comparison

```python
from vajax.debug import quick_compare

result = quick_compare(va_path, osdi_path, params, voltages)
print(result)

if not result.passed:
    print("Issues found:")
    for issue in result.issues:
        print(f"  - {issue}")
```

### 2. Cache Analysis

If residuals differ, check the cache first:

```python
from vajax.debug import ModelComparator

comparator = ModelComparator(va_path, osdi_path, params)
cache = comparator.analyze_cache()

# Check for problems
if cache.has_inf > 0:
    print(f"WARNING: {cache.has_inf} inf values in cache")
if cache.has_nan > 0:
    print(f"WARNING: {cache.has_nan} nan values in cache")
```

### 3. MIR Inspection

If cache looks OK, inspect MIR structure:

```python
from vajax.debug import MIRInspector

inspector = MIRInspector(va_path)
inspector.print_mir_stats()
inspector.print_phi_summary('eval')

# For NMOS/PMOS models, check TYPE handling
inspector.print_type_param_info()
```

### 4. PHI Node Analysis

If PHI nodes are suspected:

```python
# Find PHIs with zero operand (often indicates branch issue)
zero_phis = inspector.find_phi_nodes_with_value('v3')
print(f"Found {len(zero_phis)} PHIs with zero operand")

# Use CLI for detailed analysis
# uv run scripts/analyze_mir_cfg.py model.va --func eval --analyze-block blockXXX
```

---

## Common Issues

### 1. JAX returns near-zero current

**Symptom**: OSDI returns expected current, JAX returns ~1e-15

**Likely cause**: PHI node resolution in NMOS/PMOS branching

**Debug steps**:
1. Check TYPE parameter is passed correctly
2. Analyze PHI nodes for zero operands
3. Trace control flow with `--analyze-block`

### 2. Jacobian sparsity mismatch

**Symptom**: OSDI has N non-zeros, JAX has M << N

**Likely cause**: Branch not taken, computations skipped

**Debug steps**:
1. Use `print_jacobian_structure()` to see expected pattern
2. Check if missing entries follow a pattern (e.g., all in one row/column)

### 3. Temperature-related errors

**Symptom**: Current off by ~1% at room temperature

**Likely cause**: TNOM vs $temperature handling

**Debug steps**:
1. Check cache for VT values: `cache.temperature_related`
2. Verify expected VT at 300K: 0.02585 V
3. Check for sentinel values (1e21) in init params
