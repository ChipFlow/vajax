# VACASK .sim File Format

This document describes the VACASK simulation file format and how to adapt it for use with jax-spice using openvaf_jax models instead of OSDI.

## File Structure

VACASK .sim files are netlist files with a custom syntax defined in `lib/dflparser.y` (a Bison grammar). Key elements:

### 1. Load Statements

```
load "psp103v4.osdi"
load "spice/resistor.osdi"
```

These load OSDI compiled libraries. For jax-spice, we need to:
- Replace with openvaf_jax compilation of the corresponding .va files
- Map the module name (e.g., `psp103va`) to the JAX function

### 2. Model Definitions

```
model psp103n psp103va (
    type=1
    vfbo=-1.1
    toxo=1.5e-9
    ...
)
```

- First identifier: model name (user-defined)
- Second identifier: module name from OSDI library
- Parameters in parentheses: model parameters

### 3. Subcircuit Definitions

```
subckt nmos (d g s b)
parameters w=1u l=0.2u
  m (d g s b) psp103n w=w l=l
ends

subckt and(out in1 in2)
  mp1 (outx in1 vdd vdd) pmos w=1u l=0.2u
  mn1 (outx in1 int vss) nmos w=0.5u l=0.2u
  ...
ends
```

### 4. Instance Statements

```
v1 (1 0) vsource dc=1
r1 (1 0) resistor r=2k
m (d g s b) psp103n w=w l=l
```

Format: `name (nodes) model_name param=value ...`

### 5. Control Block

```
control
  options nr_convtol=1
  save v(p0) v(p1)
  elaborate circuit("and_test")
  analysis tranand tran stop=0.8n step=10p
  postprocess(PYTHON, "runme.py")
endc
```

## Key Test Cases

### Simple Tests (vendor/VACASK/test/)

| File | Description | Models Used |
|------|-------------|-------------|
| `test_resistor.sim` | Basic resistor | resistor.osdi |
| `test_diode.sim` | Diode I-V sweep | diode.osdi, resistor.osdi |
| `test_capacitor.sim` | Capacitor | capacitor.osdi |
| `test_op.sim` | Operating point | Various |
| `test_inverter.sim` | CMOS inverter | MOSFET model |

### Benchmark Tests (vendor/VACASK/benchmark/)

| Directory | Description | Key Files |
|-----------|-------------|-----------|
| `c6288/vacask/` | 16x16 multiplier | runme.sim, models.inc, multiplier.inc |
| `ring/` | Ring oscillator | Various |
| `graetz/` | Graetz rectifier | Various |

### C6288 AND Gate Test

The `and_test` subcircuit in `benchmark/c6288/vacask/runme.sim` is our target:

```
subckt and(out in1 in2)
  mp2 (outx in2 vdd vdd) pmos w=1u l=0.2u
  mp1 (outx in1 vdd vdd) pmos w=1u l=0.2u
  mn1 (outx in1 int vss) nmos w=0.5u l=0.2u  // <-- Floating 'int' node!
  mn2 (int  in2 vss vss) nmos w=0.5u l=0.2u
  mp3 (out outx vdd vdd) pmos w=1u l=0.2u
  mn3 (out outx vss vss) nmos w=0.5u l=0.2u
ends
```

The `int` node between mn1 and mn2 is the floating node that causes convergence issues with autodiff Jacobians.

## Adapting for jax-spice

### Step 1: Parse the .sim file

Key elements to extract:
1. `load` statements → Compile .va files with openvaf_jax
2. `model` statements → Create model parameter dictionaries
3. `subckt` definitions → Build subcircuit templates
4. Instance statements → Create circuit instances
5. `elaborate` → Build the flattened circuit
6. `analysis` → Run DC/transient analysis

### Step 2: Replace OSDI with openvaf_jax

Instead of:
```python
# VACASK: Load OSDI library
osdi_lib = load_osdi("psp103v4.osdi")
```

Use:
```python
# jax-spice: Compile to JAX
import openvaf_py
import openvaf_jax

modules = openvaf_py.compile_va("psp103v4.va")
translator = openvaf_jax.OpenVAFToJAX(modules[0])
psp103_eval = translator.translate()
```

### Step 3: Map model names to JAX functions

```python
model_registry = {
    'psp103va': psp103_eval,
    'resistor': resistor_eval,
    'diode': diode_eval,
}
```

### Step 4: Build circuit with analytical Jacobians

Use the JAX functions' returned Jacobians instead of autodiff:

```python
def evaluate_device(model_fn, voltages, params):
    inputs = build_inputs(voltages, params)
    residuals, jacobian = model_fn(inputs)  # Analytical Jacobian!
    return residuals, jacobian
```

## jax-spice Parser Compatibility

The existing jax-spice parser (`jax_spice/netlist/parser.py`) is a Python recursive descent parser
that handles the core VACASK netlist format. It successfully parses **40 out of 46 VACASK-format test files** (87%).

### Supported Features

- `load` statements
- `include` statements (with recursive file loading)
- `ground` and `global` declarations
- `model` definitions with parameters
- `subckt`/`ends` definitions
- Instance statements with terminals and parameters
- `control`/`endc` blocks (skipped, not parsed)
- `embed` blocks (skipped)
- Multi-line parameter lists in parentheses
- Comments (`//`)

### Known Limitations

The following VACASK features are not yet supported:

1. **Titles with parentheses** - A title like "SPICE JFET (verilog-A)" confuses the parser
2. **Conditional blocks (`@if/@endif`)** - Used in `test_cblock.sim`, `test_cblocksweep.sim`
3. **Vector parameters (`vec=[...]`)** - Used in `test_sweepvec.sim`
4. **Control block content** - Currently skipped; analysis commands not extracted

### Working Test Cases

All key test cases parse successfully:
- `test_resistor.sim` - Basic resistor
- `test_diode.sim` - Diode I-V sweep
- `test_capacitor.sim` - Capacitor
- `test_inverter.sim` - CMOS inverter
- `benchmark/c6288/vacask/runme.sim` - 16x16 multiplier (includes models.inc, multiplier.inc)
- `benchmark/ring/vacask/runme.sim` - Ring oscillator
- `benchmark/mul/vacask/runme.sim` - Multiplier benchmark

## VACASK Bison Grammar Reference

The full VACASK parser (`lib/dflparser.y`) uses:
- Bison 3.3+ with C++ skeleton
- Custom scanner (`lib/dflscanner.l`)
- Token types: IDENTIFIER, INTEGER, FLOAT, STRING
- Expression evaluation via RPN (Reverse Polish Notation)

Key grammar rules:
- `load`: `LOAD STRING NEWLINE`
- `model_def`: `MODEL IDENTIFIER IDENTIFIER opt_parameter_list NEWLINE`
- `subcircuit_def`: `SUBCKT IDENTIFIER terminal_def ... ENDS`
- `instance`: `IDENTIFIER terminal_def IDENTIFIER opt_parameter_list NEWLINE`

## File Locations

| Purpose | Path |
|---------|------|
| Parser grammar | `vendor/VACASK/lib/dflparser.y` |
| Scanner | `vendor/VACASK/lib/dflscanner.l` |
| Test cases | `vendor/VACASK/test/*.sim` |
| Benchmarks | `vendor/VACASK/benchmark/*/vacask/runme.sim` |
| PSP103 model | `vendor/VACASK/benchmark/c6288/vacask/models.inc` |
| Gate definitions | `vendor/VACASK/benchmark/c6288/vacask/multiplier.inc` |
