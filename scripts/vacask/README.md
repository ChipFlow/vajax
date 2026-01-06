# VACASK Ring Benchmark Analysis Tools

This directory contains Python scripts for analyzing VACASK simulation output from the ring oscillator benchmark. These tools extract structured data for comparison with other circuit simulators.

## Prerequisites

- Python 3.10+
- NumPy (`pip install numpy`)
- Graphviz (for SVG generation from DOT files)

## Scripts

### 1. parse_debug_output.py

Parses VACASK debug output to extract NR convergence and timestep control data.

**Usage:**
```bash
# Pipe debug output directly
vacask -options nr_debug=1 -options tran_debug=2 runme.sim 2>&1 | python3 parse_debug_output.py

# Or from a saved log file
vacask runme_debug.sim > debug.log 2>&1
python3 parse_debug_output.py debug.log
```

**Output:**
- `vacask_simulation_data.json` - Structured simulation data including:
  - Per-timepoint convergence info (iterations, residuals, LTE)
  - Statistics summary

**Example output:**
```
============================================================
VACASK Simulation Summary
============================================================
Integration method: trap
Total timepoints: 115
  Accepted: 115
  Rejected: 0
Total NR iterations: 367
Average iterations per timepoint: 3.19
Max iterations at a single timepoint: 4
Worst residual encountered: 5.250e-03
LTE/tol range: 3.151e-05 - 5.541e+00
============================================================
```

---

### 2. extract_comparison_data.py

Comprehensive extraction tool that combines debug log parsing with waveform data from raw files.

**Usage:**
```bash
# With both debug log and raw file
python3 extract_comparison_data.py debug_output.log tran_debug.raw

# Debug log only (waveforms optional)
python3 extract_comparison_data.py debug_output.log
```

**Output:**
- `<basename>_comparison.json` - Full comparison data:
  ```json
  {
    "simulator": "VACASK",
    "version": "0.3.1",
    "metadata": {
      "integration_method": "trap",
      "total_iterations": 367,
      "accepted_points": 115
    },
    "convergence_data": [...],
    "waveform_times": [...],
    "waveforms": {...}
  }
  ```

- `<basename>_convergence.csv` - Per-timepoint data:
  ```csv
  time,step_size,order,nr_iterations,final_residual,lte_ratio
  2.5e-11,2.5e-11,1,2,6.28e-13,
  7.5e-11,5e-11,2,2,9.16e-14,0.000129
  ```

**Data Extracted:**
| Field | Description |
|-------|-------------|
| `time` | Simulation time (seconds) |
| `step_size` | Timestep used (seconds) |
| `order` | Integration order (1=Euler/BDF1, 2=Trap/BDF2) |
| `nr_iterations` | Newton-Raphson iterations to converge |
| `final_residual` | Residual at convergence |
| `lte_ratio` | Local truncation error / tolerance |

---

### 3. generate_graph.py

Generates DOT and SVG graph visualizations of the circuit structure, Jacobian sparsity, and evaluation order.

**Usage:**
```bash
python3 generate_graph.py
```

**Output:**
- `graph_circuit.dot` / `graph_circuit.svg` - Circuit topology showing:
  - Instance nodes (boxes)
  - Circuit nodes (circles)
  - Terminal connections

- `graph_jacobian.dot` / `graph_jacobian.svg` - Jacobian sparsity pattern:
  - Nodes as unknowns
  - Edges showing non-zero Jacobian entries

- `graph_eval_order.dot` / `graph_eval_order.svg` - Evaluation dependency order:
  - Device evaluation sequence
  - Data dependencies between evaluations

- `graph_stamping.dot` / `graph_stamping.svg` - Matrix stamping pattern:
  - Which instances contribute to which matrix entries

**Requires:** Graphviz installed (`brew install graphviz` on macOS)

---

## Workflow Example

Complete analysis workflow for the ring benchmark:

```bash
cd benchmark/ring/vacask

# 1. Run simulation with debug output
../../build/simulator/vacask runme_debug.sim > debug_output.log 2>&1

# 2. Extract comparison data
python3 extract_comparison_data.py debug_output.log tran_debug.raw

# 3. Generate circuit graphs
python3 generate_graph.py

# 4. View results
cat debug_output_convergence.csv
open graph_circuit.svg
```

---

## Debug Options Reference

Enable these in your netlist to generate debug output:

```
control
  // NR solver debug (0=off, 1=summary, 2=detailed)
  options nr_debug=1

  // Transient debug (0=off, 1=basic, 2=detailed)
  options tran_debug=2

  // Integration method: "trap", "gear2", "euler", "bdf"
  options tran_method="trap"

  analysis tran step=0.1n stop=5n
endc
```

---

## Output File Summary

| File | Format | Content |
|------|--------|---------|
| `*_comparison.json` | JSON | Full structured comparison data |
| `*_convergence.csv` | CSV | Per-timepoint convergence metrics |
| `vacask_simulation_data.json` | JSON | Parsed debug output |
| `graph_*.dot` | DOT | Graphviz source files |
| `graph_*.svg` | SVG | Rendered graph images |
| `circuit_structure.json` | JSON | Circuit topology for comparison |
| `model_parameters.txt` | Text | Full PSP103 model parameter dump |
| `instance_parameters.txt` | Text | Per-instance parameter values |

---

## Comparison with Other Simulators

The extracted data is formatted for comparison with:
- **jax-spice**: Use JSON/CSV files directly
- **ngspice**: Compare waveforms from raw files
- **Xyce**: Compare convergence behavior

Key metrics for comparison:
1. **NR iterations per timepoint** - Algorithm efficiency
2. **Final residual** - Solution accuracy
3. **LTE ratio** - Timestep control behavior
4. **Step size sequence** - Adaptive stepping comparison
