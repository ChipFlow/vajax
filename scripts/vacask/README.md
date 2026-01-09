# VACASK Analysis Tools

Python scripts for analyzing VACASK simulation output.

## Documentation

Reference documentation has been moved to:
- `../../docs/reference/osdi-vacask/` - OSDI/VACASK internals and analysis

## Scripts

### parse_debug_output.py
Parses VACASK debug output to extract NR convergence and timestep data.

**Usage:**
```bash
vacask -options nr_debug=1 -options tran_debug=2 runme.sim 2>&1 | python3 parse_debug_output.py
```

**Output:** `vacask_simulation_data.json`

### extract_comparison_data.py
Extracts data for comparison with other simulators.

### generate_graph.py
Generates visualization graphs from simulation data.

## Prerequisites
- Python 3.10+
- NumPy
- Graphviz (for SVG generation)
