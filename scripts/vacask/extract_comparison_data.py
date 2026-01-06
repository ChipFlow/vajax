#!/usr/bin/env python3
"""
Extract VACASK simulation data for comparison with other simulators.

This script:
1. Parses debug output for NR convergence and timestep info
2. Reads raw file for waveform data
3. Generates JSON/CSV files suitable for comparison

Usage:
    # Run with debug enabled, capture output
    vacask -options nr_debug=1 -options tran_debug=2 runme.sim > debug.log 2>&1

    # Extract comparison data
    python3 extract_comparison_data.py debug.log tran_debug.raw
"""

import sys
import re
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Import rawfile reader if available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
    from rawfile import rawread
    HAS_RAWFILE = True
except ImportError:
    HAS_RAWFILE = False

@dataclass
class TimePointData:
    """Data for a single timepoint."""
    time: float
    step_size: float
    order: int
    nr_iterations: int
    final_residual: float
    lte_ratio: Optional[float] = None
    solution: Dict[str, float] = field(default_factory=dict)

def parse_debug_log(log_path: str) -> Tuple[List[TimePointData], Dict]:
    """Parse VACASK debug log for convergence data."""
    timepoints = []
    metadata = {
        "integration_method": "trap",
        "total_iterations": 0,
        "accepted_points": 0,
        "rejected_points": 0
    }

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Patterns
    solving_re = re.compile(r'Solving at t=([0-9.e+-]+) with hk=([0-9.e+-]+)')
    iteration_re = re.compile(r'Iteration (\d+)(, converged)?, worst residual=([0-9.e+-]+)')
    lte_re = re.compile(r'Maximal LTE/tol=([0-9.e+-]+)')
    accepted_re = re.compile(r'Point #(\d+) accepted at t=([0-9.e+-]+), dt=([0-9.e+-]+), order=(\d+)')
    method_re = re.compile(r'tran_method\s*=\s*"?(\w+)"?')

    current_time = 0.0
    current_step = 0.0
    current_iterations = 0
    current_residual = 0.0
    current_lte = None

    for line in lines:
        # Check method
        m = method_re.search(line)
        if m:
            metadata["integration_method"] = m.group(1)

        # Check solving start
        m = solving_re.search(line)
        if m:
            current_time = float(m.group(1))
            current_step = float(m.group(2))
            current_iterations = 0
            current_residual = 0.0
            current_lte = None
            continue

        # Check iteration
        m = iteration_re.search(line)
        if m:
            current_iterations = int(m.group(1))
            current_residual = float(m.group(3))
            continue

        # Check LTE
        m = lte_re.search(line)
        if m:
            current_lte = float(m.group(1))
            continue

        # Check accepted point
        m = accepted_re.search(line)
        if m:
            tp = TimePointData(
                time=float(m.group(2)),
                step_size=float(m.group(3)),
                order=int(m.group(4)),
                nr_iterations=current_iterations,
                final_residual=current_residual,
                lte_ratio=current_lte
            )
            timepoints.append(tp)
            metadata["accepted_points"] += 1
            metadata["total_iterations"] += current_iterations

    return timepoints, metadata

def read_waveforms(raw_path: str) -> Dict[str, np.ndarray]:
    """Read waveform data from raw file."""
    if not HAS_RAWFILE:
        print("Warning: rawfile module not available, skipping waveform data")
        return {}

    try:
        raw = rawread(raw_path)
        data = raw.get()  # Returns dict-like object
        # Convert to regular dict
        result = {}
        for k in data:
            result[k] = np.array(data[k])
        return result
    except Exception as e:
        print(f"Warning: Failed to read raw file: {e}")
        return {}

def generate_comparison_json(timepoints: List[TimePointData], metadata: Dict,
                             waveforms: Dict[str, np.ndarray], output_path: str):
    """Generate JSON file for comparison."""
    result = {
        "simulator": "VACASK",
        "version": "0.3.1",
        "metadata": metadata,
        "convergence_data": [
            {
                "time": tp.time,
                "step_size": tp.step_size,
                "order": tp.order,
                "nr_iterations": tp.nr_iterations,
                "final_residual": tp.final_residual,
                "lte_ratio": tp.lte_ratio
            }
            for tp in timepoints
        ]
    }

    # Add waveform sample points if available
    if waveforms and "time" in waveforms.keys():
        result["waveform_times"] = waveforms["time"].tolist()
        result["waveforms"] = {
            k: v.tolist() for k, v in waveforms.items() if k != "time"
        }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Comparison data saved to {output_path}")

def generate_summary_csv(timepoints: List[TimePointData], output_path: str):
    """Generate CSV file with timestep/convergence summary."""
    with open(output_path, 'w') as f:
        f.write("time,step_size,order,nr_iterations,final_residual,lte_ratio\n")
        for tp in timepoints:
            f.write(f"{tp.time},{tp.step_size},{tp.order},{tp.nr_iterations},"
                    f"{tp.final_residual},{tp.lte_ratio if tp.lte_ratio else ''}\n")
    print(f"CSV summary saved to {output_path}")

def print_summary(timepoints: List[TimePointData], metadata: Dict):
    """Print human-readable summary."""
    print("=" * 60)
    print("VACASK Simulation Comparison Data")
    print("=" * 60)
    print(f"Integration method: {metadata['integration_method']}")
    print(f"Total accepted timepoints: {metadata['accepted_points']}")
    print(f"Total NR iterations: {metadata['total_iterations']}")
    if timepoints:
        avg_iters = metadata['total_iterations'] / len(timepoints)
        print(f"Average iterations per point: {avg_iters:.2f}")
        max_iters = max(tp.nr_iterations for tp in timepoints)
        print(f"Max iterations at single point: {max_iters}")
        worst_res = max(tp.final_residual for tp in timepoints)
        print(f"Worst final residual: {worst_res:.3e}")
    print("=" * 60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_comparison_data.py <debug.log> [raw_file.raw]")
        print("\nTo generate debug output:")
        print("  vacask -options nr_debug=1 -options tran_debug=2 netlist.sim > debug.log 2>&1")
        sys.exit(1)

    log_path = sys.argv[1]
    raw_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Parse debug log
    print(f"Parsing debug log: {log_path}")
    timepoints, metadata = parse_debug_log(log_path)

    # Read waveforms if available
    waveforms = {}
    if raw_path:
        print(f"Reading waveforms: {raw_path}")
        waveforms = read_waveforms(raw_path)

    # Print summary
    print_summary(timepoints, metadata)

    # Generate output files
    base_name = Path(log_path).stem
    generate_comparison_json(timepoints, metadata, waveforms, f"{base_name}_comparison.json")
    generate_summary_csv(timepoints, f"{base_name}_convergence.csv")

if __name__ == "__main__":
    main()
