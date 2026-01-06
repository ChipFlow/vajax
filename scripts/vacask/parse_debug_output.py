#!/usr/bin/env python3
"""
Parse VACASK debug output for comparison with other simulators.

Usage:
    vacask -v runme_debug.sim 2>&1 | python3 parse_debug_output.py

    Or run the simulation and redirect to a file:
    vacask runme_debug.sim 2>&1 > debug_output.txt
    python3 parse_debug_output.py debug_output.txt
"""

import sys
import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

@dataclass
class NRIteration:
    iteration: int
    residual: float
    residual_node: str
    delta: Optional[float] = None
    delta_node: Optional[str] = None
    converged: bool = False

@dataclass
class TimePoint:
    time: float
    step_size: float
    order: int
    nr_iterations: List[NRIteration] = field(default_factory=list)
    lte_ratio: Optional[float] = None
    suggested_step: Optional[float] = None
    accepted: bool = True
    is_discontinuity: bool = False

@dataclass
class SimulationData:
    timepoints: List[TimePoint] = field(default_factory=list)
    total_iterations: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    integration_method: str = "unknown"

def parse_debug_output(lines: List[str]) -> SimulationData:
    """Parse VACASK debug output into structured data."""
    data = SimulationData()
    current_timepoint = None
    current_time = 0.0
    current_step = 0.0

    # Patterns
    solving_pattern = re.compile(r'Solving at t=([0-9.e+-]+)\.? with hk=([0-9.e+-]+)\.?')
    iteration_pattern = re.compile(r'Iteration (\d+)(, converged)?, worst residual=([0-9.e+-]+) @ (\S+)(?:, worst delta=([0-9.e+-]+) @ (\S+))?')
    converged_pattern = re.compile(r'NR algorithm converged in (\d+) iteration')
    lte_pattern = re.compile(r'Maximal LTE/tol=([0-9.e+-]+) suggests dt=([0-9.e+-]+)\.')
    accepted_pattern = re.compile(r'Point #(\d+) accepted at t=([0-9.e+-]+), dt=([0-9.e+-]+), order=(\d+)')
    rejected_pattern = re.compile(r'Point rejected at t=([0-9.e+-]+)')
    discontinuity_pattern = re.compile(r'Discontinuity reached')
    order_pattern = re.compile(r'Increasing order to (\d+)')
    method_pattern = re.compile(r'tran_method\s*=\s*"?(\w+)"?')

    for line in lines:
        line = line.strip()

        # Check for integration method
        m = method_pattern.search(line)
        if m:
            data.integration_method = m.group(1)

        # Check for "Solving at" line
        m = solving_pattern.search(line)
        if m:
            current_time = float(m.group(1))
            current_step = float(m.group(2))
            current_timepoint = TimePoint(
                time=current_time,
                step_size=current_step,
                order=1  # Will be updated
            )
            continue

        # Check for iteration info
        m = iteration_pattern.search(line)
        if m and current_timepoint is not None:
            iteration = NRIteration(
                iteration=int(m.group(1)),
                converged=m.group(2) is not None,
                residual=float(m.group(3)),
                residual_node=m.group(4),
                delta=float(m.group(5)) if m.group(5) else None,
                delta_node=m.group(6) if m.group(6) else None
            )
            current_timepoint.nr_iterations.append(iteration)
            continue

        # Check for LTE info
        m = lte_pattern.search(line)
        if m and current_timepoint is not None:
            current_timepoint.lte_ratio = float(m.group(1))
            current_timepoint.suggested_step = float(m.group(2))
            continue

        # Check for accepted point
        m = accepted_pattern.search(line)
        if m:
            if current_timepoint is not None:
                current_timepoint.order = int(m.group(4))
                current_timepoint.accepted = True
                data.timepoints.append(current_timepoint)
                data.total_accepted += 1
            current_timepoint = None
            continue

        # Check for rejected point
        m = rejected_pattern.search(line)
        if m:
            if current_timepoint is not None:
                current_timepoint.accepted = False
                data.timepoints.append(current_timepoint)
                data.total_rejected += 1
            current_timepoint = None
            continue

        # Check for discontinuity
        if discontinuity_pattern.search(line) and current_timepoint is not None:
            current_timepoint.is_discontinuity = True

    # Calculate total iterations
    data.total_iterations = sum(len(tp.nr_iterations) for tp in data.timepoints)

    return data

def generate_summary(data: SimulationData) -> str:
    """Generate a human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("VACASK Simulation Summary")
    lines.append("=" * 60)
    lines.append(f"Integration method: {data.integration_method}")
    lines.append(f"Total timepoints: {len(data.timepoints)}")
    lines.append(f"  Accepted: {data.total_accepted}")
    lines.append(f"  Rejected: {data.total_rejected}")
    lines.append(f"Total NR iterations: {data.total_iterations}")

    if data.timepoints:
        avg_iters = data.total_iterations / len(data.timepoints)
        lines.append(f"Average iterations per timepoint: {avg_iters:.2f}")

        # Find max iterations
        max_iters = max(len(tp.nr_iterations) for tp in data.timepoints)
        lines.append(f"Max iterations at a single timepoint: {max_iters}")

        # Find worst residual
        worst_res = max(
            (it.residual for tp in data.timepoints for it in tp.nr_iterations),
            default=0
        )
        lines.append(f"Worst residual encountered: {worst_res:.3e}")

        # LTE statistics
        lte_values = [tp.lte_ratio for tp in data.timepoints if tp.lte_ratio is not None]
        if lte_values:
            lines.append(f"LTE/tol range: {min(lte_values):.3e} - {max(lte_values):.3e}")

    lines.append("=" * 60)
    return "\n".join(lines)

def generate_comparison_data(data: SimulationData) -> Dict:
    """Generate data structure for comparison with other simulators."""
    return {
        "simulator": "VACASK",
        "integration_method": data.integration_method,
        "statistics": {
            "total_timepoints": len(data.timepoints),
            "accepted": data.total_accepted,
            "rejected": data.total_rejected,
            "total_nr_iterations": data.total_iterations
        },
        "timepoints": [
            {
                "time": tp.time,
                "step_size": tp.step_size,
                "order": tp.order,
                "accepted": tp.accepted,
                "is_discontinuity": tp.is_discontinuity,
                "nr_iterations": len(tp.nr_iterations),
                "final_residual": tp.nr_iterations[-1].residual if tp.nr_iterations else None,
                "final_residual_node": tp.nr_iterations[-1].residual_node if tp.nr_iterations else None,
                "lte_ratio": tp.lte_ratio
            }
            for tp in data.timepoints
        ]
    }

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    data = parse_debug_output(lines)

    # Print summary
    print(generate_summary(data))

    # Save detailed data to JSON
    comparison_data = generate_comparison_data(data)
    with open("vacask_simulation_data.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nDetailed data saved to vacask_simulation_data.json")

if __name__ == "__main__":
    main()
