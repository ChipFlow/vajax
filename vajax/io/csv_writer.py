"""Write simulation results to CSV format.

Simple, portable format compatible with spreadsheets and data analysis tools.
"""

import csv
from pathlib import Path
from typing import Any, Union

import numpy as np


def write_csv(
    result: Any,
    output_path: Union[str, Path],
    precision: int = 9,
) -> None:
    """Write simulation results to CSV file.

    Format:
        time,node1,node2,node3,...
        0.0,0.0,1.2,0.5,...
        1e-9,0.1,1.1,0.6,...
        ...

    Args:
        result: Simulation result with times and voltages attributes
        output_path: Path to output file
        precision: Number of decimal places for scientific notation
    """
    output_path = Path(output_path)

    # Extract data
    if hasattr(result, "frequencies"):
        x_name = "frequency"
        x_data = np.asarray(result.frequencies)
    else:
        x_name = "time"
        x_data = np.asarray(result.times)

    voltages = {k: np.asarray(v) for k, v in result.voltages.items()}

    # Sort node names for consistent output
    node_names = sorted(voltages.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        header = [x_name] + node_names
        writer.writerow(header)

        # Data rows
        fmt = f"{{:.{precision}e}}"
        for i in range(len(x_data)):
            row = [fmt.format(x_data[i])]
            for name in node_names:
                row.append(fmt.format(voltages[name][i]))
            writer.writerow(row)


def read_csv(input_path: Union[str, Path]) -> dict:
    """Read simulation results from CSV file.

    Returns dict with:
        - times or frequencies: array of x values
        - voltages: dict of node_name -> array
    """
    input_path = Path(input_path)
    result = {"voltages": {}}

    with open(input_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        x_name = header[0]
        node_names = header[1:]

        x_data = []
        data = {name: [] for name in node_names}

        for row in reader:
            x_data.append(float(row[0]))
            for i, name in enumerate(node_names):
                data[name].append(float(row[i + 1]))

    if x_name == "frequency":
        result["frequencies"] = np.array(x_data)
    else:
        result["times"] = np.array(x_data)

    result["voltages"] = {name: np.array(values) for name, values in data.items()}

    return result
