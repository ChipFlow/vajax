"""Write ngspice-compatible raw files.

The raw file format is compatible with:
- ngspice's rawread command
- gwave waveform viewer
- Other SPICE raw file tools

Supports both ASCII and binary formats.
"""

import struct
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

from jax import Array


def write_rawfile(
    result: Any,
    output_path: Union[str, Path],
    binary: bool = True,
) -> None:
    """Write simulation results to ngspice-compatible raw file.

    Args:
        result: Simulation result with times and voltages attributes
        output_path: Path to output file
        binary: If True, write binary format; else ASCII format
    """
    output_path = Path(output_path)

    # Extract data from result
    times = np.asarray(result.times)
    voltages = {k: np.asarray(v) for k, v in result.voltages.items()}

    # Determine analysis type
    if hasattr(result, 'frequencies'):
        analysis_type = 'AC Analysis'
        x_name = 'frequency'
        x_data = np.asarray(result.frequencies)
    else:
        analysis_type = 'Transient Analysis'
        x_name = 'time'
        x_data = times

    n_points = len(x_data)
    n_vars = 1 + len(voltages)  # x variable + all node voltages

    # Build variable list
    variables = [(x_name, 'time' if x_name == 'time' else 'frequency')]
    for name in voltages.keys():
        variables.append((name, 'voltage'))

    if binary:
        _write_binary_rawfile(
            output_path, analysis_type, variables,
            x_data, voltages, n_points
        )
    else:
        _write_ascii_rawfile(
            output_path, analysis_type, variables,
            x_data, voltages, n_points
        )


def _write_ascii_rawfile(
    output_path: Path,
    analysis_type: str,
    variables: list,
    x_data: np.ndarray,
    voltages: Dict[str, np.ndarray],
    n_points: int,
) -> None:
    """Write ASCII format raw file."""
    with open(output_path, 'w') as f:
        # Header
        f.write(f"Title: JAX-SPICE Simulation\n")
        f.write(f"Date: \n")
        f.write(f"Plotname: {analysis_type}\n")
        f.write(f"Flags: real\n")
        f.write(f"No. Variables: {len(variables)}\n")
        f.write(f"No. Points: {n_points}\n")

        # Variable definitions
        f.write("Variables:\n")
        for i, (name, var_type) in enumerate(variables):
            f.write(f"\t{i}\t{name}\t{var_type}\n")

        # Data values
        f.write("Values:\n")
        for i in range(n_points):
            f.write(f" {i}\t{x_data[i]:.15e}\n")
            for name, _ in variables[1:]:  # Skip x variable
                f.write(f"\t{voltages[name][i]:.15e}\n")


def _write_binary_rawfile(
    output_path: Path,
    analysis_type: str,
    variables: list,
    x_data: np.ndarray,
    voltages: Dict[str, np.ndarray],
    n_points: int,
) -> None:
    """Write binary format raw file.

    Binary format is more compact and faster to read/write.
    Uses double precision (8 bytes) for all values.
    """
    with open(output_path, 'wb') as f:
        # Header (ASCII)
        header = []
        header.append("Title: JAX-SPICE Simulation")
        header.append("Date: ")
        header.append(f"Plotname: {analysis_type}")
        header.append("Flags: real")
        header.append(f"No. Variables: {len(variables)}")
        header.append(f"No. Points: {n_points}")
        header.append("Variables:")

        for i, (name, var_type) in enumerate(variables):
            header.append(f"\t{i}\t{name}\t{var_type}")

        header.append("Binary:")

        # Write header as ASCII with newlines
        header_text = '\n'.join(header) + '\n'
        f.write(header_text.encode('ascii'))

        # Binary data: each point has all variables in sequence
        # Format: x, v1, v2, v3, ... for each time point
        for i in range(n_points):
            # Write x value (time or frequency)
            f.write(struct.pack('d', float(x_data[i])))
            # Write all voltage values
            for name, _ in variables[1:]:
                f.write(struct.pack('d', float(voltages[name][i])))


def read_rawfile_header(path: Union[str, Path]) -> Dict[str, Any]:
    """Read header information from a raw file.

    Returns dict with keys:
    - title: simulation title
    - plotname: analysis type
    - n_vars: number of variables
    - n_points: number of data points
    - variables: list of (name, type) tuples
    - binary: True if binary format
    """
    path = Path(path)
    info: Dict[str, Any] = {
        'variables': [],
        'binary': False,
    }

    with open(path, 'rb') as f:
        # Read header (ASCII portion)
        in_variables = False
        for line in f:
            try:
                text = line.decode('ascii').strip()
            except UnicodeDecodeError:
                # Hit binary data
                break

            if text.lower().startswith('title:'):
                info['title'] = text[6:].strip()
            elif text.lower().startswith('plotname:'):
                info['plotname'] = text[9:].strip()
            elif text.lower().startswith('no. variables:'):
                info['n_vars'] = int(text[14:].strip())
            elif text.lower().startswith('no. points:'):
                info['n_points'] = int(text[11:].strip())
            elif text.lower() == 'variables:':
                in_variables = True
            elif text.lower() == 'binary:':
                info['binary'] = True
                break
            elif text.lower() == 'values:':
                info['binary'] = False
                break
            elif in_variables and text.startswith('\t'):
                parts = text.split()
                if len(parts) >= 3:
                    info['variables'].append((parts[1], parts[2]))

    return info
