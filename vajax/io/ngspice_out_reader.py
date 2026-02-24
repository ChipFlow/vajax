"""Parser for ngspice .out reference files.

The .out format contains simulation results with headers and tab-separated data.

Format structure:
1. Header info (circuit name, temperature, initial solution)
2. "No. of Data Rows : N" line
3. Circuit title as separator
4. Column header line (Index, time, signals...)
5. Separator line of dashes
6. Tab-separated data rows
7. May repeat headers for long outputs
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_ngspice_out(
    filepath: Path,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Read ngspice .out reference file.

    Args:
        filepath: Path to .out file

    Returns:
        Tuple of (column_names, data_dict) where data_dict maps column name to array
    """
    content = filepath.read_text()
    lines = content.split("\n")

    columns: List[str] = []
    data_rows: List[List[float]] = []
    in_data = False
    header_seen = False

    for line in lines:
        line = line.rstrip()

        # Skip empty lines
        if not line:
            continue

        # Skip separator lines (all dashes)
        if line.startswith("---"):
            in_data = True
            continue

        # Detect column header line (starts with Index)
        if line.strip().startswith("Index"):
            # Parse column names
            parts = line.split()
            if not header_seen:
                columns = parts
                header_seen = True
            in_data = False
            continue

        # Parse data row if we're in data section
        if in_data and header_seen:
            # Skip non-numeric lines
            parts = line.split()
            if not parts:
                continue

            try:
                # Try to parse as numbers
                row = [float(p) for p in parts]
                data_rows.append(row)
            except ValueError:
                # Not a data line, end of data section
                in_data = False
                continue

    # Convert to numpy arrays
    if not data_rows or not columns:
        return [], {}

    data_array = np.array(data_rows)
    data_dict = {}

    for i, col in enumerate(columns):
        if i < data_array.shape[1]:
            data_dict[col.lower()] = data_array[:, i]

    return columns, data_dict


def read_ngspice_standard(
    filepath: Path,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Read ngspice .standard reference file.

    The .standard format is simpler: space-separated with header row.

    Args:
        filepath: Path to .standard file

    Returns:
        Tuple of (column_names, data_dict) where data_dict maps column name to array
    """
    content = filepath.read_text()
    lines = content.strip().split("\n")

    if not lines:
        return [], {}

    # First line is header
    columns = lines[0].split()

    # Rest are data rows
    data_rows = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            try:
                row = [float(p) for p in parts]
                data_rows.append(row)
            except ValueError:
                continue

    # Convert to numpy arrays
    if not data_rows:
        return columns, {}

    data_array = np.array(data_rows)
    data_dict = {}

    for i, col in enumerate(columns):
        if i < data_array.shape[1]:
            data_dict[col.lower()] = data_array[:, i]

    return columns, data_dict


def read_reference_file(
    filepath: Path,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Read ngspice reference file (auto-detect format).

    Args:
        filepath: Path to .out or .standard file

    Returns:
        Tuple of (column_names, data_dict)
    """
    if filepath.suffix == ".standard":
        return read_ngspice_standard(filepath)
    else:
        return read_ngspice_out(filepath)


def get_time_and_signals(
    data: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Extract time array and signal arrays from parsed data.

    Args:
        data: Dict from read_reference_file

    Returns:
        Tuple of (time_array, signals_dict)
    """
    time = data.get("time")
    signals = {k: v for k, v in data.items() if k not in ("index", "time")}
    return time, signals
