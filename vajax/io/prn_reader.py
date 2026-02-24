"""PRN file reader for Xyce simulation output.

Xyce outputs simulation results in PRN (print) format:
- Whitespace-separated columns
- First row contains column headers
- First column is typically INDEX or TIME
- Last line may contain "End of Xyce(TM) Simulation"
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array


def read_prn(file_path: Path | str) -> Tuple[List[str], Dict[str, Array]]:
    """Read a Xyce PRN file into JAX arrays.

    Args:
        file_path: Path to the PRN file

    Returns:
        Tuple of (column_names, data_dict) where:
        - column_names: List of column header names
        - data_dict: Dict mapping column name to JAX array of values

    Example:
        >>> names, data = read_prn("output.prn")
        >>> print(names)
        ['INDEX', 'TIME', 'V(1)', 'I(VMON)']
        >>> print(data['V(1)'])
        Array([0.0, 0.1, 0.2, ...])
    """
    file_path = Path(file_path)

    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return [], {}

    # Parse header line
    header_line = lines[0].strip()
    # Column names may contain special chars like V(1), I(VMON)
    columns = header_line.split()

    # Parse data lines, skipping header and any footer
    data_rows: List[List[float]] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # Skip Xyce footer
        if line.startswith("End of"):
            continue
        # Skip comment lines
        if line.startswith("*") or line.startswith("#"):
            continue

        try:
            values = [float(v) for v in line.split()]
            if len(values) == len(columns):
                data_rows.append(values)
        except ValueError:
            # Skip lines that can't be parsed as floats
            continue

    if not data_rows:
        return columns, {col: jnp.array([]) for col in columns}

    # Convert to column-oriented dict
    data_dict: Dict[str, Array] = {}
    for i, col in enumerate(columns):
        col_data = [row[i] for row in data_rows]
        data_dict[col] = jnp.array(col_data)

    return columns, data_dict


def normalize_column_name(name: str) -> str:
    """Normalize column name for comparison.

    Removes parentheses and special characters, converts to lowercase.

    Args:
        name: Original column name like "V(3)" or "I(VMON)"

    Returns:
        Normalized name like "v3" or "ivmon"
    """
    # Remove non-alphanumeric except underscore
    normalized = re.sub(r"[^a-zA-Z0-9_]", "", name)
    return normalized.lower()


def get_column(data: Dict[str, Array], name: str) -> Optional[Array]:
    """Get column by name with fuzzy matching.

    Tries exact match first, then normalized match.

    Args:
        data: Dict from read_prn
        name: Column name to find

    Returns:
        Array if found, None otherwise
    """
    # Exact match
    if name in data:
        return data[name]

    # Normalized match
    name_norm = normalize_column_name(name)
    for col, arr in data.items():
        if normalize_column_name(col) == name_norm:
            return arr

    return None
