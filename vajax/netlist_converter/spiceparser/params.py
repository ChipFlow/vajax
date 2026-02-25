# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parameter handling utilities for SPICE netlist processing.

This module provides functions for parsing, formatting, and transforming
SPICE parameters, including SI prefix handling and expression processing.

Extracted from netlist_converter/m_params.py to provide shared parameter handling.
"""

import re

from .elements import (
    MERGE_VECTOR_INSTANCE_PARAMS,
    REMOVE_INSTANCE_PARAMS,
)

# SI prefix pattern for Ngspice-style prefixes
# meg, g, t, mil are Ngspice-specific (case-insensitive)
_SI_PREFIX_PATTERN = re.compile(
    r"\b(\d+\.?\d*|\.\d+)(meg|g|t|mil)\b",
    re.IGNORECASE,
)

# Pattern to match 'temper' variable (replaced with $temp in VACASK)
_TEMPER_PATTERN = re.compile(r"\btemper\b")

# Pattern to match gauss/agauss function calls
# Matches: gauss(mean, sigma) or gauss(mean, sigma, n) or agauss(...)
# Examples: gauss(1k, 0.1, 3), agauss(0, 0.05, 3), GAUSS(vth0, sigma)
_GAUSS_PATTERN = re.compile(
    r"\b(a?gauss)\s*\(\s*"  # function name with open paren
    r"([^,]+)"  # mean/nominal value (first arg)
    r"\s*,\s*"  # comma separator
    r"([^,)]+)"  # std deviation (second arg)
    r"(?:\s*,\s*([^)]+))?"  # optional num_sigma (third arg)
    r"\s*\)",  # close paren
    re.IGNORECASE,
)


# SI prefix conversion map for VACASK output
_SI_OUTPUT_MAP = {
    "meg": "M",
    "g": "G",
    "t": "T",
}

# SI prefixes for numeric output formatting (value, suffix)
# Ordered from largest to smallest for proper matching
_SI_OUTPUT_PREFIXES = [
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1e-3, "m"),
    (1e-6, "u"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
]


def _si_replace_worker(match: re.Match) -> str:
    """Convert SI prefix based on regex match.

    xmeg -> xM
    xg -> xG
    xt -> xT
    xmil -> (x*25.4e-6)
    """
    num = match.group(1)
    prefix = match.group(2).lower()
    if prefix == "mil":
        return f"({num}*25.4e-6)"
    return num + _SI_OUTPUT_MAP[prefix]


def convert_si_prefixes(expr: str) -> str:
    """Convert SI prefixes to VACASK syntax.

    Converts Ngspice SI prefixes (meg, g, t, mil) to VACASK format.

    Args:
        expr: Expression string potentially containing SI prefixes

    Returns:
        Expression with converted SI prefixes
    """
    return _SI_PREFIX_PATTERN.sub(_si_replace_worker, expr)


def contains_gauss_function(expr: str) -> bool:
    """Check if expression contains a gauss or agauss function call.

    Used to detect Monte Carlo variation expressions that should be
    preserved as-is in output (pass-through for VACASK).

    Args:
        expr: Expression string to check

    Returns:
        True if expression contains gauss() or agauss() function call
    """
    return bool(_GAUSS_PATTERN.search(expr))


def extract_gauss_calls(expr: str) -> list[dict]:
    """Extract all gauss/agauss function calls from an expression.

    Parses gauss/agauss calls and returns their components for analysis
    or transformation. The raw expression is preserved for pass-through.

    Args:
        expr: Expression string potentially containing gauss calls

    Returns:
        List of dicts with keys: function, mean, std_dev, num_sigma, raw
    """
    calls = []
    for match in _GAUSS_PATTERN.finditer(expr):
        func_name = match.group(1).lower()
        mean = match.group(2).strip()
        std_dev = match.group(3).strip()
        num_sigma_str = match.group(4)
        num_sigma = float(num_sigma_str.strip()) if num_sigma_str else None

        calls.append(
            {
                "function": func_name,
                "mean": mean,
                "std_dev": std_dev,
                "num_sigma": num_sigma,
                "raw": match.group(0),
            }
        )
    return calls


def format_numeric_si(value: float) -> str:
    """Format a numeric value using SI prefixes for readability.

    Converts values like 1000.0 to "1k", 1e-12 to "1p", etc.
    Values between 0.001 and 999 are kept as-is for readability.

    Args:
        value: Numeric value to format

    Returns:
        String with SI prefix (e.g., "1k", "10p", "2.5M")
    """
    if value == 0:
        return "0"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    # Values in comfortable range (0.001 to 999) - keep as-is
    if 0.001 <= abs_value < 1000:
        if abs_value == int(abs_value):
            return f"{sign}{int(abs_value)}"
        return f"{sign}{abs_value:.6g}"

    # Find appropriate SI prefix for large/small values
    for scale, suffix in _SI_OUTPUT_PREFIXES:
        if abs_value >= scale * 0.9999:  # Allow small floating-point tolerance
            scaled = abs_value / scale
            # Format without unnecessary trailing zeros
            if scaled == int(scaled):
                return f"{sign}{int(scaled)}{suffix}"
            else:
                # Limit to reasonable precision
                formatted = f"{scaled:.6g}"
                return f"{sign}{formatted}{suffix}"

    # Very small values (< 1 femto) - use scientific notation
    return f"{value:.6g}"


def format_value(value_str: str) -> str:
    """Format a parameter value for VACASK output.

    - Removes curly braces from expressions
    - Converts SI prefixes to VACASK format
    - Converts plain numeric values to SI prefix format
    - Preserves gauss/agauss expressions as-is for VACASK pass-through

    Args:
        value_str: Raw parameter value string

    Returns:
        Formatted value string
    """
    if value_str.startswith("{") and value_str.endswith("}"):
        value_str = value_str[1:-1]
    # Preserve gauss expressions without SI prefix conversion
    # to avoid mangling the function arguments
    if contains_gauss_function(value_str):
        return value_str

    # Try to convert plain numeric values to SI format
    try:
        num_value = float(value_str)
        return format_numeric_si(num_value)
    except ValueError:
        pass

    return convert_si_prefixes(value_str)


def split_params(params: list[str], handle_m: bool = False) -> list[tuple[str, str]]:
    """Split parameter strings into (name, value) tuples.

    Treats boolean parameters (without =) as <param>=1.

    Args:
        params: List of parameter strings (e.g., ["r=1k", "tc1=0", "off"])
        handle_m: If True, rename 'm' parameter to '$mfactor'

    Returns:
        List of (name, value) tuples

    Raises:
        ValueError: If parameter format is invalid
    """
    result = []
    for p in params:
        parts = p.split("=", 1)
        if len(parts) == 1:
            # Boolean parameter (no value)
            name, value = parts[0], "1"
        elif len(parts) == 2:
            name, value = parts
        else:
            raise ValueError(f"Malformed parameter '{p}'")

        # Handle mfactor renaming
        if handle_m and name.lower() in ("m", "_mfactor"):
            name = "$mfactor"

        result.append((name, value))

    return result


def remove_params(params: list[tuple[str, str]], to_remove: set[str]) -> list[tuple[str, str]]:
    """Remove specified parameters from a parameter list.

    Args:
        params: List of (name, value) tuples
        to_remove: Set of parameter names to remove (lowercase)

    Returns:
        Filtered parameter list
    """
    return [(name, value) for name, value in params if name.lower() not in to_remove]


def merge_vector_params(params: list[str], vector_names: set[str]) -> list[str]:
    """Merge vector parameters into single strings.

    Vector parameters in SPICE can span multiple tokens with commas.
    E.g., ic=1,2,3 might be split into ["ic=1,", "2,", "3"]

    Args:
        params: List of unsplit parameter strings
        vector_names: Set of parameter names that are vectors

    Returns:
        Parameter list with vectors merged
    """
    result = []
    i = 0
    while i < len(params):
        param = params[i]
        parts = param.split("=", 1)
        if len(parts) > 1 and parts[0].lower() in vector_names:
            # Start merging
            merged = param
            while i < len(params) and params[i].strip().endswith(","):
                i += 1
                if i < len(params):
                    merged += params[i]
            result.append(merged)
        else:
            result.append(param)
        i += 1

    return result


def process_expressions(params: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Process parameter expressions.

    Replaces SPICE-specific variables with VACASK equivalents:
    - temper -> $temp

    Args:
        params: List of (name, value) tuples

    Returns:
        Processed parameter list
    """
    result = []
    for name, value in params:
        # Replace temper with $temp
        value = _TEMPER_PATTERN.sub("$temp", value)
        result.append((name, value))
    return result


def process_instance_params(
    params: list[str],
    device_type: str,
    handle_m: bool = False,
) -> list[tuple[str, str]]:
    """Process instance parameters for output.

    Applies the full processing pipeline:
    1. Merge vector parameters
    2. Split into (name, value) tuples
    3. Remove device-specific unwanted parameters
    4. Process expressions

    Args:
        params: List of raw parameter strings
        device_type: Device type (e.g., "q", "d", "m") for filtering
        handle_m: If True, rename 'm' parameter to '$mfactor'

    Returns:
        Processed list of (name, value) tuples
    """
    # Merge vector parameters
    vec_names = MERGE_VECTOR_INSTANCE_PARAMS.get(device_type.lower(), set())
    if vec_names:
        params = merge_vector_params(params, vec_names)

    # Split parameters
    split = split_params(params, handle_m=handle_m)

    # Remove unwanted parameters
    to_remove = REMOVE_INSTANCE_PARAMS.get(device_type.lower(), set())
    if to_remove:
        split = remove_params(split, to_remove)

    # Process expressions
    split = process_expressions(split)

    return split


def process_terminals(terminals: list[str]) -> list[str]:
    """Process terminal/node names for VACASK output.

    Replaces characters that are invalid in VACASK identifiers.

    Args:
        terminals: List of terminal names

    Returns:
        Processed terminal names
    """
    return [t.replace("!", "_") for t in terminals]


def format_params_line(
    params: list[tuple[str, str]],
    max_columns: int = 80,
    indent: int = 2,
) -> tuple[str, bool]:
    """Format parameters into a line or multiple lines.

    Args:
        params: List of (name, value) tuples
        max_columns: Maximum line width
        indent: Indentation for continuation lines

    Returns:
        Tuple of (formatted string, was_split)
    """
    if not params:
        return "", False

    # Format each parameter
    formatted = []
    for name, value in params:
        value = format_value(value)
        formatted.append(f"{name}={value}")

    # Check if it fits on one line
    one_line = " ".join(formatted)
    if len(one_line) <= max_columns:
        return one_line, False

    # Split into multiple lines
    lines = []
    current_line = ""
    indent_str = " " * indent

    for param in formatted:
        if not current_line:
            current_line = param
        elif len(current_line) + 1 + len(param) <= max_columns:
            current_line += " " + param
        else:
            lines.append(current_line)
            current_line = param

    if current_line:
        lines.append(current_line)

    return ("\n" + indent_str).join(lines), len(lines) > 1
