"""Safe expression evaluator for SPICE parameter expressions.

Uses simpleeval to safely evaluate arithmetic expressions without
allowing arbitrary code execution. Supports SPICE SI suffixes and
common math functions.

Example:
    >>> from jax_spice.utils.safe_eval import safe_eval_expr
    >>> safe_eval_expr("2*wmin + 1u", {"wmin": 1e-6})
    3e-06
"""

import math
import re
from typing import Dict, Union

from simpleeval import EvalWithCompoundTypes, InvalidExpression

# SPICE SI suffixes (order matters - check longer suffixes first)
SI_SUFFIXES = [
    ("meg", 1e6),
    ("mil", 25.4e-6),
    ("g", 1e9),
    ("t", 1e12),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
]

# Safe math functions available in expressions
SAFE_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}


def parse_spice_number(s: str) -> tuple[float, bool]:
    """Parse a SPICE number with optional SI suffix.

    Args:
        s: String like "1u", "100n", "1.5meg", "2.5e-6"

    Returns:
        Tuple of (value, success). If parsing fails, returns (0.0, False).
    """
    s = s.strip().lower()
    if not s:
        return 0.0, False

    # Try direct float parse first
    try:
        return float(s), True
    except ValueError:
        pass

    # Try with SI suffix
    for suffix, mult in SI_SUFFIXES:
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * mult, True
            except ValueError:
                pass

    return 0.0, False


def _substitute_params(expr: str, params: Dict[str, float]) -> str:
    """Substitute parameter names with their values in an expression.

    Substitutes longer names first to avoid partial replacements
    (e.g., 'wmin' before 'w').
    """
    result = expr
    for name, value in sorted(params.items(), key=lambda x: -len(x[0])):
        # Use word boundaries to avoid partial matches
        # But SPICE params can contain underscores, so we do simple replace
        result = result.replace(name, str(value))
    return result


def _expand_si_suffixes(expr: str) -> str:
    """Expand SI suffixes in an expression to their numeric values.

    Converts "1u" to "1e-6", "100n" to "100e-9", etc.
    """
    # Pattern: number followed by SI suffix (not followed by more alphanumeric)
    # e.g., "1u", "100n", "1.5meg"
    for suffix, mult in SI_SUFFIXES:
        # Match: optional sign, digits/decimal, suffix, not followed by alphanumeric
        pattern = rf"(\d+\.?\d*|\.\d+)({suffix})(?![a-zA-Z0-9])"
        replacement = rf"\1*{mult}"
        expr = re.sub(pattern, replacement, expr, flags=re.IGNORECASE)
    return expr


# Create a reusable evaluator instance
_evaluator = EvalWithCompoundTypes(functions=SAFE_FUNCTIONS)


def safe_eval_expr(
    expr: str, params: Dict[str, float], default: float = 0.0
) -> Union[float, str]:
    """Safely evaluate a SPICE parameter expression.

    Supports:
    - Arithmetic operators: +, -, *, /, **, ()
    - Math functions: sin, cos, exp, log, sqrt, abs, etc.
    - SPICE SI suffixes: u, n, p, f, k, meg, etc.
    - Parameter substitution from the params dict

    Args:
        expr: Expression string like "2*wmin + 1u" or "sin(2*pi*f)"
        params: Dict of parameter names to values for substitution
        default: Value to return if evaluation fails

    Returns:
        Evaluated float value, or default if evaluation fails

    Example:
        >>> safe_eval_expr("2*w + l", {"w": 1e-6, "l": 0.5e-6})
        2.5e-06
        >>> safe_eval_expr("1u + 100n", {})
        1.1e-06
    """
    if not isinstance(expr, str):
        try:
            return float(expr)
        except (ValueError, TypeError):
            return default

    expr = expr.strip()
    if not expr:
        return default

    # Try direct SPICE number parse first (most common case)
    val, success = parse_spice_number(expr)
    if success:
        return val

    try:
        # Substitute parameters
        eval_expr = _substitute_params(expr, params)

        # Expand SI suffixes
        eval_expr = _expand_si_suffixes(eval_expr)

        # Evaluate safely
        result = _evaluator.eval(eval_expr)
        return float(result)

    except (InvalidExpression, ValueError, TypeError, KeyError, SyntaxError):
        return default
    except Exception:
        # Catch any other unexpected errors from simpleeval
        return default
