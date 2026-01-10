"""Code generation context for tracking state during translation.

This module provides CodeGenContext which tracks:
- Defined variables
- Constants (float, bool, int)
- Input array mappings
- Variable naming prefixes
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Union, List
import sys
import os

# Add parent directory to path to import openvaf_ast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from openvaf_ast import (
    name as ast_name, const as ast_const,
    jnp_call, jnp_bool, jnp_inf, jnp_nan,
    subscript, unaryop,
)

# Import pre-allocated constant ValueIds
from mir.types import (
    V_GRAVESTONE, V_FALSE, V_TRUE, V_F_ZERO,
    V_ZERO, V_ONE, V_F_ONE, V_F_N_ONE,
)


@dataclass
class CodeGenContext:
    """Context for code generation tracking state.

    Tracks defined variables, constants, and provides operand resolution.
    """

    # Constants from MIR
    constants: Dict[str, float] = field(default_factory=dict)
    bool_constants: Dict[str, bool] = field(default_factory=dict)
    int_constants: Dict[str, int] = field(default_factory=dict)

    # Variables that have been defined in generated code
    defined_vars: Set[str] = field(default_factory=set)

    # Prefix for variable names (e.g., 'init_' for init function variables)
    var_prefix: str = ''

    # Input array name
    input_array: str = 'inputs'

    # String constants (for $simparam, etc.)
    str_constants: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize pre-allocated OpenVAF constants."""
        # Pre-allocated MIR constants (from OpenVAF mir/src/dfg/values.rs:57-76)
        # These are always present at fixed positions but not shown in MIR dumps
        self.bool_constants.setdefault(V_FALSE, False)
        self.bool_constants.setdefault(V_TRUE, True)
        self.constants.setdefault(V_F_ZERO, 0.0)
        self.int_constants.setdefault(V_ZERO, 0)
        self.int_constants.setdefault(V_ONE, 1)
        self.constants.setdefault(V_F_ONE, 1.0)
        self.constants.setdefault(V_F_N_ONE, -1.0)

    def define_var(self, var: str) -> str:
        """Mark a variable as defined and return its prefixed name."""
        prefixed = f"{self.var_prefix}{var}"
        self.defined_vars.add(prefixed)
        return prefixed

    def is_defined(self, var: str) -> bool:
        """Check if a variable is defined."""
        prefixed = f"{self.var_prefix}{var}"
        return (prefixed in self.defined_vars or
                var in self.constants or
                var in self.bool_constants or
                var in self.int_constants)

    def get_operand(self, op: str, allow_undefined: bool = False) -> ast.expr:
        """Resolve an operand to an AST expression.

        Order of resolution:
        1. Check constants (float, bool, int) - includes pre-allocated v0-v7
        2. Check if it's a defined variable (with prefix)
        3. If allow_undefined=True, return as bare name
        4. Otherwise return zero as fallback

        Args:
            op: The operand string (e.g., 'v123')
            allow_undefined: If True, allow referencing undefined vars

        Returns:
            AST expression for the operand
        """
        # Check float constants (includes pre-allocated v3, v6, v7)
        if op in self.constants:
            value = self.constants[op]
            if value == float('inf'):
                return jnp_inf()
            elif value == float('-inf'):
                return unaryop(ast.USub(), jnp_inf())
            elif value != value:  # NaN check
                return jnp_nan()
            else:
                return ast_const(value)

        # Check bool constants (includes pre-allocated v1, v2)
        if op in self.bool_constants:
            return jnp_bool(ast_const(self.bool_constants[op]))

        # Check int constants (includes pre-allocated v4, v5)
        if op in self.int_constants:
            return ast_const(self.int_constants[op])

        # Check prefixed defined variable
        prefixed = f"{self.var_prefix}{op}"
        if prefixed in self.defined_vars:
            return ast_name(prefixed)

        # Check unprefixed defined variable (for PHI references)
        if op in self.defined_vars:
            return ast_name(op)

        # If allow_undefined, return name (will be defined later)
        if allow_undefined:
            return ast_name(prefixed)

        # Fallback to zero for undefined operands
        # This happens when PHI references values from unexecuted branches
        return ast_const(0.0)

    def get_operand_str(self, op: str) -> str:
        """Get the string representation of an operand.

        Used for code generation that doesn't use AST.
        """
        # Check float constants (includes pre-allocated v3, v6, v7)
        if op in self.constants:
            value = self.constants[op]
            if value == float('inf'):
                return 'jnp.inf'
            elif value == float('-inf'):
                return '-jnp.inf'
            elif value != value:
                return 'jnp.nan'
            else:
                return repr(value)

        # Check bool constants (includes pre-allocated v1, v2)
        if op in self.bool_constants:
            return f"jnp.bool_({self.bool_constants[op]})"

        # Check int constants (includes pre-allocated v4, v5)
        if op in self.int_constants:
            return repr(self.int_constants[op])

        # Check defined variables
        prefixed = f"{self.var_prefix}{op}"
        if prefixed in self.defined_vars:
            return prefixed

        return prefixed

    def get_input(self, index: int) -> ast.expr:
        """Get AST expression for inputs[index]."""
        return subscript(ast_name(self.input_array), ast_const(index))

    def get_input_negative(self, offset: int) -> ast.expr:
        """Get AST expression for inputs[-offset] (e.g., inputs[-1])."""
        return subscript(
            ast_name(self.input_array),
            unaryop(ast.USub(), ast_const(offset))
        )

    def zero(self) -> ast.expr:
        """Get AST expression for zero constant (0.0)."""
        return ast_const(0.0)

    def one(self) -> ast.expr:
        """Get AST expression for one constant (1.0)."""
        return ast_const(1.0)


@dataclass
class SplitParamContext(CodeGenContext):
    """Context for split parameter code generation.

    Extends CodeGenContext to support split shared/device params.
    """

    # Mapping from original param index to (source, new_index)
    # source is 'shared' or 'device'
    param_mapping: Dict[int, tuple] = field(default_factory=dict)

    # Cache mapping from original index to (source, new_index)
    cache_mapping: Dict[int, tuple] = field(default_factory=dict)

    # Array names
    shared_params_array: str = 'shared_params'
    device_params_array: str = 'device_params'
    shared_cache_array: str = 'shared_cache'
    device_cache_array: str = 'device_cache'
    cache_array: str = 'cache'

    # Whether cache is split
    use_cache_split: bool = False

    def get_param(self, orig_index: int) -> ast.expr:
        """Get AST expression for parameter by original index."""
        if orig_index in self.param_mapping:
            source, new_idx = self.param_mapping[orig_index]
            if source == 'shared':
                return subscript(ast_name(self.shared_params_array), ast_const(new_idx))
            else:
                return subscript(ast_name(self.device_params_array), ast_const(new_idx))
        # Fallback to device params
        return subscript(ast_name(self.device_params_array), ast_const(orig_index))

    def get_cache(self, cache_index: int) -> ast.expr:
        """Get AST expression for cache value by index."""
        if self.use_cache_split and cache_index in self.cache_mapping:
            source, new_idx = self.cache_mapping[cache_index]
            if source == 'shared_cache':
                return subscript(ast_name(self.shared_cache_array), ast_const(new_idx))
            else:
                return subscript(ast_name(self.device_cache_array), ast_const(new_idx))
        return subscript(ast_name(self.cache_array), ast_const(cache_index))


def build_context_from_mir(mir_func, var_prefix: str = '') -> CodeGenContext:
    """Build a CodeGenContext from a MIRFunction.

    Args:
        mir_func: MIRFunction to extract constants from
        var_prefix: Prefix for variable names

    Returns:
        Initialized CodeGenContext
    """
    return CodeGenContext(
        constants=dict(mir_func.constants),
        bool_constants=dict(mir_func.bool_constants),
        int_constants=dict(mir_func.int_constants),
        defined_vars=set(),
        var_prefix=var_prefix,
    )
