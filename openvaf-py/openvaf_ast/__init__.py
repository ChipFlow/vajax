"""AST-based code generation for OpenVAF to JAX translation.

This package provides utilities for building Python AST nodes
programmatically, enabling cleaner and more maintainable code
generation compared to string concatenation.
"""

from .expressions import (
    name,
    const,
    binop,
    unaryop,
    compare,
    boolop,
    call,
    attr,
    subscript,
    list_expr,
    tuple_expr,
    jnp_call,
    jnp_where,
    jnp_float64,
    jnp_bool,
    jnp_inf,
    jnp_nan,
    lax_call,
    lax_while_loop,
    safe_divide,
    nested_where,
)

from .statements import (
    assign,
    assign_tuple,
    aug_assign,
    function_def,
    return_stmt,
    import_stmt,
    import_from,
    expr_stmt,
    if_stmt,
    pass_stmt,
)

from .builder import ASTBuilder, ExpressionBuilder

__all__ = [
    # Expressions
    "name",
    "const",
    "binop",
    "unaryop",
    "compare",
    "boolop",
    "call",
    "attr",
    "subscript",
    "list_expr",
    "tuple_expr",
    "jnp_call",
    "jnp_where",
    "jnp_float64",
    "jnp_bool",
    "jnp_inf",
    "jnp_nan",
    "lax_call",
    "lax_while_loop",
    "safe_divide",
    "nested_where",
    # Statements
    "assign",
    "assign_tuple",
    "aug_assign",
    "function_def",
    "return_stmt",
    "import_stmt",
    "import_from",
    "expr_stmt",
    "if_stmt",
    "pass_stmt",
    # Builder
    "ASTBuilder",
    "ExpressionBuilder",
]
