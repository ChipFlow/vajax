"""AST expression builders.

Helper functions for creating Python AST expression nodes.
Each function returns a properly constructed ast.expr subclass.
"""

import ast
from typing import Any, List, Optional, Union


def name(id: str, ctx: Optional[ast.expr_context] = None) -> ast.Name:
    """Create an ast.Name node (variable reference).

    Args:
        id: Variable name
        ctx: Context (Load, Store, Del). Defaults to Load.

    Returns:
        ast.Name node

    Example:
        >>> name('x')  # Generates: x
    """
    return ast.Name(id=id, ctx=ctx or ast.Load())


def const(value: Any) -> ast.Constant:
    """Create an ast.Constant node (literal value).

    Args:
        value: The constant value (int, float, str, bool, None, etc.)

    Returns:
        ast.Constant node

    Example:
        >>> const(42)     # Generates: 42
        >>> const(3.14)   # Generates: 3.14
        >>> const('hi')   # Generates: 'hi'
    """
    return ast.Constant(value=value)


def binop(left: ast.expr, op: ast.operator, right: ast.expr) -> ast.BinOp:
    """Create a binary operation node.

    Args:
        left: Left operand expression
        op: Operator (ast.Add(), ast.Sub(), ast.Mult(), ast.Div(), etc.)
        right: Right operand expression

    Returns:
        ast.BinOp node

    Example:
        >>> binop(name('x'), ast.Add(), name('y'))  # Generates: (x + y)
    """
    return ast.BinOp(left=left, op=op, right=right)


def unaryop(op: ast.unaryop, operand: ast.expr) -> ast.UnaryOp:
    """Create a unary operation node.

    Args:
        op: Operator (ast.UAdd(), ast.USub(), ast.Not(), ast.Invert())
        operand: Operand expression

    Returns:
        ast.UnaryOp node

    Example:
        >>> unaryop(ast.USub(), name('x'))  # Generates: -x
    """
    return ast.UnaryOp(op=op, operand=operand)


def compare(left: ast.expr, op: ast.cmpop, right: ast.expr) -> ast.Compare:
    """Create a comparison operation node.

    Args:
        left: Left operand expression
        op: Comparison operator (ast.Eq(), ast.Lt(), ast.Gt(), etc.)
        right: Right operand expression

    Returns:
        ast.Compare node

    Example:
        >>> compare(name('x'), ast.Lt(), const(0))  # Generates: x < 0
    """
    return ast.Compare(left=left, ops=[op], comparators=[right])


def compare_chain(left: ast.expr, ops: List[ast.cmpop],
                  comparators: List[ast.expr]) -> ast.Compare:
    """Create a chained comparison (e.g., 0 < x < 10).

    Args:
        left: Leftmost operand
        ops: List of comparison operators
        comparators: List of comparators (one per operator)

    Returns:
        ast.Compare node with chained comparisons
    """
    return ast.Compare(left=left, ops=ops, comparators=comparators)


def boolop(op: ast.boolop, values: List[ast.expr]) -> ast.BoolOp:
    """Create a boolean operation node (and/or).

    Args:
        op: Boolean operator (ast.And() or ast.Or())
        values: List of operand expressions

    Returns:
        ast.BoolOp node

    Example:
        >>> boolop(ast.And(), [name('a'), name('b')])  # Generates: a and b
    """
    return ast.BoolOp(op=op, values=values)


def call(func: ast.expr, args: Optional[List[ast.expr]] = None,
         keywords: Optional[List[ast.keyword]] = None) -> ast.Call:
    """Create a function call node.

    Args:
        func: Function expression (Name or Attribute)
        args: Positional arguments
        keywords: Keyword arguments

    Returns:
        ast.Call node

    Example:
        >>> call(name('print'), [const('hello')])  # Generates: print('hello')
    """
    return ast.Call(
        func=func,
        args=args or [],
        keywords=keywords or []
    )


def attr(value: ast.expr, attr_name: str,
         ctx: Optional[ast.expr_context] = None) -> ast.Attribute:
    """Create an attribute access node.

    Args:
        value: Object expression
        attr_name: Attribute name
        ctx: Context (Load, Store, Del). Defaults to Load.

    Returns:
        ast.Attribute node

    Example:
        >>> attr(name('jnp'), 'exp')  # Generates: jnp.exp
    """
    return ast.Attribute(value=value, attr=attr_name, ctx=ctx or ast.Load())


def subscript(value: ast.expr, index: ast.expr,
              ctx: Optional[ast.expr_context] = None) -> ast.Subscript:
    """Create a subscript access node.

    Args:
        value: Container expression
        index: Index expression
        ctx: Context (Load, Store, Del). Defaults to Load.

    Returns:
        ast.Subscript node

    Example:
        >>> subscript(name('arr'), const(0))  # Generates: arr[0]
    """
    return ast.Subscript(value=value, slice=index, ctx=ctx or ast.Load())


def list_expr(elts: List[ast.expr],
              ctx: Optional[ast.expr_context] = None) -> ast.List:
    """Create a list literal node.

    Args:
        elts: List elements
        ctx: Context. Defaults to Load.

    Returns:
        ast.List node

    Example:
        >>> list_expr([const(1), const(2)])  # Generates: [1, 2]
    """
    return ast.List(elts=elts, ctx=ctx or ast.Load())


def tuple_expr(elts: List[ast.expr],
               ctx: Optional[ast.expr_context] = None) -> ast.Tuple:
    """Create a tuple literal node.

    Args:
        elts: Tuple elements
        ctx: Context. Defaults to Load.

    Returns:
        ast.Tuple node

    Example:
        >>> tuple_expr([name('a'), name('b')])  # Generates: (a, b)
    """
    return ast.Tuple(elts=elts, ctx=ctx or ast.Load())


# =============================================================================
# JAX-specific expression builders
# =============================================================================

def jnp_call(method: str, *args: ast.expr) -> ast.Call:
    """Create a jnp.method(*args) call.

    Args:
        method: JAX numpy method name (e.g., 'exp', 'sqrt', 'array')
        *args: Arguments to pass to the method

    Returns:
        ast.Call node for jnp.method(*args)

    Example:
        >>> jnp_call('exp', name('x'))  # Generates: jnp.exp(x)
        >>> jnp_call('array', list_expr([name('a'), name('b')]))
        # Generates: jnp.array([a, b])
    """
    return call(attr(name('jnp'), method), list(args))


def jnp_where(cond: ast.expr, true_val: ast.expr,
              false_val: ast.expr) -> ast.Call:
    """Create a jnp.where(cond, true_val, false_val) call.

    Args:
        cond: Condition expression
        true_val: Value when condition is True
        false_val: Value when condition is False

    Returns:
        ast.Call node for jnp.where(...)

    Example:
        >>> jnp_where(name('cond'), name('a'), name('b'))
        # Generates: jnp.where(cond, a, b)
    """
    return jnp_call('where', cond, true_val, false_val)


def jnp_float64(value: ast.expr) -> ast.Call:
    """Create a jnp.float64(value) call.

    Args:
        value: Value to convert

    Returns:
        ast.Call node for jnp.float64(value)
    """
    return jnp_call('float64', value)


def jnp_bool(value: ast.expr) -> ast.Call:
    """Create a jnp.bool_(value) call for JIT-compatible booleans.

    Args:
        value: Boolean value expression

    Returns:
        ast.Call node for jnp.bool_(value)
    """
    return call(attr(name('jnp'), 'bool_'), [value])


def jnp_inf() -> ast.Attribute:
    """Create jnp.inf reference."""
    return attr(name('jnp'), 'inf')


def jnp_nan() -> ast.Attribute:
    """Create jnp.nan reference."""
    return attr(name('jnp'), 'nan')


def lax_call(method: str, *args: ast.expr) -> ast.Call:
    """Create a lax.method(*args) call.

    Args:
        method: JAX lax method name (e.g., 'while_loop', 'cond')
        *args: Arguments to pass to the method

    Returns:
        ast.Call node for lax.method(*args)

    Example:
        >>> lax_call('while_loop', name('cond'), name('body'), name('init'))
        # Generates: lax.while_loop(cond, body, init)
    """
    return call(attr(name('lax'), method), list(args))


def lax_while_loop(cond_fn: Union[str, ast.expr],
                   body_fn: Union[str, ast.expr],
                   init_state: ast.expr) -> ast.Call:
    """Create a lax.while_loop(cond_fn, body_fn, init_state) call.

    Args:
        cond_fn: Condition function name or expression
        body_fn: Body function name or expression
        init_state: Initial state expression

    Returns:
        ast.Call node for lax.while_loop(...)
    """
    cond = name(cond_fn) if isinstance(cond_fn, str) else cond_fn
    body = name(body_fn) if isinstance(body_fn, str) else body_fn
    return lax_call('while_loop', cond, body, init_state)


# =============================================================================
# Safe operation patterns
# =============================================================================

def safe_divide(dividend: ast.expr, divisor: ast.expr,
                zero_ref: Optional[ast.expr] = None,
                one_ref: Optional[ast.expr] = None) -> ast.Call:
    """Create a safe division that handles divide-by-zero.

    Generates: jnp.where(divisor == 0, 0, dividend / jnp.where(divisor == 0, 1, divisor))

    Args:
        dividend: Numerator expression
        divisor: Denominator expression
        zero_ref: Reference to zero value (defaults to _ZERO constant)
        one_ref: Reference to one value (defaults to _ONE constant)

    Returns:
        ast.Call node for safe division
    """
    zero = zero_ref or name('_ZERO')
    one = one_ref or name('_ONE')

    is_zero = compare(divisor, ast.Eq(), zero)
    safe_divisor = jnp_where(is_zero, one, divisor)
    division = binop(dividend, ast.Div(), safe_divisor)
    return jnp_where(is_zero, zero, division)


def nested_where(conditions: List[tuple], default: ast.expr) -> ast.expr:
    """Build nested jnp.where for multi-way conditionals.

    Args:
        conditions: List of (condition, value) tuples
        default: Default value if no conditions match

    Returns:
        Nested jnp.where expression

    Example:
        >>> nested_where([(cond1, val1), (cond2, val2)], default)
        # Generates: jnp.where(cond1, val1, jnp.where(cond2, val2, default))
    """
    result = default
    for cond, val in reversed(conditions):
        result = jnp_where(cond, val, result)
    return result
