"""AST statement builders.

Helper functions for creating Python AST statement nodes.
Each function returns a properly constructed ast.stmt subclass.
"""

import ast
from typing import Any, List, Optional, Union

from .expressions import name, const


def assign(targets: Union[str, ast.expr, List[Union[str, ast.expr]]],
           value: ast.expr) -> ast.Assign:
    """Create an assignment statement.

    Args:
        targets: Target variable name(s) or expression(s)
        value: Value expression to assign

    Returns:
        ast.Assign node

    Example:
        >>> assign('x', const(42))  # Generates: x = 42
        >>> assign(['a', 'b'], tuple_expr([const(1), const(2)]))
        # Generates: a, b = (1, 2)
    """
    target_nodes: List[ast.expr] = []
    if isinstance(targets, str):
        target_nodes = [name(targets, ast.Store())]
    elif isinstance(targets, ast.expr):
        # Ensure the expression has Store context
        if isinstance(targets, (ast.Name, ast.Attribute, ast.Subscript)):
            targets.ctx = ast.Store()
        target_nodes = [targets]
    else:
        # List of targets
        for t in targets:
            if isinstance(t, str):
                target_nodes.append(name(t, ast.Store()))
            elif isinstance(t, (ast.Name, ast.Attribute, ast.Subscript)):
                t.ctx = ast.Store()
                target_nodes.append(t)
            else:
                target_nodes.append(t)

    return ast.Assign(targets=target_nodes, value=value)


def assign_tuple(target_names: List[str], value: ast.expr) -> ast.Assign:
    """Create a tuple unpacking assignment.

    Args:
        target_names: List of variable names to unpack into
        value: Value expression (should produce a tuple)

    Returns:
        ast.Assign node with tuple target

    Example:
        >>> assign_tuple(['a', 'b'], name('result'))
        # Generates: a, b = result
    """
    target = ast.Tuple(
        elts=[name(n, ast.Store()) for n in target_names],
        ctx=ast.Store()
    )
    return ast.Assign(targets=[target], value=value)


def aug_assign(target: Union[str, ast.Name, ast.Attribute, ast.Subscript],
               op: ast.operator,
               value: ast.expr) -> ast.AugAssign:
    """Create an augmented assignment statement.

    Args:
        target: Target variable name or expression (must be Name, Attribute, or Subscript)
        op: Operator (ast.Add(), ast.Sub(), etc.)
        value: Value expression

    Returns:
        ast.AugAssign node

    Example:
        >>> aug_assign('x', ast.Add(), const(1))  # Generates: x += 1
    """
    target_node: Union[ast.Name, ast.Attribute, ast.Subscript]
    if isinstance(target, str):
        target_node = name(target, ast.Store())
    else:
        target.ctx = ast.Store()
        target_node = target

    return ast.AugAssign(target=target_node, op=op, value=value)


def function_def(func_name: str,
                 args: Optional[List[str]] = None,
                 body: Optional[List[ast.stmt]] = None,
                 decorator_list: Optional[List[ast.expr]] = None,
                 returns: Optional[ast.expr] = None,
                 defaults: Optional[List[ast.expr]] = None) -> ast.FunctionDef:
    """Create a function definition.

    Args:
        func_name: Function name
        args: List of parameter names
        body: List of body statements (defaults to [pass])
        decorator_list: List of decorator expressions
        returns: Return type annotation
        defaults: List of default values for parameters

    Returns:
        ast.FunctionDef node

    Example:
        >>> function_def('add', ['a', 'b'], [return_stmt(binop(name('a'), ast.Add(), name('b')))])
        # Generates:
        # def add(a, b):
        #     return a + b
    """
    arg_nodes = [ast.arg(arg=a) for a in (args or [])]

    return ast.FunctionDef(
        name=func_name,
        args=ast.arguments(
            posonlyargs=[],
            args=arg_nodes,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=defaults or []
        ),
        body=body or [ast.Pass()],
        decorator_list=decorator_list or [],
        returns=returns
    )


def return_stmt(value: Optional[ast.expr] = None) -> ast.Return:
    """Create a return statement.

    Args:
        value: Return value expression (None for bare return)

    Returns:
        ast.Return node

    Example:
        >>> return_stmt(name('result'))  # Generates: return result
        >>> return_stmt()                # Generates: return
    """
    return ast.Return(value=value)


def import_stmt(names: List[Union[str, tuple]]) -> ast.Import:
    """Create an import statement.

    Args:
        names: List of module names or (name, asname) tuples

    Returns:
        ast.Import node

    Example:
        >>> import_stmt(['os', 'sys'])  # Generates: import os, sys
        >>> import_stmt([('jax.numpy', 'jnp')])  # Generates: import jax.numpy as jnp
    """
    aliases = []
    for n in names:
        if isinstance(n, tuple):
            aliases.append(ast.alias(name=n[0], asname=n[1]))
        else:
            aliases.append(ast.alias(name=n, asname=None))
    return ast.Import(names=aliases)


def import_from(module: str, names: List[Union[str, tuple]],
                level: int = 0) -> ast.ImportFrom:
    """Create a from ... import statement.

    Args:
        module: Module name
        names: List of names to import or (name, asname) tuples
        level: Relative import level (0 for absolute)

    Returns:
        ast.ImportFrom node

    Example:
        >>> import_from('jax', ['lax'])  # Generates: from jax import lax
        >>> import_from('typing', [('List', 'L')])
        # Generates: from typing import List as L
    """
    aliases = []
    for n in names:
        if isinstance(n, tuple):
            aliases.append(ast.alias(name=n[0], asname=n[1]))
        else:
            aliases.append(ast.alias(name=n, asname=None))
    return ast.ImportFrom(module=module, names=aliases, level=level)


def expr_stmt(value: ast.expr) -> ast.Expr:
    """Create an expression statement (expression used as statement).

    Args:
        value: Expression to use as statement

    Returns:
        ast.Expr node

    Example:
        >>> expr_stmt(call(name('print'), [const('hello')]))
        # Generates: print('hello')
    """
    return ast.Expr(value=value)


def if_stmt(test: ast.expr,
            body: List[ast.stmt],
            orelse: Optional[List[ast.stmt]] = None) -> ast.If:
    """Create an if statement.

    Args:
        test: Condition expression
        body: List of statements for the if body
        orelse: List of statements for else body (or elif/else chain)

    Returns:
        ast.If node

    Example:
        >>> if_stmt(compare(name('x'), ast.Gt(), const(0)),
        ...         [return_stmt(const(1))],
        ...         [return_stmt(const(-1))])
        # Generates:
        # if x > 0:
        #     return 1
        # else:
        #     return -1
    """
    return ast.If(test=test, body=body, orelse=orelse or [])


def for_stmt(target: Union[str, ast.Name, ast.Tuple, ast.List],
             iter_expr: ast.expr,
             body: List[ast.stmt],
             orelse: Optional[List[ast.stmt]] = None) -> ast.For:
    """Create a for loop statement.

    Args:
        target: Loop variable name or expression (Name, Tuple, or List for unpacking)
        iter_expr: Iterator expression
        body: List of body statements
        orelse: List of else statements

    Returns:
        ast.For node

    Example:
        >>> for_stmt('i', call(name('range'), [const(10)]),
        ...          [expr_stmt(call(name('print'), [name('i')]))])
        # Generates:
        # for i in range(10):
        #     print(i)
    """
    target_node: Union[ast.Name, ast.Tuple, ast.List]
    if isinstance(target, str):
        target_node = name(target, ast.Store())
    else:
        target.ctx = ast.Store()
        target_node = target

    return ast.For(target=target_node, iter=iter_expr,
                   body=body, orelse=orelse or [])


def while_stmt(test: ast.expr,
               body: List[ast.stmt],
               orelse: Optional[List[ast.stmt]] = None) -> ast.While:
    """Create a while loop statement.

    Args:
        test: Condition expression
        body: List of body statements
        orelse: List of else statements

    Returns:
        ast.While node
    """
    return ast.While(test=test, body=body, orelse=orelse or [])


def pass_stmt() -> ast.Pass:
    """Create a pass statement.

    Returns:
        ast.Pass node
    """
    return ast.Pass()


def comment_as_string_stmt(text: str) -> ast.Expr:
    """Create a comment-like string statement (docstring-style).

    Note: Python AST doesn't have true comments. This creates a string
    expression that serves as documentation but may have runtime overhead.
    For pure comments, consider using the builder's comment tracking instead.

    Args:
        text: Comment text

    Returns:
        ast.Expr with string constant
    """
    return ast.Expr(value=const(text))


# =============================================================================
# Compound statement builders
# =============================================================================

def nested_function(func_name: str,
                    args: List[str],
                    body_stmts: List[ast.stmt]) -> ast.FunctionDef:
    """Create a nested function definition (for lax.while_loop callbacks).

    Args:
        func_name: Function name (e.g., '_loop_cond', '_loop_body')
        args: Parameter names
        body_stmts: Function body statements

    Returns:
        ast.FunctionDef node
    """
    return function_def(func_name, args, body_stmts)


def build_module(body: List[ast.stmt]) -> ast.Module:
    """Create a complete module AST.

    Args:
        body: List of module-level statements

    Returns:
        ast.Module node ready for compilation
    """
    return ast.Module(body=body, type_ignores=[])


def fix_and_compile(module: ast.Module, filename: str = '<generated>') -> Any:
    """Fix missing locations and compile an AST module.

    Args:
        module: AST module to compile
        filename: Filename for error messages

    Returns:
        Compiled code object ready for exec()
    """
    ast.fix_missing_locations(module)
    return compile(module, filename, 'exec')
