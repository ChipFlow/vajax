"""AST Builder for OpenVAF to JAX code generation.

Provides a structured way to build Python AST for device evaluation functions.
"""

import ast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .expressions import (
    name, const, binop, unaryop, compare, call, attr, subscript,
    list_expr, tuple_expr, jnp_call, jnp_where, lax_while_loop,
    jnp_float64, jnp_bool, jnp_inf, jnp_nan, safe_divide, nested_where,
)
from .statements import (
    assign, assign_tuple, function_def, return_stmt, import_stmt,
    import_from, expr_stmt, pass_stmt, build_module, fix_and_compile,
)


class ASTBuilder:
    """Builder for constructing Python AST for JAX device functions.

    This class provides a structured way to build device evaluation functions
    that are compatible with JAX's JIT compilation and vmap.

    Example usage:
        builder = ASTBuilder()
        builder.start_function('device_eval', ['inputs'])
        builder.add_jax_imports()
        builder.add_constant('_ZERO', 0.0, is_jax_float=True)
        builder.add_assignment('x', subscript(name('inputs'), const(0)))
        builder.add_return(name('x'))
        module = builder.build()

        # Compile and execute
        code = fix_and_compile(module)
        namespace = {}
        exec(code, namespace)
        fn = namespace['device_eval']
    """

    def __init__(self):
        """Initialize the builder."""
        self._body: List[ast.stmt] = []
        self._defined_vars: Set[str] = set()
        self._current_function: Optional[str] = None
        self._function_body: List[ast.stmt] = []
        self._indent_level: int = 0

    @property
    def defined_vars(self) -> Set[str]:
        """Get the set of defined variable names."""
        return self._defined_vars.copy()

    def start_function(self, func_name: str, args: List[str]) -> None:
        """Start building a new function.

        Args:
            func_name: Name of the function
            args: List of parameter names
        """
        self._current_function = func_name
        self._function_args = args
        self._function_body = []
        self._defined_vars.clear()

    def add_jax_imports(self) -> None:
        """Add standard JAX imports (jnp and lax)."""
        self._function_body.append(
            import_stmt([('jax.numpy', 'jnp')])
        )
        self._function_body.append(
            import_from('jax', ['lax'])
        )

    def add_blank_line(self) -> None:
        """Add a blank line (as a pass statement placeholder).

        Note: AST doesn't directly support blank lines, but we can
        use ast.unparse() formatting or just skip this.
        """
        # In AST, we don't need explicit blank lines
        pass

    def add_constant(self, var_name: str, value: Any,
                     is_jax_float: bool = False,
                     is_jax_bool: bool = False) -> None:
        """Add a constant assignment.

        Args:
            var_name: Variable name to assign
            value: The constant value
            is_jax_float: If True, wrap in jnp.float64()
            is_jax_bool: If True, wrap in jnp.bool_()
        """
        if value == float('inf'):
            value_expr = jnp_inf()
        elif value == float('-inf'):
            value_expr = unaryop(ast.USub(), jnp_inf())
        elif isinstance(value, float) and value != value:  # NaN check
            value_expr = jnp_nan()
        elif is_jax_float:
            value_expr = jnp_float64(const(value))
        elif is_jax_bool:
            value_expr = jnp_bool(const(value))
        else:
            value_expr = const(value)

        self._function_body.append(assign(var_name, value_expr))
        self._defined_vars.add(var_name)

    def add_assignment(self, target: str, value: ast.expr) -> None:
        """Add a simple assignment statement.

        Args:
            target: Variable name to assign
            value: Expression to assign
        """
        self._function_body.append(assign(target, value))
        self._defined_vars.add(target)

    def add_statement(self, stmt: ast.stmt) -> None:
        """Add a raw statement to the current function body.

        Args:
            stmt: AST statement node
        """
        self._function_body.append(stmt)

    def add_nested_function(self, func_name: str, args: List[str],
                            body: List[ast.stmt]) -> None:
        """Add a nested function definition.

        Args:
            func_name: Name of the nested function
            args: Parameter names
            body: Function body statements
        """
        self._function_body.append(function_def(func_name, args, body))

    def add_while_loop(self, cond_fn_name: str, cond_body: List[ast.stmt],
                       body_fn_name: str, loop_body: List[ast.stmt],
                       state_vars: List[str],
                       init_values: List[ast.expr],
                       result_var: str = '_loop_result') -> None:
        """Add a lax.while_loop construct with condition and body functions.

        Args:
            cond_fn_name: Name for the condition function
            cond_body: Body of the condition function (must return bool)
            body_fn_name: Name for the body function
            loop_body: Body of the loop function (must return updated state)
            state_vars: Names of state variables in the loop
            init_values: Initial values for state variables
            result_var: Variable name to store loop result
        """
        # Build condition function
        cond_fn = function_def(cond_fn_name, ['_loop_state'], [
            assign_tuple(state_vars, subscript(name('_loop_state'), const(0)))
            if len(state_vars) > 1 else
            assign(state_vars[0], subscript(name('_loop_state'), const(0))),
            *cond_body
        ])
        self._function_body.append(cond_fn)

        # Build body function
        body_fn = function_def(body_fn_name, ['_loop_state'], [
            assign_tuple(state_vars, subscript(name('_loop_state'), const(0)))
            if len(state_vars) > 1 else
            assign(state_vars[0], subscript(name('_loop_state'), const(0))),
            *loop_body
        ])
        self._function_body.append(body_fn)

        # Build initial state tuple
        if len(init_values) == 1:
            init_state = tuple_expr([init_values[0]])
        else:
            init_state = tuple_expr(init_values)

        # Add the while_loop call
        loop_call = lax_while_loop(cond_fn_name, body_fn_name,
                                   tuple_expr([init_state]))
        self._function_body.append(assign(result_var, loop_call))
        self._defined_vars.add(result_var)

    def add_input_param(self, var_name: str, index: int,
                        input_name: str = 'inputs') -> None:
        """Add an input parameter assignment.

        Args:
            var_name: Variable name to assign
            index: Index into the inputs array
            input_name: Name of the input array parameter
        """
        self.add_assignment(var_name, subscript(name(input_name), const(index)))

    def add_array_return(self, arrays: Dict[str, List[str]]) -> None:
        """Add a return statement returning multiple arrays.

        Args:
            arrays: Dict mapping array names to lists of variable names
        """
        return_items = []
        for array_name, var_names in arrays.items():
            elements = [name(v) for v in var_names]
            return_items.append(jnp_call('array', list_expr(elements)))

        if len(return_items) == 1:
            self._function_body.append(return_stmt(return_items[0]))
        else:
            self._function_body.append(return_stmt(tuple_expr(return_items)))

    def add_return(self, value: ast.expr) -> None:
        """Add a return statement.

        Args:
            value: Expression to return
        """
        self._function_body.append(return_stmt(value))

    def end_function(self) -> ast.FunctionDef:
        """End the current function and return the AST node.

        Returns:
            The completed FunctionDef node
        """
        func = function_def(
            self._current_function,
            self._function_args,
            self._function_body
        )
        self._current_function = None
        return func

    def build(self) -> ast.Module:
        """Build the complete module AST.

        Returns:
            ast.Module ready for compilation

        Note: If a function is currently being built, it will be finalized.
        """
        body = list(self._body)
        if self._current_function:
            body.append(self.end_function())
        return build_module(body)

    def build_and_compile(self, filename: str = '<generated>') -> Any:
        """Build and compile the module.

        Args:
            filename: Filename for error messages

        Returns:
            Compiled code object ready for exec()
        """
        module = self.build()
        return fix_and_compile(module, filename)


class ExpressionBuilder:
    """Builds AST expressions from MIR instructions.

    This class handles the translation of MIR opcodes to JAX AST expressions.
    """

    # Map MIR opcodes to (ast.operator, is_binary)
    ARITHMETIC_OPS = {
        'fadd': (ast.Add, True),
        'fsub': (ast.Sub, True),
        'fmul': (ast.Mult, True),
        'fneg': (ast.USub, False),
        'iadd': (ast.Add, True),
        'isub': (ast.Sub, True),
        'imul': (ast.Mult, True),
        'idiv': (ast.FloorDiv, True),
        'irem': (ast.Mod, True),
        'ineg': (ast.USub, False),
    }

    # Map MIR opcodes to bitwise operators
    BITWISE_OPS = {
        'iand': (ast.BitAnd, True),
        'ior': (ast.BitOr, True),
        'ixor': (ast.BitXor, True),
    }

    # Map MIR opcodes to jnp function names
    # Note: 'sqrt' and 'ln' are handled specially to avoid NaN for invalid inputs
    TRANSCENDENTAL_OPS = {
        'exp': 'exp',
        'sin': 'sin',
        'cos': 'cos',
        'tan': 'tan',
        'asin': 'arcsin',
        'acos': 'arccos',
        'atan': 'arctan',
        'sinh': 'sinh',
        'cosh': 'cosh',
        'tanh': 'tanh',
        'asinh': 'arcsinh',
        'acosh': 'arccosh',
        'atanh': 'arctanh',
        'floor': 'floor',
        'ceil': 'ceil',
        'abs': 'abs',
        'fabs': 'abs',
    }

    # Map MIR opcodes to comparison operators
    COMPARISON_OPS = {
        'feq': ast.Eq,
        'fne': ast.NotEq,
        'flt': ast.Lt,
        'fgt': ast.Gt,
        'fle': ast.LtE,
        'fge': ast.GtE,
        'ieq': ast.Eq,
        'ine': ast.NotEq,
        'ilt': ast.Lt,
        'igt': ast.Gt,
        'ile': ast.LtE,
        'ige': ast.GtE,
        'beq': ast.Eq,
        'bne': ast.NotEq,
    }

    def __init__(self, zero_name: str = '_ZERO', one_name: str = '_ONE'):
        """Initialize the expression builder.

        Args:
            zero_name: Variable name for zero constant
            one_name: Variable name for one constant
        """
        self.zero_name = zero_name
        self.one_name = one_name

    def zero(self) -> ast.Name:
        """Return reference to zero constant."""
        return name(self.zero_name)

    def one(self) -> ast.Name:
        """Return reference to one constant."""
        return name(self.one_name)

    def translate_opcode(self, opcode: str,
                         operands: List[ast.expr]) -> Optional[ast.expr]:
        """Translate a MIR opcode to an AST expression.

        Args:
            opcode: MIR opcode (lowercase)
            operands: List of operand expressions

        Returns:
            AST expression node, or None if opcode not recognized
        """
        opcode = opcode.lower()

        # Arithmetic operations
        if opcode in self.ARITHMETIC_OPS:
            op_class, is_binary = self.ARITHMETIC_OPS[opcode]
            if is_binary:
                return binop(operands[0], op_class(), operands[1])
            else:
                return unaryop(op_class(), operands[0])

        # Bitwise operations
        if opcode in self.BITWISE_OPS:
            op_class, _ = self.BITWISE_OPS[opcode]
            return binop(operands[0], op_class(), operands[1])

        # Safe division
        if opcode == 'fdiv':
            return safe_divide(operands[0], operands[1],
                               self.zero(), self.one())

        # Safe sqrt: clamp negative inputs to zero to avoid NaN
        if opcode == 'sqrt':
            return jnp_call('sqrt', jnp_call('maximum', operands[0], self.zero()))

        # Safe ln (log): clamp non-positive inputs to small positive value to avoid NaN/inf
        if opcode == 'ln':
            # Use a small epsilon (1e-300) to avoid log(0) = -inf
            small_eps = const(1e-300)
            return jnp_call('log', jnp_call('maximum', operands[0], small_eps))

        # Transcendental functions
        if opcode in self.TRANSCENDENTAL_OPS:
            return jnp_call(self.TRANSCENDENTAL_OPS[opcode], operands[0])

        # Two-argument transcendental functions
        if opcode == 'pow':
            return jnp_call('power', operands[0], operands[1])
        if opcode == 'hypot':
            return jnp_call('hypot', operands[0], operands[1])
        if opcode == 'atan2':
            return jnp_call('arctan2', operands[0], operands[1])

        # Comparison operations
        if opcode in self.COMPARISON_OPS:
            op_class = self.COMPARISON_OPS[opcode]
            return compare(operands[0], op_class(), operands[1])

        # Boolean not - use jnp.logical_not for JIT compatibility
        if opcode == 'bnot':
            return jnp_call('logical_not', operands[0])

        # Optimization barrier (pass through)
        if opcode == 'optbarrier':
            return operands[0] if operands else self.zero()

        # Type casts
        if opcode == 'ifcast':  # int to float
            return jnp_call('float64', operands[0])
        if opcode == 'ficast':  # float to int
            return call(attr(name('jnp'), 'int32'),
                        [jnp_call('floor', operands[0])])
        if opcode == 'fbcast':  # float to bool
            return compare(operands[0], ast.NotEq(), self.zero())
        if opcode == 'ibcast':  # int to bool
            return compare(operands[0], ast.NotEq(), const(0))
        if opcode == 'bfcast':  # bool to float
            return jnp_where(operands[0], const(1.0), const(0.0))
        if opcode == 'bicast':  # bool to int
            return call(attr(name('jnp'), 'int32'), [operands[0]])

        # Control flow (handled at block level)
        if opcode in ('br', 'jmp', 'exit'):
            return None

        return None

    def translate_phi(self, cond: ast.expr, true_val: ast.expr,
                      false_val: ast.expr) -> ast.expr:
        """Translate a two-way PHI node to jnp.where.

        Args:
            cond: Condition expression
            true_val: Value when condition is True
            false_val: Value when condition is False

        Returns:
            jnp.where expression
        """
        return jnp_where(cond, true_val, false_val)

    def translate_multi_way_phi(self,
                                cases: List[Tuple[ast.expr, ast.expr]],
                                default: ast.expr) -> ast.expr:
        """Translate a multi-way PHI node to nested jnp.where.

        Args:
            cases: List of (condition, value) pairs
            default: Default value if no conditions match

        Returns:
            Nested jnp.where expression
        """
        return nested_where(cases, default)

    def jax_bool(self, value: bool) -> ast.Call:
        """Create a JIT-compatible boolean constant.

        Args:
            value: Python bool value

        Returns:
            jnp.bool_(value) expression
        """
        return jnp_bool(const(value))

    def inputs_subscript(self, index: Union[int, ast.expr]) -> ast.Subscript:
        """Create inputs[index] expression.

        Args:
            index: Integer index or expression

        Returns:
            subscript expression for inputs[index]
        """
        if isinstance(index, int):
            return subscript(name('inputs'), const(index))
        return subscript(name('inputs'), index)

    def analysis_comparison(self, type_code: int) -> ast.Compare:
        """Create comparison for analysis() function.

        Args:
            type_code: Analysis type code (0=dc, 1=ac, 2=tran, 3=noise, 4=nodeset)

        Returns:
            (inputs[-2] == type_code) expression
        """
        return compare(
            subscript(name('inputs'), unaryop(ast.USub(), const(2))),
            ast.Eq(),
            const(type_code)
        )
