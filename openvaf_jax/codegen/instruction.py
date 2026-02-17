"""Instruction translator for MIR to JAX.

This module translates individual MIR instructions to JAX AST expressions.
"""

import ast
import re
from typing import Dict, Optional

from ..mir.ssa import PHIResolution, PHIResolutionType, SSAAnalyzer
from ..mir.types import V_F_ZERO, MIRInstruction
from ..openvaf_ast import (
    attr,
    binop,
    compare,
    jnp_bool,
    jnp_call,
    jnp_where,
    nested_where,
    safe_divide,
    subscript,
    unaryop,
)
from ..openvaf_ast import (
    call as ast_call,
)
from ..openvaf_ast import (
    const as ast_const,
)
from ..openvaf_ast import (
    name as ast_name,
)
from .context import CodeGenContext


class InstructionTranslator:
    """Translates MIR instructions to JAX AST expressions.

    Uses dispatch tables for efficient opcode handling.
    """

    # Binary arithmetic operators: opcode -> ast operator class
    BINARY_ARITH_OPS: Dict[str, type] = {
        "fadd": ast.Add,
        "fsub": ast.Sub,
        "fmul": ast.Mult,
        "frem": ast.Mod,  # Float remainder
        "iadd": ast.Add,
        "isub": ast.Sub,
        "imul": ast.Mult,
        "idiv": ast.FloorDiv,
        "irem": ast.Mod,
    }

    # Bitwise operators
    BITWISE_OPS: Dict[str, type] = {
        "iand": ast.BitAnd,
        "ior": ast.BitOr,
        "ixor": ast.BitXor,
        "ishl": ast.LShift,
        "ishr": ast.RShift,
    }

    # Comparison operators
    COMPARE_OPS: Dict[str, type] = {
        "feq": ast.Eq,
        "fne": ast.NotEq,
        "flt": ast.Lt,
        "fgt": ast.Gt,
        "fle": ast.LtE,
        "fge": ast.GtE,
        "ieq": ast.Eq,
        "ine": ast.NotEq,
        "ilt": ast.Lt,
        "igt": ast.Gt,
        "ile": ast.LtE,
        "ige": ast.GtE,
        "beq": ast.Eq,
        "bne": ast.NotEq,
        "seq": ast.Eq,  # String equal
        "sne": ast.NotEq,  # String not equal
    }

    # Unary jnp functions with same name
    UNARY_JNP_SAME = {
        "exp",
        "floor",
        "ceil",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "hypot",
        "abs",
    }

    # Unary jnp functions with different name
    UNARY_JNP_MAP = {
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        "asinh": "arcsinh",
        "acosh": "arccosh",
        "atanh": "arctanh",
        "fabs": "abs",
    }

    # Binary jnp functions
    BINARY_JNP_MAP = {
        "pow": "power",
        "atan2": "arctan2",
        "hypot": "hypot",
        "fmin": "minimum",
        "fmax": "maximum",
    }

    def __init__(
        self,
        ctx: CodeGenContext,
        ssa: Optional[SSAAnalyzer] = None,
        emit_debug_prints: bool = False,
    ):
        """Initialize instruction translator.

        Args:
            ctx: Code generation context
            ssa: SSA analyzer for PHI resolution (optional)
            emit_debug_prints: If True, emit jax.debug.print for $display calls.
                              Default False because debug.print causes slow JIT tracing.
        """
        self.ctx = ctx
        self.ssa = ssa
        self.emit_debug_prints = emit_debug_prints

        # Track flags for special features
        self.uses_simparam_gmin = False  # Deprecated - use ctx.simparam_registry
        self.uses_analysis = False
        self.uses_display = False  # Whether $display is used
        self.simparam_warnings: list[str] = []  # Unknown $simparam names
        self.discontinuity_warnings: list[str] = []  # $discontinuity usage
        self.display_calls: list[tuple[str, list[str]]] = []  # (format_str, [var_names])

        # Analysis type codes
        self.analysis_type_map = {
            "dc": 0,
            "static": 0,
            "ac": 1,
            "tran": 2,
            "transient": 2,
            "noise": 3,
            "nodeset": 4,
        }

    def translate(self, inst: MIRInstruction, loop_info=None) -> Optional[ast.expr]:
        """Translate a MIR instruction to AST expression.

        Args:
            inst: The instruction to translate
            loop_info: LoopInfo if instruction is in a loop header

        Returns:
            AST expression, or None if instruction produces no value
        """
        opcode = inst.opcode.lower()

        # Skip terminators - they don't produce values
        if inst.is_terminator:
            return None

        # PHI nodes
        if inst.is_phi:
            return self._translate_phi(inst, loop_info)

        # Binary arithmetic
        if opcode in self.BINARY_ARITH_OPS:
            return self._translate_binary_arith(inst)

        # Bitwise ops
        if opcode in self.BITWISE_OPS:
            return self._translate_bitwise(inst)

        # Comparison ops
        if opcode in self.COMPARE_OPS:
            return self._translate_compare(inst)

        # Division (special handling for safety)
        if opcode == "fdiv":
            return self._translate_fdiv(inst)

        # Unary negation
        if opcode == "fneg":
            return self._translate_fneg(inst)
        if opcode == "ineg":
            return self._translate_ineg(inst)

        # Boolean not
        if opcode == "bnot":
            return self._translate_bnot(inst)

        # Integer bitwise not
        if opcode == "inot":
            return self._translate_inot(inst)

        # Square root (safe)
        if opcode == "sqrt":
            return self._translate_sqrt(inst)

        # Natural log (safe)
        if opcode == "ln":
            return self._translate_ln(inst)

        # Base-10 log
        if opcode == "log":
            return self._translate_log10(inst)

        # Ceiling log base 2
        if opcode == "clog2":
            return self._translate_clog2(inst)

        # Unary jnp functions
        if opcode in self.UNARY_JNP_SAME:
            return self._translate_unary_jnp(inst, opcode)
        if opcode in self.UNARY_JNP_MAP:
            return self._translate_unary_jnp(inst, self.UNARY_JNP_MAP[opcode])

        # Binary jnp functions
        if opcode in self.BINARY_JNP_MAP:
            return self._translate_binary_jnp(inst, self.BINARY_JNP_MAP[opcode])

        # Type casts
        if opcode == "ifcast":  # int to float
            return self._translate_ifcast(inst)
        if opcode == "ficast":  # float to int
            return self._translate_ficast(inst)
        if opcode == "fbcast":  # float to bool
            return self._translate_fbcast(inst)
        if opcode == "ibcast":  # int to bool
            return self._translate_ibcast(inst)
        if opcode == "bfcast":  # bool to float
            return self._translate_bfcast(inst)
        if opcode == "bicast":  # bool to int
            return self._translate_bicast(inst)

        # Optimization barrier (pass through)
        if opcode == "optbarrier":
            return self._translate_optbarrier(inst)

        # Select (conditional)
        if opcode == "select":
            return self._translate_select(inst)

        # Function calls
        if opcode == "call":
            return self._translate_call(inst)

        # Copy (simple assignment)
        if opcode == "copy":
            return self._translate_copy(inst)

        # Unknown opcode - raise error
        raise ValueError(f"Unknown opcode: {inst.opcode}")

    def _translate_phi(self, inst: MIRInstruction, loop_info=None) -> ast.expr:
        """Translate a PHI node using SSA analysis."""
        if self.ssa is None or inst.phi_operands is None:
            # Fallback: use first operand
            if inst.phi_operands:
                return self.ctx.get_operand(inst.phi_operands[0].value)
            return self.ctx.zero()

        resolution = self.ssa.resolve_phi(inst, loop_info)
        return self._apply_phi_resolution(resolution)

    def _apply_phi_resolution(self, res: PHIResolution) -> ast.expr:
        """Generate AST from PHI resolution."""
        if res.type == PHIResolutionType.FALLBACK:
            return self.ctx.get_operand(res.single_value or V_F_ZERO)

        if res.type == PHIResolutionType.TWO_WAY:
            assert res.condition is not None
            # Handle potentially negated conditions
            cond_str = res.condition
            if cond_str.startswith("!"):
                cond = jnp_call("logical_not", self.ctx.get_operand(cond_str[1:]))
            else:
                cond = self.ctx.get_operand(cond_str)

            # Support nested resolutions: branch can be either a value or another PHIResolution
            if res.nested_true is not None:
                true_val = self._apply_phi_resolution(res.nested_true)
            else:
                assert res.true_value is not None
                true_val = self.ctx.get_operand(res.true_value)

            if res.nested_false is not None:
                false_val = self._apply_phi_resolution(res.nested_false)
            else:
                assert res.false_value is not None
                false_val = self.ctx.get_operand(res.false_value)

            return jnp_where(cond, true_val, false_val)

        if res.type == PHIResolutionType.MULTI_WAY:
            assert res.cases is not None
            assert res.default is not None
            # Build nested where
            cases = []
            for cond_str, val_str in res.cases:
                # Handle negated conditions
                if cond_str.startswith("!"):
                    cond = jnp_call("logical_not", self.ctx.get_operand(cond_str[1:]))
                else:
                    cond = self.ctx.get_operand(cond_str)
                val = self.ctx.get_operand(val_str)
                cases.append((cond, val))
            default = self.ctx.get_operand(res.default)
            return nested_where(cases, default)

        if res.type in (PHIResolutionType.LOOP_INIT, PHIResolutionType.LOOP_UPDATE):
            assert res.init_value is not None
            # For loop PHIs, we return the init value
            # The update is handled by the loop body
            return self.ctx.get_operand(res.init_value)

        return self.ctx.zero()

    def _translate_binary_arith(self, inst: MIRInstruction) -> ast.expr:
        """Translate binary arithmetic operation."""
        op_class = self.BINARY_ARITH_OPS[inst.opcode.lower()]
        left = self.ctx.get_operand(inst.operands[0])
        right = self.ctx.get_operand(inst.operands[1])
        return binop(left, op_class(), right)

    def _translate_bitwise(self, inst: MIRInstruction) -> ast.expr:
        """Translate bitwise operation."""
        op_class = self.BITWISE_OPS[inst.opcode.lower()]
        left = self.ctx.get_operand(inst.operands[0])
        right = self.ctx.get_operand(inst.operands[1])
        return binop(left, op_class(), right)

    def _translate_compare(self, inst: MIRInstruction) -> ast.expr:
        """Translate comparison operation."""
        op_class = self.COMPARE_OPS[inst.opcode.lower()]
        left = self.ctx.get_operand(inst.operands[0])
        right = self.ctx.get_operand(inst.operands[1])
        return compare(left, op_class(), right)

    def _translate_fdiv(self, inst: MIRInstruction) -> ast.expr:
        """Translate division with safe divide for JAX compatibility."""
        dividend = self.ctx.get_operand(inst.operands[0])
        divisor = self.ctx.get_operand(inst.operands[1])
        return safe_divide(dividend, divisor, self.ctx.zero(), self.ctx.one())

    def _translate_fneg(self, inst: MIRInstruction) -> ast.expr:
        """Translate float negation."""
        operand = self.ctx.get_operand(inst.operands[0])
        return unaryop(ast.USub(), operand)

    def _translate_ineg(self, inst: MIRInstruction) -> ast.expr:
        """Translate integer negation."""
        operand = self.ctx.get_operand(inst.operands[0])
        return unaryop(ast.USub(), operand)

    def _translate_bnot(self, inst: MIRInstruction) -> ast.expr:
        """Translate boolean not (use jnp.logical_not for JIT)."""
        operand = self.ctx.get_operand(inst.operands[0])
        return jnp_call("logical_not", operand)

    def _translate_inot(self, inst: MIRInstruction) -> ast.expr:
        """Translate integer bitwise NOT."""
        operand = self.ctx.get_operand(inst.operands[0])
        return unaryop(ast.Invert(), operand)

    def _translate_sqrt(self, inst: MIRInstruction) -> ast.expr:
        """Translate sqrt with safe clamping for negative inputs."""
        operand = self.ctx.get_operand(inst.operands[0])
        # Clamp to zero to avoid NaN
        clamped = jnp_call("maximum", operand, self.ctx.zero())
        return jnp_call("sqrt", clamped)

    def _translate_ln(self, inst: MIRInstruction) -> ast.expr:
        """Translate natural log with safe clamping."""
        operand = self.ctx.get_operand(inst.operands[0])
        # Clamp to small epsilon to avoid -inf
        small_eps = ast_const(1e-300)
        clamped = jnp_call("maximum", operand, small_eps)
        return jnp_call("log", clamped)

    def _translate_log10(self, inst: MIRInstruction) -> ast.expr:
        """Translate base-10 log with safe clamping."""
        operand = self.ctx.get_operand(inst.operands[0])
        # Clamp to small epsilon to avoid -inf
        small_eps = ast_const(1e-300)
        clamped = jnp_call("maximum", operand, small_eps)
        return jnp_call("log10", clamped)

    def _translate_clog2(self, inst: MIRInstruction) -> ast.expr:
        """Translate ceiling of log base 2.

        Computes ceil(log2(x)) for integer x.
        """
        operand = self.ctx.get_operand(inst.operands[0])
        # log2(x) = log(x) / log(2)
        # Then ceil the result
        small_eps = ast_const(1e-300)
        clamped = jnp_call("maximum", operand, small_eps)
        log2_val = jnp_call("log2", clamped)
        return jnp_call("ceil", log2_val)

    def _translate_unary_jnp(self, inst: MIRInstruction, func_name: str) -> ast.expr:
        """Translate unary jnp function."""
        operand = self.ctx.get_operand(inst.operands[0])
        return jnp_call(func_name, operand)

    def _translate_binary_jnp(self, inst: MIRInstruction, func_name: str) -> ast.expr:
        """Translate binary jnp function."""
        left = self.ctx.get_operand(inst.operands[0])
        right = self.ctx.get_operand(inst.operands[1])
        return jnp_call(func_name, left, right)

    def _translate_ifcast(self, inst: MIRInstruction) -> ast.expr:
        """Translate int to float cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        return jnp_call("float64", operand)

    def _translate_ficast(self, inst: MIRInstruction) -> ast.expr:
        """Translate float to int cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        floored = jnp_call("floor", operand)
        return ast_call(attr(ast_name("jnp"), "int32"), [floored])

    def _translate_fbcast(self, inst: MIRInstruction) -> ast.expr:
        """Translate float to bool cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        return compare(operand, ast.NotEq(), self.ctx.zero())

    def _translate_ibcast(self, inst: MIRInstruction) -> ast.expr:
        """Translate int to bool cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        return compare(operand, ast.NotEq(), ast_const(0))

    def _translate_bfcast(self, inst: MIRInstruction) -> ast.expr:
        """Translate bool to float cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        return jnp_where(operand, ast_const(1.0), ast_const(0.0))

    def _translate_bicast(self, inst: MIRInstruction) -> ast.expr:
        """Translate bool to int cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        return ast_call(attr(ast_name("jnp"), "int32"), [operand])

    def _translate_optbarrier(self, inst: MIRInstruction) -> ast.expr:
        """Translate optimization barrier (pass through)."""
        if inst.operands:
            return self.ctx.get_operand(inst.operands[0])
        return self.ctx.zero()

    def _translate_select(self, inst: MIRInstruction) -> ast.expr:
        """Translate select (conditional) operation."""
        if len(inst.operands) >= 3:
            cond = self.ctx.get_operand(inst.operands[0])
            true_val = self.ctx.get_operand(inst.operands[1])
            false_val = self.ctx.get_operand(inst.operands[2])
            return jnp_where(cond, true_val, false_val)
        return self.ctx.zero()

    def _translate_call(self, inst: MIRInstruction) -> ast.expr:
        """Translate function call.

        Handles OpenVAF CallbackKind callbacks:
        - SimParam/SimParamOpt/SimParamStr: $simparam("name") -> simparams[idx]
        - Analysis: analysis("type") -> comparison with $analysis_type
        - TimeDerivative: ddt() -> returns charge (transient handles dQ/dt)
        - Derivative/NodeDerivative: ddx() -> returns 0 (not supported)
        - StoreLimit/BuiltinLimit: $limit -> passthrough (convergence help)
        - LimDiscontinuity: $discontinuity -> ignored for DC
        - WhiteNoise/FlickerNoise/NoiseTable: noise -> returns 0
        - Print: $display/$strobe -> jax.debug.print (disabled by default)
        - CollapseHint: node collapse -> no-op (handled at build time)
        - SetRetFlag: $finish/$stop -> no-op
        """
        func_name = inst.func_name or ""

        # Empty func_name indicates an inlined or optimized-away function
        # This can happen when function declarations don't resolve
        if not func_name:
            return self.ctx.zero()

        # $simparam - access via simparams array using dynamic registry
        # Supported simparams (VAMS-LRM Table 9-27): gmin, abstol, vntol, reltol, tnom, scale, shrink, imax
        if "simparam" in func_name.lower():
            if inst.operands and inst.operands[0] in self.ctx.str_constants:
                param_name = self.ctx.str_constants[inst.operands[0]]

                # Register the simparam and get its index
                simparam_idx = self.ctx.register_simparam(param_name)

                # Backward compatibility flag
                if param_name == "gmin":
                    self.uses_simparam_gmin = True

                # Return simparams[idx]
                return subscript(ast_name("simparams"), ast_const(simparam_idx))
            return self.ctx.zero()

        # $discontinuity (LRM 9.17.1) - hint for timestep control
        if "discontinuity" in func_name.lower():
            # Get the discontinuity order if provided
            order = -1
            if inst.operands:
                op = inst.operands[0]
                if isinstance(op, (int, float)):
                    order = int(op)
            if not self.discontinuity_warnings:
                self.discontinuity_warnings.append(
                    f"$discontinuity({order}) used - ignored for DC analysis"
                )
            return self.ctx.zero()  # No-op, but return something

        # $display / $strobe / $write / $debug - check if this is a display function
        # Only treat as display if func_name explicitly indicates it, or if first operand
        # is a format string WITH format specifiers (contains %)
        is_display_func = func_name and any(
            x in func_name.lower() for x in ("display", "strobe", "write", "debug")
        )
        if inst.operands and inst.operands[0] in self.ctx.str_constants:
            fmt_str = self.ctx.str_constants[inst.operands[0]]
            # Only treat as display call if:
            # 1. Function name indicates display/strobe/write/debug, OR
            # 2. Format string contains % (likely a format specifier)
            has_format_specifiers = "%" in fmt_str
            if is_display_func or has_format_specifiers:
                self.uses_display = True

                # Track for metadata even if not emitting
                value_operands = inst.operands[1:]
                var_names = [str(op) for op in value_operands]
                self.display_calls.append((fmt_str, var_names))

                # Skip actual debug.print emission if disabled (default)
                # jax.debug.print causes very slow JIT tracing
                if not self.emit_debug_prints:
                    return self.ctx.zero()

                # Get the value operands (skip the format string)
                value_exprs = [self.ctx.get_operand(op) for op in value_operands]

                # Convert Verilog-A format string to Python format
                # Use regex to handle format specifiers with width/precision like %12.5e, %18.10g
                # Pattern matches: % followed by optional width.precision, then type specifier
                # Types: g, e, f, E, G (float), d, i (int), s (string)
                py_fmt = re.sub(r"%[-+0 #]*(\d+)?(\.\d+)?[gGeEfFdis]", "{}", fmt_str)
                py_fmt = py_fmt.rstrip("\n")  # jax.debug.print adds newline

                # Build jax.debug.print(fmt, val1, val2, ...)
                fmt_const = ast.Constant(value=py_fmt)
                args = [fmt_const] + value_exprs

                debug_print = ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(value=ast_name("jax"), attr="debug", ctx=ast.Load()),
                        attr="print",
                        ctx=ast.Load(),
                    ),
                    args=args,
                    keywords=[ast.keyword(arg="ordered", value=ast.Constant(value=True))],
                )

                # jax.debug.print returns None, so we use 'or' to execute it
                # and return 0.0: (jax.debug.print(...) or 0.0)
                # None is falsy, so this evaluates print, gets None, then returns 0.0
                return ast.BoolOp(op=ast.Or(), values=[debug_print, self.ctx.zero()])

        # analysis() - access via simparams array using registered $analysis_type
        # $analysis_type: 0=DC, 1=AC, 2=transient, 3=noise
        if "analysis" in func_name.lower():
            self.uses_analysis = True
            if inst.operands and inst.operands[0] in self.ctx.str_constants:
                analysis_name = self.ctx.str_constants[inst.operands[0]]
                type_code = self.analysis_type_map.get(analysis_name.lower(), 0)
                # $analysis_type is pre-registered at index 0
                analysis_idx = self.ctx.get_simparam_index("$analysis_type")
                # Return (simparams[analysis_idx] == type_code)
                analysis_input = subscript(ast_name("simparams"), ast_const(analysis_idx))
                return compare(analysis_input, ast.Eq(), ast_const(type_code))
            return jnp_bool(ast_const(False))

        # ddt() - time derivative
        if "ddt" in func_name.lower():
            # Return the charge directly - transient integration handles dQ/dt
            if inst.operands:
                return self.ctx.get_operand(inst.operands[0])
            return self.ctx.zero()

        # ddx() - spatial derivative (not supported)
        if "ddx" in func_name.lower():
            return self.ctx.zero()

        # noise functions - return 0
        noise_funcs = {"white_noise", "flicker_noise", "noise_table"}
        if any(nf in func_name.lower() for nf in noise_funcs):
            return self.ctx.zero()

        # abs function
        if func_name.lower() == "abs":
            if inst.operands:
                return jnp_call("abs", self.ctx.get_operand(inst.operands[0]))
            return self.ctx.zero()

        # min/max functions
        if func_name.lower() == "min":
            if len(inst.operands) >= 2:
                return jnp_call(
                    "minimum",
                    self.ctx.get_operand(inst.operands[0]),
                    self.ctx.get_operand(inst.operands[1]),
                )
        if func_name.lower() == "max":
            if len(inst.operands) >= 2:
                return jnp_call(
                    "maximum",
                    self.ctx.get_operand(inst.operands[0]),
                    self.ctx.get_operand(inst.operands[1]),
                )

        # $limit related functions - StoreLimit, LoadLimit, BuiltinLimit
        # These are used for Newton-Raphson convergence help.
        # The MIR has: StoreLimit(lim_stateN), LoadLimit(lim_stateN), $limit[pnjlim], etc.
        #
        # When use_limit_functions is True:
        #   - BuiltinLimit generates calls to limit_funcs['pnjlim'](vnew, vold, vt, vcrit)
        #   - StoreLimit writes to limit_state_out[idx] and returns the value
        #   - LoadLimit reads from limit_state_in[idx]
        #
        # When use_limit_functions is False (default):
        #   - All limit operations pass through the input voltage unchanged
        if "storelimit" in func_name.lower():
            if self.ctx.use_limit_functions:
                return self._translate_store_limit(func_name, inst)
            # Pass through the input voltage when limiting disabled
            if inst.operands:
                return self.ctx.get_operand(inst.operands[0])
            return self.ctx.zero()

        if "loadlimit" in func_name.lower():
            if self.ctx.use_limit_functions:
                return self._translate_load_limit(func_name, inst)
            # Pass through the input voltage when limiting disabled
            if inst.operands:
                return self.ctx.get_operand(inst.operands[0])
            return self.ctx.zero()

        # $limit[pnjlim], $limit[fetlim], etc. - BuiltinLimit callbacks
        # These implement NR convergence algorithms (SPICE pnjlim/fetlim).
        if "$limit" in func_name.lower() or "builtinlimit" in func_name.lower():
            if inst.operands:
                # Check if we should generate limit function calls
                if self.ctx.use_limit_functions:
                    return self._translate_builtin_limit(func_name, inst)
                # Default: pass through first operand (vnew)
                return self.ctx.get_operand(inst.operands[0])
            return self.ctx.zero()

        # SimParamOpt - $simparam("name", default) - with optional default
        if "simparam_opt" in func_name.lower():
            if inst.operands and inst.operands[0] in self.ctx.str_constants:
                param_name = self.ctx.str_constants[inst.operands[0]]
                simparam_idx = self.ctx.register_simparam(param_name)
                return subscript(ast_name("simparams"), ast_const(simparam_idx))
            # If param name unknown, return default (second operand) if available
            if len(inst.operands) >= 2:
                return self.ctx.get_operand(inst.operands[1])
            return self.ctx.zero()

        # SimParamStr - string $simparam (rare)
        if "simparam_str" in func_name.lower():
            # String simparams not supported - return 0
            return self.ctx.zero()

        # CollapseHint - node collapse hints (handled at model build time)
        if "collapse" in func_name.lower():
            # No-op for eval - node collapse is structural
            return self.ctx.zero()

        # ParamInfo - parameter bounds validation (set_MinInclusive, etc.)
        if func_name.lower().startswith("set_"):
            # No-op for eval - parameter validation is done at model build
            return self.ctx.zero()

        # SetRetFlag - $finish, $stop, $abort
        # These should terminate simulation but for eval we just continue
        if "setretflag" in func_name.lower():
            # TODO: Track these and potentially stop iteration
            return self.ctx.zero()

        # Unknown function - log warning and return zero
        # This helps identify new CallbackKind values we need to handle
        import warnings

        warnings.warn(f"Unknown MIR function '{func_name}' - returning zero", stacklevel=2)
        return self.ctx.zero()

    def _translate_builtin_limit(self, func_name: str, inst: MIRInstruction) -> ast.expr:
        """Translate BuiltinLimit to a limit function call.

        Parses the limit function name from MIR and generates a call like:
            limit_funcs['pnjlim'](vnew, vold, vt, vcrit)

        BuiltinLimit MIR format:
            BuiltinLimit { name: Spur(N), num_args: 4 }
        where Spur(N) is an interned string reference to the limit name.

        Standard SPICE limit functions:
            - pnjlim(vnew, vold, vt, vcrit) - PN junction limiting
            - fetlim(vnew, vold, vto) - FET gate limiting

        Args:
            func_name: The MIR function name (e.g., "BuiltinLimit { name: Spur(1), num_args: 4 }")
            inst: The MIR instruction with operands

        Returns:
            AST expression for the limit function call
        """
        # Parse limit function name from MIR func_name
        # Format: "BuiltinLimit { name: Spur(N), num_args: M }"
        # The actual name (pnjlim, fetlim) is stored in str_constants
        limit_name = "pnjlim"  # Default

        # Try to extract the interned string reference
        # Look for patterns like "Spur(1)" or just the limit name directly
        import re

        spur_match = re.search(r"Spur\((\d+)\)", func_name)
        if spur_match:
            spur_idx = int(spur_match.group(1))
            # Look up in string constants (the index is the string ID)
            for str_id, str_val in self.ctx.str_constants.items():
                # String IDs are like 's1', 's2', etc.
                if str_id.startswith("s") and str_id[1:].isdigit():
                    if int(str_id[1:]) == spur_idx:
                        limit_name = str_val.lower()
                        break

        # Also check for direct name match
        for known_limit in ["pnjlim", "fetlim", "limexp", "limvds"]:
            if known_limit in func_name.lower():
                limit_name = known_limit
                break

        # Build the function call
        # limit_funcs['pnjlim'](vnew, vold, vt, vcrit)
        limit_func = subscript(ast_name("limit_funcs"), ast_const(limit_name))

        # Get operands
        # For pnjlim: (vnew, vold, vt, vcrit)
        # For fetlim: (vnew, vold, vto)
        args = [self.ctx.get_operand(op) for op in inst.operands]

        # Track which V_raw feeds into this BuiltinLimit result
        # Used by StoreLimit to build limit_state_idx -> V_raw mapping
        if inst.result and inst.operands:
            self.ctx.builtin_limit_raw[str(inst.result)] = str(inst.operands[0])

        return ast_call(limit_func, args)

    def _parse_limit_state_index(self, func_name: str) -> int:
        """Parse limit state index from StoreLimit/LoadLimit MIR function name.

        MIR format examples:
            "StoreLimit { state: LimState(0) }"
            "LoadLimit { state: LimState(1) }"

        Args:
            func_name: The MIR function name

        Returns:
            Limit state index (0, 1, 2, ...), or 0 as default
        """
        import re

        # Look for LimState(N) pattern
        match = re.search(r"LimState\((\d+)\)", func_name)
        if match:
            return int(match.group(1))
        # Fallback: look for any number
        match = re.search(r"(\d+)", func_name)
        if match:
            return int(match.group(1))
        return 0

    def _translate_store_limit(self, func_name: str, inst: MIRInstruction) -> ast.expr:
        """Translate StoreLimit to store value in limit_state_out.

        StoreLimit stores the limited voltage value to be used as vold
        in the next NR iteration.

        Args:
            func_name: MIR function name containing the limit state index
            inst: The MIR instruction with the value to store

        Returns:
            AST expression that stores to limit_state_out and returns the value
        """
        idx = self._parse_limit_state_index(func_name)

        # Register this limit state
        state_name = f"lim_state{idx}"
        self.ctx.register_limit(state_name)

        # Get the value to store (first operand)
        if not inst.operands:
            return self.ctx.zero()

        # Track this store for building limit_state_out later
        # We store the MIR operand ID so we can reference it in function_builder
        operand_id = inst.operands[0]
        self.ctx.limit_store_operands[idx] = operand_id

        # Link this LimState to the raw voltage that was limited
        # The operand to StoreLimit is V_lim (result of BuiltinLimit).
        # Look up the V_raw that was the first operand to BuiltinLimit.
        v_lim_id = str(operand_id)
        v_raw_id = self.ctx.builtin_limit_raw.get(v_lim_id)
        if v_raw_id:
            self.ctx.limit_to_raw_operand[idx] = v_raw_id

        # Get and return the value expression (StoreLimit passes through)
        value_expr = self.ctx.get_operand(operand_id)
        return value_expr

    def _translate_load_limit(self, func_name: str, inst: MIRInstruction) -> ast.expr:
        """Translate LoadLimit to read from limit_state_in.

        LoadLimit retrieves the previous NR iteration's limited voltage
        to be used as vold in BuiltinLimit.

        Args:
            func_name: MIR function name containing the limit state index
            inst: The MIR instruction (operands not used for load)

        Returns:
            AST expression reading from limit_state_in[idx]
        """
        idx = self._parse_limit_state_index(func_name)

        # Register this limit state
        state_name = f"lim_state{idx}"
        self.ctx.register_limit(state_name)

        # Return limit_state_in[idx]
        return subscript(ast_name("limit_state_in"), ast_const(idx))

    def _translate_copy(self, inst: MIRInstruction) -> ast.expr:
        """Translate copy (simple value forwarding)."""
        if inst.operands:
            return self.ctx.get_operand(inst.operands[0])
        return self.ctx.zero()
