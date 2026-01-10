"""Instruction translator for MIR to JAX.

This module translates individual MIR instructions to JAX AST expressions.
"""

import ast
from typing import Dict, List, Optional, Callable, Tuple
import sys
import os

# Add parent directory to path to import openvaf_ast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from openvaf_ast import (
    name as ast_name, const as ast_const, binop, unaryop, compare,
    call as ast_call, attr, subscript,
    jnp_call, jnp_where, jnp_float64, jnp_bool,
    safe_divide, nested_where,
)

from ..mir.types import MIRInstruction
from ..mir.ssa import SSAAnalyzer, PHIResolution, PHIResolutionType
from .context import CodeGenContext


class InstructionTranslator:
    """Translates MIR instructions to JAX AST expressions.

    Uses dispatch tables for efficient opcode handling.
    """

    # Binary arithmetic operators: opcode -> ast operator class
    BINARY_ARITH_OPS: Dict[str, type] = {
        'fadd': ast.Add,
        'fsub': ast.Sub,
        'fmul': ast.Mult,
        'iadd': ast.Add,
        'isub': ast.Sub,
        'imul': ast.Mult,
        'idiv': ast.FloorDiv,
        'irem': ast.Mod,
    }

    # Bitwise operators
    BITWISE_OPS: Dict[str, type] = {
        'iand': ast.BitAnd,
        'ior': ast.BitOr,
        'ixor': ast.BitXor,
    }

    # Comparison operators
    COMPARE_OPS: Dict[str, type] = {
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

    # Unary jnp functions with same name
    UNARY_JNP_SAME = {'exp', 'floor', 'ceil', 'sin', 'cos', 'tan',
                      'sinh', 'cosh', 'tanh', 'hypot', 'abs'}

    # Unary jnp functions with different name
    UNARY_JNP_MAP = {
        'asin': 'arcsin',
        'acos': 'arccos',
        'atan': 'arctan',
        'asinh': 'arcsinh',
        'acosh': 'arccosh',
        'atanh': 'arctanh',
        'fabs': 'abs',
    }

    # Binary jnp functions
    BINARY_JNP_MAP = {
        'pow': 'power',
        'atan2': 'arctan2',
        'hypot': 'hypot',
        'fmin': 'minimum',
        'fmax': 'maximum',
    }

    def __init__(self, ctx: CodeGenContext, ssa: Optional[SSAAnalyzer] = None):
        """Initialize instruction translator.

        Args:
            ctx: Code generation context
            ssa: SSA analyzer for PHI resolution (optional)
        """
        self.ctx = ctx
        self.ssa = ssa

        # Track flags for special features
        self.uses_simparam_gmin = False
        self.uses_analysis = False

        # Analysis type codes
        self.analysis_type_map = {
            'dc': 0, 'static': 0,
            'ac': 1,
            'tran': 2, 'transient': 2,
            'noise': 3,
            'nodeset': 4,
        }

    def translate(self, inst: MIRInstruction,
                  loop_info=None) -> Optional[ast.expr]:
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
        if opcode == 'fdiv':
            return self._translate_fdiv(inst)

        # Unary negation
        if opcode == 'fneg':
            return self._translate_fneg(inst)
        if opcode == 'ineg':
            return self._translate_ineg(inst)

        # Boolean not
        if opcode == 'bnot':
            return self._translate_bnot(inst)

        # Square root (safe)
        if opcode == 'sqrt':
            return self._translate_sqrt(inst)

        # Natural log (safe)
        if opcode == 'ln':
            return self._translate_ln(inst)

        # Unary jnp functions
        if opcode in self.UNARY_JNP_SAME:
            return self._translate_unary_jnp(inst, opcode)
        if opcode in self.UNARY_JNP_MAP:
            return self._translate_unary_jnp(inst, self.UNARY_JNP_MAP[opcode])

        # Binary jnp functions
        if opcode in self.BINARY_JNP_MAP:
            return self._translate_binary_jnp(inst, self.BINARY_JNP_MAP[opcode])

        # Type casts
        if opcode == 'ifcast':  # int to float
            return self._translate_ifcast(inst)
        if opcode == 'ficast':  # float to int
            return self._translate_ficast(inst)
        if opcode == 'fbcast':  # float to bool
            return self._translate_fbcast(inst)
        if opcode == 'ibcast':  # int to bool
            return self._translate_ibcast(inst)
        if opcode == 'bfcast':  # bool to float
            return self._translate_bfcast(inst)
        if opcode == 'bicast':  # bool to int
            return self._translate_bicast(inst)

        # Optimization barrier (pass through)
        if opcode == 'optbarrier':
            return self._translate_optbarrier(inst)

        # Select (conditional)
        if opcode == 'select':
            return self._translate_select(inst)

        # Function calls
        if opcode == 'call':
            return self._translate_call(inst)

        # Copy (simple assignment)
        if opcode == 'copy':
            return self._translate_copy(inst)

        # Unknown opcode - return zero as fallback
        return self.ctx.zero()

    def _translate_phi(self, inst: MIRInstruction,
                       loop_info=None) -> ast.expr:
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
            return self.ctx.get_operand(res.single_value or '_ZERO')

        if res.type == PHIResolutionType.TWO_WAY:
            assert res.condition is not None
            assert res.true_value is not None
            assert res.false_value is not None
            cond = self.ctx.get_operand(res.condition)
            true_val = self.ctx.get_operand(res.true_value)
            false_val = self.ctx.get_operand(res.false_value)
            return jnp_where(cond, true_val, false_val)

        if res.type == PHIResolutionType.MULTI_WAY:
            assert res.cases is not None
            assert res.default is not None
            # Build nested where
            cases = []
            for cond_str, val_str in res.cases:
                # Handle negated conditions
                if cond_str.startswith('!'):
                    cond = jnp_call('logical_not',
                                    self.ctx.get_operand(cond_str[1:]))
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
        return jnp_call('logical_not', operand)

    def _translate_sqrt(self, inst: MIRInstruction) -> ast.expr:
        """Translate sqrt with safe clamping for negative inputs."""
        operand = self.ctx.get_operand(inst.operands[0])
        # Clamp to zero to avoid NaN
        clamped = jnp_call('maximum', operand, self.ctx.zero())
        return jnp_call('sqrt', clamped)

    def _translate_ln(self, inst: MIRInstruction) -> ast.expr:
        """Translate natural log with safe clamping."""
        operand = self.ctx.get_operand(inst.operands[0])
        # Clamp to small epsilon to avoid -inf
        small_eps = ast_const(1e-300)
        clamped = jnp_call('maximum', operand, small_eps)
        return jnp_call('log', clamped)

    def _translate_unary_jnp(self, inst: MIRInstruction,
                              func_name: str) -> ast.expr:
        """Translate unary jnp function."""
        operand = self.ctx.get_operand(inst.operands[0])
        return jnp_call(func_name, operand)

    def _translate_binary_jnp(self, inst: MIRInstruction,
                               func_name: str) -> ast.expr:
        """Translate binary jnp function."""
        left = self.ctx.get_operand(inst.operands[0])
        right = self.ctx.get_operand(inst.operands[1])
        return jnp_call(func_name, left, right)

    def _translate_ifcast(self, inst: MIRInstruction) -> ast.expr:
        """Translate int to float cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        return jnp_call('float64', operand)

    def _translate_ficast(self, inst: MIRInstruction) -> ast.expr:
        """Translate float to int cast."""
        operand = self.ctx.get_operand(inst.operands[0])
        floored = jnp_call('floor', operand)
        return ast_call(attr(ast_name('jnp'), 'int32'), [floored])

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
        return ast_call(attr(ast_name('jnp'), 'int32'), [operand])

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

        Handles special functions:
        - $simparam("gmin") -> inputs[-1]
        - analysis("type") -> comparison with inputs[-2]
        - ddt() -> returns charge directly (for reactive component)
        - ddx() -> returns 0 (spatial derivative not supported)
        - noise functions -> returns 0
        """
        func_name = inst.func_name or ''

        # $simparam
        if 'simparam' in func_name.lower():
            # Check if it's gmin
            if inst.operands and inst.operands[0] in self.ctx.str_constants:
                param_name = self.ctx.str_constants[inst.operands[0]]
                if param_name == 'gmin':
                    self.uses_simparam_gmin = True
                    return self.ctx.get_input_negative(1)
            return self.ctx.zero()

        # analysis()
        if 'analysis' in func_name.lower():
            self.uses_analysis = True
            if inst.operands and inst.operands[0] in self.ctx.str_constants:
                analysis_name = self.ctx.str_constants[inst.operands[0]]
                type_code = self.analysis_type_map.get(analysis_name.lower(), 0)
                # Return (inputs[-2] == type_code)
                analysis_input = self.ctx.get_input_negative(2)
                return compare(analysis_input, ast.Eq(), ast_const(type_code))
            return jnp_bool(ast_const(False))

        # ddt() - time derivative
        if 'ddt' in func_name.lower():
            # Return the charge directly - transient integration handles dQ/dt
            if inst.operands:
                return self.ctx.get_operand(inst.operands[0])
            return self.ctx.zero()

        # ddx() - spatial derivative (not supported)
        if 'ddx' in func_name.lower():
            return self.ctx.zero()

        # noise functions - return 0
        noise_funcs = {'white_noise', 'flicker_noise', 'noise_table'}
        if any(nf in func_name.lower() for nf in noise_funcs):
            return self.ctx.zero()

        # abs function
        if func_name.lower() == 'abs':
            if inst.operands:
                return jnp_call('abs', self.ctx.get_operand(inst.operands[0]))
            return self.ctx.zero()

        # min/max functions
        if func_name.lower() == 'min':
            if len(inst.operands) >= 2:
                return jnp_call('minimum',
                                self.ctx.get_operand(inst.operands[0]),
                                self.ctx.get_operand(inst.operands[1]))
        if func_name.lower() == 'max':
            if len(inst.operands) >= 2:
                return jnp_call('maximum',
                                self.ctx.get_operand(inst.operands[0]),
                                self.ctx.get_operand(inst.operands[1]))

        # Unknown function - return zero
        return self.ctx.zero()

    def _translate_copy(self, inst: MIRInstruction) -> ast.expr:
        """Translate copy (simple value forwarding)."""
        if inst.operands:
            return self.ctx.get_operand(inst.operands[0])
        return self.ctx.zero()
