"""Function builders for generating complete JAX functions.

This module provides builders for:
- Init functions (compute cache from parameters)
- Eval functions (compute residuals/Jacobian from cache + voltages)
"""

import ast
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from openvaf_ast import (
    name as ast_name, const as ast_const, binop, unaryop, compare,
    call as ast_call, attr, subscript, list_expr, tuple_expr,
    jnp_call, jnp_where, jnp_float64, jnp_bool,
    assign, assign_tuple, function_def, return_stmt,
    import_stmt, import_from, expr_stmt, pass_stmt,
)
from openvaf_ast.statements import build_module, fix_and_compile

from ..mir.types import MIRFunction, MIRInstruction, Block, parse_mir_function
from ..mir.cfg import CFGAnalyzer, LoopInfo
from ..mir.ssa import SSAAnalyzer, PHIResolutionType
from .context import CodeGenContext, SplitParamContext, build_context_from_mir
from .instruction import InstructionTranslator


class FunctionBuilder:
    """Base class for function builders."""

    def __init__(self, mir_func: MIRFunction):
        """Initialize builder with MIR function.

        Args:
            mir_func: Parsed MIR function
        """
        self.mir_func = mir_func
        self.cfg = CFGAnalyzer(mir_func)
        self.ssa = SSAAnalyzer(mir_func, self.cfg)

    def _emit_preamble(self, body: List[ast.stmt], ctx: CodeGenContext):
        """Emit function preamble: imports and constants."""
        # JAX imports: from jax import numpy as jnp, lax
        body.append(import_from('jax', [('numpy', 'jnp'), 'lax']))

        # Zero and one constants
        body.append(assign(ctx.zero_var, jnp_float64(ast_const(0.0))))
        body.append(assign(ctx.one_var, jnp_float64(ast_const(1.0))))
        ctx.defined_vars.add(ctx.zero_var)
        ctx.defined_vars.add(ctx.one_var)

    def _emit_constants(self, body: List[ast.stmt], ctx: CodeGenContext):
        """Emit constant definitions."""
        # Float constants
        for name, value in self.mir_func.constants.items():
            var_name = f"{ctx.var_prefix}{name}"
            if value == float('inf'):
                expr = attr(ast_name('jnp'), 'inf')  # jnp.inf (constant, not call)
            elif value == float('-inf'):
                expr = unaryop(ast.USub(), attr(ast_name('jnp'), 'inf'))
            elif value != value:  # NaN
                expr = attr(ast_name('jnp'), 'nan')  # jnp.nan (constant, not call)
            else:
                expr = ast_const(value)
            body.append(assign(var_name, expr))
            ctx.defined_vars.add(var_name)

        # Boolean constants
        for name, value in self.mir_func.bool_constants.items():
            var_name = f"{ctx.var_prefix}{name}"
            body.append(assign(var_name, jnp_bool(ast_const(value))))
            ctx.defined_vars.add(var_name)

        # Integer constants
        for name, value in self.mir_func.int_constants.items():
            var_name = f"{ctx.var_prefix}{name}"
            body.append(assign(var_name, ast_const(value)))
            ctx.defined_vars.add(var_name)

    def _emit_block(self, body: List[ast.stmt], ctx: CodeGenContext,
                    block: Block, translator: InstructionTranslator,
                    loop_info: Optional[LoopInfo] = None):
        """Emit code for a basic block."""
        for inst in block.instructions:
            if inst.is_terminator:
                continue

            expr = translator.translate(inst, loop_info)
            if expr and inst.result:
                var_name = ctx.define_var(inst.result)
                body.append(assign(var_name, expr))

    def _emit_loop(self, body: List[ast.stmt], ctx: CodeGenContext,
                   loop: LoopInfo, translator: InstructionTranslator) -> None:
        """Emit code for a loop using lax.while_loop."""
        header_block = self.mir_func.blocks.get(loop.header)
        if not header_block:
            return

        # Find PHI nodes in header
        phi_nodes = header_block.phi_nodes
        if not phi_nodes:
            # No loop-carried values, emit blocks linearly
            for block_name in sorted(loop.body):
                block = self.mir_func.blocks.get(block_name)
                if block:
                    self._emit_block(body, ctx, block, translator)
            return

        # Extract loop-carried state from PHIs
        loop_state: List[Tuple[str, str, str]] = []  # (result, init_val, update_val)
        for phi in phi_nodes:
            resolution = self.ssa.resolve_phi(phi, loop)
            if resolution.type == PHIResolutionType.LOOP_INIT:
                loop_state.append((
                    phi.result,
                    resolution.init_value,
                    resolution.update_value
                ))

        if not loop_state:
            # Couldn't extract loop state, emit linearly
            for block_name in sorted(loop.body):
                block = self.mir_func.blocks.get(block_name)
                if block:
                    self._emit_block(body, ctx, block, translator)
            return

        # Build initial state tuple
        init_vals = [ctx.get_operand(lc[1]) for lc in loop_state]
        init_state = tuple_expr(init_vals) if len(init_vals) > 1 else init_vals[0]

        # Find branch condition in header
        branch_cond = None
        header_term = header_block.terminator
        if header_term and header_term.is_branch:
            branch_cond = header_term.condition

        # Build condition function
        cond_body = self._build_loop_cond(ctx, loop, loop_state, translator, branch_cond)
        cond_fn = function_def('_loop_cond', ['_state'], cond_body)
        body.append(cond_fn)

        # Build body function
        loop_body_stmts = self._build_loop_body(ctx, loop, loop_state, translator)
        body_fn = function_def('_loop_body', ['_state'], loop_body_stmts)
        body.append(body_fn)

        # Call while_loop
        loop_call = ast_call(
            attr(ast_name('lax'), 'while_loop'),
            [ast_name('_loop_cond'), ast_name('_loop_body'), init_state]
        )
        body.append(assign('_loop_result', loop_call))

        # Unpack results
        for i, (result, _, _) in enumerate(loop_state):
            var_name = ctx.define_var(result)
            if len(loop_state) > 1:
                body.append(assign(var_name, subscript(ast_name('_loop_result'), ast_const(i))))
            else:
                body.append(assign(var_name, ast_name('_loop_result')))

    def _build_loop_cond(self, ctx: CodeGenContext, loop: LoopInfo,
                         loop_state: List[Tuple[str, str, str]],
                         translator: InstructionTranslator,
                         branch_cond: Optional[str]) -> List[ast.stmt]:
        """Build loop condition function body."""
        cond_body = []

        # Unpack state
        state_vars = [f"{ctx.var_prefix}{ls[0]}" for ls in loop_state]
        if len(state_vars) > 1:
            cond_body.append(assign_tuple(state_vars, ast_name('_state')))
        else:
            cond_body.append(assign(state_vars[0], ast_name('_state')))

        # Mark state vars as defined for this scope
        local_defined = set(state_vars)

        # Process header block instructions to compute condition
        header_block = self.mir_func.blocks.get(loop.header)
        if header_block:
            for inst in header_block.body_instructions:
                expr = translator.translate(inst)
                if expr and inst.result:
                    var_name = f"{ctx.var_prefix}{inst.result}"
                    cond_body.append(assign(var_name, expr))
                    local_defined.add(var_name)

        # Return condition
        if branch_cond:
            cond_var = f"{ctx.var_prefix}{branch_cond}"
            if cond_var in local_defined or cond_var in ctx.defined_vars:
                cond_body.append(return_stmt(ast_name(cond_var)))
            else:
                cond_body.append(return_stmt(ctx.get_operand(branch_cond)))
        else:
            cond_body.append(return_stmt(jnp_bool(ast_const(False))))

        return cond_body

    def _build_loop_body(self, ctx: CodeGenContext, loop: LoopInfo,
                         loop_state: List[Tuple[str, str, str]],
                         translator: InstructionTranslator) -> List[ast.stmt]:
        """Build loop body function."""
        loop_body = []

        # Unpack state
        state_vars = [f"{ctx.var_prefix}{ls[0]}" for ls in loop_state]
        if len(state_vars) > 1:
            loop_body.append(assign_tuple(state_vars, ast_name('_state')))
        else:
            loop_body.append(assign(state_vars[0], ast_name('_state')))

        # Mark state vars as defined
        local_defined = set(state_vars)

        # Process all blocks in loop
        for block_name in sorted(loop.body):
            block = self.mir_func.blocks.get(block_name)
            if not block:
                continue

            for inst in block.instructions:
                if inst.is_terminator or inst.is_phi:
                    continue
                expr = translator.translate(inst)
                if expr and inst.result:
                    var_name = f"{ctx.var_prefix}{inst.result}"
                    loop_body.append(assign(var_name, expr))
                    local_defined.add(var_name)

        # Build return tuple with updated values
        update_vals = []
        for result, _, update in loop_state:
            update_var = f"{ctx.var_prefix}{update}"
            if update_var in local_defined:
                update_vals.append(ast_name(update_var))
            else:
                # Fallback to operand resolution
                update_vals.append(ctx.get_operand(update))

        if len(update_vals) > 1:
            loop_body.append(return_stmt(tuple_expr(update_vals)))
        else:
            loop_body.append(return_stmt(update_vals[0]))

        return loop_body


class InitFunctionBuilder(FunctionBuilder):
    """Builder for init functions."""

    def __init__(self, mir_func: MIRFunction,
                 cache_mapping: List[Dict[str, Any]],
                 collapse_decision_outputs: List[Tuple[int, str]]):
        """Initialize init function builder.

        Args:
            mir_func: Parsed init MIR function
            cache_mapping: List of {init_value, eval_param} mappings
            collapse_decision_outputs: List of (pair_idx, value_name) tuples
        """
        super().__init__(mir_func)
        self.cache_mapping = cache_mapping
        self.collapse_decision_outputs = collapse_decision_outputs

    def build_simple(self, param_indices: List[int]) -> Tuple[str, List[str]]:
        """Build init function with single input array.

        Args:
            param_indices: List of param indices (usually range(n_params))

        Returns:
            Tuple of (function_name, code_lines)
        """
        # Create context
        ctx = build_context_from_mir(self.mir_func, var_prefix='')

        # Build function body
        body: List[ast.stmt] = []

        # Preamble
        self._emit_preamble(body, ctx)

        # Constants
        self._emit_constants(body, ctx)

        # Ensure v3 exists (commonly used for zero)
        if 'v3' not in ctx.defined_vars:
            body.append(assign('v3', ast_name(ctx.zero_var)))
            ctx.defined_vars.add('v3')

        # Map init params from input array
        self._emit_simple_param_mapping(body, ctx, param_indices)

        # Process blocks
        translator = InstructionTranslator(ctx, self.ssa)
        block_order = self.cfg.topological_order()

        for item in block_order:
            if isinstance(item, LoopInfo):
                self._emit_loop(body, ctx, item, translator)
            else:
                block = self.mir_func.blocks.get(item)
                if block:
                    self._emit_block(body, ctx, block, translator)

        # Build cache output array
        self._emit_cache_output(body, ctx)

        # Build collapse decisions
        self._emit_collapse_decisions(body, ctx)

        # Return statement
        body.append(return_stmt(tuple_expr([
            ast_name('cache'),
            ast_name('collapse_decisions')
        ])))

        # Build function
        func = function_def('init_fn', ['inputs'], body)

        # Compile to code
        module = build_module([func])
        ast.fix_missing_locations(module)
        code_str = ast.unparse(module)

        return 'init_fn', code_str.split('\n')

    def _emit_simple_param_mapping(self, body: List[ast.stmt],
                                    ctx: CodeGenContext,
                                    param_indices: List[int]):
        """Emit parameter mapping from single input array."""
        for init_idx, param in enumerate(self.mir_func.params):
            var_name = f"{ctx.var_prefix}{param}"

            if init_idx < len(param_indices):
                body.append(assign(var_name,
                    subscript(ast_name('inputs'), ast_const(init_idx))))
            else:
                # Fallback to zero
                body.append(assign(var_name, ast_name(ctx.zero_var)))

            ctx.defined_vars.add(var_name)

    def build_split(self, shared_indices: List[int],
                    varying_indices: List[int],
                    init_to_eval: List[int]) -> Tuple[str, List[str]]:
        """Build init function with split shared/device params.

        Args:
            shared_indices: Eval param indices that are constant across devices
            varying_indices: Eval param indices that vary per device
            init_to_eval: Mapping from init param index to eval param index

        Returns:
            Tuple of (function_name, code_lines)
        """
        # Build index mappings
        shared_set = set(shared_indices)
        varying_set = set(varying_indices)
        shared_to_pos = {idx: pos for pos, idx in enumerate(shared_indices)}
        varying_to_pos = {idx: pos for pos, idx in enumerate(varying_indices)}

        # Create context
        ctx = build_context_from_mir(self.mir_func, var_prefix='')

        # Build function body
        body: List[ast.stmt] = []

        # Preamble
        self._emit_preamble(body, ctx)

        # Constants
        self._emit_constants(body, ctx)

        # Ensure v3 exists (commonly used for zero)
        if 'v3' not in ctx.defined_vars:
            body.append(assign('v3', ast_name(ctx.zero_var)))
            ctx.defined_vars.add('v3')

        # Map init params from split arrays
        self._emit_split_param_mapping(body, ctx, shared_set, varying_set,
                                       shared_to_pos, varying_to_pos, init_to_eval)

        # Process blocks
        translator = InstructionTranslator(ctx, self.ssa)
        block_order = self.cfg.topological_order()

        for item in block_order:
            if isinstance(item, LoopInfo):
                self._emit_loop(body, ctx, item, translator)
            else:
                block = self.mir_func.blocks.get(item)
                if block:
                    self._emit_block(body, ctx, block, translator)

        # Build cache output array
        self._emit_cache_output(body, ctx)

        # Build collapse decisions
        self._emit_collapse_decisions(body, ctx)

        # Return statement
        body.append(return_stmt(tuple_expr([
            ast_name('cache'),
            ast_name('collapse_decisions')
        ])))

        # Build function
        func = function_def('init_fn_split',
                            ['shared_params', 'device_params'],
                            body)

        # Compile to code
        module = build_module([func])
        ast.fix_missing_locations(module)
        code_str = ast.unparse(module)

        return 'init_fn_split', code_str.split('\n')

    def _emit_split_param_mapping(self, body: List[ast.stmt],
                                   ctx: CodeGenContext,
                                   shared_set: Set[int],
                                   varying_set: Set[int],
                                   shared_to_pos: Dict[int, int],
                                   varying_to_pos: Dict[int, int],
                                   init_to_eval: List[int]):
        """Emit parameter mapping from split arrays."""
        for init_idx, param in enumerate(self.mir_func.params):
            eval_idx = init_to_eval[init_idx] if init_idx < len(init_to_eval) else -1
            var_name = f"{ctx.var_prefix}{param}"

            if eval_idx in shared_set:
                pos = shared_to_pos[eval_idx]
                body.append(assign(var_name,
                    subscript(ast_name('shared_params'), ast_const(pos))))
            elif eval_idx in varying_set:
                pos = varying_to_pos[eval_idx]
                body.append(assign(var_name,
                    subscript(ast_name('device_params'), ast_const(pos))))
            else:
                # Fallback to zero
                body.append(assign(var_name, ast_name(ctx.zero_var)))

            ctx.defined_vars.add(var_name)

    def _emit_cache_output(self, body: List[ast.stmt], ctx: CodeGenContext):
        """Emit cache array construction."""
        cache_vals: List[ast.expr] = []

        for mapping in self.cache_mapping:
            init_val = mapping['init_value']
            var_name = f"{ctx.var_prefix}{init_val}"
            if var_name in ctx.defined_vars or init_val in ctx.defined_vars:
                cache_vals.append(ast_name(var_name if var_name in ctx.defined_vars else init_val))
            else:
                cache_vals.append(ast_name(ctx.zero_var))

        if cache_vals:
            body.append(assign('cache', jnp_call('array', list_expr(cache_vals))))
        else:
            body.append(assign('cache', jnp_call('array', list_expr([]))))

    def _emit_collapse_decisions(self, body: List[ast.stmt], ctx: CodeGenContext):
        """Emit collapse decision array construction."""
        collapse_vals: List[ast.expr] = []

        for pair_idx, val_name in self.collapse_decision_outputs:
            if val_name.startswith('!'):
                actual_val = val_name[1:]
                negate = True
            else:
                actual_val = val_name
                negate = False

            var_name = f"{ctx.var_prefix}{actual_val}"
            if var_name in ctx.defined_vars or actual_val in ctx.defined_vars:
                val_expr = ast_name(var_name if var_name in ctx.defined_vars else actual_val)
                if negate:
                    val_expr = jnp_call('logical_not', val_expr)
                collapse_vals.append(ast_call(
                    attr(ast_name('jnp'), 'float32'), [val_expr]))
            else:
                # Default based on negation
                collapse_vals.append(ast_name(ctx.one_var if negate else ctx.zero_var))

        if collapse_vals:
            body.append(assign('collapse_decisions',
                jnp_call('array', list_expr(collapse_vals))))
        else:
            body.append(assign('collapse_decisions', jnp_call('array', list_expr([]))))


class EvalFunctionBuilder(FunctionBuilder):
    """Builder for eval functions."""

    def __init__(self, mir_func: MIRFunction,
                 dae_data: Dict[str, Any],
                 cache_mapping: List[Dict[str, Any]],
                 param_idx_to_val: Dict[int, str]):
        """Initialize eval function builder.

        Args:
            mir_func: Parsed eval MIR function
            dae_data: DAE system data (residuals, jacobian)
            cache_mapping: Cache slot to eval param mapping
            param_idx_to_val: Maps eval param index to value name
        """
        super().__init__(mir_func)
        self.dae_data = dae_data
        self.cache_mapping = cache_mapping
        self.param_idx_to_val = param_idx_to_val

    def build_with_cache_split(self, shared_indices: List[int],
                                varying_indices: List[int],
                                shared_cache_indices: Optional[List[int]] = None,
                                varying_cache_indices: Optional[List[int]] = None
                                ) -> Tuple[str, List[str]]:
        """Build eval function with split params and optional split cache.

        Args:
            shared_indices: Param indices constant across devices
            varying_indices: Param indices that vary per device
            shared_cache_indices: Cache indices constant across devices
            varying_cache_indices: Cache indices that vary per device

        Returns:
            Tuple of (function_name, code_lines)
        """
        use_cache_split = (shared_cache_indices is not None and
                           varying_cache_indices is not None)

        # Build index mappings
        idx_mapping: Dict[int, Tuple[str, int]] = {}
        for new_idx, orig_idx in enumerate(shared_indices):
            idx_mapping[orig_idx] = ('shared', new_idx)
        for new_idx, orig_idx in enumerate(varying_indices):
            idx_mapping[orig_idx] = ('device', new_idx)

        cache_idx_mapping: Dict[int, Tuple[str, int]] = {}
        if use_cache_split:
            for new_idx, orig_idx in enumerate(shared_cache_indices):
                cache_idx_mapping[orig_idx] = ('shared_cache', new_idx)
            for new_idx, orig_idx in enumerate(varying_cache_indices):
                cache_idx_mapping[orig_idx] = ('device_cache', new_idx)

        # Create context
        ctx = build_context_from_mir(self.mir_func, var_prefix='')

        # Build function body
        body: List[ast.stmt] = []

        # Preamble
        self._emit_preamble(body, ctx)

        # Constants
        self._emit_constants(body, ctx)

        # Ensure v3 exists
        if 'v3' not in ctx.defined_vars:
            body.append(assign('v3', ast_name(ctx.zero_var)))
            ctx.defined_vars.add('v3')

        # Map params from split arrays
        self._emit_param_mapping(body, ctx, idx_mapping)

        # Map cache values
        self._emit_cache_mapping(body, ctx, cache_idx_mapping, use_cache_split)

        # Process blocks
        translator = InstructionTranslator(ctx, self.ssa)
        block_order = self.cfg.topological_order()

        for item in block_order:
            if isinstance(item, LoopInfo):
                self._emit_loop(body, ctx, item, translator)
            else:
                block = self.mir_func.blocks.get(item)
                if block:
                    self._emit_block(body, ctx, block, translator)

        # Build output arrays
        self._emit_residual_arrays(body, ctx)
        self._emit_jacobian_arrays(body, ctx)
        self._emit_lim_rhs_arrays(body, ctx)

        # Return statement: (res_resist, res_react, jac_resist, jac_react, lim_rhs_resist, lim_rhs_react)
        body.append(return_stmt(tuple_expr([
            ast_name('residuals_resist'),
            ast_name('residuals_react'),
            ast_name('jacobian_resist'),
            ast_name('jacobian_react'),
            ast_name('lim_rhs_resist'),
            ast_name('lim_rhs_react'),
        ])))

        # Build function
        fn_name = 'eval_fn_with_cache_split_cache' if use_cache_split else 'eval_fn_with_cache_split'
        if use_cache_split:
            args = ['shared_params', 'device_params', 'shared_cache', 'device_cache']
        else:
            args = ['shared_params', 'device_params', 'cache']

        func = function_def(fn_name, args, body)

        # Compile to code
        module = build_module([func])
        ast.fix_missing_locations(module)
        code_str = ast.unparse(module)

        return fn_name, code_str.split('\n')

    def _emit_param_mapping(self, body: List[ast.stmt],
                             ctx: CodeGenContext,
                             idx_mapping: Dict[int, Tuple[str, int]]):
        """Emit parameter mapping from split arrays."""
        for i, param in enumerate(self.mir_func.params):
            var_name = f"{ctx.var_prefix}{param}"

            if i in idx_mapping:
                source, new_idx = idx_mapping[i]
                if source == 'shared':
                    body.append(assign(var_name,
                        subscript(ast_name('shared_params'), ast_const(new_idx))))
                else:
                    body.append(assign(var_name,
                        subscript(ast_name('device_params'), ast_const(new_idx))))
            else:
                # Fallback: derivative selector params default to 0
                body.append(assign(var_name, ast_const(0)))

            ctx.defined_vars.add(var_name)

    def _emit_cache_mapping(self, body: List[ast.stmt],
                             ctx: CodeGenContext,
                             cache_idx_mapping: Dict[int, Tuple[str, int]],
                             use_cache_split: bool):
        """Emit cache value mapping."""
        for cache_idx, mapping in enumerate(self.cache_mapping):
            eval_param_idx = mapping['eval_param']
            eval_val = self.param_idx_to_val.get(eval_param_idx, f"cached_{eval_param_idx}")
            var_name = f"{ctx.var_prefix}{eval_val}"

            if use_cache_split and cache_idx in cache_idx_mapping:
                source, new_idx = cache_idx_mapping[cache_idx]
                if source == 'shared_cache':
                    body.append(assign(var_name,
                        subscript(ast_name('shared_cache'), ast_const(new_idx))))
                else:
                    body.append(assign(var_name,
                        subscript(ast_name('device_cache'), ast_const(new_idx))))
            else:
                body.append(assign(var_name,
                    subscript(ast_name('cache'), ast_const(cache_idx))))

            ctx.defined_vars.add(var_name)

    def _emit_residual_arrays(self, body: List[ast.stmt], ctx: CodeGenContext):
        """Emit residual output arrays."""
        resist_exprs: List[ast.expr] = []
        react_exprs: List[ast.expr] = []

        for res in self.dae_data['residuals']:
            resist_var = self._mir_to_var(res['resist_var'], ctx)
            react_var = self._mir_to_var(res['react_var'], ctx)

            resist_exprs.append(
                ast_name(resist_var) if resist_var in ctx.defined_vars
                else ast_name(ctx.zero_var))
            react_exprs.append(
                ast_name(react_var) if react_var in ctx.defined_vars
                else ast_name(ctx.zero_var))

        body.append(assign('residuals_resist', jnp_call('array', list_expr(resist_exprs))))
        body.append(assign('residuals_react', jnp_call('array', list_expr(react_exprs))))

    def _emit_jacobian_arrays(self, body: List[ast.stmt], ctx: CodeGenContext):
        """Emit Jacobian output arrays."""
        resist_exprs: List[ast.expr] = []
        react_exprs: List[ast.expr] = []

        for entry in self.dae_data['jacobian']:
            resist_var = self._mir_to_var(entry['resist_var'], ctx)
            react_var = self._mir_to_var(entry['react_var'], ctx)

            resist_exprs.append(
                ast_name(resist_var) if resist_var in ctx.defined_vars
                else ast_name(ctx.zero_var))
            react_exprs.append(
                ast_name(react_var) if react_var in ctx.defined_vars
                else ast_name(ctx.zero_var))

        body.append(assign('jacobian_resist', jnp_call('array', list_expr(resist_exprs))))
        body.append(assign('jacobian_react', jnp_call('array', list_expr(react_exprs))))

    def _emit_lim_rhs_arrays(self, body: List[ast.stmt], ctx: CodeGenContext):
        """Emit limiting RHS correction arrays.

        These corrections are subtracted from residuals during Newton-Raphson iteration
        when limiting is applied. The formula is:
            lim_rhs = J(lim_x) * (lim_x - x)
        where lim_x is the limited voltage and x is the actual voltage.

        The corrected residual becomes:
            f_corrected = f_computed - lim_rhs
        """
        resist_exprs: List[ast.expr] = []
        react_exprs: List[ast.expr] = []

        for res in self.dae_data['residuals']:
            # Get lim_rhs variables if they exist
            resist_lim_rhs_var = self._mir_to_var(res.get('resist_lim_rhs_var', ''), ctx)
            react_lim_rhs_var = self._mir_to_var(res.get('react_lim_rhs_var', ''), ctx)

            resist_exprs.append(
                ast_name(resist_lim_rhs_var) if resist_lim_rhs_var in ctx.defined_vars
                else ast_name(ctx.zero_var))
            react_exprs.append(
                ast_name(react_lim_rhs_var) if react_lim_rhs_var in ctx.defined_vars
                else ast_name(ctx.zero_var))

        body.append(assign('lim_rhs_resist', jnp_call('array', list_expr(resist_exprs))))
        body.append(assign('lim_rhs_react', jnp_call('array', list_expr(react_exprs))))

    def _mir_to_var(self, mir_ref: str, ctx: CodeGenContext) -> str:
        """Convert MIR reference (e.g., 'mir_123') to variable name."""
        if mir_ref and mir_ref.startswith('mir_'):
            # Extract value ID
            val_id = mir_ref[4:]  # Remove 'mir_' prefix
            return f"{ctx.var_prefix}v{val_id}"
        return mir_ref or ''
