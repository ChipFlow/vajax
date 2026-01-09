"""JAX emitter for OpenVAF MIR.

This module generates JAX-traceable functions from OpenVAF MIR using a
branchless execution model suitable for GPU acceleration.

Key design principles:
1. Execute ALL instructions unconditionally (no actual branching)
2. Track block reachability via float arrays (0.0/1.0)
3. PHI nodes select values using jnp.where based on predecessor reachability
4. Branches update reachability masks but don't skip instructions

This converts control flow to data flow, making the code GPU-friendly.
"""

from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import lax


# Pre-allocated MIR constants (always present, indices 0-7)
# See docs/reference/openvaf/OPENVAF_MIR_CONSTANTS.md
PREALLOCATED_CONSTANTS = {
    'v0': 0.0,    # GRAVESTONE - dead value placeholder
    'v1': 0.0,    # FALSE (as float for uniform array type)
    'v2': 1.0,    # TRUE
    'v3': 0.0,    # F_ZERO
    'v4': 0.0,    # ZERO (integer as float)
    'v5': 1.0,    # ONE
    'v6': 1.0,    # F_ONE
    'v7': -1.0,   # F_N_ONE
}


# Safe math operations that handle edge cases common in circuit simulation

def safe_add(a, b):
    """Addition with inf + (-inf) = 0 handling."""
    result = a + b
    is_indeterminate = jnp.isnan(result) & jnp.isinf(a) & jnp.isinf(b)
    return jnp.where(is_indeterminate, 0.0, result)


def safe_sub(a, b):
    """Subtraction with inf - inf = 0 handling."""
    result = a - b
    is_indeterminate = jnp.isnan(result) & jnp.isinf(a) & jnp.isinf(b)
    return jnp.where(is_indeterminate, 0.0, result)


def safe_mul(a, b):
    """Multiplication with 0 * inf = 0 handling."""
    result = a * b
    is_zero_times_inf = jnp.isnan(result) & ((a == 0) | (b == 0))
    return jnp.where(is_zero_times_inf, 0.0, result)


def safe_div(a, b):
    """Division with 0/0 = 0 handling."""
    return jnp.where(
        b != 0, a / b,
        jnp.where(a > 0, jnp.inf,
                  jnp.where(a < 0, -jnp.inf, 0.0))
    )


def safe_ln(x):
    """Logarithm with non-positive handling."""
    return jnp.log(jnp.maximum(x, 1e-300))


def safe_exp(x):
    """Exponential with overflow protection."""
    return jnp.exp(jnp.clip(x, -700, 700))


def safe_sqrt(x):
    """Square root with negative handling."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


def safe_pow(base, exp):
    """Power with negative base handling."""
    return jnp.power(jnp.maximum(base, 0.0), exp)


# Opcode mappings for the instruction interpreter
BINARY_OPS = {
    'fadd': safe_add,
    'fsub': safe_sub,
    'fmul': safe_mul,
    'fdiv': safe_div,
    'flt': lambda a, b: jnp.float64(a < b),
    'fle': lambda a, b: jnp.float64(a <= b),
    'fgt': lambda a, b: jnp.float64(a > b),
    'fge': lambda a, b: jnp.float64(a >= b),
    'feq': lambda a, b: jnp.float64(a == b),
    'fne': lambda a, b: jnp.float64(a != b),
    'ilt': lambda a, b: jnp.float64(a < b),
    'ile': lambda a, b: jnp.float64(a <= b),
    'igt': lambda a, b: jnp.float64(a > b),
    'ige': lambda a, b: jnp.float64(a >= b),
    'ieq': lambda a, b: jnp.float64(a == b),
    'ine': lambda a, b: jnp.float64(a != b),
    'pow': safe_pow,
    'atan2': jnp.arctan2,
    'hypot': jnp.hypot,
    'min': jnp.minimum,
    'max': jnp.maximum,
}

UNARY_OPS = {
    'fneg': lambda a: -a,
    'ineg': lambda a: -a,
    'bnot': lambda a: jnp.float64(a == 0.0),
    'inot': lambda a: jnp.float64(a == 0.0),
    'ln': safe_ln,
    'exp': safe_exp,
    'sqrt': safe_sqrt,
    'log': lambda x: jnp.log10(jnp.maximum(x, 1e-300)),
    'abs': jnp.abs,
    'floor': jnp.floor,
    'ceil': jnp.ceil,
    'sin': jnp.sin,
    'cos': jnp.cos,
    'tan': jnp.tan,
    'asin': jnp.arcsin,
    'acos': jnp.arccos,
    'atan': jnp.arctan,
    'sinh': jnp.sinh,
    'cosh': jnp.cosh,
    'tanh': jnp.tanh,
    'asinh': jnp.arcsinh,
    'acosh': jnp.arccosh,
    'atanh': jnp.arctanh,
    'ifcast': lambda x: jnp.float64(x),
    'ficast': lambda x: jnp.float64(jnp.int32(x)),
    'bfcast': lambda x: jnp.float64(x),
    'bicast': lambda x: jnp.float64(jnp.int32(x)),
    'ibcast': lambda x: jnp.float64(x != 0),
    'fbcast': lambda x: jnp.float64(x != 0.0),
    'optbarrier': lambda x: x,
}


@dataclass
class EmitContext:
    """Context for code emission containing all metadata."""
    # Variable index mapping
    var_to_idx: Dict[str, int]
    n_vars: int

    # Block index mapping
    block_to_idx: Dict[str, int]
    n_blocks: int

    # Constant array (pre-computed)
    const_array: jnp.ndarray

    # Param indices for input loading
    param_indices: List[int]
    n_params: int

    # Cache mappings
    cache_to_var_idx: Dict[int, int]

    # Output indices
    resist_res_indices: List[int]
    react_res_indices: List[int]
    resist_jac_indices: List[int]
    react_jac_indices: List[int]

    # Function declarations for call handling
    function_decls: Dict[str, Dict]


def _build_emit_context(module) -> EmitContext:
    """Build emission context from module metadata."""
    eval_mir = module.get_mir_instructions()
    metadata = module.get_codegen_metadata()

    constants = eval_mir['constants']
    params = eval_mir.get('params', [])
    instructions = eval_mir['instructions']
    blocks_info = eval_mir['blocks']
    function_decls = eval_mir.get('function_decls', {})

    residuals_meta = metadata['residuals']
    jacobian_meta = metadata['jacobian']
    cache_info = metadata['cache_info']

    # Collect all variables for indexing
    all_vars = set()
    for i in range(8):
        all_vars.add(f'v{i}')
    all_vars.update(constants.keys())
    all_vars.update(params)
    for inst in instructions:
        result = inst.get('result')
        if result:
            all_vars.add(result)
        for op in inst.get('operands', []):
            if op:
                all_vars.add(op)
        for phi_op in inst.get('phi_operands', []):
            val = phi_op.get('value')
            if val:
                all_vars.add(val)
    for ci in cache_info:
        all_vars.add(ci['eval_param'])

    var_list = sorted(all_vars)
    var_to_idx = {v: i for i, v in enumerate(var_list)}
    n_vars = len(var_list)

    # Build constant array
    import numpy as np
    const_vals = np.zeros(n_vars, dtype=np.float64)

    # Pre-allocated constants
    for var, val in PREALLOCATED_CONSTANTS.items():
        if var in var_to_idx:
            const_vals[var_to_idx[var]] = val

    # MIR constants
    for var, val in constants.items():
        if val is not None and var in var_to_idx:
            fval = float(val) if isinstance(val, (int, float)) else 0.0
            if isinstance(fval, float) and np.isnan(fval):
                fval = 0.0  # Sanitize NaN constants
            const_vals[var_to_idx[var]] = fval

    const_array = jnp.array(const_vals)

    # Param indices
    param_indices = [var_to_idx[p] for p in params]

    # Cache mappings - eval_param is variable name like "v2616"
    cache_to_var_idx = {}
    for ci in cache_info:
        eval_param = ci['eval_param']
        if eval_param in var_to_idx:
            cache_to_var_idx[ci['cache_idx']] = var_to_idx[eval_param]

    # Output indices
    resist_res_indices = [var_to_idx.get(r['resist_var'], 0) for r in residuals_meta]
    react_res_indices = [var_to_idx.get(r['react_var'], 0) for r in residuals_meta]
    resist_jac_indices = [var_to_idx.get(j['resist_var'], 0) for j in jacobian_meta]
    react_jac_indices = [var_to_idx.get(j['react_var'], 0) for j in jacobian_meta]

    # Block index mapping
    block_to_idx = {name: i for i, name in enumerate(blocks_info.keys())}

    return EmitContext(
        var_to_idx=var_to_idx,
        n_vars=n_vars,
        block_to_idx=block_to_idx,
        n_blocks=len(blocks_info),
        const_array=const_array,
        param_indices=param_indices,
        n_params=len(params),
        cache_to_var_idx=cache_to_var_idx,
        resist_res_indices=resist_res_indices,
        react_res_indices=react_res_indices,
        resist_jac_indices=resist_jac_indices,
        react_jac_indices=react_jac_indices,
        function_decls=function_decls,
    )


@dataclass
class CompiledInstruction:
    """Pre-compiled instruction for efficient execution."""
    opcode: str
    result_idx: int
    block_idx: int
    operand_indices: List[int] = field(default_factory=list)

    # For PHI nodes: list of (value_idx, predecessor_block_idx)
    phi_operands: List[Tuple[int, int]] = field(default_factory=list)

    # For branches: (condition_idx, true_block_idx, false_block_idx)
    branch_info: Optional[Tuple[int, int, int]] = None

    # For jumps: destination_block_idx
    jump_dest: Optional[int] = None

    # For calls: function name
    func_name: str = ""


def _compile_instructions(module, ctx: EmitContext) -> List[CompiledInstruction]:
    """Compile MIR instructions into efficient form."""
    eval_mir = module.get_mir_instructions()
    instructions = eval_mir['instructions']

    compiled = []
    for inst in instructions:
        opcode = inst['opcode']
        result_idx = ctx.var_to_idx.get(inst.get('result'), -1)
        block_idx = ctx.block_to_idx.get(inst.get('block', 'default'), 0)

        ci = CompiledInstruction(
            opcode=opcode,
            result_idx=result_idx,
            block_idx=block_idx,
        )

        if opcode == 'phi':
            for phi_op in inst.get('phi_operands', []):
                val_idx = ctx.var_to_idx.get(phi_op.get('value'), 0)
                pred_idx = ctx.block_to_idx.get(phi_op.get('block'), 0)
                ci.phi_operands.append((val_idx, pred_idx))

        elif opcode == 'br':
            cond_idx = ctx.var_to_idx.get(inst.get('condition'), 0)
            true_idx = ctx.block_to_idx.get(inst.get('true_block'), 0)
            false_idx = ctx.block_to_idx.get(inst.get('false_block'), 0)
            ci.branch_info = (cond_idx, true_idx, false_idx)

        elif opcode == 'jmp':
            ci.jump_dest = ctx.block_to_idx.get(inst.get('destination'), 0)

        elif opcode == 'call':
            func_ref = inst.get('func_ref', '')
            ci.func_name = ctx.function_decls.get(func_ref, {}).get('name', '')
            ci.operand_indices = [ctx.var_to_idx.get(op, 0) for op in inst.get('operands', [])]

        else:
            ci.operand_indices = [ctx.var_to_idx.get(op, 0) for op in inst.get('operands', [])]

        compiled.append(ci)

    return compiled


def emit_eval(module) -> Callable:
    """Generate JAX eval function from OpenVAF module.

    Uses branchless execution model:
    - Execute ALL instructions unconditionally
    - Track block reachability via float array
    - PHI nodes select values using jnp.where
    - Branches update reachability masks

    Args:
        module: openvaf_py VaModule

    Returns:
        eval_fn(input_array, cache_array) -> ((resist_res, react_res), (resist_jac, react_jac))
    """
    ctx = _build_emit_context(module)
    compiled = _compile_instructions(module, ctx)

    # Convert to static arrays for closure
    const_array = ctx.const_array
    param_indices = list(ctx.param_indices)
    n_params = ctx.n_params
    cache_mappings = list(ctx.cache_to_var_idx.items())
    n_blocks = ctx.n_blocks

    resist_res_idx = jnp.array(ctx.resist_res_indices, dtype=jnp.int32)
    react_res_idx = jnp.array(ctx.react_res_indices, dtype=jnp.int32)
    resist_jac_idx = jnp.array(ctx.resist_jac_indices, dtype=jnp.int32)
    react_jac_idx = jnp.array(ctx.react_jac_indices, dtype=jnp.int32)

    def eval_fn(input_array: jnp.ndarray, cache: jnp.ndarray):
        """Evaluate device model using branchless execution."""
        # Initialize values from constants
        vals = const_array.copy()

        # Load params from input array
        for i, param_idx in enumerate(param_indices):
            if i < input_array.shape[0]:
                vals = vals.at[param_idx].set(input_array[i])

        # Override with cache values
        for cache_idx, var_idx in cache_mappings:
            if cache_idx < cache.shape[0]:
                vals = vals.at[var_idx].set(cache[cache_idx])

        # Block reachability tracking
        # Entry block (block 0) is always reached
        block_reached = jnp.zeros(n_blocks).at[0].set(1.0)

        # Execute all instructions (branchless)
        for inst in compiled:
            opcode = inst.opcode

            if opcode == 'phi':
                if inst.result_idx < 0 or not inst.phi_operands:
                    continue

                # Select value from predecessor that was reached
                first_val_idx, first_pred_idx = inst.phi_operands[0]
                result = vals[first_val_idx]

                for val_idx, pred_idx in inst.phi_operands:
                    pred_reached = block_reached[pred_idx]
                    result = jnp.where(pred_reached > 0.5, vals[val_idx], result)

                vals = vals.at[inst.result_idx].set(result)

            elif opcode == 'br':
                if inst.branch_info is None:
                    continue
                cond_idx, true_idx, false_idx = inst.branch_info
                this_reached = block_reached[inst.block_idx]
                cond = vals[cond_idx]

                # Propagate reachability
                true_reaches = this_reached * cond
                false_reaches = this_reached * (1.0 - cond)

                block_reached = block_reached.at[true_idx].set(
                    jnp.maximum(block_reached[true_idx], true_reaches)
                )
                block_reached = block_reached.at[false_idx].set(
                    jnp.maximum(block_reached[false_idx], false_reaches)
                )

            elif opcode == 'jmp':
                if inst.jump_dest is None:
                    continue
                this_reached = block_reached[inst.block_idx]
                block_reached = block_reached.at[inst.jump_dest].set(
                    jnp.maximum(block_reached[inst.jump_dest], this_reached)
                )

            elif opcode == 'call':
                if inst.result_idx < 0:
                    continue

                op_vals = [vals[i] for i in inst.operand_indices]
                func_name = inst.func_name

                if 'SimParam' in func_name:
                    result = jnp.array(1e-12)  # gmin
                elif 'TimeDerivative' in func_name:
                    result = jnp.array(0.0)  # DC analysis
                elif 'NodeDerivative' in func_name:
                    result = jnp.array(0.0)  # DC analysis
                elif 'limexp' in func_name.lower():
                    result = safe_exp(jnp.minimum(op_vals[0], 700.0)) if op_vals else jnp.array(1.0)
                else:
                    result = jnp.array(0.0)

                vals = vals.at[inst.result_idx].set(result)

            else:
                # Regular arithmetic/math instruction
                if inst.result_idx < 0:
                    continue

                op_vals = [vals[i] for i in inst.operand_indices]

                if opcode in BINARY_OPS and len(op_vals) >= 2:
                    result = BINARY_OPS[opcode](op_vals[0], op_vals[1])
                elif opcode in UNARY_OPS and len(op_vals) >= 1:
                    result = UNARY_OPS[opcode](op_vals[0])
                else:
                    result = op_vals[0] if op_vals else jnp.array(0.0)

                vals = vals.at[inst.result_idx].set(result)

        # Extract outputs
        resist_res = vals[resist_res_idx]
        react_res = vals[react_res_idx]
        resist_jac = vals[resist_jac_idx]
        react_jac = vals[react_jac_idx]

        return (resist_res, react_res), (resist_jac, react_jac)

    return eval_fn


def emit_eval_lax_loop(module) -> Callable:
    """Generate eval function using lax.fori_loop for large models.

    For models with >1000 instructions, using a loop-based interpreter
    avoids XLA compilation timeout from unrolling.

    Args:
        module: openvaf_py VaModule

    Returns:
        eval_fn(input_array, cache_array) -> ((resist_res, react_res), (resist_jac, react_jac))
    """
    import numpy as np

    ctx = _build_emit_context(module)
    compiled = _compile_instructions(module, ctx)
    n_instructions = len(compiled)

    # Encode instructions as arrays for lax.fori_loop
    # Fixed-size encoding: [opcode, result_idx, block_idx, op1, op2, extra1, extra2, extra3]
    INST_SIZE = 8

    OPCODE_MAP = {
        'fadd': 0, 'fsub': 1, 'fmul': 2, 'fdiv': 3,
        'flt': 4, 'fle': 5, 'fgt': 6, 'fge': 7, 'feq': 8, 'fne': 9,
        'fneg': 10, 'ln': 11, 'exp': 12, 'sqrt': 13,
        'sin': 14, 'cos': 15, 'tan': 16, 'atan': 17, 'abs': 18,
        'pow': 19, 'min': 20, 'max': 21, 'atan2': 22,
        'ifcast': 23, 'optbarrier': 24, 'call': 25,
        'ilt': 26, 'ile': 27, 'igt': 28, 'ige': 29, 'ieq': 30, 'ine': 31,
        'ineg': 32, 'bnot': 33, 'inot': 34, 'bfcast': 35,
        'asin': 36, 'acos': 37, 'sinh': 38, 'cosh': 39,
        'tanh': 40, 'asinh': 41, 'acosh': 42, 'atanh': 43, 'log': 44, 'hypot': 45,
        'ficast': 46, 'bicast': 47, 'ibcast': 48, 'fbcast': 49,
        'floor': 50, 'ceil': 51,
        'phi': 100, 'br': 101, 'jmp': 102, 'noop': 127,
    }

    # Encode instructions
    inst_data = []
    phi_data = []  # Separate array for variable-length PHI operands

    for inst in compiled:
        opcode = inst.opcode
        opcode_int = OPCODE_MAP.get(opcode, 127)

        if opcode == 'phi':
            phi_start = len(phi_data)
            phi_data.append(len(inst.phi_operands))
            for val_idx, pred_idx in inst.phi_operands:
                phi_data.extend([val_idx, pred_idx])
            inst_data.extend([100, inst.result_idx, phi_start, 0, inst.block_idx, 0, 0, 0])

        elif opcode == 'br':
            if inst.branch_info is not None:
                cond_idx, true_idx, false_idx = inst.branch_info
                inst_data.extend([101, 0, cond_idx, 0, inst.block_idx, true_idx, false_idx, 0])

        elif opcode == 'jmp':
            jump_dest = inst.jump_dest if inst.jump_dest is not None else 0
            inst_data.extend([102, 0, 0, 0, inst.block_idx, jump_dest, jump_dest, 0])

        else:
            ops = inst.operand_indices
            op1 = ops[0] if len(ops) > 0 else 0
            op2 = ops[1] if len(ops) > 1 else 0
            inst_data.extend([opcode_int, inst.result_idx, op1, op2, inst.block_idx, 0, 0, 0])

    inst_array = jnp.array(inst_data, dtype=jnp.int32)
    phi_array = jnp.array(phi_data if phi_data else [0], dtype=jnp.int32)

    # Static context
    const_array = ctx.const_array
    param_indices = jnp.array(ctx.param_indices, dtype=jnp.int32)
    n_params = ctx.n_params
    cache_indices = jnp.array([k for k, _v in ctx.cache_to_var_idx.items()], dtype=jnp.int32)
    cache_var_indices = jnp.array([v for _k, v in ctx.cache_to_var_idx.items()], dtype=jnp.int32)
    n_cache = len(ctx.cache_to_var_idx)
    n_blocks = ctx.n_blocks

    resist_res_idx = jnp.array(ctx.resist_res_indices, dtype=jnp.int32)
    react_res_idx = jnp.array(ctx.react_res_indices, dtype=jnp.int32)
    resist_jac_idx = jnp.array(ctx.resist_jac_indices, dtype=jnp.int32)
    react_jac_idx = jnp.array(ctx.react_jac_indices, dtype=jnp.int32)

    def eval_fn(input_array: jnp.ndarray, cache: jnp.ndarray):
        """Evaluate using lax.fori_loop interpreter."""
        vals = const_array.copy()

        # Load params
        def load_param(i, v):
            return v.at[param_indices[i]].set(input_array[i])
        vals = lax.fori_loop(0, jnp.minimum(input_array.shape[0], n_params), load_param, vals)

        # Load cache
        def load_cache(i, v):
            return v.at[cache_var_indices[i]].set(cache[cache_indices[i]])
        if n_cache > 0:
            vals = lax.fori_loop(0, n_cache, load_cache, vals)

        block_reached = jnp.zeros(n_blocks).at[0].set(1.0)

        def exec_inst(i, state):
            vals, block_reached = state
            base = i * INST_SIZE

            opcode = inst_array[base]
            result_idx = inst_array[base + 1]
            op1_idx = inst_array[base + 2]
            op2_idx = inst_array[base + 3]
            block_idx = inst_array[base + 4]
            extra1 = inst_array[base + 5]
            extra2 = inst_array[base + 6]

            a = vals[op1_idx]
            b = vals[op2_idx]

            # Compute result based on opcode
            is_binary = (opcode <= 3) | ((opcode >= 19) & (opcode <= 22)) | \
                       ((opcode >= 26) & (opcode <= 31)) | (opcode == 45)

            binary_result = jnp.where(opcode == 0, safe_add(a, b),
                           jnp.where(opcode == 1, safe_sub(a, b),
                           jnp.where(opcode == 2, safe_mul(a, b),
                           jnp.where(opcode == 3, safe_div(a, b),
                           jnp.where(opcode == 4, jnp.float64(a < b),
                           jnp.where(opcode == 5, jnp.float64(a <= b),
                           jnp.where(opcode == 6, jnp.float64(a > b),
                           jnp.where(opcode == 7, jnp.float64(a >= b),
                           jnp.where(opcode == 8, jnp.float64(a == b),
                           jnp.where(opcode == 9, jnp.float64(a != b),
                           jnp.where(opcode == 19, safe_pow(a, b),
                           jnp.where(opcode == 20, jnp.minimum(a, b),
                           jnp.where(opcode == 21, jnp.maximum(a, b),
                           jnp.where(opcode == 22, jnp.arctan2(a, b),
                           jnp.where(opcode == 45, jnp.hypot(a, b),
                           0.0)))))))))))))))

            unary_result = jnp.where(opcode == 10, -a,
                          jnp.where(opcode == 11, safe_ln(a),
                          jnp.where(opcode == 12, safe_exp(a),
                          jnp.where(opcode == 13, safe_sqrt(a),
                          jnp.where(opcode == 14, jnp.sin(a),
                          jnp.where(opcode == 15, jnp.cos(a),
                          jnp.where(opcode == 16, jnp.tan(a),
                          jnp.where(opcode == 17, jnp.arctan(a),
                          jnp.where(opcode == 18, jnp.abs(a),
                          jnp.where(opcode == 23, jnp.float64(a),
                          jnp.where(opcode == 24, a,
                          jnp.where(opcode == 25, safe_exp(jnp.minimum(a, 700.0)),
                          jnp.where(opcode == 33, jnp.float64(a == 0.0),
                          jnp.where(opcode == 34, jnp.float64(a == 0.0),
                          jnp.where(opcode == 35, jnp.float64(a),
                          jnp.where(opcode == 36, jnp.arcsin(a),
                          jnp.where(opcode == 37, jnp.arccos(a),
                          jnp.where(opcode == 38, jnp.sinh(a),
                          jnp.where(opcode == 39, jnp.cosh(a),
                          jnp.where(opcode == 40, jnp.tanh(a),
                          jnp.where(opcode == 41, jnp.arcsinh(a),
                          jnp.where(opcode == 42, jnp.arccosh(a),
                          jnp.where(opcode == 43, jnp.arctanh(a),
                          jnp.where(opcode == 44, jnp.log10(jnp.maximum(a, 1e-300)),
                          jnp.where(opcode == 50, jnp.floor(a),
                          jnp.where(opcode == 51, jnp.ceil(a),
                          0.0))))))))))))))))))))))))))

            arith_result = jnp.where(is_binary, binary_result, unary_result)

            # PHI handling
            is_phi = (opcode == 100)
            phi_start = op1_idx
            n_phi_ops = phi_array[phi_start]

            first_val_idx = phi_array[phi_start + 1]
            phi_result = vals[first_val_idx]

            def select_phi(result, op_num):
                val_idx = phi_array[phi_start + 1 + op_num * 2]
                pred_idx = phi_array[phi_start + 2 + op_num * 2]
                return jnp.where(block_reached[pred_idx] > 0.5, vals[val_idx], result)

            phi_result = jnp.where(n_phi_ops > 0, select_phi(phi_result, 0), phi_result)
            phi_result = jnp.where(n_phi_ops > 1, select_phi(phi_result, 1), phi_result)
            phi_result = jnp.where(n_phi_ops > 2, select_phi(phi_result, 2), phi_result)
            phi_result = jnp.where(n_phi_ops > 3, select_phi(phi_result, 3), phi_result)

            # Branch handling
            is_br = (opcode == 101)
            is_jmp = (opcode == 102)
            cond = vals[op1_idx]
            this_reached = block_reached[block_idx]

            true_reaches = jnp.where(is_br, this_reached * cond,
                           jnp.where(is_jmp, this_reached, 0.0))
            false_reaches = jnp.where(is_br, this_reached * (1.0 - cond), 0.0)

            new_block_reached = jnp.where(
                is_br | is_jmp,
                block_reached.at[extra1].set(jnp.maximum(block_reached[extra1], true_reaches)),
                block_reached
            )
            new_block_reached = jnp.where(
                is_br,
                new_block_reached.at[extra2].set(jnp.maximum(new_block_reached[extra2], false_reaches)),
                new_block_reached
            )

            # Choose result
            final_val = jnp.where(is_phi, phi_result, arith_result)

            is_arith = (opcode < 100)
            should_update = (is_arith | is_phi) & (result_idx >= 0)
            new_vals = jnp.where(should_update, vals.at[result_idx].set(final_val), vals)

            return (new_vals, new_block_reached)

        vals, _ = lax.fori_loop(0, n_instructions, exec_inst, (vals, block_reached))

        return (vals[resist_res_idx], vals[react_res_idx]), (vals[resist_jac_idx], vals[react_jac_idx])

    return eval_fn


def build_eval_fn(module, force_lax_loop: bool = False) -> Tuple[Callable, Dict]:
    """Build JAX eval function from module.

    Args:
        module: openvaf_py VaModule
        force_lax_loop: Force use of lax_loop interpreter

    Returns:
        (eval_fn, metadata)
    """
    eval_mir = module.get_mir_instructions()
    n_instructions = len(eval_mir['instructions'])

    # Use lax_loop for large models (>1000 instructions)
    use_lax_loop = force_lax_loop or n_instructions > 1000

    if use_lax_loop:
        eval_fn = emit_eval_lax_loop(module)
        strategy = 'lax_loop'
    else:
        eval_fn = emit_eval(module)
        strategy = 'branchless'

    metadata = {
        'strategy': strategy,
        'n_instructions': n_instructions,
    }

    return eval_fn, metadata


# Test entry point
if __name__ == '__main__':
    import openvaf_py
    from pathlib import Path

    REPO_ROOT = Path(__file__).parent.parent
    VACASK = REPO_ROOT / "vendor" / "VACASK" / "devices"

    print("Testing JAX emitter...")
    print()

    # Test resistor
    print("=== Resistor ===")
    modules = openvaf_py.compile_va(str(VACASK / "resistor.va"))
    resistor = modules[0]

    eval_fn, meta = build_eval_fn(resistor)
    print(f"Strategy: {meta['strategy']}, Instructions: {meta['n_instructions']}")

    params = jnp.zeros(10)
    cache = jnp.zeros(5)

    jit_eval = jax.jit(eval_fn)
    result = jit_eval(params, cache)
    print(f"Result shapes: res={result[0][0].shape}, jac={result[1][0].shape}")
    print()

    # Test diode (has control flow)
    print("=== Diode ===")
    modules = openvaf_py.compile_va(str(VACASK / "diode.va"))
    diode = modules[0]

    eval_fn, meta = build_eval_fn(diode)
    print(f"Strategy: {meta['strategy']}, Instructions: {meta['n_instructions']}")

    n_params = len(list(diode.param_names))
    n_cache = diode.num_cached_values
    params = jnp.zeros(n_params)
    cache = jnp.zeros(n_cache)

    jit_eval = jax.jit(eval_fn)
    result = jit_eval(params, cache)
    print(f"Result shapes: res={result[0][0].shape}, jac={result[1][0].shape}")
    print()

    print("Done!")
