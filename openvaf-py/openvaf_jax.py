"""OpenVAF to JAX translator

Compiles Verilog-A models to JAX functions using openvaf-py.
"""

from typing import Dict, List, Callable, Any, Tuple, Set, Optional
from dataclasses import dataclass
import openvaf_py


@dataclass
class CompiledDevice:
    """A compiled Verilog-A device with JAX evaluation function"""
    name: str
    module_name: str
    nodes: List[str]
    param_names: List[str]
    param_kinds: List[str]
    eval_fn: Callable
    num_residuals: int
    num_jacobian: int


class OpenVAFToJAX:
    """Translates OpenVAF MIR to JAX functions"""

    def __init__(self, module):
        """Initialize with a compiled VaModule from openvaf_py

        Args:
            module: VaModule from openvaf_py.compile_va()
        """
        self.module = module
        self.mir_data = module.get_mir_instructions()
        self.dae_data = module.get_dae_system()
        self.init_mir_data = module.get_init_mir_instructions()

        # Build value tracking
        self.constants = dict(self.mir_data['constants'])
        self.bool_constants = dict(self.mir_data.get('bool_constants', {}))
        self.int_constants = dict(self.mir_data.get('int_constants', {}))
        self.params = list(self.mir_data['params'])

        # Init function data
        self.init_constants = dict(self.init_mir_data['constants'])
        self.init_bool_constants = dict(self.init_mir_data.get('bool_constants', {}))
        self.init_int_constants = dict(self.init_mir_data.get('int_constants', {}))
        self.init_params = list(self.init_mir_data['params'])
        self.cache_mapping = list(self.init_mir_data['cache_mapping'])

    @classmethod
    def from_file(cls, va_path: str) -> "OpenVAFToJAX":
        """Create translator from a Verilog-A file

        Args:
            va_path: Path to the .va file

        Returns:
            OpenVAFToJAX instance
        """
        modules = openvaf_py.compile_va(va_path)
        if not modules:
            raise ValueError(f"No modules found in {va_path}")
        return cls(modules[0])

    def translate(self) -> Callable:
        """Generate a JAX function from the MIR

        Returns a function with signature:
            f(inputs: List[float]) -> (residuals: Dict, jacobian: Dict)

        The inputs should be ordered according to self.params
        """
        code_lines = self._generate_code()
        code = '\n'.join(code_lines)

        # Compile and return
        import jax.numpy as jnp
        from jax import lax
        local_ns = {'jnp': jnp, 'lax': lax}
        exec(code, local_ns)
        return local_ns['device_eval']

    def _generate_code(self) -> List[str]:
        """Generate the JAX function code"""
        lines = []
        lines.append("def device_eval(inputs):")
        lines.append("    import jax.numpy as jnp")
        lines.append("    from jax import lax")
        lines.append("")

        # Only use eval constants (init constants are prefixed separately)
        # Initialize constants for eval function
        lines.append("    # Constants (eval function)")
        for name, value in self.constants.items():
            # Handle special float values
            if value == float('inf'):
                lines.append(f"    {name} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {name} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {name} = jnp.nan")
            else:
                lines.append(f"    {name} = {repr(value)}")

        # Boolean constants
        lines.append("    # Boolean constants")
        for name, value in self.bool_constants.items():
            lines.append(f"    {name} = {repr(value)}")

        # Int constants (from both eval and init MIR)
        lines.append("    # Int constants")
        # First add eval int constants
        for name, value in self.int_constants.items():
            if name not in self.constants:
                lines.append(f"    {name} = {repr(value)}")
        # Then add any init int constants not already defined
        for name, value in self.init_int_constants.items():
            if name not in self.constants and name not in self.int_constants:
                lines.append(f"    {name} = {repr(value)}")

        # Ensure v3 exists (commonly used for zero)
        if 'v3' not in self.constants:
            lines.append("    v3 = 0.0")

        lines.append("")

        # Map function parameters to inputs
        # Named eval params from user inputs, derivative selectors default to 0
        lines.append("    # Input parameters (eval function)")
        num_named_params = len(self.module.param_names)
        for i, param in enumerate(self.params[:num_named_params]):
            lines.append(f"    {param} = inputs[{i}]")

        # Derivative selector params (used internally for Jacobian computation)
        # Default to 0 for DC analysis (no derivatives enabled)
        lines.append("    # Derivative selector params (default to 0)")
        for param in self.params[num_named_params:]:
            lines.append(f"    {param} = 0")

        # Process init function first to compute cached values
        # Use init_ prefix to avoid name collisions with eval function
        lines.append("    # Init function computation")
        init_defined = set()

        # Add prefixed init constants
        lines.append("    # Init constants (prefixed)")
        for name, value in self.init_constants.items():
            prefixed = f"init_{name}"
            if value == float('inf'):
                lines.append(f"    {prefixed} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {prefixed} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {prefixed} = jnp.nan")
            else:
                lines.append(f"    {prefixed} = {repr(value)}")
            init_defined.add(prefixed)

        # Map init params from inputs (they overlap with eval params)
        # Init params are: R, $temperature, tnom, zeta, mfactor (for resistor)
        # These correspond to certain eval params
        init_param_mapping = self._build_init_param_mapping()
        for init_param, eval_idx in init_param_mapping.items():
            if eval_idx is not None:
                prefixed = f"init_{init_param}"
                lines.append(f"    {prefixed} = inputs[{eval_idx}]")
                init_defined.add(prefixed)

        # Process init instructions in block order (like eval function)
        # This ensures variables are defined before use
        init_block_order = self._topological_sort_init_blocks()
        init_by_block = self._group_init_instructions_by_block()

        for item in init_block_order:
            if isinstance(item, tuple) and item[0] == 'loop':
                # Handle loop structure
                _, header, loop_blocks, exit_blocks = item
                loop_lines = self._generate_init_loop(
                    header, loop_blocks, exit_blocks,
                    init_by_block, init_defined
                )
                lines.extend(loop_lines)
            else:
                # Regular block
                block_name = item
                lines.append(f"")
                lines.append(f"    # init {block_name}")

                for inst in init_by_block.get(block_name, []):
                    expr = self._translate_init_instruction(inst, init_defined)
                    if expr and 'result' in inst:
                        prefixed_result = f"init_{inst['result']}"
                        lines.append(f"    {prefixed_result} = {expr}")
                        init_defined.add(prefixed_result)

        lines.append("")

        # Map cached values from init to eval params
        # Need to find the actual Value names for cached params
        lines.append("    # Cached values from init -> eval params")
        all_func_params = self.module.get_all_func_params()
        param_idx_to_val = {p[0]: f"v{p[1]}" for p in all_func_params}

        for mapping in self.cache_mapping:
            init_val = f"init_{mapping['init_value']}"
            eval_param_idx = mapping['eval_param']
            # Look up the actual Value name for this param index
            eval_val = param_idx_to_val.get(eval_param_idx, f"cached_{eval_param_idx}")
            if init_val in init_defined:
                lines.append(f"    {eval_val} = {init_val}")

        lines.append("")

        # Process eval blocks in topological order
        block_order = self._topological_sort()
        defined_vars: Set[str] = set(self.constants.keys())
        defined_vars.update(self.params)
        defined_vars.update(init_defined)  # Include init-computed values
        defined_vars.add('v3')

        lines.append("    # Eval function computation")

        for block_name in block_order:
            block_data = self.mir_data['blocks'].get(block_name, {})
            lines.append(f"")
            lines.append(f"    # {block_name}")

            # Get instructions for this block
            for inst in self.mir_data['instructions']:
                if inst.get('block') != block_name:
                    continue

                expr = self._translate_instruction(inst, defined_vars)
                if expr and 'result' in inst:
                    lines.append(f"    {inst['result']} = {expr}")
                    defined_vars.add(inst['result'])

        lines.append("")

        # Build output expressions
        lines.append("    # Build outputs")
        lines.append("    residuals = {")
        for node, res in self.dae_data['residuals'].items():
            resist_val = res['resist'] if res['resist'] in defined_vars else '0.0'
            react_val = res['react'] if res['react'] in defined_vars else '0.0'
            lines.append(f"        '{node}': {{'resist': {resist_val}, 'react': {react_val}}},")
        lines.append("    }")

        lines.append("    jacobian = {")
        for entry in self.dae_data['jacobian']:
            key = f"('{entry['row']}', '{entry['col']}')"
            resist_val = entry['resist'] if entry['resist'] in defined_vars else '0.0'
            react_val = entry['react'] if entry['react'] in defined_vars else '0.0'
            lines.append(f"        {key}: {{'resist': {resist_val}, 'react': {react_val}}},")
        lines.append("    }")

        lines.append("    return residuals, jacobian")

        return lines

    def _build_init_param_mapping(self) -> Dict[str, Optional[int]]:
        """Build mapping from init params to eval input indices

        Init params (like R, $temperature, tnom) need to come from the inputs.
        We find which eval params correspond to each init param.
        """
        mapping = {}

        # Get init param names
        init_param_names = list(self.module.init_param_names)
        eval_param_names = list(self.module.param_names)

        for i, init_name in enumerate(init_param_names):
            # Find matching eval param
            init_param_val = self.init_params[i] if i < len(self.init_params) else None
            if init_param_val:
                # Look for this param name in eval params
                try:
                    eval_idx = eval_param_names.index(init_name)
                    # Get the eval param's value name
                    eval_param_val = self.params[eval_idx] if eval_idx < len(self.params) else None
                    if eval_param_val:
                        # Map init value name to eval input index
                        mapping[init_param_val] = eval_idx
                except ValueError:
                    # Not found in eval params
                    pass

        return mapping

    def _topological_sort(self) -> List[str]:
        """Sort eval blocks in topological order"""
        return self._topological_sort_blocks(self.mir_data.get('blocks', {}))

    def _topological_sort_init_blocks(self) -> List[str]:
        """Sort init blocks in topological order"""
        return self._topological_sort_blocks(self.init_mir_data.get('blocks', {}))

    def _topological_sort_blocks(self, blocks: Dict) -> List[str]:
        """Sort blocks in topological order using Kahn's algorithm, handling loops.

        Uses Kahn's algorithm to ensure a block is only processed after ALL its
        predecessors have been processed. Loops are collapsed into single nodes.

        Returns a list where each element is either:
        - A block name (str) for non-loop blocks
        - A tuple ('loop', header_block, body_blocks, exit_blocks) for loops
        """
        if not blocks:
            return []

        # First, find loops (SCCs with >1 node)
        loops = self._find_loops(blocks)

        # Create mapping from block to its loop (if any)
        block_to_loop: Dict[str, int] = {}
        for i, loop in enumerate(loops):
            for block in loop:
                block_to_loop[block] = i

        # Build condensed graph:
        # - Each non-loop block is a node
        # - Each loop is collapsed into a single node (represented by its header)
        # The "condensed" node name is either the block name or f"loop_{i}" for loops

        def get_condensed_node(block: str) -> str:
            if block in block_to_loop:
                return f"loop_{block_to_loop[block]}"
            return block

        # Compute in-degrees for condensed graph (Kahn's algorithm)
        in_degree: Dict[str, int] = {}
        condensed_successors: Dict[str, Set[str]] = {}

        # Initialize all nodes
        for block in blocks:
            cn = get_condensed_node(block)
            if cn not in in_degree:
                in_degree[cn] = 0
                condensed_successors[cn] = set()

        # Build edges in condensed graph
        for block, data in blocks.items():
            cn = get_condensed_node(block)
            for succ in data.get('successors', []):
                if succ not in blocks:
                    continue
                succ_cn = get_condensed_node(succ)
                if cn != succ_cn:  # Skip edges within same loop
                    if succ_cn not in condensed_successors[cn]:
                        condensed_successors[cn].add(succ_cn)
                        in_degree[succ_cn] = in_degree.get(succ_cn, 0) + 1

        # Kahn's algorithm: process nodes with in-degree 0
        queue = [cn for cn, deg in in_degree.items() if deg == 0]
        # Sort for deterministic order
        queue.sort(key=lambda x: (
            not x.startswith('loop_'),  # Loops last among same in-degree
            int(x.replace('block', '').replace('loop_', '')) if x.replace('block', '').replace('loop_', '').isdigit() else 0
        ))

        result = []
        processed = set()

        while queue:
            cn = queue.pop(0)
            if cn in processed:
                continue
            processed.add(cn)

            # Add to result
            if cn.startswith('loop_'):
                loop_idx = int(cn.replace('loop_', ''))
                loop = loops[loop_idx]
                header = self._find_loop_header(loop, blocks)
                exits = self._find_loop_exits(loop, blocks)
                result.append(('loop', header, loop, exits))
            else:
                result.append(cn)

            # Decrease in-degree of successors
            for succ_cn in condensed_successors.get(cn, []):
                in_degree[succ_cn] -= 1
                if in_degree[succ_cn] == 0:
                    queue.append(succ_cn)

            # Keep queue sorted for determinism
            queue.sort(key=lambda x: (
                not x.startswith('loop_'),
                int(x.replace('block', '').replace('loop_', '')) if x.replace('block', '').replace('loop_', '').isdigit() else 0
            ))

        # Handle any remaining blocks (shouldn't happen in well-formed CFG)
        for cn in in_degree:
            if cn not in processed:
                if cn.startswith('loop_'):
                    loop_idx = int(cn.replace('loop_', ''))
                    loop = loops[loop_idx]
                    header = self._find_loop_header(loop, blocks)
                    exits = self._find_loop_exits(loop, blocks)
                    result.append(('loop', header, loop, exits))
                else:
                    result.append(cn)

        return result

    def _find_loops(self, blocks: Dict) -> List[Set[str]]:
        """Find all loops (SCCs with >1 node) using Tarjan's algorithm"""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in blocks.get(v, {}).get('successors', []):
                if w not in blocks:
                    continue
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], index[w])

            if lowlinks[v] == index[v]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in blocks:
            if v not in index:
                strongconnect(v)

        # Return only non-trivial SCCs (loops with >1 node)
        return [scc for scc in sccs if len(scc) > 1]

    def _find_loop_header(self, loop: Set[str], blocks: Dict) -> str:
        """Find the loop header (entry point from outside the loop)"""
        for block in loop:
            preds = blocks.get(block, {}).get('predecessors', [])
            for pred in preds:
                if pred not in loop:
                    return block
        # Fallback: return block with lowest number
        return min(loop, key=lambda x: int(x.replace('block', '')) if x.startswith('block') else 0)

    def _find_loop_exits(self, loop: Set[str], blocks: Dict) -> List[str]:
        """Find blocks that exit the loop (successors outside the loop)"""
        exits = []
        for block in loop:
            succs = blocks.get(block, {}).get('successors', [])
            for succ in succs:
                if succ not in loop and succ not in exits:
                    exits.append(succ)
        return exits

    def _generate_init_loop(self, header: str, loop_blocks: Set[str],
                            exit_blocks: List[str], init_by_block: Dict[str, List[dict]],
                            init_defined: Set[str]) -> List[str]:
        """Generate JAX code for a loop using jax.lax.while_loop

        Args:
            header: The loop header block name
            loop_blocks: Set of all blocks in the loop
            exit_blocks: List of blocks that are exited to
            init_by_block: Dict mapping block names to their instructions
            init_defined: Set of already-defined variable names (will be updated)

        Returns:
            List of code lines to add
        """
        lines = []
        lines.append("")
        lines.append(f"    # Loop: {header} with blocks {sorted(loop_blocks)}")

        # Find PHI nodes in the header - these are loop-carried values
        header_insts = init_by_block.get(header, [])
        phi_nodes = [inst for inst in header_insts if inst.get('opcode', '').lower() == 'phi']

        # Get loop-carried variables: PHI results and their incoming values from the loop
        loop_carried = []  # (result_var, init_value, loop_value)
        for phi in phi_nodes:
            result = phi.get('result', '')
            phi_ops = phi.get('phi_operands', [])
            init_val = None
            loop_val = None
            for op in phi_ops:
                if op['block'] in loop_blocks:
                    loop_val = op['value']
                else:
                    init_val = op['value']
            if result and init_val and loop_val:
                loop_carried.append((result, init_val, loop_val))

        if not loop_carried:
            # No loop-carried values - something is wrong
            lines.append("    # WARNING: No loop-carried values found, skipping loop")
            return lines

        # Find the condition and body instructions
        condition_inst = None
        body_insts = []

        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'br' and 'condition' in inst:
                condition_inst = inst
            elif op != 'phi':
                body_insts.append(inst)

        # Also get instructions from other loop blocks (the body)
        for block in sorted(loop_blocks):
            if block != header:
                for inst in init_by_block.get(block, []):
                    op = inst.get('opcode', '').lower()
                    if op not in ('br', 'jmp'):
                        body_insts.append(inst)

        # Helper to get operand with init_ prefix
        def get_operand(op: str, local_vars: Set[str] = None) -> str:
            if local_vars is None:
                local_vars = set()
            # Check local loop variables first (no prefix)
            if op in local_vars:
                return op
            prefixed = f"init_{op}"
            if prefixed in init_defined:
                return prefixed
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            return prefixed

        # Generate initial state tuple
        init_state_parts = []
        for result, init_val, _ in loop_carried:
            init_state_parts.append(get_operand(init_val))

        lines.append(f"    _loop_state_init = ({', '.join(init_state_parts)},)")

        # Generate condition function
        lines.append("")
        lines.append("    def _loop_cond(_loop_state):")

        # Unpack state
        state_vars = [lc[0] for lc in loop_carried]  # Use original var names inside loop
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        # Generate condition computation (instructions before the branch in header)
        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Return the condition
        if condition_inst:
            cond_var = condition_inst.get('condition', '')
            # Check if condition is in local vars or needs prefix
            if cond_var in local_vars:
                lines.append(f"        return {cond_var}")
            else:
                lines.append(f"        return {get_operand(cond_var, local_vars)}")
        else:
            lines.append("        return False")

        # Generate body function
        lines.append("")
        lines.append("    def _loop_body(_loop_state):")
        lines.append(f"        {', '.join(state_vars)}, = _loop_state")

        # Recompute header instructions (needed for body)
        local_vars = set(state_vars)
        for inst in header_insts:
            op = inst.get('opcode', '').lower()
            if op == 'phi' or op == 'br':
                continue
            result = inst.get('result', '')
            expr = self._translate_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Generate body instructions
        for inst in body_insts:
            result = inst.get('result', '')
            expr = self._translate_loop_instruction(inst, local_vars, init_defined)
            if expr and result:
                lines.append(f"        {result} = {expr}")
                local_vars.add(result)

        # Return new state (the loop_val from each PHI)
        new_state_parts = []
        for _, _, loop_val in loop_carried:
            if loop_val in local_vars:
                new_state_parts.append(loop_val)
            else:
                new_state_parts.append(get_operand(loop_val, local_vars))

        lines.append(f"        return ({', '.join(new_state_parts)},)")

        # Call while_loop
        lines.append("")
        lines.append("    _loop_result = lax.while_loop(_loop_cond, _loop_body, _loop_state_init)")

        # Unpack results to prefixed variables
        for i, (result, _, _) in enumerate(loop_carried):
            prefixed = f"init_{result}"
            lines.append(f"    {prefixed} = _loop_result[{i}]")
            init_defined.add(prefixed)

        return lines

    def _translate_loop_instruction(self, inst: dict, local_vars: Set[str],
                                    init_defined: Set[str]) -> Optional[str]:
        """Translate an instruction for use inside a loop

        Args:
            inst: The instruction to translate
            local_vars: Variables defined locally in the loop (no prefix)
            init_defined: Variables defined in init scope (with prefix)
        """
        def get_operand(op: str) -> str:
            # Local vars first (no prefix)
            if op in local_vars:
                return op
            prefixed = f"init_{op}"
            if prefixed in init_defined:
                return prefixed
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            return prefixed

        return self._translate_instruction_impl(inst, get_operand)

    def _group_init_instructions_by_block(self) -> Dict[str, List[dict]]:
        """Group init instructions by their block"""
        by_block: Dict[str, List[dict]] = {}
        for inst in self.init_mir_data.get('instructions', []):
            block = inst.get('block', 'block0')
            if block not in by_block:
                by_block[block] = []
            by_block[block].append(inst)
        return by_block

    def _build_branch_conditions(self) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Build a map of (block -> successor -> (condition, polarity)) for eval function"""
        return self._build_branch_conditions_impl(self.mir_data.get('instructions', []))

    def _build_init_branch_conditions(self) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Build a map of (block -> successor -> (condition, polarity)) for init function"""
        return self._build_branch_conditions_impl(self.init_mir_data.get('instructions', []))

    def _build_branch_conditions_impl(self, instructions: List) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Build a map of (block -> successor -> (condition, polarity))

        For each block with 2 successors, find the condition that determines the branch.
        Returns a dict mapping: block -> {successor: (condition_var, is_true_branch)}
        """
        conditions = {}

        # Find branch instructions with explicit conditions
        for inst in instructions:
            op = inst.get('opcode', '').lower()
            if op == 'br' and 'condition' in inst:
                block = inst.get('block', '')
                cond = inst['condition']
                true_block = inst.get('true_block', '')
                false_block = inst.get('false_block', '')
                if block and cond and true_block and false_block:
                    conditions[block] = {
                        true_block: (cond, True),
                        false_block: (cond, False),
                    }

        return conditions

    def _get_phi_condition(self, phi_block: str, pred_blocks: List[str]) -> Optional[Tuple[str, str, str]]:
        """Get the condition for a PHI node in eval function

        Returns (condition_var, true_value_block, false_value_block) or None
        """
        blocks = self.mir_data.get('blocks', {})
        branch_conds = self._build_branch_conditions()
        return self._get_phi_condition_impl(phi_block, pred_blocks, blocks, branch_conds, prefix='')

    def _get_init_phi_condition(self, phi_block: str, pred_blocks: List[str]) -> Optional[Tuple[str, str, str]]:
        """Get the condition for a PHI node in init function

        Returns (condition_var, true_value_block, false_value_block) or None
        """
        blocks = self.init_mir_data.get('blocks', {})
        branch_conds = self._build_init_branch_conditions()
        return self._get_phi_condition_impl(phi_block, pred_blocks, blocks, branch_conds, prefix='init_')

    def _get_phi_condition_impl(self, phi_block: str, pred_blocks: List[str],
                                 blocks: Dict, branch_conds: Dict,
                                 prefix: str = '') -> Optional[Tuple[str, str, str]]:
        """Implementation of PHI condition finding

        Returns (condition_var, true_value_block, false_value_block) or None
        The condition_var is prefixed with prefix if provided.
        """
        if len(pred_blocks) != 2:
            return None

        pred0, pred1 = pred_blocks

        # Check if either predecessor is the branching block
        if pred0 in branch_conds:
            # pred0 branches to phi_block and pred1
            cond_info = branch_conds[pred0].get(phi_block)
            if cond_info:
                cond_var, is_true = cond_info
                cond_var = f"{prefix}{cond_var}" if prefix else cond_var
                if is_true:
                    return (cond_var, pred0, pred1)
                else:
                    return (cond_var, pred1, pred0)

        if pred1 in branch_conds:
            cond_info = branch_conds[pred1].get(phi_block)
            if cond_info:
                cond_var, is_true = cond_info
                cond_var = f"{prefix}{cond_var}" if prefix else cond_var
                if is_true:
                    return (cond_var, pred1, pred0)
                else:
                    return (cond_var, pred0, pred1)

        # Check if there's a common dominator that branches
        # Look for a block that is predecessor to both pred0 and pred1
        for block_name, block_data in blocks.items():
            succs = block_data.get('successors', [])
            if set(succs) == set(pred_blocks) and block_name in branch_conds:
                cond_info = branch_conds[block_name]
                if pred0 in cond_info and pred1 in cond_info:
                    cond_var, is_true0 = cond_info[pred0]
                    cond_var = f"{prefix}{cond_var}" if prefix else cond_var
                    if is_true0:
                        return (cond_var, pred0, pred1)
                    else:
                        return (cond_var, pred1, pred0)

        return None

    def _translate_init_instruction(self, inst: dict, defined_vars: Set[str]) -> Optional[str]:
        """Translate an init function instruction with prefixed variables"""

        def get_operand(op: str) -> str:
            prefixed = f"init_{op}"
            if prefixed in defined_vars:
                return prefixed
            if op in self.init_constants:
                return repr(self.init_constants[op])
            if op in self.init_bool_constants:
                return repr(self.init_bool_constants[op])
            if op in self.init_int_constants:
                return repr(self.init_int_constants[op])
            # Fallback to prefixed anyway
            return prefixed

        # Handle PHI nodes specially for init function
        opcode = inst.get('opcode', '').lower()
        if opcode == 'phi':
            phi_ops = inst.get('phi_operands', [])
            phi_block = inst.get('block', '')
            if phi_ops and len(phi_ops) >= 2:
                # Get the predecessor blocks from PHI operands
                pred_blocks = [op['block'] for op in phi_ops]
                val_by_block = {op['block']: get_operand(op['value']) for op in phi_ops}

                # Try to find the condition that determines the branch (using init version)
                cond_info = self._get_init_phi_condition(phi_block, pred_blocks)
                if cond_info:
                    cond_var, true_block, false_block = cond_info
                    true_val = val_by_block.get(true_block, '0.0')
                    false_val = val_by_block.get(false_block, '0.0')
                    return f"jnp.where({cond_var}, {true_val}, {false_val})"
                else:
                    # Fallback: just use first value (may be incorrect)
                    val0 = get_operand(phi_ops[0]['value'])
                    return val0
            elif phi_ops:
                return get_operand(phi_ops[0]['value'])
            return '0.0'

        return self._translate_instruction_impl(inst, get_operand)

    def _translate_instruction(self, inst: dict, defined_vars: Set[str]) -> Optional[str]:
        """Translate a single instruction to JAX expression"""

        def get_operand(op: str) -> str:
            if op in defined_vars:
                return op
            if op in self.constants:
                return repr(self.constants[op])
            return op

        return self._translate_instruction_impl(inst, get_operand)

    def _translate_instruction_impl(self, inst: dict, get_operand: Callable[[str], str]) -> Optional[str]:
        """Implementation of instruction translation with custom operand resolver"""
        opcode = inst.get('opcode', '').lower()
        operands = inst.get('operands', [])

        if opcode == 'fadd':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} + {ops[1]})"

        elif opcode == 'fsub':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} - {ops[1]})"

        elif opcode == 'fmul':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} * {ops[1]})"

        elif opcode == 'fdiv':
            ops = [get_operand(op) for op in operands]
            # Use safe division to avoid div-by-zero when conditionals are linearized
            return f"jnp.where({ops[1]} == 0.0, 0.0, {ops[0]} / jnp.where({ops[1]} == 0.0, 1.0, {ops[1]}))"

        elif opcode == 'fneg':
            ops = [get_operand(op) for op in operands]
            return f"(-{ops[0]})"

        elif opcode == 'exp':
            ops = [get_operand(op) for op in operands]
            return f"jnp.exp({ops[0]})"

        elif opcode == 'ln':
            ops = [get_operand(op) for op in operands]
            return f"jnp.log({ops[0]})"

        elif opcode == 'sqrt':
            ops = [get_operand(op) for op in operands]
            return f"jnp.sqrt({ops[0]})"

        elif opcode == 'pow':
            ops = [get_operand(op) for op in operands]
            return f"jnp.power({ops[0]}, {ops[1]})"

        elif opcode in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'):
            ops = [get_operand(op) for op in operands]
            return f"jnp.{opcode}({ops[0]})"

        elif opcode == 'floor':
            ops = [get_operand(op) for op in operands]
            return f"jnp.floor({ops[0]})"

        elif opcode == 'ceil':
            ops = [get_operand(op) for op in operands]
            return f"jnp.ceil({ops[0]})"

        elif opcode == 'hypot':
            ops = [get_operand(op) for op in operands]
            return f"jnp.hypot({ops[0]}, {ops[1]})"

        elif opcode == 'atan2':
            ops = [get_operand(op) for op in operands]
            return f"jnp.arctan2({ops[0]}, {ops[1]})"

        elif opcode == 'feq':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} == {ops[1]})"

        elif opcode == 'flt':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} < {ops[1]})"

        elif opcode == 'fgt':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} > {ops[1]})"

        elif opcode == 'fle':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} <= {ops[1]})"

        elif opcode == 'fge':
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} >= {ops[1]})"

        elif opcode == 'optbarrier':
            # Optimization barrier - just pass through
            ops = [get_operand(op) for op in operands]
            return ops[0] if ops else '0.0'

        elif opcode == 'phi':
            # PHI node - select value based on control flow
            phi_ops = inst.get('phi_operands', [])
            phi_block = inst.get('block', '')
            if phi_ops and len(phi_ops) >= 2:
                # Get the predecessor blocks from PHI operands
                pred_blocks = [op['block'] for op in phi_ops]
                val_by_block = {op['block']: get_operand(op['value']) for op in phi_ops}

                # Try to find the condition that determines the branch
                cond_info = self._get_phi_condition(phi_block, pred_blocks)
                if cond_info:
                    cond_var, true_block, false_block = cond_info
                    true_val = val_by_block.get(true_block, '0.0')
                    false_val = val_by_block.get(false_block, '0.0')
                    return f"jnp.where({cond_var}, {true_val}, {false_val})"
                else:
                    # Fallback: just use first value (may be incorrect)
                    val0 = get_operand(phi_ops[0]['value'])
                    return val0
            elif phi_ops:
                return get_operand(phi_ops[0]['value'])
            elif operands:
                return get_operand(operands[0])
            return '0.0'

        elif opcode == 'call':
            # Function call - handle known functions
            func_ref = inst.get('func_ref', '')
            func_decls = self.mir_data.get('function_decls', {})

            if func_ref in func_decls:
                fn_name = func_decls[func_ref].get('name', '')

                if 'simparam' in fn_name.lower():
                    # $simparam("name", default) - return the default value
                    if len(operands) >= 2:
                        return get_operand(operands[1])
                    return '1e-12'  # gmin default

                elif 'ddt' in fn_name.lower() or 'TimeDerivative' in fn_name:
                    # Time derivative - for DC analysis, return 0
                    return '0.0'

                elif 'ddx' in fn_name.lower() or 'NodeDerivative' in fn_name:
                    # Derivative with respect to a variable - return 0 for now
                    return '0.0'

                elif 'noise' in fn_name.lower():
                    # Noise functions - return 0 for DC analysis
                    return '0.0'

                elif 'collapse' in fn_name.lower():
                    # Node collapsing - side effect only
                    return None

            # Unknown function - return 0
            return '0.0'

        elif opcode == 'ifcast':
            # Integer to float cast - just pass through the operand
            # In JAX, integers and floats are often compatible
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"jnp.float64({ops[0]})"
            return '0.0'

        elif opcode == 'ibcast':
            # Integer to bool cast - check if non-zero
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"({ops[0]} != 0)"
            return 'False'

        elif opcode == 'ficast':
            # Float to integer cast
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"jnp.int32({ops[0]})"
            return '0'

        elif opcode == 'ige':
            # Integer greater-than-or-equal
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} >= {ops[1]})"

        elif opcode == 'igt':
            # Integer greater-than
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} > {ops[1]})"

        elif opcode == 'ile':
            # Integer less-than-or-equal
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} <= {ops[1]})"

        elif opcode == 'ilt':
            # Integer less-than
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} < {ops[1]})"

        elif opcode == 'ieq':
            # Integer equal
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} == {ops[1]})"

        elif opcode == 'ine':
            # Integer not-equal
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} != {ops[1]})"

        elif opcode == 'fne':
            # Float not-equal
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} != {ops[1]})"

        elif opcode == 'bnot':
            # Boolean not
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"(not {ops[0]})"
            return 'True'

        elif opcode == 'iand':
            # Integer AND
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} & {ops[1]})"

        elif opcode == 'ior':
            # Integer OR
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} | {ops[1]})"

        elif opcode == 'ixor':
            # Integer XOR
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} ^ {ops[1]})"

        elif opcode == 'ineg':
            # Integer negate
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"(-{ops[0]})"
            return '0'

        elif opcode == 'iadd':
            # Integer add
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} + {ops[1]})"

        elif opcode == 'isub':
            # Integer subtract
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} - {ops[1]})"

        elif opcode == 'imul':
            # Integer multiply
            ops = [get_operand(op) for op in operands]
            return f"({ops[0]} * {ops[1]})"

        elif opcode == 'bicast':
            # Bool to int cast
            ops = [get_operand(op) for op in operands]
            if ops:
                return f"jnp.int32({ops[0]})"
            return '0'

        elif opcode in ('br', 'jmp', 'exit'):
            # Control flow - handled at block level
            return None

        return None

    def get_parameter_info(self) -> List[Tuple[str, str, str]]:
        """Get parameter information

        Returns:
            List of (value_name, param_name, param_kind) tuples
        """
        result = []
        for i, param in enumerate(self.params):
            if i < len(self.module.param_names):
                result.append((param, self.module.param_names[i], self.module.param_kinds[i]))
            else:
                result.append((param, f"cached_{i}", "cached"))
        return result

    def get_generated_code(self) -> str:
        """Get the generated JAX code as a string"""
        return '\n'.join(self._generate_code())


def compile_va(va_path: str) -> CompiledDevice:
    """Compile a Verilog-A file to a JAX-compatible device

    Args:
        va_path: Path to the .va file

    Returns:
        CompiledDevice with eval function and metadata
    """
    translator = OpenVAFToJAX.from_file(va_path)
    eval_fn = translator.translate()

    return CompiledDevice(
        name=va_path,
        module_name=translator.module.name,
        nodes=translator.module.nodes,
        param_names=translator.module.param_names,
        param_kinds=translator.module.param_kinds,
        eval_fn=eval_fn,
        num_residuals=translator.module.num_residuals,
        num_jacobian=translator.module.num_jacobian,
    )


if __name__ == "__main__":
    import numpy as np

    # Test with resistor
    print("="*60)
    print("Testing OpenVAF to JAX translator with resistor")
    print("="*60)

    translator = OpenVAFToJAX.from_file(
        "/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF/integration_tests/RESISTOR/resistor.va"
    )

    print("\nParameter mapping:")
    for val_name, param_name, kind in translator.get_parameter_info():
        print(f"  {val_name} -> {param_name} ({kind})")

    print("\nGenerated JAX code:")
    print("-"*60)
    print(translator.get_generated_code())
    print("-"*60)

    # Compile and test
    eval_fn = translator.translate()

    # Build inputs array
    # v16=V(A,B), v17=vres, v18=R, v19=$temp, v20=tnom, v22=zeta, v25=res, v28=mfactor
    V = 1.0
    R = 1000.0
    inputs = [
        V,      # v16 = V(A,B)
        V,      # v17 = vres (hidden state = V)
        R,      # v18 = R
        300.15, # v19 = $temperature
        300.0,  # v20 = tnom
        0.0,    # v22 = zeta
        R,      # v25 = res (hidden state = R)
        1.0,    # v28 = mfactor
    ]

    residuals, jacobian = eval_fn(inputs)

    print(f"\nInputs: V={V}, R={R}")
    print(f"Residuals: {residuals}")
    print(f"Expected: I = V/R = {V/R}")

    # Test with diode
    print("\n" + "="*60)
    print("Testing with diode")
    print("="*60)

    translator = OpenVAFToJAX.from_file(
        "/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF/integration_tests/DIODE/diode.va"
    )

    print(f"\nModule: {translator.module.name}")
    print(f"Parameters: {len(translator.params)}")
    print(f"Constants: {len(translator.constants)}")
    print(f"Blocks: {len(translator.mir_data['blocks'])}")

    eval_fn = translator.translate()

    # For diode, need to set many hidden states
    # This is complex - the translator generates code but hidden states must be pre-computed
    print("\nNote: Diode evaluation requires pre-computed hidden states")
    print("The JAX code is generated but evaluation needs proper setup")
