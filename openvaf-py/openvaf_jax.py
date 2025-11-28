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
        self.params = list(self.mir_data['params'])

        # Init function data
        self.init_constants = dict(self.init_mir_data['constants'])
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

        # Combine all constants (init + eval)
        all_constants = {}
        all_constants.update(self.init_constants)
        all_constants.update(self.constants)

        # Initialize constants
        lines.append("    # Constants")
        for name, value in all_constants.items():
            # Handle special float values
            if value == float('inf'):
                lines.append(f"    {name} = jnp.inf")
            elif value == float('-inf'):
                lines.append(f"    {name} = -jnp.inf")
            elif value != value:  # NaN check
                lines.append(f"    {name} = jnp.nan")
            else:
                lines.append(f"    {name} = {repr(value)}")

        # Ensure v3 exists (commonly used for zero)
        if 'v3' not in all_constants:
            lines.append("    v3 = 0.0")

        lines.append("")

        # Map function parameters to inputs
        # Named eval params first
        lines.append("    # Input parameters (eval function)")
        num_named_params = len(self.module.param_names)
        for i, param in enumerate(self.params[:num_named_params]):
            lines.append(f"    {param} = inputs[{i}]")

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

        # Process init instructions with prefixed variable names
        for inst in self.init_mir_data.get('instructions', []):
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
        defined_vars: Set[str] = set(all_constants.keys())
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
        """Sort blocks in topological order"""
        blocks = self.mir_data['blocks']
        if not blocks:
            return []

        # Find entry blocks (no predecessors)
        entry_blocks = [name for name, block in blocks.items()
                       if not block.get('predecessors', [])]
        if not entry_blocks:
            # Fall back to sorted order
            entry_blocks = sorted(blocks.keys(),
                                 key=lambda x: int(x.replace('block', '')) if x.startswith('block') else 0)

        visited = set()
        result = []

        def visit(name: str):
            if name in visited or name not in blocks:
                return
            visited.add(name)
            for succ in blocks[name].get('successors', []):
                visit(succ)
            result.append(name)

        for entry in entry_blocks:
            visit(entry)

        return list(reversed(result))

    def _translate_init_instruction(self, inst: dict, defined_vars: Set[str]) -> Optional[str]:
        """Translate an init function instruction with prefixed variables"""

        def get_operand(op: str) -> str:
            prefixed = f"init_{op}"
            if prefixed in defined_vars:
                return prefixed
            if op in self.init_constants:
                return repr(self.init_constants[op])
            # Fallback to prefixed anyway
            return prefixed

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
            return f"({ops[0]} / {ops[1]})"

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
            if phi_ops and len(phi_ops) >= 2:
                # For a 2-way PHI, use jnp.where with first predecessor's condition
                # This is a simplification - full impl would track branch conditions
                val0 = get_operand(phi_ops[0]['value'])
                val1 = get_operand(phi_ops[1]['value'])
                # Without branch condition tracking, just use first value
                # TODO: Properly track branch conditions for PHI resolution
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
