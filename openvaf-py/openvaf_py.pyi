"""Type stubs for the openvaf_py maturin module.

This module provides Python bindings for OpenVAF Verilog-A compilation.
"""

from typing import Any, Dict, List, Optional, Tuple


class VaModule:
    """Python wrapper for a compiled Verilog-A module."""

    # Basic module info
    name: str
    """Module name from Verilog-A source."""

    # Eval function parameters
    param_names: List[str]
    """Parameter names in order (for eval function)."""

    param_kinds: List[str]
    """Parameter kinds: 'param', 'param_given', 'voltage', 'current', 'temperature', etc."""

    param_value_indices: List[int]
    """MIR Value indices for parameters (for debugging)."""

    # Node information
    nodes: List[str]
    """Node names from unknowns."""

    num_residuals: int
    """Number of residual equations."""

    num_jacobian: int
    """Number of Jacobian entries."""

    func_num_params: int
    """Number of eval function parameters."""

    callback_names: List[str]
    """Callback function names (for debugging)."""

    # Init function parameters
    init_param_names: List[str]
    """Init function parameter names."""

    init_param_kinds: List[str]
    """Init function parameter kinds."""

    init_param_value_indices: List[int]
    """Init function parameter Value indices."""

    init_num_params: int
    """Number of init function parameters."""

    num_cached_values: int
    """Number of cached values from init."""

    # Node collapse support
    collapsible_pairs: List[Tuple[int, int]]
    """Collapsible node pairs: (node1_idx, node2_idx). node2=MAX means ground."""

    num_collapsible: int
    """Number of collapsible pairs."""

    collapse_decision_outputs: List[Tuple[int, str]]
    """Collapse decision outputs: (pair_index, decision_variable)."""

    # OSDI metadata
    num_terminals: int
    """Number of terminal nodes (ports)."""

    num_states: int
    """Number of limiting states."""

    has_bound_step: bool
    """Whether module has bound_step."""

    def get_all_func_params(self) -> List[Tuple[int, int]]:
        """Get all Param-defined values in the function.

        Returns:
            List of (param_index, value_index) tuples.
        """
        ...

    def get_mir(self, literals: List[str]) -> str:
        """Get MIR function as string for debugging.

        Args:
            literals: List of literal strings for resolution.

        Returns:
            String representation of MIR.
        """
        ...

    def get_num_func_calls(self) -> int:
        """Get number of function calls (built-in functions) in the MIR.

        Returns:
            Number of function signatures.
        """
        ...

    def get_cache_mapping(self) -> List[Tuple[int, int]]:
        """Get cache mapping as list of (init_value_idx, eval_param_idx).

        Returns:
            List of (init_value_index, eval_param_index) tuples.
        """
        ...

    def get_param_defaults(self) -> Dict[str, float]:
        """Get parameter defaults extracted from Verilog-A source.

        Returns:
            Dict mapping parameter name (lowercase) to default value.
            Only includes parameters with literal (constant) default values.
        """
        ...

    def get_str_constants(self) -> Dict[str, str]:
        """Get resolved string constant values.

        Returns:
            Dict mapping operand name (e.g., 'v123') to actual string (e.g., 'gmin').
        """
        ...

    def get_osdi_descriptor(self) -> Dict[str, Any]:
        """Get OSDI-compatible descriptor metadata.

        Returns:
            Dict with 'params', 'nodes', 'jacobian', 'collapsible', 'noise_sources',
            'num_terminals', 'num_states', 'has_bound_step', etc.
        """
        ...

    def get_mir_instructions(self) -> Dict[str, Any]:
        """Export MIR instructions for JAX translation.

        Returns:
            Dict with 'constants', 'bool_constants', 'int_constants', 'str_constants',
            'params', 'instructions', 'blocks', 'function_decls'.
        """
        ...

    def get_init_mir_instructions(self) -> Dict[str, Any]:
        """Export init function MIR instructions for JAX translation.

        Returns:
            Dict with 'constants', 'bool_constants', 'int_constants', 'params',
            'instructions', 'blocks', 'cache_mapping'.
        """
        ...

    def get_dae_system(self) -> Dict[str, Any]:
        """Export DAE system (residuals and Jacobian) with clear naming.

        Returns:
            Dict with 'nodes', 'terminals', 'internal_nodes', 'num_terminals',
            'num_internal', 'residuals', 'jacobian', 'collapsible_pairs', 'num_collapsible'.
        """
        ...

    def run_init_eval(
        self, params: Dict[str, float]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, float, float]]]:
        """Run init function and then eval function.

        This is the proper way to evaluate - init computes cached values that eval needs.

        Args:
            params: Dict mapping parameter names to values.
                    Should include both init and eval parameters.

        Returns:
            Tuple of (residuals, jacobian) where:
            - residuals: List of (resist, react) tuples
            - jacobian: List of (row, col, resist, react) tuples
        """
        ...

    def evaluate(
        self, params: Dict[str, float]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Evaluate the module with given parameter values.

        Args:
            params: Dict mapping parameter names to values (floats).

        Returns:
            Dict with 'residuals', 'jacobian_resist', 'jacobian_react'.
        """
        ...

    def evaluate_full(
        self, params: Dict[str, float], extra_params: Optional[List[float]] = None
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, float, float]]]:
        """Evaluate and return full results as nested structure.

        Args:
            params: Dict mapping parameter names to values.
            extra_params: Optional list of values for extra unnamed parameters
                         (indexed by Param index - len(param_names)).

        Returns:
            Tuple of (residuals, jacobian) where:
            - residuals: List of (resist, react) tuples
            - jacobian: List of (row, col, resist, react) tuples
        """
        ...

    def __repr__(self) -> str:
        """String representation of the module."""
        ...


def compile_va(
    path: str,
    allow_analog_in_cond: bool = False,
    allow_builtin_primitives: bool = False,
) -> List[VaModule]:
    """Compile a Verilog-A file and return module information.

    Args:
        path: Path to the .va file.
        allow_analog_in_cond: Allow analog operators (limexp, ddt, idt) in conditionals.
                              Default is False. Set to True for foundry models that use
                              non-standard Verilog-A (like GF130 PDK).
        allow_builtin_primitives: Allow built-in primitives like `nmos` and `pmos`.
                                  Default is False.

    Returns:
        List of compiled VaModule objects (one per module in the file).
    """
    ...
