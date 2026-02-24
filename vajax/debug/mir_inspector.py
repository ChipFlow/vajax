"""MIR inspection utilities for debugging openvaf_jax translation.

This module provides tools for inspecting MIR (Mid-level IR) data
from OpenVAF compiled models to understand parameter handling,
control flow, and PHI node resolution.

Usage:
    from vajax.debug.mir_inspector import MIRInspector

    inspector = MIRInspector(va_path)
    inspector.print_param_summary()
    inspector.find_phi_nodes_with_value('v3')
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "true")


@dataclass
class ParamSummary:
    """Summary of parameters by kind."""

    total: int
    by_kind: Dict[str, int]
    voltage_params: List[Tuple[int, str]]
    hidden_state_params: List[Tuple[int, str]]
    missing_defaults: List[Tuple[int, str]]


@dataclass
class PHIInfo:
    """Information about a PHI node."""

    result: str
    block: str
    operands: List[Tuple[str, str]]  # [(predecessor_block, value), ...]
    has_zero_operand: bool


class MIRInspector:
    """Inspect MIR data from OpenVAF compiled models."""

    def __init__(self, va_path: Path):
        """Initialize inspector.

        Args:
            va_path: Path to Verilog-A source file
        """
        import openvaf_py

        self.va_path = Path(va_path)
        self.modules = openvaf_py.compile_va(str(self.va_path))
        self.module = self.modules[0]

        # Get MIR data
        self._mir_data = self.module.get_mir_instructions()
        self._init_mir_data = self.module.get_init_mir_instructions()

        # Cache parsed data
        self._eval_params = None
        self._init_params = None

    @property
    def eval_params(self) -> List[Tuple[str, str]]:
        """Get eval function parameters as (name, kind) tuples."""
        if self._eval_params is None:
            names = list(self.module.param_names)
            kinds = list(self.module.param_kinds)
            self._eval_params = list(zip(names, kinds))
        return self._eval_params

    @property
    def init_params(self) -> List[Tuple[str, str]]:
        """Get init function parameters as (name, kind) tuples."""
        if self._init_params is None:
            names = list(self.module.init_param_names)
            kinds = list(self.module.init_param_kinds)
            self._init_params = list(zip(names, kinds))
        return self._init_params

    def get_param_summary(self, func: str = "eval") -> ParamSummary:
        """Get summary of parameters.

        Args:
            func: 'eval' or 'init'

        Returns:
            ParamSummary with categorized parameters
        """
        params = self.eval_params if func == "eval" else self.init_params

        by_kind: Dict[str, int] = {}
        voltage_params = []
        hidden_state_params = []

        for i, (name, kind) in enumerate(params):
            by_kind[kind] = by_kind.get(kind, 0) + 1

            if kind == "voltage":
                voltage_params.append((i, name))
            elif kind == "hidden_state":
                hidden_state_params.append((i, name))

        # Check for missing defaults (init only)
        missing_defaults = []
        if func == "init":
            import openvaf_jax

            translator = openvaf_jax.OpenVAFToJAX(self.module)
            _, init_metadata = translator.translate_init()
            defaults = init_metadata.get("param_defaults", {})
            param_names = init_metadata.get("param_names", [])

            for i, name in enumerate(param_names):
                if name not in defaults and name.lower() not in defaults:
                    if name not in ["$temperature", "mfactor"]:
                        missing_defaults.append((i, name))

        return ParamSummary(
            total=len(params),
            by_kind=by_kind,
            voltage_params=voltage_params,
            hidden_state_params=hidden_state_params,
            missing_defaults=missing_defaults,
        )

    def print_param_summary(self, func: str = "eval"):
        """Print parameter summary."""
        summary = self.get_param_summary(func)

        print(f"\n=== {func.upper()} Parameter Summary ===")
        print(f"Total parameters: {summary.total}")
        print("\nBy kind:")
        for kind, count in sorted(summary.by_kind.items()):
            print(f"  {kind}: {count}")

        if summary.voltage_params:
            print(f"\nVoltage parameters ({len(summary.voltage_params)}):")
            for i, name in summary.voltage_params[:10]:
                print(f"  [{i}] {name}")
            if len(summary.voltage_params) > 10:
                print(f"  ... and {len(summary.voltage_params) - 10} more")

        if summary.missing_defaults:
            print(f"\nMissing defaults ({len(summary.missing_defaults)}):")
            for i, name in summary.missing_defaults:
                print(f"  [{i}] {name}")

    def get_constants(self, func: str = "eval") -> Dict[str, float]:
        """Get MIR constants.

        Args:
            func: 'eval' or 'init'

        Returns:
            Dict mapping value names to constant values
        """
        mir_data = self._mir_data if func == "eval" else self._init_mir_data
        return dict(mir_data.get("constants", {}))

    def find_constants_near(
        self,
        target: float,
        tolerance: float = 0.01,
        func: str = "eval",
    ) -> List[Tuple[str, float]]:
        """Find constants near a target value.

        Useful for finding temperature-related constants (273.15, 298.15, etc.)

        Args:
            target: Target value to search for
            tolerance: Relative tolerance
            func: 'eval' or 'init'

        Returns:
            List of (value_name, constant_value) tuples
        """
        constants = self.get_constants(func)
        results = []

        for name, value in constants.items():
            if abs(value - target) < abs(target) * tolerance:
                results.append((name, value))

        return results

    def get_phi_nodes(self, func: str = "eval") -> List[PHIInfo]:
        """Get all PHI nodes.

        Args:
            func: 'eval' or 'init'

        Returns:
            List of PHIInfo for each PHI node
        """
        mir_data = self._mir_data if func == "eval" else self._init_mir_data
        instructions = mir_data.get("instructions", [])

        phi_nodes = []
        for inst in instructions:
            if inst.get("opcode") == "phi":
                operands = []
                phi_ops = inst.get("phi_operands", [])
                for op in phi_ops:
                    operands.append((op.get("block", ""), op.get("value", "")))

                has_zero = any(v == "v3" for _, v in operands)  # v3 is typically 0.0

                phi_nodes.append(
                    PHIInfo(
                        result=inst.get("result", ""),
                        block=inst.get("block", ""),
                        operands=operands,
                        has_zero_operand=has_zero,
                    )
                )

        return phi_nodes

    def find_phi_nodes_with_value(
        self,
        value: str,
        func: str = "eval",
    ) -> List[PHIInfo]:
        """Find PHI nodes that have a specific value as an operand.

        Useful for finding PHI nodes with v3 (0.0) which often indicate
        conditional branches where one path doesn't compute a value.

        Args:
            value: Value name to search for (e.g., 'v3')
            func: 'eval' or 'init'

        Returns:
            List of PHIInfo for matching PHI nodes
        """
        all_phis = self.get_phi_nodes(func)
        return [phi for phi in all_phis if any(v == value for _, v in phi.operands)]

    def print_phi_summary(self, func: str = "eval"):
        """Print summary of PHI nodes."""
        phi_nodes = self.get_phi_nodes(func)

        print(f"\n=== {func.upper()} PHI Node Summary ===")
        print(f"Total PHI nodes: {len(phi_nodes)}")

        # Count by number of operands
        by_operand_count: Dict[int, int] = {}
        for phi in phi_nodes:
            n = len(phi.operands)
            by_operand_count[n] = by_operand_count.get(n, 0) + 1

        print("\nBy operand count:")
        for n, count in sorted(by_operand_count.items()):
            print(f"  {n} operands: {count}")

        # PHIs with zero operand
        zero_phis = [phi for phi in phi_nodes if phi.has_zero_operand]
        print(f"\nPHIs with zero (v3) operand: {len(zero_phis)}")

        if zero_phis:
            print("\nFirst 5 PHIs with zero operand:")
            for phi in zero_phis[:5]:
                print(f"  {phi.result} in {phi.block}:")
                for pred, val in phi.operands:
                    marker = " <-- ZERO" if val == "v3" else ""
                    print(f"    {pred} -> {val}{marker}")

    def get_block_count(self, func: str = "eval") -> int:
        """Get number of basic blocks."""
        mir_data = self._mir_data if func == "eval" else self._init_mir_data
        return len(mir_data.get("blocks", {}))

    def print_mir_stats(self):
        """Print overall MIR statistics."""
        print("\n=== MIR Statistics ===")

        for func in ["init", "eval"]:
            mir_data = self._mir_data if func == "eval" else self._init_mir_data

            blocks = mir_data.get("blocks", {})
            instructions = mir_data.get("instructions", [])
            constants = mir_data.get("constants", {})

            phi_count = sum(1 for i in instructions if i.get("opcode") == "phi")

            print(f"\n{func.upper()}:")
            print(f"  Blocks: {len(blocks)}")
            print(f"  Instructions: {len(instructions)}")
            print(f"  PHI nodes: {phi_count}")
            print(f"  Constants: {len(constants)}")

    def find_type_param(self) -> Optional[Tuple[int, str, str]]:
        """Find TYPE parameter (for NMOS/PMOS models).

        Returns:
            Tuple of (index, name, kind) or None if not found
        """
        for i, (name, kind) in enumerate(self.eval_params):
            if name.upper() == "TYPE":
                return (i, name, kind)
        return None

    def print_type_param_info(self):
        """Print information about TYPE parameter handling."""
        type_info = self.find_type_param()

        print("\n=== TYPE Parameter Info ===")
        if type_info:
            idx, name, kind = type_info
            print(f"Found: [{idx}] {name} ({kind})")

            # Check if TYPE is in init params too
            for i, (n, k) in enumerate(self.init_params):
                if n.upper() == "TYPE":
                    print(f"In init: [{i}] {n} ({k})")
        else:
            print("TYPE parameter not found")


def inspect_model(va_path: Path):
    """Quick inspection of a model.

    Usage:
        inspect_model("vendor/OpenVAF/integration_tests/PSP102/psp102.va")
    """
    inspector = MIRInspector(va_path)
    inspector.print_mir_stats()
    inspector.print_param_summary("eval")
    inspector.print_phi_summary("eval")
    inspector.print_type_param_info()
