"""MIR Value Flow Tracer.

Tools for tracing value flow through OpenVAF MIR to understand
where values come from and how they're used.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import openvaf_py


@dataclass
class ValueInfo:
    """Information about a MIR value."""

    name: str  # e.g., 'v7780'
    value_id: int  # e.g., 7780

    # Source information
    is_param: bool = False
    param_index: Optional[int] = None
    param_kind: Optional[str] = None  # 'param', 'voltage', 'hidden_state', etc.

    is_computed: bool = False
    producer_block: Optional[str] = None
    producer_opcode: Optional[str] = None
    producer_operands: List[str] = field(default_factory=list)

    is_constant: bool = False
    constant_value: Optional[float] = None

    is_cache: bool = False
    cache_index: Optional[int] = None

    # Usage information
    consumer_blocks: List[str] = field(default_factory=list)
    consumer_count: int = 0


@dataclass
class ParamAnalysis:
    """Analysis of all params in a model."""

    total_params: int = 0
    by_kind: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)  # kind -> [(idx, name)]
    unknown_params: List[Tuple[int, str]] = field(default_factory=list)
    cache_mapped_params: Set[int] = field(default_factory=set)


class MIRTracer:
    """Trace value flow through MIR."""

    def __init__(self, va_path: Path):
        """Initialize with a Verilog-A file path."""
        self.va_path = Path(va_path).absolute()

        # Compile the model
        modules = openvaf_py.compile_va(str(self.va_path))
        self.module = modules[0]

        # Get MIR data
        self.mir_data = self.module.get_mir_instructions()
        self.init_mir_data = self.module.get_init_mir_instructions()
        self.dae_data = self.module.get_dae_system()

        # Extract useful data
        self.params = list(self.mir_data.get("params", []))
        self.instructions = self.mir_data.get("instructions", [])
        self.constants = dict(self.mir_data.get("constants", {}))
        self.blocks = self.mir_data.get("blocks", {})

        self.init_params = list(self.init_mir_data.get("params", []))

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for efficient querying."""
        # Map value name to param index
        self.param_name_to_idx = {name: idx for idx, name in enumerate(self.params)}

        # Map value name to producer instruction
        self.producers: Dict[str, Dict] = {}
        for inst in self.instructions:
            result = inst.get("result", "")
            if result:
                self.producers[result] = inst

        # Map value name to consumer instructions
        self.consumers: Dict[str, List[Dict]] = {}
        for inst in self.instructions:
            for operand in inst.get("operands", []):
                if operand not in self.consumers:
                    self.consumers[operand] = []
                self.consumers[operand].append(inst)

        # Get all_func_params for param_idx -> value mapping
        all_func_params = self.module.get_all_func_params()
        self.param_idx_to_val = {p[0]: f"v{p[1]}" for p in all_func_params}
        self.val_to_param_idx = {f"v{p[1]}": p[0] for p in all_func_params}

    def trace_value(self, value_name: str) -> ValueInfo:
        """Trace where a value comes from and how it's used.

        Args:
            value_name: The value name (e.g., 'v7780')

        Returns:
            ValueInfo with source and usage information
        """
        # Extract value ID
        if value_name.startswith("v"):
            try:
                value_id = int(value_name[1:])
            except ValueError:
                value_id = -1
        else:
            value_id = -1

        info = ValueInfo(name=value_name, value_id=value_id)

        # Check if it's a constant
        if value_name in self.constants:
            info.is_constant = True
            info.constant_value = self.constants[value_name]
            return info

        # Check if it's produced by an instruction
        if value_name in self.producers:
            producer = self.producers[value_name]
            info.is_computed = True
            info.producer_block = producer.get("block", "")
            info.producer_opcode = producer.get("opcode", "")
            info.producer_operands = producer.get("operands", [])

        # Check if it's a param
        if value_name in self.param_name_to_idx:
            info.is_param = True
            info.param_index = self.param_name_to_idx[value_name]

        # Get consumers
        if value_name in self.consumers:
            consumers = self.consumers[value_name]
            info.consumer_count = len(consumers)
            info.consumer_blocks = list(set(c.get("block", "") for c in consumers))

        return info

    def analyze_params(self, translator=None) -> ParamAnalysis:
        """Analyze all params and their kinds.

        Args:
            translator: Optional OpenVAFToJAX translator with param kind info

        Returns:
            ParamAnalysis with categorized params
        """
        analysis = ParamAnalysis()
        analysis.total_params = len(self.params)

        # Get param kinds if translator provided
        if translator:
            # Use translate_eval to get param info
            eval_fn, eval_meta = translator.translate_eval(params={}, temperature=300.0)
            param_kinds = eval_meta.get("param_kinds", [])
            param_names = eval_meta.get("param_names", [])

            for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
                if kind not in analysis.by_kind:
                    analysis.by_kind[kind] = []
                analysis.by_kind[kind].append((i, name))

                if kind == "unknown":
                    analysis.unknown_params.append((i, name))

            # Check cache mapping
            cache_mapping = translator.cache_mapping
            for m in cache_mapping:
                eval_param_idx = m.get("eval_param", -1)
                analysis.cache_mapped_params.add(eval_param_idx)

        return analysis

    def find_unproduced_params(self) -> List[Tuple[int, str]]:
        """Find params that are used but never produced.

        These are potential problem params - they're inputs that
        need to come from somewhere.

        Returns:
            List of (param_index, param_name) for unproduced params
        """
        unproduced = []

        for idx, name in enumerate(self.params):
            # Skip if it's produced by an instruction
            if name in self.producers:
                continue

            # Skip if it's a constant
            if name in self.constants:
                continue

            # Check if it's actually used
            if name in self.consumers and len(self.consumers[name]) > 0:
                unproduced.append((idx, name))

        return unproduced

    def get_value_dependency_chain(self, value_name: str, max_depth: int = 10) -> Dict:
        """Get the dependency chain for a value.

        Args:
            value_name: Starting value
            max_depth: Maximum recursion depth

        Returns:
            Nested dict representing the dependency tree
        """

        def trace(name: str, depth: int, visited: Set[str]) -> Dict:
            if depth > max_depth or name in visited:
                return {"name": name, "truncated": True}

            visited.add(name)
            result = {"name": name}

            if name in self.producers:
                producer = self.producers[name]
                result["opcode"] = producer.get("opcode", "")
                result["block"] = producer.get("block", "")
                result["operands"] = []

                for op in producer.get("operands", []):
                    result["operands"].append(trace(op, depth + 1, visited.copy()))
            elif name in self.constants:
                result["constant"] = self.constants[name]
            elif name in self.param_name_to_idx:
                result["param_index"] = self.param_name_to_idx[name]

            return result

        return trace(value_name, 0, set())

    def print_value_trace(self, value_name: str):
        """Print a human-readable trace of a value."""
        info = self.trace_value(value_name)

        print(f"\n{'=' * 60}")
        print(f"Value: {info.name} (id={info.value_id})")
        print(f"{'=' * 60}")

        print("\nSource:")
        if info.is_constant:
            print(f"  Constant: {info.constant_value}")
        elif info.is_computed:
            print(f"  Computed in {info.producer_block}:")
            print(f"    {info.producer_opcode}({', '.join(info.producer_operands)})")
        elif info.is_param:
            print(f"  Parameter at index {info.param_index}")
            if info.param_kind:
                print(f"    Kind: {info.param_kind}")
        else:
            print("  UNKNOWN SOURCE")

        print("\nUsage:")
        print(f"  Used {info.consumer_count} times in blocks: {info.consumer_blocks}")

        if info.is_cache:
            print("\nCache:")
            print(f"  Mapped to cache index {info.cache_index}")


def trace_model(va_path: Path, values: List[str] = None):
    """Convenience function to trace values in a model.

    Args:
        va_path: Path to Verilog-A file
        values: List of value names to trace (if None, traces unknown params)
    """
    tracer = MIRTracer(va_path)

    if values is None:
        # Find and trace unproduced params
        unproduced = tracer.find_unproduced_params()
        print(f"Found {len(unproduced)} unproduced params that are used:")
        for idx, name in unproduced[:10]:
            tracer.print_value_trace(name)
        if len(unproduced) > 10:
            print(f"\n... and {len(unproduced) - 10} more")
    else:
        for name in values:
            tracer.print_value_trace(name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mir_tracer.py <va_file> [value1] [value2] ...")
        sys.exit(1)

    va_path = Path(sys.argv[1])
    values = sys.argv[2:] if len(sys.argv) > 2 else None
    trace_model(va_path, values)
