"""Parameter Kind Analyzer.

Tools for understanding parameter kinds in OpenVAF models and
comparing with OSDI descriptors.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import openvaf_py

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ParamInfo:
    """Detailed information about a parameter."""

    index: int
    name: str
    kind: str
    value_name: str  # e.g., 'v7780'

    # Source information
    default_value: Optional[float] = None
    is_computed_in_eval: bool = False
    is_computed_in_init: bool = False
    is_in_cache: bool = False
    cache_index: Optional[int] = None

    # OSDI comparison
    osdi_name: Optional[str] = None
    osdi_units: Optional[str] = None


@dataclass
class KindSummary:
    """Summary of params by kind."""

    kind: str
    count: int
    params: List[ParamInfo] = field(default_factory=list)


class ParamAnalyzer:
    """Analyze parameters in an OpenVAF model."""

    def __init__(self, va_path: Path):
        """Initialize with a Verilog-A file path."""
        self.va_path = Path(va_path).absolute()

        # Compile the model
        modules = openvaf_py.compile_va(str(self.va_path))
        self.module = modules[0]

        # Get data
        self.mir_data = self.module.get_mir_instructions()
        self.init_mir_data = self.module.get_init_mir_instructions()
        self.osdi_descriptor = self.module.get_osdi_descriptor()

        # Extract useful data
        self.eval_params = list(self.mir_data.get("params", []))
        self.init_params = list(self.init_mir_data.get("params", []))
        self.eval_instructions = self.mir_data.get("instructions", [])
        self.init_instructions = self.init_mir_data.get("instructions", [])

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices."""
        # Map value to param index
        all_func_params = self.module.get_all_func_params()
        self.param_idx_to_val = {p[0]: f"v{p[1]}" for p in all_func_params}
        self.val_to_param_idx = {f"v{p[1]}": p[0] for p in all_func_params}

        # Track which values are produced by instructions
        self.eval_producers = set()
        for inst in self.eval_instructions:
            result = inst.get("result", "")
            if result:
                self.eval_producers.add(result)

        self.init_producers = set()
        for inst in self.init_instructions:
            result = inst.get("result", "")
            if result:
                self.init_producers.add(result)

        # OSDI params by name
        self.osdi_params = {}
        for p in self.osdi_descriptor.get("params", []):
            self.osdi_params[p["name"].upper()] = p

    def _detect_param_kind(self, idx: int, name: str, value_name: str) -> str:
        """Detect the kind of a parameter.

        This mirrors the logic in OpenVAFToJAX._build_param_inputs.
        """
        # Check for voltage params: V(node) or V(node1, node2)
        if name.startswith("V(") and name.endswith(")"):
            return "voltage"

        # Check for hidden_state params (end with _i or similar patterns)
        if "_i" in name.lower() or name.endswith("_i"):
            return "hidden_state"

        # Check for param_given params
        if name.endswith("_given"):
            return "param_given"

        # Check for temperature
        if name.lower() == "temperature" or name.lower() == "$temperature":
            return "temperature"

        # Check for system functions
        if name.lower() == "mfactor" or name.lower() == "$mfactor":
            return "sysfun"

        # Check if it's a known model param from OSDI
        upper_name = name.upper()
        if upper_name in self.osdi_params:
            return "param"

        # Check if the value is computed
        if value_name in self.eval_producers:
            return "computed"

        # Default to unknown
        return "unknown"

    def analyze_eval_params(self) -> Dict[str, KindSummary]:
        """Analyze all eval function parameters.

        Returns:
            Dict mapping kind -> KindSummary
        """
        # Import translator to get proper param analysis
        from openvaf_jax import OpenVAFToJAX

        translator = OpenVAFToJAX(self.module)

        # Get param info from translate_eval
        eval_fn, eval_meta = translator.translate_eval(params={}, temperature=300.0)

        param_names = eval_meta.get("param_names", [])
        param_kinds = eval_meta.get("param_kinds", [])
        shared_inputs = eval_meta.get("shared_inputs", [])
        shared_indices = eval_meta.get("shared_indices", [])

        # Get cache mapping
        cache_mapping = translator.cache_mapping
        cache_mapped_params = {}
        for i, m in enumerate(cache_mapping):
            eval_param_idx = m.get("eval_param", -1)
            cache_mapped_params[eval_param_idx] = i

        # Build summaries by kind
        summaries: Dict[str, KindSummary] = {}

        for i, (name, kind) in enumerate(zip(param_names, param_kinds)):
            value_name = self.eval_params[i] if i < len(self.eval_params) else f"v{i}"

            info = ParamInfo(
                index=i,
                name=name,
                kind=kind,
                value_name=value_name,
            )

            # Check if in cache
            if i in cache_mapped_params:
                info.is_in_cache = True
                info.cache_index = cache_mapped_params[i]

            # Check if computed
            if value_name in self.eval_producers:
                info.is_computed_in_eval = True
            if value_name in self.init_producers:
                info.is_computed_in_init = True

            # Get shared input value if available
            if i in shared_indices:
                shared_idx = shared_indices.index(i)
                if shared_idx < len(shared_inputs):
                    info.default_value = shared_inputs[shared_idx]

            # Add to summary
            if kind not in summaries:
                summaries[kind] = KindSummary(kind=kind, count=0, params=[])
            summaries[kind].count += 1
            summaries[kind].params.append(info)

        return summaries

    def find_problematic_params(self) -> List[ParamInfo]:
        """Find params that might cause issues.

        These are params with kind='unknown' that are:
        - Not computed in eval
        - Not computed in init
        - Not in cache
        - But are used in the model

        Returns:
            List of problematic ParamInfo
        """
        summaries = self.analyze_eval_params()
        problems = []

        unknown_summary = summaries.get("unknown", KindSummary(kind="unknown", count=0))
        for info in unknown_summary.params:
            # Check if it's used but not sourced
            if not info.is_computed_in_eval and not info.is_in_cache:
                # Check if it's actually used
                value_name = info.value_name
                used = any(
                    value_name in inst.get("operands", []) for inst in self.eval_instructions
                )
                if used:
                    problems.append(info)

        return problems

    def compare_with_osdi(self) -> Dict[str, Any]:
        """Compare param structure with OSDI descriptor.

        Returns:
            Dict with comparison results
        """
        osdi_params = self.osdi_descriptor.get("params", [])
        osdi_nodes = self.osdi_descriptor.get("nodes", [])

        return {
            "osdi_param_count": len(osdi_params),
            "eval_param_count": len(self.eval_params),
            "osdi_node_count": len(osdi_nodes),
            "osdi_params": [p["name"] for p in osdi_params],
            "osdi_nodes": osdi_nodes,
        }

    def print_summary(self):
        """Print a human-readable summary of param analysis."""
        summaries = self.analyze_eval_params()

        print(f"\n{'=' * 60}")
        print(f"Parameter Analysis: {self.va_path.name}")
        print(f"{'=' * 60}")

        print(f"\nTotal eval params: {len(self.eval_params)}")
        print(f"Total init params: {len(self.init_params)}")

        print("\nParams by kind:")
        for kind, summary in sorted(summaries.items(), key=lambda x: -x[1].count):
            print(f"  {kind}: {summary.count}")

        # Show problematic params
        problems = self.find_problematic_params()
        if problems:
            print("\nProblematic params (unknown, unused but referenced):")
            for p in problems[:10]:
                print(f"  [{p.index}] {p.name} ({p.value_name})")
                print(
                    f"      computed_in_eval={p.is_computed_in_eval}, "
                    f"computed_in_init={p.is_computed_in_init}, "
                    f"in_cache={p.is_in_cache}"
                )
            if len(problems) > 10:
                print(f"  ... and {len(problems) - 10} more")

        # OSDI comparison
        osdi_cmp = self.compare_with_osdi()
        print("\nOSDI comparison:")
        print(f"  OSDI params: {osdi_cmp['osdi_param_count']}")
        print(f"  Eval params: {osdi_cmp['eval_param_count']}")
        print(f"  OSDI nodes: {osdi_cmp['osdi_nodes']}")


def analyze_model(va_path: Path):
    """Convenience function to analyze a model's params."""
    analyzer = ParamAnalyzer(va_path)
    analyzer.print_summary()
    return analyzer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python param_analyzer.py <va_file>")
        sys.exit(1)

    va_path = Path(sys.argv[1])
    analyze_model(va_path)
