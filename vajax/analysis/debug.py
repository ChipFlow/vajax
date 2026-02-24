"""Debug and print commands for circuit inspection.

Implements VACASK-style print commands:
- print stats: Circuit statistics
- print devices: All device instances
- print models: All model definitions
- print instance("name"): Specific instance parameters
- print model("name"): Specific model parameters

Also provides parameter tracing utilities for debugging the 4-layer
parameter flow:
    Netlist -> Circuit AST -> param_array -> OpenVAF evaluation
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from vajax.analysis.engine import CircuitEngine
    from vajax.netlist.circuit import Circuit, PrintDirective

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Tracing
# =============================================================================


@dataclass
class ParamTrace:
    """Trace of a parameter through all mapping layers.

    Tracks how a parameter flows from netlist to OpenVAF evaluation:
    1. Instance params (from netlist device line)
    2. Model params (from model card)
    3. OpenVAF param_kinds (param, hidden_state, voltage, etc.)
    4. Final value in shared_params or device_params arrays
    """

    param_name: str
    instance_name: str

    # Layer 1: Netlist instance
    in_instance: bool = False
    instance_value: Any = None

    # Layer 2: Model card
    in_model: bool = False
    model_value: Any = None
    model_name: Optional[str] = None

    # Layer 3: OpenVAF compiled model
    param_kind: Optional[str] = None  # 'param', 'hidden_state', 'voltage', etc.
    param_index: Optional[int] = None  # Index in param_names array

    # Layer 4: Final value
    final_value: Any = None
    source: Optional[str] = None  # Where final value came from

    def __str__(self) -> str:
        lines = [f"Parameter trace: {self.instance_name}.{self.param_name}"]
        lines.append("-" * 50)

        # Layer 1
        if self.in_instance:
            lines.append(f"  [Instance] {self.param_name} = {self.instance_value}")
        else:
            lines.append("  [Instance] (not set)")

        # Layer 2
        if self.in_model:
            lines.append(f"  [Model:{self.model_name}] {self.param_name} = {self.model_value}")
        else:
            lines.append(f"  [Model:{self.model_name}] (not set)")

        # Layer 3
        if self.param_kind:
            lines.append(f"  [OpenVAF] kind={self.param_kind}, index={self.param_index}")
        else:
            lines.append("  [OpenVAF] (not mapped)")

        # Layer 4
        if self.final_value is not None:
            lines.append(f"  [Final] value={self.final_value} (from {self.source})")
        else:
            lines.append("  [Final] (no value)")

        return "\n".join(lines)


def trace_param(
    engine: "CircuitEngine",
    instance_name: str,
    param_name: str,
) -> ParamTrace:
    """Trace a parameter through all mapping layers.

    Args:
        engine: CircuitEngine with parsed circuit and compiled models
        instance_name: Device instance name (e.g., "u1.mp.m")
        param_name: Parameter name to trace (e.g., "w", "l", "type")

    Returns:
        ParamTrace with values at each layer
    """
    trace = ParamTrace(param_name=param_name, instance_name=instance_name)
    param_lower = param_name.lower()

    # Find the device in engine.devices
    device = None
    for dev in engine.devices:
        if dev.get("name") == instance_name:
            device = dev
            break

    if not device:
        return trace

    # Layer 1: Instance params - check original netlist instance
    # The device dict 'original_params' contains the actual netlist params
    original_params = device.get("original_params", {})
    for k, v in original_params.items():
        if k.lower() == param_lower:
            trace.in_instance = True
            trace.instance_value = v
            break

    # Layer 2: Model params from model card
    # Try to find the model card - may be model name or variant (e.g., psp103n/psp103p)
    model_name = device.get("model")
    model_card_name = device.get("model_card")  # May be stored explicitly
    trace.model_name = model_card_name or model_name

    if engine.circuit:
        # Try direct lookup first
        model = engine.circuit.models.get(model_name)

        # If not found, try to find a variant (e.g., psp103 -> psp103n, psp103p)
        if not model and model_name:
            for candidate in engine.circuit.models.keys():
                if candidate.startswith(model_name):
                    model = engine.circuit.models.get(candidate)
                    trace.model_name = candidate
                    break

        if model and model.params:
            for k, v in model.params.items():
                if k.lower() == param_lower:
                    trace.in_model = True
                    trace.model_value = v
                    break

    # If not found in original_params or model, get value from merged params
    if not trace.in_instance and not trace.in_model:
        merged_params = device.get("params", {})
        for k, v in merged_params.items():
            if k.lower() == param_lower:
                trace.instance_value = v  # Store the value but mark as default
                break

    # Layer 3 & 4: OpenVAF mapping
    # Get the model type (e.g., 'psp103' from 'psp103n')
    model_type = device.get("model_type", model_name)
    compiled = engine._compiled_models.get(model_type)

    if compiled:
        param_names = compiled.get("param_names", [])
        param_kinds = compiled.get("param_kinds", [])

        # Find the parameter in OpenVAF's param list
        for idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if name.lower() == param_lower:
                trace.param_kind = kind
                trace.param_index = idx
                break

    # Layer 4: Get final value from split eval params (shared_params/device_params)
    if compiled and trace.param_index is not None:
        shared_indices = compiled.get("shared_indices", [])
        varying_indices = compiled.get("varying_indices", [])
        shared_params = compiled.get("shared_params")
        device_params = compiled.get("device_params")

        if shared_params is not None and device_params is not None:
            # Find device index
            model_devices = [
                d for d in engine.devices if d.get("model_type", d.get("model")) == model_type
            ]
            for dev_idx, dev in enumerate(model_devices):
                if dev.get("name") == instance_name:
                    param_idx = trace.param_index
                    if param_idx in shared_indices:
                        # Constant param - same for all devices
                        shared_pos = shared_indices.index(param_idx)
                        trace.final_value = float(shared_params[shared_pos])
                    elif param_idx in varying_indices:
                        # Varying param - different per device
                        varying_pos = varying_indices.index(param_idx)
                        if dev_idx < device_params.shape[0]:
                            trace.final_value = float(device_params[dev_idx, varying_pos])

                    # Determine source
                    if trace.final_value is not None:
                        if trace.in_instance:
                            trace.source = "instance"
                        elif trace.in_model:
                            trace.source = "model"
                        else:
                            trace.source = "default"
                    break

    return trace


def trace_all_params(
    engine: "CircuitEngine",
    instance_name: str,
) -> List[ParamTrace]:
    """Trace all parameters for a device instance.

    Args:
        engine: CircuitEngine with parsed circuit and compiled models
        instance_name: Device instance name

    Returns:
        List of ParamTrace for each parameter
    """
    traces = []

    # Find the device
    device = None
    for dev in engine.devices:
        if dev.get("name") == instance_name:
            device = dev
            break

    if not device:
        return traces

    # Get all params from instance and model
    all_params = set()

    instance_params = device.get("params", {})
    all_params.update(k.lower() for k in instance_params.keys())

    model_name = device.get("model")
    if model_name and engine.circuit:
        model = engine.circuit.models.get(model_name)
        if model and model.params:
            all_params.update(k.lower() for k in model.params.keys())

    # Trace each param
    for param in sorted(all_params):
        traces.append(trace_param(engine, instance_name, param))

    return traces


def check_param_coverage(
    engine: "CircuitEngine",
    instance_name: str,
) -> Dict[str, Any]:
    """Check that all netlist parameters are mapped to OpenVAF.

    Args:
        engine: CircuitEngine with parsed circuit and compiled models
        instance_name: Device instance name

    Returns:
        Dict with 'mapped', 'unmapped', and 'coverage' fields
    """
    traces = trace_all_params(engine, instance_name)

    mapped = []
    unmapped = []

    for trace in traces:
        if trace.param_kind is not None:
            mapped.append(trace.param_name)
        else:
            unmapped.append(trace.param_name)

    total = len(traces)
    coverage = len(mapped) / total if total > 0 else 1.0

    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "total": total,
        "coverage": coverage,
        "coverage_pct": f"{coverage * 100:.1f}%",
        "traces": traces,
    }


def get_coverage_breakdown(
    engine: "CircuitEngine",
    instance_name: str,
) -> Dict[str, Any]:
    """Get detailed coverage breakdown by param_kind and source.

    Args:
        engine: CircuitEngine with parsed circuit and compiled models
        instance_name: Device instance name

    Returns:
        Dict with breakdowns by param_kind and source
    """
    coverage = check_param_coverage(engine, instance_name)
    traces = coverage["traces"]

    # Count by param_kind
    by_kind: Dict[str, int] = {}
    for trace in traces:
        kind = trace.param_kind or "unmapped"
        by_kind[kind] = by_kind.get(kind, 0) + 1

    # Count by source (instance, model, default, unmapped)
    by_source: Dict[str, int] = {}
    for trace in traces:
        if trace.param_kind is None:
            source = "unmapped"
        elif trace.in_instance:
            source = "instance"
        elif trace.in_model:
            source = "model"
        else:
            source = "default"
        by_source[source] = by_source.get(source, 0) + 1

    return {
        **coverage,
        "by_kind": by_kind,
        "by_source": by_source,
    }


def format_coverage_chart(
    engine: "CircuitEngine",
    instance_name: str,
    width: int = 40,
) -> str:
    """Format coverage breakdown as ASCII bar charts.

    Args:
        engine: CircuitEngine
        instance_name: Device instance name
        width: Chart width in characters

    Returns:
        Formatted ASCII chart string
    """
    breakdown = get_coverage_breakdown(engine, instance_name)
    total = breakdown["total"]
    if total == 0:
        return f"No parameters found for {instance_name}"

    lines = []
    lines.append(f"Parameter Coverage: {instance_name}")
    lines.append("=" * 60)
    lines.append(f"Total: {total} parameters, {breakdown['coverage_pct']} mapped")
    lines.append("")

    # By param_kind chart
    lines.append("By OpenVAF param_kind:")
    lines.append("-" * 60)
    by_kind = breakdown["by_kind"]
    max_count = max(by_kind.values()) if by_kind else 1
    for kind in sorted(by_kind.keys()):
        count = by_kind[kind]
        pct = count / total * 100
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len
        lines.append(f"  {kind:15} {bar:40} {count:4} ({pct:5.1f}%)")

    lines.append("")

    # By source chart
    lines.append("By value source:")
    lines.append("-" * 60)
    by_source = breakdown["by_source"]
    source_order = ["instance", "model", "default", "unmapped"]
    source_labels = {
        "instance": "Instance params",
        "model": "Model card",
        "default": "OpenVAF default",
        "unmapped": "Not mapped",
    }
    max_count = max(by_source.values()) if by_source else 1
    for source in source_order:
        if source in by_source:
            count = by_source[source]
            pct = count / total * 100
            bar_len = int(count / max_count * width)
            bar = "█" * bar_len
            label = source_labels.get(source, source)
            lines.append(f"  {label:15} {bar:40} {count:4} ({pct:5.1f}%)")

    return "\n".join(lines)


def format_param_trace(
    engine: "CircuitEngine",
    instance_name: str,
    param_name: Optional[str] = None,
) -> str:
    """Format parameter trace for display.

    Args:
        engine: CircuitEngine
        instance_name: Device instance name
        param_name: Specific param to trace, or None for all

    Returns:
        Formatted trace string
    """
    if param_name:
        trace = trace_param(engine, instance_name, param_name)
        return str(trace)
    else:
        traces = trace_all_params(engine, instance_name)
        coverage = check_param_coverage(engine, instance_name)

        lines = [f"Parameter coverage for {instance_name}: {coverage['coverage_pct']}"]
        lines.append(f"  Mapped: {len(coverage['mapped'])}")
        lines.append(f"  Unmapped: {len(coverage['unmapped'])}")

        if coverage["unmapped"]:
            lines.append("\nUnmapped parameters:")
            for p in coverage["unmapped"]:
                lines.append(f"  - {p}")

        lines.append("\nAll parameters:")
        for trace in traces:
            status = "✓" if trace.param_kind else "✗"
            value = (
                trace.final_value
                if trace.final_value is not None
                else trace.instance_value or trace.model_value
            )
            lines.append(f"  {status} {trace.param_name}: {value} ({trace.source or 'unmapped'})")

        return "\n".join(lines)


def format_stats(circuit: "Circuit", engine: "CircuitEngine" = None) -> str:
    """Format circuit statistics (print stats command).

    Args:
        circuit: Parsed Circuit object
        engine: Optional CircuitEngine for additional stats

    Returns:
        Formatted statistics string
    """
    lines = ["Circuit Statistics:"]
    lines.append("-" * 40)

    # Basic circuit info
    if circuit.title:
        lines.append(f"Title: {circuit.title}")

    stats = circuit.stats()
    lines.append(f"Number of subcircuits: {stats['num_subckts']}")
    lines.append(f"Number of models: {stats['num_models']}")
    lines.append(f"Number of top instances: {stats['num_top_instances']}")
    lines.append(f"Number of globals: {stats['num_globals']}")

    # Engine-level stats if available
    if engine:
        lines.append(f"Number of nodes: {engine.num_nodes}")
        lines.append(f"Number of devices: {len(engine.devices)}")
        lines.append(f"Number of flat instances: {len(engine.flat_instances)}")

        # Count device types
        device_counts = {}
        for dev in engine.devices:
            model = dev.get("model", "unknown")
            device_counts[model] = device_counts.get(model, 0) + 1

        if device_counts:
            lines.append("\nDevice breakdown:")
            for model, count in sorted(device_counts.items()):
                lines.append(f"  {model}: {count}")

    return "\n".join(lines)


def format_devices(circuit: "Circuit", engine: "CircuitEngine" = None) -> str:
    """Format all device instances (print devices command).

    Args:
        circuit: Parsed Circuit object
        engine: Optional CircuitEngine for flattened device info

    Returns:
        Formatted device list string
    """
    lines = ["Device Instances:"]
    lines.append("-" * 60)

    if engine and engine.devices:
        for dev in engine.devices:
            name = dev.get("name", "unknown")
            model = dev.get("model", "unknown")
            terminals = dev.get("terminals", [])
            lines.append(f"{name}: {model} ({', '.join(terminals)})")
    else:
        # Use top-level instances
        for inst in circuit.top_instances:
            lines.append(f"{inst.name}: {inst.model} ({', '.join(inst.terminals)})")

    return "\n".join(lines)


def format_models(circuit: "Circuit") -> str:
    """Format all model definitions (print models command).

    Args:
        circuit: Parsed Circuit object

    Returns:
        Formatted model list string
    """
    lines = ["Model Definitions:"]
    lines.append("-" * 60)

    for name, model in sorted(circuit.models.items()):
        lines.append(f"\n{name}:")
        lines.append(f"  Module: {model.module}")
        if model.params:
            lines.append("  Parameters:")
            for k, v in sorted(model.params.items()):
                lines.append(f"    {k} = {v}")

    return "\n".join(lines)


def format_instance(
    instance_names: list[str],
    circuit: "Circuit",
    engine: "CircuitEngine" = None,
) -> str:
    """Format specific instance parameters (print instance command).

    Args:
        instance_names: List of instance names to print
        circuit: Parsed Circuit object
        engine: Optional CircuitEngine for device lookup

    Returns:
        Formatted instance info string
    """
    lines = []

    for name in instance_names:
        # Strip quotes if present
        name = name.strip("\"'")
        lines.append(f"\nInstance: {name}")
        lines.append("-" * 40)

        # Look up in engine devices first
        found = False
        if engine and engine.devices:
            for dev in engine.devices:
                if dev.get("name") == name:
                    lines.append(f"Model: {dev.get('model')}")
                    lines.append(f"Terminals: {', '.join(dev.get('terminals', []))}")
                    if dev.get("params"):
                        lines.append("Parameters:")
                        for k, v in sorted(dev["params"].items()):
                            lines.append(f"  {k} = {v}")
                    found = True
                    break

        # Fall back to top instances
        if not found:
            for inst in circuit.top_instances:
                if inst.name == name:
                    lines.append(f"Model: {inst.model}")
                    lines.append(f"Terminals: {', '.join(inst.terminals)}")
                    if inst.params:
                        lines.append("Parameters:")
                        for k, v in sorted(inst.params.items()):
                            lines.append(f"  {k} = {v}")
                    found = True
                    break

        if not found:
            lines.append("  (not found)")

    return "\n".join(lines)


def format_model(model_names: list[str], circuit: "Circuit") -> str:
    """Format specific model parameters (print model command).

    Args:
        model_names: List of model names to print
        circuit: Parsed Circuit object

    Returns:
        Formatted model info string
    """
    lines = []

    for name in model_names:
        # Strip quotes if present
        name = name.strip("\"'")
        lines.append(f"\nModel: {name}")
        lines.append("-" * 40)

        if name in circuit.models:
            model = circuit.models[name]
            lines.append(f"Module: {model.module}")
            if model.params:
                lines.append("Parameters:")
                for k, v in sorted(model.params.items()):
                    lines.append(f"  {k} = {v}")
        else:
            lines.append("  (not found)")

    return "\n".join(lines)


def execute_print_directive(
    directive: "PrintDirective",
    circuit: "Circuit",
    engine: "CircuitEngine" = None,
) -> str:
    """Execute a print directive and return the output.

    Args:
        directive: Parsed PrintDirective
        circuit: Parsed Circuit object
        engine: Optional CircuitEngine for device info

    Returns:
        Formatted output string
    """
    subcommand = directive.subcommand.lower()

    if subcommand == "stats":
        return format_stats(circuit, engine)
    elif subcommand == "devices":
        return format_devices(circuit, engine)
    elif subcommand == "models":
        return format_models(circuit)
    elif subcommand == "instance":
        return format_instance(directive.args, circuit, engine)
    elif subcommand == "model":
        return format_model(directive.args, circuit)
    else:
        return f"Unknown print subcommand: {subcommand}"


def execute_all_print_directives(
    circuit: "Circuit",
    engine: "CircuitEngine" = None,
) -> list[str]:
    """Execute all print directives in the control block.

    Args:
        circuit: Parsed Circuit object
        engine: Optional CircuitEngine

    Returns:
        List of output strings from each print directive
    """
    outputs = []

    if circuit.control and circuit.control.prints:
        for directive in circuit.control.prints:
            output = execute_print_directive(directive, circuit, engine)
            outputs.append(output)
            logger.info(f"\n{output}")

    return outputs
