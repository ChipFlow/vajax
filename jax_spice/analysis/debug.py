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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from jax_spice.netlist.circuit import Circuit, PrintDirective
    from jax_spice.analysis.engine import CircuitEngine

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
    4. Final value in static_inputs array
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
            lines.append(f"  [Instance] (not set)")

        # Layer 2
        if self.in_model:
            lines.append(f"  [Model:{self.model_name}] {self.param_name} = {self.model_value}")
        else:
            lines.append(f"  [Model:{self.model_name}] (not set)")

        # Layer 3
        if self.param_kind:
            lines.append(f"  [OpenVAF] kind={self.param_kind}, index={self.param_index}")
        else:
            lines.append(f"  [OpenVAF] (not mapped)")

        # Layer 4
        if self.final_value is not None:
            lines.append(f"  [Final] value={self.final_value} (from {self.source})")
        else:
            lines.append(f"  [Final] (no value)")

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
        if dev.get('name') == instance_name:
            device = dev
            break

    if not device:
        return trace

    # Layer 1: Instance params
    instance_params = device.get('params', {})
    for k, v in instance_params.items():
        if k.lower() == param_lower:
            trace.in_instance = True
            trace.instance_value = v
            break

    # Layer 2: Model params
    model_name = device.get('model')
    trace.model_name = model_name
    if model_name and engine.circuit:
        model = engine.circuit.models.get(model_name)
        if model and model.params:
            for k, v in model.params.items():
                if k.lower() == param_lower:
                    trace.in_model = True
                    trace.model_value = v
                    break

    # Layer 3 & 4: OpenVAF mapping
    # Get the model type (e.g., 'psp103' from 'psp103n')
    model_type = device.get('model_type', model_name)
    compiled = engine._compiled_models.get(model_type)

    if compiled:
        param_names = compiled.get('param_names', [])
        param_kinds = compiled.get('param_kinds', [])

        # Find the parameter in OpenVAF's param list
        for idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
            if name.lower() == param_lower:
                trace.param_kind = kind
                trace.param_index = idx
                break

    # Layer 4: Get final value from static_inputs if available
    if hasattr(engine, '_transient_setup_cache') and engine._transient_setup_cache:
        static_inputs_cache = engine._transient_setup_cache.get('static_inputs_cache', {})
        if model_type in static_inputs_cache:
            static_inputs = static_inputs_cache[model_type][0]

            # Find device index
            model_devices = [d for d in engine.devices if d.get('model_type', d.get('model')) == model_type]
            for dev_idx, dev in enumerate(model_devices):
                if dev.get('name') == instance_name:
                    if trace.param_index is not None and dev_idx < static_inputs.shape[0]:
                        trace.final_value = float(static_inputs[dev_idx, trace.param_index])
                        # Determine source
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
        if dev.get('name') == instance_name:
            device = dev
            break

    if not device:
        return traces

    # Get all params from instance and model
    all_params = set()

    instance_params = device.get('params', {})
    all_params.update(k.lower() for k in instance_params.keys())

    model_name = device.get('model')
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
        'mapped': mapped,
        'unmapped': unmapped,
        'total': total,
        'coverage': coverage,
        'coverage_pct': f"{coverage * 100:.1f}%",
    }


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

        if coverage['unmapped']:
            lines.append(f"\nUnmapped parameters:")
            for p in coverage['unmapped']:
                lines.append(f"  - {p}")

        lines.append(f"\nAll parameters:")
        for trace in traces:
            status = "✓" if trace.param_kind else "✗"
            value = trace.final_value if trace.final_value is not None else trace.instance_value or trace.model_value
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
            model = dev.get('model', 'unknown')
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
            name = dev.get('name', 'unknown')
            model = dev.get('model', 'unknown')
            terminals = dev.get('terminals', [])
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
            lines.append(f"  Parameters:")
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
        name = name.strip('"\'')
        lines.append(f"\nInstance: {name}")
        lines.append("-" * 40)

        # Look up in engine devices first
        found = False
        if engine and engine.devices:
            for dev in engine.devices:
                if dev.get('name') == name:
                    lines.append(f"Model: {dev.get('model')}")
                    lines.append(f"Terminals: {', '.join(dev.get('terminals', []))}")
                    if dev.get('params'):
                        lines.append("Parameters:")
                        for k, v in sorted(dev['params'].items()):
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
            lines.append(f"  (not found)")

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
        name = name.strip('"\'')
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
