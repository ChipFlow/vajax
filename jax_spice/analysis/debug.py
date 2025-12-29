"""Debug and print commands for circuit inspection.

Implements VACASK-style print commands:
- print stats: Circuit statistics
- print devices: All device instances
- print models: All model definitions
- print instance("name"): Specific instance parameters
- print model("name"): Specific model parameters
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax_spice.netlist.circuit import Circuit, PrintDirective
    from jax_spice.analysis.engine import CircuitEngine

logger = logging.getLogger(__name__)


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
