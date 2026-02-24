"""Tests for device parameter tracing and debug commands.

These tests verify that:
1. Parameter tracing works through all mapping layers
2. Print directives execute correctly
3. Parameter coverage detection works
"""

from pathlib import Path

import pytest

from vajax.analysis import CircuitEngine
from vajax.analysis.debug import (
    ParamTrace,
    check_param_coverage,
    execute_all_print_directives,
    execute_print_directive,
    format_devices,
    format_instance,
    format_model,
    format_models,
    format_param_trace,
    format_stats,
    trace_all_params,
    trace_param,
)
from vajax.netlist.circuit import PrintDirective
from vajax.netlist.parser import parse_netlist

# Paths to test fixtures
VACASK_BENCHMARK = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark"
VACASK_TEST = Path(__file__).parent.parent / "vendor" / "VACASK" / "test"


class TestParamTrace:
    """Tests for ParamTrace dataclass and trace_param function."""

    @pytest.fixture(scope="class")
    def ring_engine(self):
        """Parse and prepare ring benchmark."""
        sim_path = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"
        engine = CircuitEngine(sim_path)
        engine.parse()
        return engine

    def test_param_trace_instance_param(self, ring_engine):
        """Parameter from instance is correctly traced."""
        # u1.mp.m has w=w*pfact (10u * 2 = 20u)
        trace = trace_param(ring_engine, "u1.mp.m", "w")

        assert trace.param_name == "w"
        assert trace.instance_name == "u1.mp.m"
        assert trace.in_instance is True
        # w should be ~2e-5 (10u * pfact=2)
        assert trace.instance_value is not None
        assert abs(trace.instance_value - 2e-5) < 1e-10

    def test_param_trace_model_param(self, ring_engine):
        """Parameter from model card is correctly traced."""
        # Check for a model param like toxo (oxide thickness)
        trace = trace_param(ring_engine, "u1.mp.m", "toxo")

        assert trace.param_name == "toxo"
        # toxo comes from model card, not instance
        if trace.in_model:
            assert trace.model_value is not None

    def test_param_trace_openvaf_mapping(self, ring_engine):
        """Parameters are mapped to OpenVAF indices."""
        trace = trace_param(ring_engine, "u1.mp.m", "w")

        # w should be mapped to an OpenVAF param
        assert trace.param_kind is not None
        assert trace.param_index is not None
        assert trace.param_kind == "param"

    def test_param_trace_str_format(self, ring_engine):
        """ParamTrace __str__ produces readable output."""
        trace = trace_param(ring_engine, "u1.mp.m", "w")
        output = str(trace)

        assert "u1.mp.m" in output
        assert "w" in output
        assert "[Instance]" in output
        assert "[Model:" in output
        assert "[OpenVAF]" in output

    def test_param_trace_unknown_instance(self, ring_engine):
        """Tracing unknown instance returns empty trace."""
        trace = trace_param(ring_engine, "nonexistent", "w")

        assert trace.in_instance is False
        assert trace.in_model is False
        assert trace.param_kind is None


class TestTraceAllParams:
    """Tests for trace_all_params function."""

    @pytest.fixture(scope="class")
    def ring_engine(self):
        sim_path = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"
        engine = CircuitEngine(sim_path)
        engine.parse()
        return engine

    def test_trace_all_returns_list(self, ring_engine):
        """trace_all_params returns list of ParamTrace."""
        traces = trace_all_params(ring_engine, "u1.mp.m")

        assert isinstance(traces, list)
        assert len(traces) > 0
        assert all(isinstance(t, ParamTrace) for t in traces)

    def test_trace_all_unknown_instance(self, ring_engine):
        """Tracing unknown instance returns empty list."""
        traces = trace_all_params(ring_engine, "nonexistent")
        assert traces == []


class TestParamCoverage:
    """Tests for parameter coverage checking."""

    @pytest.fixture(scope="class")
    def ring_engine(self):
        sim_path = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"
        engine = CircuitEngine(sim_path)
        engine.parse()
        return engine

    def test_coverage_returns_dict(self, ring_engine):
        """check_param_coverage returns expected dict structure."""
        coverage = check_param_coverage(ring_engine, "u1.mp.m")

        assert isinstance(coverage, dict)
        assert "mapped" in coverage
        assert "unmapped" in coverage
        assert "total" in coverage
        assert "coverage" in coverage
        assert "coverage_pct" in coverage

    def test_coverage_values(self, ring_engine):
        """Coverage values are reasonable."""
        coverage = check_param_coverage(ring_engine, "u1.mp.m")

        assert isinstance(coverage["mapped"], list)
        assert isinstance(coverage["unmapped"], list)
        assert coverage["total"] == len(coverage["mapped"]) + len(coverage["unmapped"])
        assert 0.0 <= coverage["coverage"] <= 1.0

    def test_coverage_unknown_instance(self, ring_engine):
        """Coverage for unknown instance returns 100% (no params)."""
        coverage = check_param_coverage(ring_engine, "nonexistent")
        assert coverage["coverage"] == 1.0
        assert coverage["total"] == 0


class TestFormatFunctions:
    """Tests for formatting functions."""

    @pytest.fixture(scope="class")
    def ring_circuit(self):
        sim_path = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"
        return parse_netlist(sim_path)

    @pytest.fixture(scope="class")
    def ring_engine(self):
        sim_path = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"
        engine = CircuitEngine(sim_path)
        engine.parse()
        return engine

    def test_format_stats(self, ring_circuit, ring_engine):
        """format_stats produces readable output."""
        output = format_stats(ring_circuit, ring_engine)

        assert "Circuit Statistics" in output
        assert "Number of nodes" in output
        assert "Number of devices" in output

    def test_format_stats_without_engine(self, ring_circuit):
        """format_stats works without engine."""
        output = format_stats(ring_circuit)

        assert "Circuit Statistics" in output
        assert "Number of subcircuits" in output
        assert "Number of models" in output

    def test_format_devices(self, ring_circuit, ring_engine):
        """format_devices lists all devices."""
        output = format_devices(ring_circuit, ring_engine)

        assert "Device Instances" in output
        # Should contain some device names
        assert "u1." in output or "vdd" in output.lower()

    def test_format_models(self, ring_circuit):
        """format_models lists all models."""
        output = format_models(ring_circuit)

        assert "Model Definitions" in output
        # Should contain model info
        assert "psp103" in output.lower() or "Module" in output

    def test_format_instance(self, ring_circuit, ring_engine):
        """format_instance shows instance details."""
        output = format_instance(["u1.mp.m"], ring_circuit, ring_engine)

        assert "Instance:" in output
        assert "u1.mp.m" in output

    def test_format_model_by_name(self, ring_circuit):
        """format_model shows model details."""
        # Get a model name that exists
        model_names = list(ring_circuit.models.keys())
        if model_names:
            output = format_model([model_names[0]], ring_circuit)
            assert "Model:" in output

    def test_format_param_trace_single(self, ring_engine):
        """format_param_trace for single param."""
        output = format_param_trace(ring_engine, "u1.mp.m", "w")

        assert "Parameter trace" in output
        assert "u1.mp.m" in output
        assert "w" in output

    def test_format_param_trace_all(self, ring_engine):
        """format_param_trace for all params."""
        output = format_param_trace(ring_engine, "u1.mp.m")

        assert "Parameter coverage" in output
        assert "u1.mp.m" in output


class TestPrintDirectives:
    """Tests for print directive execution."""

    @pytest.fixture(scope="class")
    def ring_circuit(self):
        sim_path = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"
        return parse_netlist(sim_path)

    @pytest.fixture(scope="class")
    def ring_engine(self):
        sim_path = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"
        engine = CircuitEngine(sim_path)
        engine.parse()
        return engine

    def test_execute_print_stats(self, ring_circuit, ring_engine):
        """Execute 'print stats' directive."""
        directive = PrintDirective(subcommand="stats", args=[])
        output = execute_print_directive(directive, ring_circuit, ring_engine)

        assert "Circuit Statistics" in output

    def test_execute_print_devices(self, ring_circuit, ring_engine):
        """Execute 'print devices' directive."""
        directive = PrintDirective(subcommand="devices", args=[])
        output = execute_print_directive(directive, ring_circuit, ring_engine)

        assert "Device Instances" in output

    def test_execute_print_models(self, ring_circuit, ring_engine):
        """Execute 'print models' directive."""
        directive = PrintDirective(subcommand="models", args=[])
        output = execute_print_directive(directive, ring_circuit, ring_engine)

        assert "Model Definitions" in output

    def test_execute_print_instance(self, ring_circuit, ring_engine):
        """Execute 'print instance' directive."""
        directive = PrintDirective(subcommand="instance", args=["u1.mp.m"])
        output = execute_print_directive(directive, ring_circuit, ring_engine)

        assert "Instance" in output
        assert "u1.mp.m" in output

    def test_execute_print_unknown(self, ring_circuit, ring_engine):
        """Execute unknown print directive."""
        directive = PrintDirective(subcommand="unknown", args=[])
        output = execute_print_directive(directive, ring_circuit, ring_engine)

        assert "Unknown" in output

    def test_execute_all_print_directives(self, ring_circuit, ring_engine):
        """Execute all print directives from control block."""
        # Ring benchmark has 'print stats' in control block
        outputs = execute_all_print_directives(ring_circuit, ring_engine)

        # Should have at least one output
        assert isinstance(outputs, list)
        # Ring benchmark has 'print stats'
        if ring_circuit.control and ring_circuit.control.prints:
            assert len(outputs) == len(ring_circuit.control.prints)


class TestRCBenchmark:
    """Test parameter tracing on RC benchmark (simpler circuit)."""

    @pytest.fixture(scope="class")
    def rc_engine(self):
        sim_path = VACASK_BENCHMARK / "rc" / "vacask" / "runme.sim"
        engine = CircuitEngine(sim_path)
        engine.parse()
        return engine

    def test_resistor_param_trace(self, rc_engine):
        """Trace resistor 'r' parameter."""
        # Find a resistor instance
        resistors = [d for d in rc_engine.devices if "resistor" in d.get("model", "").lower()]
        if resistors:
            name = resistors[0]["name"]
            trace = trace_param(rc_engine, name, "r")

            assert trace.param_name == "r"
            if trace.in_instance or trace.in_model:
                assert trace.param_kind is not None

    def test_capacitor_param_trace(self, rc_engine):
        """Trace capacitor 'c' parameter."""
        # Find a capacitor instance
        capacitors = [d for d in rc_engine.devices if "capacitor" in d.get("model", "").lower()]
        if capacitors:
            name = capacitors[0]["name"]
            trace = trace_param(rc_engine, name, "c")

            assert trace.param_name == "c"
            if trace.in_instance or trace.in_model:
                assert trace.param_kind is not None
