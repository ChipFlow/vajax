"""Tests for DC sweep analysis.

Tests against known analytical solutions for simple circuits.
"""

import pytest

from vajax.analysis import CircuitEngine, DCSweepResult


class TestDCSweepResistorDivider:
    """Test DC sweep with a simple resistor divider."""

    @pytest.fixture
    def divider_netlist(self, tmp_path):
        """Resistor divider: V1 -- R1(1k) -- node2 -- R2(1k) -- GND.

        V(node2) = V1 * R2/(R1+R2) = V1 * 0.5
        """
        netlist = tmp_path / "divider_sweep.sim"
        netlist.write_text("""Resistor Divider DC Sweep

ground 0

load "resistor.va"

model resistor resistor
model vsource vsource

v1 (1 0) vsource dc=1
r1 (1 2) resistor r=1k
r2 (2 0) resistor r=1k

control
endc
""")
        return netlist

    def test_dc_sweep_with_points(self, divider_netlist):
        """Sweep V1 from 0 to 2V, check V(2) = V1 * 0.5."""
        engine = CircuitEngine(divider_netlist)
        engine.parse()
        result = engine.run_dc_sweep(source="v1", start=0.0, stop=2.0, points=5)

        assert isinstance(result, DCSweepResult)
        assert result.num_points == 5
        assert result.sweep_parameter == "v1"

        # V(node 2) should be half of V1 at each sweep point
        v2 = result.voltages["2"]
        for i, v1_val in enumerate(result.sweep_values):
            expected = float(v1_val) * 0.5
            assert float(v2[i]) == pytest.approx(expected, abs=1e-6)

    def test_dc_sweep_with_step(self, divider_netlist):
        """Sweep V1 from 0 to 1V with 0.5V steps."""
        engine = CircuitEngine(divider_netlist)
        engine.parse()
        result = engine.run_dc_sweep(source="v1", start=0.0, stop=1.0, step=0.5)

        # Should have 3 points: 0.0, 0.5, 1.0
        assert result.num_points == 3
        assert float(result.sweep_values[0]) == pytest.approx(0.0)
        assert float(result.sweep_values[1]) == pytest.approx(0.5)
        assert float(result.sweep_values[2]) == pytest.approx(1.0)

    def test_dc_sweep_has_branch_currents(self, divider_netlist):
        """DC sweep should include branch currents through voltage sources."""
        engine = CircuitEngine(divider_netlist)
        engine.parse()
        result = engine.run_dc_sweep(source="v1", start=0.0, stop=1.0, points=3)

        assert "v1" in result.currents
        # Current through divider: I = V1 / (R1 + R2) = V1 / 2000
        for i, v1_val in enumerate(result.sweep_values):
            expected_current = float(v1_val) / 2000.0
            # Current sign convention may differ, check magnitude
            assert abs(float(result.currents["v1"][i])) == pytest.approx(expected_current, abs=1e-9)

    def test_dc_sweep_source_not_found(self, divider_netlist):
        """Should raise ValueError for nonexistent source."""
        engine = CircuitEngine(divider_netlist)
        engine.parse()
        with pytest.raises(ValueError, match="not found"):
            engine.run_dc_sweep(source="vx", start=0.0, stop=1.0, points=3)

    def test_dc_sweep_requires_step_or_points(self, divider_netlist):
        """Should raise ValueError if neither step nor points given."""
        engine = CircuitEngine(divider_netlist)
        engine.parse()
        with pytest.raises(ValueError, match="Must provide"):
            engine.run_dc_sweep(source="v1", start=0.0, stop=1.0)

    def test_dc_sweep_rejects_both_step_and_points(self, divider_netlist):
        """Should raise ValueError if both step and points given."""
        engine = CircuitEngine(divider_netlist)
        engine.parse()
        with pytest.raises(ValueError, match="not both"):
            engine.run_dc_sweep(source="v1", start=0.0, stop=1.0, step=0.1, points=10)


class TestDCSweepDiode:
    """Test DC sweep with a diode circuit."""

    @pytest.fixture
    def diode_netlist(self, tmp_path):
        """V1 -- R1(1k) -- node2 -- D1 -- GND."""
        netlist = tmp_path / "diode_sweep.sim"
        netlist.write_text("""Diode DC Sweep

ground 0

load "resistor.va"
load "diode.va"

model resistor resistor
model vsource vsource
model d diode is=1e-12 n=2

v1 (1 0) vsource dc=0
r1 (1 2) resistor r=1k
d1 (2 0) d

control
endc
""")
        return netlist

    def test_diode_iv_sweep(self, diode_netlist):
        """Sweep V1 from 0 to 1V, verify diode forward characteristic."""
        engine = CircuitEngine(diode_netlist)
        engine.parse()
        result = engine.run_dc_sweep(source="v1", start=0.0, stop=1.0, points=11)

        assert result.num_points == 11

        # V(node 2) should increase monotonically but saturate at diode forward voltage
        v2 = result.voltages["2"]
        for i in range(1, len(v2)):
            # Each point should be >= previous (monotonically increasing)
            assert float(v2[i]) >= float(v2[i - 1]) - 1e-10

        # At V1=1V, V(node 2) should be around diode forward voltage
        # With n=2 ideality factor, forward voltage is higher (~0.9V)
        assert 0.3 < float(v2[-1]) < 1.0
