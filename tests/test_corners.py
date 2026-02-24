"""Tests for corner analysis functionality.

Tests the PVT (Process/Voltage/Temperature) corner sweep capabilities.
"""

from vajax.analysis.corners import (
    PROCESS_CORNERS,
    TEMPERATURE_CORNERS,
    CornerConfig,
    CornerResult,
    CornerSweepResult,
    VoltageCorner,
    create_pvt_corners,
    create_standard_corners,
)


class TestProcessCorner:
    """Tests for ProcessCorner dataclass."""

    def test_ff_corner_values(self):
        """FF corner should have faster characteristics."""
        ff = PROCESS_CORNERS["FF"]
        assert ff.mobility_scale > 1.0  # Higher mobility = faster
        assert ff.vth_shift < 0  # Lower threshold = faster
        assert ff.tox_scale < 1.0  # Thinner oxide = faster

    def test_ss_corner_values(self):
        """SS corner should have slower characteristics."""
        ss = PROCESS_CORNERS["SS"]
        assert ss.mobility_scale < 1.0  # Lower mobility = slower
        assert ss.vth_shift > 0  # Higher threshold = slower
        assert ss.tox_scale > 1.0  # Thicker oxide = slower

    def test_tt_corner_is_nominal(self):
        """TT corner should have all nominal values."""
        tt = PROCESS_CORNERS["TT"]
        assert tt.mobility_scale == 1.0
        assert tt.vth_shift == 0.0
        assert tt.tox_scale == 1.0


class TestVoltageCorner:
    """Tests for VoltageCorner dataclass."""

    def test_nominal_voltage(self):
        """Nominal voltage corner should have scale 1.0."""
        v = VoltageCorner(name="nom", vdd_scale=1.0)
        assert v.vdd_scale == 1.0

    def test_explicit_source_values(self):
        """Can specify explicit values for specific sources."""
        v = VoltageCorner(name="custom", vdd_scale=1.0, source_values={"vdd": 1.8, "vss": -1.8})
        assert v.source_values["vdd"] == 1.8
        assert v.source_values["vss"] == -1.8


class TestCornerConfig:
    """Tests for CornerConfig dataclass."""

    def test_default_values(self):
        """Default corner should be nominal at room temperature."""
        c = CornerConfig(name="default")
        assert c.temperature == 300.15
        assert c.process is None
        assert c.voltage is None

    def test_full_config(self):
        """Can create fully specified corner."""
        c = CornerConfig(
            name="FF_hot_high",
            process=PROCESS_CORNERS["FF"],
            voltage=VoltageCorner(name="high", vdd_scale=1.1),
            temperature=398.15,
        )
        assert c.name == "FF_hot_high"
        assert c.process.name == "FF"
        assert c.voltage.vdd_scale == 1.1
        assert c.temperature == 398.15


class TestTemperatureCorners:
    """Tests for temperature corner definitions."""

    def test_cold_temperature(self):
        """Cold corner should be -40C."""
        assert TEMPERATURE_CORNERS["cold"] == 233.15

    def test_room_temperature(self):
        """Room corner should be 27C."""
        assert TEMPERATURE_CORNERS["room"] == 300.15

    def test_hot_temperature(self):
        """Hot corner should be 125C."""
        assert TEMPERATURE_CORNERS["hot"] == 398.15


class TestCreateStandardCorners:
    """Tests for create_standard_corners function."""

    def test_single_corner(self):
        """Single corner should produce one config."""
        corners = create_standard_corners(processes=["TT"], temperatures=[300.15], vdd_scales=[1.0])
        assert len(corners) == 1
        assert corners[0].process.name == "TT"

    def test_multiple_corners(self):
        """Multiple options should produce cartesian product."""
        corners = create_standard_corners(
            processes=["FF", "TT", "SS"], temperatures=[233.15, 300.15], vdd_scales=[0.9, 1.0]
        )
        # 3 processes * 2 temps * 2 voltages = 12 corners
        assert len(corners) == 12

    def test_corner_names_are_descriptive(self):
        """Corner names should describe the configuration."""
        corners = create_standard_corners(processes=["FF"], temperatures=[233.15], vdd_scales=[0.9])
        assert "FF" in corners[0].name
        assert "m40C" in corners[0].name or "40" in corners[0].name

    def test_defaults(self):
        """Default should be single TT at room temp."""
        corners = create_standard_corners()
        assert len(corners) == 1
        assert corners[0].temperature == 300.15


class TestCreatePVTCorners:
    """Tests for create_pvt_corners convenience function."""

    def test_default_matrix(self):
        """Default should be 3x3x3 = 27 corners."""
        corners = create_pvt_corners()
        assert len(corners) == 27

    def test_custom_matrix(self):
        """Can customize the corner matrix."""
        corners = create_pvt_corners(
            processes=["FF", "SS"], temperatures=["cold", "hot"], voltages=[0.9, 1.1]
        )
        # 2 * 2 * 2 = 8 corners
        assert len(corners) == 8


class TestCornerSweepResult:
    """Tests for CornerSweepResult dataclass."""

    def test_empty_results(self):
        """Empty sweep should work."""
        result = CornerSweepResult(corners=[], results=[])
        assert result.num_corners == 0
        assert result.num_converged == 0
        assert result.all_converged

    def test_all_converged(self):
        """All converged flag should reflect results."""
        corners = [
            CornerConfig(name="c1"),
            CornerConfig(name="c2"),
        ]
        results = [
            CornerResult(corner=corners[0], result=None, converged=True),
            CornerResult(corner=corners[1], result=None, converged=True),
        ]
        sweep = CornerSweepResult(corners=corners, results=results)
        assert sweep.all_converged

    def test_partial_convergence(self):
        """Partial convergence should be detected."""
        corners = [
            CornerConfig(name="c1"),
            CornerConfig(name="c2"),
        ]
        results = [
            CornerResult(corner=corners[0], result=None, converged=True),
            CornerResult(corner=corners[1], result=None, converged=False),
        ]
        sweep = CornerSweepResult(corners=corners, results=results)
        assert not sweep.all_converged
        assert sweep.num_converged == 1

    def test_get_result_by_name(self):
        """Can retrieve result by corner name."""
        corner = CornerConfig(name="test_corner")
        result = CornerResult(corner=corner, result="test_data", converged=True)
        sweep = CornerSweepResult(corners=[corner], results=[result])

        found = sweep.get_result("test_corner")
        assert found is not None
        assert found.result == "test_data"

        not_found = sweep.get_result("nonexistent")
        assert not_found is None

    def test_converged_results_filter(self):
        """Can filter to only converged results."""
        corners = [
            CornerConfig(name="c1"),
            CornerConfig(name="c2"),
            CornerConfig(name="c3"),
        ]
        results = [
            CornerResult(corner=corners[0], result=None, converged=True),
            CornerResult(corner=corners[1], result=None, converged=False),
            CornerResult(corner=corners[2], result=None, converged=True),
        ]
        sweep = CornerSweepResult(corners=corners, results=results)

        converged = sweep.converged_results()
        assert len(converged) == 2
        assert all(r.converged for r in converged)
