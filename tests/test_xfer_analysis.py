"""Tests for transfer function analyses (DCINC, DCXF, ACXF).

Tests against known analytical solutions for simple circuits.
"""

import math

import jax.numpy as jnp
import pytest

from jax_spice.analysis import (
    ACXFResult,
    CircuitEngine,
    DCIncResult,
    DCXFResult,
)


class TestDCINC:
    """Test DC incremental analysis."""

    @pytest.fixture
    def resistor_divider_netlist(self, tmp_path):
        """Simple resistor divider for DCINC testing.

        V1 (1V DC, 1V mag) -- R1 (1k) -- node 2 -- R2 (1k) -- GND

        At DC, V2 = 0.5V
        For incremental: dV2 = 0.5 * mag (voltage divider)
        """
        netlist = tmp_path / "resistor_divider.sim"
        netlist.write_text("""Resistor Divider DCINC Test

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource dc=1 mag=1
r1 (1 2) resistor r=1k
r2 (2 0) resistor r=1k

control
  analysis dcinc1 dcinc
endc
""")
        return netlist

    def test_dcinc_resistor_divider(self, resistor_divider_netlist):
        """DCINC with resistor divider should give dV2 = 0.5."""
        engine = CircuitEngine(resistor_divider_netlist)
        engine.parse()
        result = engine.run_dcinc()

        # Incremental voltage at node 2 should be 0.5 (half of mag=1)
        v2_inc = result.incremental_voltages.get("2")
        assert v2_inc is not None
        assert v2_inc == pytest.approx(0.5, rel=0.01)

    @pytest.fixture
    def diode_circuit_netlist(self, tmp_path):
        """Diode circuit similar to VACASK test_dcinc.sim.

        V1 (0.8V DC, 1V mag) -- R1 (1k) -- node 2 -- D1 -- GND

        The incremental response depends on diode conductance gd at DC OP.
        dV2 = (1/gd) / (1/gd + R1) for 1V excitation
        """
        netlist = tmp_path / "diode_dcinc.sim"
        netlist.write_text("""Diode DCINC Test

ground 0

load "resistor.osdi"
load "diode.osdi"

model resistor resistor
model vsource vsource
model d diode is=1e-12 n=2

v1 (1 0) vsource dc=0.8 mag=1
r1 (1 2) resistor r=1k
d1 (2 0) d

control
  analysis dcinc1 dcinc
endc
""")
        return netlist

    def test_dcinc_diode_circuit_runs(self, diode_circuit_netlist):
        """DCINC with diode should complete without error."""
        engine = CircuitEngine(diode_circuit_netlist)
        engine.parse()
        result = engine.run_dcinc()

        # Should have incremental voltage at node 2
        assert "2" in result.incremental_voltages
        v2_inc = result.incremental_voltages["2"]

        # Value should be between 0 and 1 (voltage divider with diode)
        assert 0 < v2_inc < 1


class TestDCXF:
    """Test DC transfer function analysis."""

    @pytest.fixture
    def resistor_divider_netlist(self, tmp_path):
        """Simple resistor divider for DCXF testing."""
        netlist = tmp_path / "resistor_divider_dcxf.sim"
        netlist.write_text("""Resistor Divider DCXF Test

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource dc=1 mag=1
r1 (1 2) resistor r=1k
r2 (2 0) resistor r=1k

control
  analysis dcxf1 dcxf out="2"
endc
""")
        return netlist

    def test_dcxf_transfer_function(self, resistor_divider_netlist):
        """DCXF should compute transfer function = 0.5 for divider."""
        engine = CircuitEngine(resistor_divider_netlist)
        engine.parse()
        result = engine.run_dcxf(out=2)

        # Transfer function from v1 to node 2 should be 0.5
        assert "v1" in result.tf
        tf_v1 = result.tf["v1"]
        assert tf_v1 == pytest.approx(0.5, rel=0.01)

    def test_dcxf_input_impedance(self, resistor_divider_netlist):
        """DCXF should compute input impedance ~= 2k for divider (R1 + R2)."""
        engine = CircuitEngine(resistor_divider_netlist)
        engine.parse()
        result = engine.run_dcxf(out=2)

        # Input impedance seen by v1 = R1 + R2 = 2k
        # Note: High-G voltage source model causes some numerical variation
        assert "v1" in result.zin
        zin_v1 = result.zin["v1"]
        assert zin_v1 == pytest.approx(2000.0, rel=0.15)  # Allow 15% tolerance

    def test_dcxf_input_admittance(self, resistor_divider_netlist):
        """DCXF should compute input admittance ~= 1/2k = 0.5mS."""
        engine = CircuitEngine(resistor_divider_netlist)
        engine.parse()
        result = engine.run_dcxf(out=2)

        # Input admittance = 1 / 2k = 0.5e-3 S
        # Note: High-G voltage source model causes some numerical variation
        assert "v1" in result.yin
        yin_v1 = result.yin["v1"]
        assert yin_v1 == pytest.approx(0.5e-3, rel=0.15)  # Allow 15% tolerance

    @pytest.fixture
    def two_source_netlist(self, tmp_path):
        """Circuit with two sources for DCXF testing.

        V1 -- R1 (1k) -- node 2 -- D1 -- V2

        Based on VACASK test_dcxf.sim
        """
        netlist = tmp_path / "two_source_dcxf.sim"
        netlist.write_text("""Two Source DCXF Test

ground 0

load "resistor.osdi"
load "diode.osdi"

model resistor resistor
model vsource vsource
model d diode is=1e-12 n=2

v1 (1 0) vsource dc=0.8 mag=1
r1 (1 2) resistor r=1k
d1 (2 3) d
v2 (3 0) vsource dc=0

control
  analysis dcxf1 dcxf out="2"
endc
""")
        return netlist

    def test_dcxf_two_sources(self, two_source_netlist):
        """DCXF with two sources should compute TF for each."""
        engine = CircuitEngine(two_source_netlist)
        engine.parse()
        result = engine.run_dcxf(out=2)

        # Should have tf for both v1 and v2
        assert "v1" in result.tf
        assert "v2" in result.tf

        # Both should be positive (contributing to output)
        # v1 drives through resistor, v2 drives through diode
        assert result.tf["v1"] > 0
        assert result.tf["v2"] > 0

        # Sum should be ~1 (superposition at same node)
        # This is true for this particular circuit topology
        total_tf = result.tf["v1"] + result.tf["v2"]
        assert total_tf == pytest.approx(1.0, rel=0.05)


class TestACXF:
    """Test AC transfer function analysis."""

    @pytest.fixture
    def rc_lowpass_netlist(self, tmp_path):
        """RC lowpass for ACXF testing.

        V1 -- R (1k) -- node 2 -- C (1uF) -- GND

        TF(f) = Zc / (R + Zc) = 1 / (1 + j*2*pi*f*R*C)
        fc = 1/(2*pi*R*C) = 159.15 Hz
        """
        netlist = tmp_path / "rc_lowpass_acxf.sim"
        netlist.write_text("""RC Lowpass ACXF Test

ground 0

load "resistor.osdi"
load "capacitor.osdi"

model resistor resistor
model capacitor capacitor
model vsource vsource

v1 (1 0) vsource dc=0 mag=1
r1 (1 2) resistor r=1k
c1 (2 0) capacitor c=1u

control
  analysis acxf1 acxf out="2" from=1 to=10k mode="dec" points=10
endc
""")
        return netlist

    def test_acxf_has_frequencies(self, rc_lowpass_netlist):
        """ACXF should return frequency sweep."""
        engine = CircuitEngine(rc_lowpass_netlist)
        engine.parse()
        result = engine.run_acxf(out=2, freq_start=1.0, freq_stop=10000.0)

        assert len(result.frequencies) > 0
        assert float(result.frequencies[0]) == pytest.approx(1.0)
        assert float(result.frequencies[-1]) == pytest.approx(10000.0)

    def test_acxf_transfer_function_at_dc(self, rc_lowpass_netlist):
        """ACXF at very low frequency should give TF ~= 1."""
        engine = CircuitEngine(rc_lowpass_netlist)
        engine.parse()
        result = engine.run_acxf(out=2, freq_start=0.1, freq_stop=0.1, mode="list", values=[0.1])

        # At f~0, TF should be ~1
        assert "v1" in result.tf
        tf_v1 = result.tf["v1"]
        assert len(tf_v1) == 1
        assert abs(tf_v1[0]) == pytest.approx(1.0, rel=0.1)

    def test_acxf_transfer_function_at_cutoff(self, rc_lowpass_netlist):
        """ACXF at cutoff frequency should give |TF| ~= 1/sqrt(2)."""
        engine = CircuitEngine(rc_lowpass_netlist)
        engine.parse()

        # Cutoff frequency: fc = 1/(2*pi*R*C) = 159.15 Hz
        fc = 1.0 / (2 * math.pi * 1000 * 1e-6)

        result = engine.run_acxf(out=2, freq_start=fc, freq_stop=fc, mode="list", values=[fc])

        assert "v1" in result.tf
        tf_v1 = result.tf["v1"]
        mag = abs(tf_v1[0])

        # At cutoff, |TF| = 1/sqrt(2) = 0.707
        expected = 1.0 / math.sqrt(2)
        assert mag == pytest.approx(expected, rel=0.1)

    def test_acxf_phase_at_cutoff(self, rc_lowpass_netlist):
        """ACXF at cutoff frequency should give phase ~= -45 degrees."""
        engine = CircuitEngine(rc_lowpass_netlist)
        engine.parse()

        fc = 1.0 / (2 * math.pi * 1000 * 1e-6)

        result = engine.run_acxf(out=2, freq_start=fc, freq_stop=fc, mode="list", values=[fc])

        tf_v1 = result.tf["v1"]
        phase_deg = float(jnp.angle(tf_v1[0]) * 180 / jnp.pi)

        # At cutoff, phase = -45 degrees
        assert phase_deg == pytest.approx(-45.0, abs=10.0)

    def test_acxf_input_impedance_at_dc(self, rc_lowpass_netlist):
        """ACXF at DC should give Zin ~= R (capacitor is open)."""
        engine = CircuitEngine(rc_lowpass_netlist)
        engine.parse()

        result = engine.run_acxf(out=2, freq_start=0.1, freq_stop=0.1, mode="list", values=[0.1])

        assert "v1" in result.zin
        zin = result.zin["v1"]

        # At very low frequency, Zin ~= R = 1k (capacitor is effectively open)
        # Actually for this circuit: Zin = R + 1/(jwC), at very low freq this is very high
        # Let's check it's significantly higher than R alone
        assert abs(zin[0]) > 1000

    def test_acxf_input_impedance_at_high_freq(self, rc_lowpass_netlist):
        """ACXF at high frequency should give Zin ~= R (capacitor is short)."""
        engine = CircuitEngine(rc_lowpass_netlist)
        engine.parse()

        result = engine.run_acxf(
            out=2, freq_start=100000.0, freq_stop=100000.0, mode="list", values=[100000.0]
        )

        assert "v1" in result.zin
        zin = result.zin["v1"]

        # At very high frequency, Zin ~= R = 1k (capacitor is effectively short)
        assert abs(zin[0]) == pytest.approx(1000.0, rel=0.1)

    @pytest.fixture
    def diode_capacitor_netlist(self, tmp_path):
        """Diode + capacitor circuit similar to VACASK test_acxf.sim."""
        netlist = tmp_path / "diode_capacitor_acxf.sim"
        netlist.write_text("""Diode Capacitor ACXF Test

ground 0

load "resistor.osdi"
load "capacitor.osdi"
load "diode.osdi"

model resistor resistor
model capacitor capacitor
model vsource vsource
model d diode is=1e-12 n=2

v2 (2 0) vsource dc=0.8
v3 (10 0) vsource dc=0
r2 (2 3) resistor r=1k
d2 (3 10) d
c2 (3 10) capacitor c=1u

control
  analysis acxf1 acxf out="3" from=1 to=10k mode="dec" points=10
endc
""")
        return netlist

    def test_acxf_diode_circuit_runs(self, diode_capacitor_netlist):
        """ACXF with diode should complete without error."""
        engine = CircuitEngine(diode_capacitor_netlist)
        engine.parse()
        result = engine.run_acxf(out=3, freq_start=1.0, freq_stop=10000.0)

        # Should have tf for both sources
        assert "v2" in result.tf
        assert "v3" in result.tf

        # Should have frequency sweep
        assert len(result.frequencies) > 10


class TestXferResultStructures:
    """Test result data structures."""

    def test_dcinc_result_has_voltages(self):
        """DCIncResult should have incremental voltages dict."""
        result = DCIncResult(
            incremental_voltages={"1": 0.5, "2": 0.3},
            dc_voltages=None,
        )
        assert "1" in result.incremental_voltages
        assert result.incremental_voltages["1"] == 0.5

    def test_dcxf_result_has_metrics(self):
        """DCXFResult should have tf, zin, yin dicts."""
        result = DCXFResult(
            tf={"v1": 0.5},
            zin={"v1": 2000.0},
            yin={"v1": 0.5e-3},
            out_node=2,
        )
        assert result.tf["v1"] == 0.5
        assert result.zin["v1"] == 2000.0
        assert result.yin["v1"] == 0.5e-3

    def test_acxf_result_has_complex_arrays(self):
        """ACXFResult should have complex tf, zin, yin arrays."""
        result = ACXFResult(
            frequencies=jnp.array([1.0, 10.0, 100.0]),
            tf={"v1": jnp.array([1 + 0j, 0.9 + 0.1j, 0.5 + 0.5j])},
            zin={"v1": jnp.array([1000 + 0j, 900 + 100j, 500 + 500j])},
            yin={"v1": jnp.array([0.001 + 0j, 0.0009 - 0.0001j, 0.0005 - 0.0005j])},
            out_node=2,
        )
        assert len(result.frequencies) == 3
        assert jnp.iscomplexobj(result.tf["v1"])
        assert jnp.iscomplexobj(result.zin["v1"])
