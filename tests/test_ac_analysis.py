"""Tests for AC (small-signal) analysis.

Tests AC frequency sweep analysis against known analytical solutions
and VACASK reference results.
"""

import math
import sys
from pathlib import Path

import pytest

# Add openvaf_jax and openvaf_py to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "openvaf_jax" / "openvaf_py"))

import jax.numpy as jnp

from vajax.analysis import ACConfig, ACResult, CircuitEngine, generate_frequencies


class TestFrequencyGeneration:
    """Test frequency sweep generation."""

    def test_linear_sweep(self):
        """Linear frequency sweep with fixed step."""
        config = ACConfig(
            freq_start=100.0,
            freq_stop=1000.0,
            mode="lin",
            step=100.0,
        )
        freqs = generate_frequencies(config)

        assert len(freqs) == 10  # 100, 200, ..., 1000
        assert float(freqs[0]) == pytest.approx(100.0)
        assert float(freqs[-1]) == pytest.approx(1000.0)

    def test_decade_sweep(self):
        """Logarithmic sweep with points per decade."""
        config = ACConfig(
            freq_start=1.0,
            freq_stop=1000.0,
            mode="dec",
            points=10,
        )
        freqs = generate_frequencies(config)

        # 3 decades * 10 points + 1 = 31 points
        assert len(freqs) == 31
        assert float(freqs[0]) == pytest.approx(1.0)
        assert float(freqs[-1]) == pytest.approx(1000.0)

        # Check logarithmic spacing
        log_freqs = jnp.log10(freqs)
        diffs = jnp.diff(log_freqs)
        assert jnp.allclose(diffs, diffs[0], rtol=0.01)

    def test_octave_sweep(self):
        """Logarithmic sweep with points per octave."""
        config = ACConfig(
            freq_start=100.0,
            freq_stop=1600.0,  # 4 octaves (100 -> 200 -> 400 -> 800 -> 1600)
            mode="oct",
            points=2,
        )
        freqs = generate_frequencies(config)

        assert float(freqs[0]) == pytest.approx(100.0)
        assert float(freqs[-1]) == pytest.approx(1600.0)

    def test_list_mode(self):
        """Explicit frequency list."""
        config = ACConfig(
            freq_start=0.0,  # Ignored
            freq_stop=0.0,  # Ignored
            mode="list",
            values=[10.0, 100.0, 1000.0, 10000.0],
        )
        freqs = generate_frequencies(config)

        assert len(freqs) == 4
        assert float(freqs[0]) == pytest.approx(10.0)
        assert float(freqs[3]) == pytest.approx(10000.0)


class TestRCLowpass:
    """Test AC analysis on simple RC lowpass filter.

    Circuit:
        Vin (1V AC) -- R (1k) -- Vout -- C (1uF) -- GND

    Transfer function: H(f) = 1 / (1 + j*2*pi*f*R*C)
    Cutoff frequency: fc = 1 / (2*pi*R*C) = 159.15 Hz
    """

    @pytest.fixture
    def rc_circuit_netlist(self, tmp_path):
        """Create a simple RC lowpass filter netlist."""
        netlist = tmp_path / "rc_lowpass.sim"
        netlist.write_text("""RC Lowpass Filter

ground 0

load "resistor.osdi"
load "capacitor.osdi"

model resistor resistor
model capacitor capacitor
model vsource vsource

v1 (1 0) vsource dc=0 mag=1.0
r1 (1 2) resistor r=1k
c1 (2 0) capacitor c=1u

control
  analysis ac1 ac from=1 to=10k mode="dec" points=10
endc
""")
        return netlist

    def test_rc_magnitude_at_dc(self, rc_circuit_netlist):
        """At DC (f=0), output should equal input (0dB)."""
        engine = CircuitEngine(rc_circuit_netlist)
        engine.parse()
        result = engine.run_ac(
            freq_start=0.1,  # Near DC
            freq_stop=0.1,
            mode="list",
            values=[0.1],
        )

        # At very low frequency, Vout ~= Vin
        v_out = result.voltages.get("2")
        assert v_out is not None
        mag = float(jnp.abs(v_out[0]))
        assert mag == pytest.approx(1.0, rel=0.1)

    def test_rc_at_cutoff(self, rc_circuit_netlist):
        """At cutoff frequency: magnitude -3dB (1/sqrt(2)), phase -45 degrees."""
        engine = CircuitEngine(rc_circuit_netlist)
        engine.parse()

        # Cutoff: fc = 1/(2*pi*R*C) = 1/(2*pi*1000*1e-6) = 159.15 Hz
        fc = 1.0 / (2 * math.pi * 1000 * 1e-6)

        result = engine.run_ac(
            freq_start=fc,
            freq_stop=fc,
            mode="list",
            values=[fc],
        )

        v_out = result.voltages.get("2")
        assert v_out is not None

        # At cutoff, |H| = 1/sqrt(2) = 0.707
        mag = float(jnp.abs(v_out[0]))
        expected_mag = 1.0 / math.sqrt(2)
        assert mag == pytest.approx(expected_mag, rel=0.1)

        # At cutoff, phase = -45 degrees
        phase_deg = float(jnp.angle(v_out[0]) * 180 / jnp.pi)
        assert phase_deg == pytest.approx(-45.0, abs=10.0)

    def test_rc_frequency_sweep(self, rc_circuit_netlist):
        """Test full frequency sweep with expected rolloff."""
        engine = CircuitEngine(rc_circuit_netlist)
        engine.parse()

        result = engine.run_ac(
            freq_start=1.0,
            freq_stop=10000.0,
            mode="dec",
            points=10,
        )

        assert len(result.frequencies) > 20
        assert float(result.frequencies[0]) == pytest.approx(1.0)
        assert float(result.frequencies[-1]) == pytest.approx(10000.0)

        v_out = result.voltages.get("2")
        assert v_out is not None

        # Magnitude should decrease with frequency
        mags = jnp.abs(v_out)
        assert float(mags[0]) > float(mags[-1])

        # At high frequency (10kHz >> 159Hz), rolloff is -20dB/decade
        # So at 10kHz (factor of 63x above cutoff), we expect ~-36dB
        high_freq_mag_db = 20 * jnp.log10(mags[-1])
        assert float(high_freq_mag_db) < -30  # At least -30dB


class TestDiodeCircuit:
    """Test AC analysis with nonlinear device (diode).

    Based on VACASK test_ac.sim:
        v2 (2 0) vsource dc=0.8 mag=2.0
        r2 (2 3) resistor r=1k
        d2 (3 0) d  (diode)
        c2 (3 0) capacitor c=1u

    The diode is biased at V=0.8V through 1k resistor.
    At small signal, the diode acts as a conductance gd = Is*exp(Vd/nVt)/nVt
    in parallel with junction capacitance.
    """

    @pytest.fixture
    def diode_circuit_netlist(self, tmp_path):
        """Create diode circuit netlist."""
        netlist = tmp_path / "diode_ac.sim"
        netlist.write_text("""Diode AC analysis

ground 0

load "resistor.osdi"
load "capacitor.osdi"
load "diode.osdi"

model resistor resistor
model capacitor capacitor
model vsource vsource
model d diode is=1e-12 n=2 rs=0.1 cjo=100p vj=1 m=0.5

v2 (2 0) vsource dc=0.8 mag=2.0
r2 (2 3) resistor r=1k
d2 (3 0) d
c2 (3 0) capacitor c=1u

control
  analysis ac1 ac from=1 to=10k mode="dec" points=10
endc
""")
        return netlist

    def test_diode_ac_parses(self, diode_circuit_netlist):
        """Diode AC circuit should parse without error."""
        engine = CircuitEngine(diode_circuit_netlist)
        engine.parse()
        assert engine.num_nodes > 0

    def test_diode_ac_runs(self, diode_circuit_netlist):
        """Diode AC analysis should run."""
        engine = CircuitEngine(diode_circuit_netlist)
        engine.parse()
        result = engine.run_ac(
            freq_start=1.0,
            freq_stop=10000.0,
            mode="dec",
            points=10,
        )

        assert len(result.frequencies) > 0
        assert result.dc_voltages is not None


class TestACResult:
    """Test ACResult data structure."""

    def test_result_has_frequencies(self):
        """ACResult should have frequency array."""
        result = ACResult(
            frequencies=jnp.array([1.0, 10.0, 100.0]),
            voltages={"1": jnp.array([1 + 0j, 0.9 + 0.1j, 0.5 + 0.5j])},
            currents={},
            dc_voltages=jnp.array([0.0, 1.0]),
        )
        assert len(result.frequencies) == 3

    def test_result_voltages_complex(self):
        """Voltages should be complex arrays."""
        result = ACResult(
            frequencies=jnp.array([100.0]),
            voltages={"1": jnp.array([0.707 + 0.707j])},
            currents={},
            dc_voltages=jnp.array([0.0]),
        )
        assert jnp.iscomplexobj(result.voltages["1"])
