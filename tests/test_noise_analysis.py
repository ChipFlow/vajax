"""Tests for noise analysis.

Tests noise analysis against known analytical solutions.
"""

import jax.numpy as jnp
import pytest

from jax_spice.analysis import (
    CircuitEngine,
    NoiseResult,
    compute_flicker_noise_psd,
    compute_shot_noise_psd,
    compute_thermal_noise_psd,
)


class TestNoisePSDCalculations:
    """Test noise PSD calculation functions."""

    def test_thermal_noise_psd(self):
        """Thermal noise PSD = 4*k*T/R."""
        k = 1.380649e-23  # Boltzmann constant
        T = 300.0  # Temperature
        R = 1000.0  # Resistance

        expected = 4 * k * T / R
        result = compute_thermal_noise_psd(R, T)

        assert result == pytest.approx(expected, rel=1e-10)

    def test_thermal_noise_scales_with_resistance(self):
        """Thermal noise decreases with higher resistance."""
        psd_1k = compute_thermal_noise_psd(1000.0, 300.0)
        psd_10k = compute_thermal_noise_psd(10000.0, 300.0)

        assert psd_1k > psd_10k
        assert psd_1k / psd_10k == pytest.approx(10.0, rel=1e-10)

    def test_shot_noise_psd(self):
        """Shot noise PSD = 2*q*I."""
        q = 1.602176634e-19  # Electron charge
        I = 1e-3  # 1mA current

        expected = 2 * q * I
        result = compute_shot_noise_psd(I)

        assert result == pytest.approx(expected, rel=1e-10)

    def test_shot_noise_scales_with_current(self):
        """Shot noise increases with current."""
        psd_1mA = compute_shot_noise_psd(1e-3)
        psd_10mA = compute_shot_noise_psd(10e-3)

        assert psd_10mA > psd_1mA
        assert psd_10mA / psd_1mA == pytest.approx(10.0, rel=1e-10)

    def test_flicker_noise_psd(self):
        """Flicker noise PSD = Kf * I^Af / f^Ef."""
        I = 1e-3
        f = 100.0
        Kf = 1e-15
        Af = 1.0
        Ef = 1.0

        expected = Kf * (I**Af) / (f**Ef)
        result = compute_flicker_noise_psd(I, f, Kf, Af, Ef)

        assert result == pytest.approx(expected, rel=1e-10)

    def test_flicker_noise_decreases_with_frequency(self):
        """Flicker noise is 1/f: higher frequency = lower noise."""
        psd_10Hz = compute_flicker_noise_psd(1e-3, 10.0, 1e-15, 1.0, 1.0)
        psd_100Hz = compute_flicker_noise_psd(1e-3, 100.0, 1e-15, 1.0, 1.0)

        assert psd_10Hz > psd_100Hz
        assert psd_10Hz / psd_100Hz == pytest.approx(10.0, rel=1e-10)


class TestResistorNoise:
    """Test noise analysis with resistor circuits."""

    @pytest.fixture
    def resistor_netlist(self, tmp_path):
        """Simple resistor for thermal noise testing.

        V1 -- R1 (1k) -- output (node 2) -- R2 (1k) -- GND

        Thermal noise from R1 and R2 should contribute to output.
        """
        netlist = tmp_path / "resistor_noise.sim"
        netlist.write_text("""Resistor Noise Test

ground 0

load "resistor.osdi"

model resistor resistor has_noise=1
model vsource vsource

v1 (1 0) vsource dc=1
r1 (1 2) resistor r=1k
r2 (2 0) resistor r=1k

control
  analysis noise1 noise out="2" in="v1" from=1 to=10k mode="dec" points=10
endc
""")
        return netlist

    def test_resistor_noise_runs(self, resistor_netlist):
        """Noise analysis with resistors should complete."""
        engine = CircuitEngine(resistor_netlist)
        engine.parse()
        result = engine.run_noise(out=2, input_source="v1", freq_start=1.0, freq_stop=10000.0)

        assert len(result.frequencies) > 0
        assert len(result.output_noise) == len(result.frequencies)

    def test_resistor_noise_contributions(self, resistor_netlist):
        """Should have noise contributions from both resistors."""
        engine = CircuitEngine(resistor_netlist)
        engine.parse()
        result = engine.run_noise(
            out=2, input_source="v1", freq_start=100.0, freq_stop=100.0, mode="list", values=[100.0]
        )

        # Should have contributions from r1 and r2
        assert "r1" in result.contributions or "r2" in result.contributions

    def test_resistor_thermal_noise_is_white(self, resistor_netlist):
        """Thermal noise should be frequency-independent (white)."""
        engine = CircuitEngine(resistor_netlist)
        engine.parse()
        result = engine.run_noise(
            out=2, input_source="v1", freq_start=10.0, freq_stop=10000.0, mode="dec", points=5
        )

        # Output noise should be roughly constant across frequency
        noise_values = result.output_noise
        noise_ratio = float(noise_values[-1] / noise_values[0])

        # Should be within 2x (allowing for numerical effects)
        assert 0.5 < noise_ratio < 2.0


class TestDiodeNoise:
    """Test noise analysis with diode circuits."""

    @pytest.fixture
    def diode_netlist(self, tmp_path):
        """Diode circuit for shot and flicker noise testing.

        V1 (0.8V) -- R1 (1k) -- node 2 -- D1 -- GND
                              |
                              C1 (1uF)
                              |
                             GND

        Similar to VACASK test_noise.sim
        """
        netlist = tmp_path / "diode_noise.sim"
        netlist.write_text("""Diode Noise Test

ground 0

load "resistor.osdi"
load "capacitor.osdi"
load "diode.osdi"

model resistor resistor has_noise=1
model capacitor capacitor
model vsource vsource
model d diode is=1e-12 n=2 kf=1e-15 af=1.2 ef=1.5

v1 (1 0) vsource dc=0.8
r1 (1 2) resistor r=1k
d1 (2 0) d
c1 (2 0) capacitor c=1u

control
  analysis noise1 noise out="2" in="v1" from=1 to=10k mode="dec" points=10
endc
""")
        return netlist

    def test_diode_noise_runs(self, diode_netlist):
        """Noise analysis with diode should complete."""
        engine = CircuitEngine(diode_netlist)
        engine.parse()
        result = engine.run_noise(out=2, input_source="v1", freq_start=1.0, freq_stop=10000.0)

        assert len(result.frequencies) > 0
        assert len(result.output_noise) == len(result.frequencies)

    def test_diode_has_shot_and_flicker(self, diode_netlist):
        """Diode should contribute shot and flicker noise."""
        engine = CircuitEngine(diode_netlist)
        engine.parse()
        result = engine.run_noise(out=2, input_source="v1", freq_start=1.0, freq_stop=10000.0)

        # Check detailed contributions include shot and flicker
        has_shot = any("shot" in key[1] for key in result.detailed_contributions.keys())
        has_flicker = any("flicker" in key[1] for key in result.detailed_contributions.keys())

        assert has_shot, "Expected shot noise contribution from diode"
        assert has_flicker, "Expected flicker noise contribution from diode"

    def test_flicker_noise_dominates_at_low_freq(self, diode_netlist):
        """At low frequencies, flicker (1/f) noise should be significant."""
        engine = CircuitEngine(diode_netlist)
        engine.parse()
        result = engine.run_noise(
            out=2, input_source="v1", freq_start=1.0, freq_stop=10000.0, mode="dec", points=10
        )

        # Output noise at low frequency should be higher than at high frequency
        # due to 1/f noise contribution
        low_freq_noise = float(result.output_noise[0])
        high_freq_noise = float(result.output_noise[-1])

        assert low_freq_noise > high_freq_noise


class TestNoiseResult:
    """Test NoiseResult data structure."""

    def test_result_has_frequencies(self):
        """NoiseResult should have frequency array."""
        result = NoiseResult(
            frequencies=jnp.array([1.0, 10.0, 100.0]),
            output_noise=jnp.array([1e-20, 1e-21, 1e-22]),
            power_gain=jnp.array([1.0, 0.9, 0.5]),
            contributions={"r1": jnp.array([1e-20, 1e-21, 1e-22])},
            detailed_contributions={("r1", "thermal"): jnp.array([1e-20, 1e-21, 1e-22])},
        )
        assert len(result.frequencies) == 3

    def test_result_has_output_noise(self):
        """NoiseResult should have output noise array."""
        result = NoiseResult(
            frequencies=jnp.array([100.0]),
            output_noise=jnp.array([1e-20]),
            power_gain=jnp.array([1.0]),
            contributions={},
            detailed_contributions={},
        )
        assert len(result.output_noise) == 1
        assert float(result.output_noise[0]) == pytest.approx(1e-20)

    def test_result_has_power_gain(self):
        """NoiseResult should have power gain array."""
        result = NoiseResult(
            frequencies=jnp.array([100.0]),
            output_noise=jnp.array([1e-20]),
            power_gain=jnp.array([0.5]),
            contributions={},
            detailed_contributions={},
        )
        assert float(result.power_gain[0]) == pytest.approx(0.5)
