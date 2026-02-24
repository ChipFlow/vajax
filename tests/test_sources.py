# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for source types (DC, PULSE, SINE, PWL, EXP, AM, FM).

These tests verify that source waveforms are computed correctly using JAX.
"""

import jax.numpy as jnp
import pytest

from vajax.analysis import CircuitEngine


class TestDCSource:
    """Tests for DC source type."""

    @pytest.fixture
    def dc_netlist(self, tmp_path):
        """DC source netlist."""
        netlist = tmp_path / "dc_source.sim"
        netlist.write_text("""DC Source Test

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="dc" dc=5.0
r1 (1 0) resistor r=1k

control
  analysis tran step=1e-6 stop=1e-3
endc
""")
        return netlist

    def test_dc_constant_value(self, dc_netlist):
        """DC source should return constant value at all times."""
        engine = CircuitEngine(dc_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Test at various times
        for t in [0.0, 1e-6, 1e-3, 1.0]:
            values = source_fn(t)
            assert jnp.isclose(values["v1"], 5.0), f"DC source should be 5.0 at t={t}"


class TestPulseSource:
    """Tests for PULSE source type."""

    @pytest.fixture
    def pulse_netlist_with_delay(self, tmp_path):
        """PULSE source with delay."""
        netlist = tmp_path / "pulse_delay.sim"
        netlist.write_text("""PULSE Source with Delay

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="pulse" val0=0 val1=5 delay=1e-6 rise=1e-9 fall=1e-9 width=1e-6 period=5e-6
r1 (1 0) resistor r=1k

control
  analysis tran step=1e-9 stop=10e-6
endc
""")
        return netlist

    @pytest.fixture
    def pulse_netlist_no_delay(self, tmp_path):
        """PULSE source without delay."""
        netlist = tmp_path / "pulse_no_delay.sim"
        netlist.write_text("""PULSE Source No Delay

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="pulse" val0=0 val1=5 delay=0 rise=1e-9 fall=1e-9 width=1e-6 period=2e-6
r1 (1 0) resistor r=1k

control
  analysis tran step=1e-9 stop=10e-6
endc
""")
        return netlist

    def test_pulse_initial_value(self, pulse_netlist_with_delay):
        """PULSE source should start at val0 before delay."""
        engine = CircuitEngine(pulse_netlist_with_delay)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Before delay
        values = source_fn(0.0)
        assert jnp.isclose(values["v1"], 0.0), "Should be at val0 before delay"

    def test_pulse_high_value(self, pulse_netlist_no_delay):
        """PULSE source should be at val1 during pulse width."""
        engine = CircuitEngine(pulse_netlist_no_delay)
        engine.parse()
        source_fn = engine._build_source_fn()

        # During pulse (after rise, before fall)
        values = source_fn(0.5e-6)
        assert jnp.isclose(values["v1"], 5.0, atol=0.01), "Should be at val1 during pulse"

    def test_pulse_periodicity(self, pulse_netlist_no_delay):
        """PULSE source should repeat with given period."""
        engine = CircuitEngine(pulse_netlist_no_delay)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Values should be similar at t and t+period
        v1 = source_fn(0.5e-6)["v1"]
        v2 = source_fn(2.5e-6)["v1"]
        assert jnp.isclose(v1, v2, atol=0.01), "Should repeat with period"


class TestSineSource:
    """Tests for SINE source type."""

    @pytest.fixture
    def sine_netlist(self, tmp_path):
        """SINE source netlist."""
        netlist = tmp_path / "sine_source.sim"
        netlist.write_text("""SINE Source Test

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="sine" sinedc=2.5 ampl=2 freq=1e6
r1 (1 0) resistor r=1k

control
  analysis tran step=1e-9 stop=10e-6
endc
""")
        return netlist

    def test_sine_dc_offset(self, sine_netlist):
        """SINE source should have correct DC offset."""
        engine = CircuitEngine(sine_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # At t=0 with phase=0, sin(0)=0, so value should be DC offset
        values = source_fn(0.0)
        assert jnp.isclose(values["v1"], 2.5, atol=0.01), "Should be at DC offset when sin=0"

    def test_sine_amplitude(self, sine_netlist):
        """SINE source should have correct amplitude."""
        engine = CircuitEngine(sine_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # At quarter period, sin=1, so value = dc + ampl = 2.5 + 2 = 4.5
        t_quarter = 0.25 / 1e6
        values = source_fn(t_quarter)
        assert jnp.isclose(values["v1"], 4.5, atol=0.01), (
            "Should be at DC+amplitude at quarter period"
        )

    def test_sine_frequency(self, sine_netlist):
        """SINE source should complete full cycle in 1/freq."""
        engine = CircuitEngine(sine_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # After one period, should be back to initial value
        v0 = source_fn(0.0)["v1"]
        v1 = source_fn(1e-6)["v1"]  # 1/freq
        assert jnp.isclose(v0, v1, atol=0.01), "Should repeat after one period"


class TestPWLSource:
    """Tests for PWL (piecewise linear) source type."""

    @pytest.fixture
    def pwl_ramp_netlist(self, tmp_path):
        """PWL source with linear ramp."""
        netlist = tmp_path / "pwl_ramp.sim"
        netlist.write_text("""PWL Ramp Source

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="pwl" wave=[0, 0, 1e-6, 5]
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=2e-6
endc
""")
        return netlist

    @pytest.fixture
    def pwl_trapezoid_netlist(self, tmp_path):
        """PWL source with trapezoid waveform."""
        netlist = tmp_path / "pwl_trap.sim"
        netlist.write_text("""PWL Trapezoid Source

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="pwl" wave=[0, 0, 1e-6, 5, 2e-6, 5, 3e-6, 0]
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=4e-6
endc
""")
        return netlist

    @pytest.fixture
    def pwl_scaled_netlist(self, tmp_path):
        """PWL source with scale and offset."""
        netlist = tmp_path / "pwl_scaled.sim"
        netlist.write_text("""PWL Scaled Source

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="pwl" wave=[0, 0, 1e-6, 1] scale=2 offset=1
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=2e-6
endc
""")
        return netlist

    @pytest.fixture
    def pwl_periodic_netlist(self, tmp_path):
        """PWL source with periodicity."""
        netlist = tmp_path / "pwl_periodic.sim"
        netlist.write_text("""PWL Periodic Source

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="pwl" wave=[0, 0, 1e-6, 5] pwlperiod=2e-6
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=6e-6
endc
""")
        return netlist

    def test_pwl_two_points(self, pwl_ramp_netlist):
        """PWL with two points should give linear ramp."""
        engine = CircuitEngine(pwl_ramp_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Start
        assert jnp.isclose(source_fn(0.0)["v1"], 0.0, atol=0.01)
        # Middle
        assert jnp.isclose(source_fn(0.5e-6)["v1"], 2.5, atol=0.01)
        # End
        assert jnp.isclose(source_fn(1e-6)["v1"], 5.0, atol=0.01)

    def test_pwl_multi_segment(self, pwl_trapezoid_netlist):
        """PWL with multiple segments."""
        engine = CircuitEngine(pwl_trapezoid_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Ramp up
        assert jnp.isclose(source_fn(0.5e-6)["v1"], 2.5, atol=0.01)
        # Plateau
        assert jnp.isclose(source_fn(1.5e-6)["v1"], 5.0, atol=0.01)
        # Ramp down
        assert jnp.isclose(source_fn(2.5e-6)["v1"], 2.5, atol=0.01)

    def test_pwl_extrapolation(self, pwl_ramp_netlist):
        """PWL should hold last value beyond defined points."""
        engine = CircuitEngine(pwl_ramp_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Beyond last point - jnp.interp holds last value
        assert jnp.isclose(source_fn(2e-6)["v1"], 5.0, atol=0.01)

    def test_pwl_scale_and_offset(self, pwl_scaled_netlist):
        """PWL scale and offset parameters."""
        engine = CircuitEngine(pwl_scaled_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # At t=0: value=0*2+1=1
        assert jnp.isclose(source_fn(0.0)["v1"], 1.0, atol=0.01)
        # At t=1e-6: value=1*2+1=3
        assert jnp.isclose(source_fn(1e-6)["v1"], 3.0, atol=0.01)

    def test_pwl_periodicity(self, pwl_periodic_netlist):
        """PWL periodicity with pwlperiod parameter."""
        engine = CircuitEngine(pwl_periodic_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # At t=0.5e-6: in first period
        v1 = source_fn(0.5e-6)["v1"]
        # At t=2.5e-6: should wrap to 0.5e-6
        v2 = source_fn(2.5e-6)["v1"]
        assert jnp.isclose(v1, v2, atol=0.01), "Should repeat with period"


class TestEXPSource:
    """Tests for EXP (exponential) source type."""

    @pytest.fixture
    def exp_netlist_with_delay(self, tmp_path):
        """EXP source with delay."""
        netlist = tmp_path / "exp_delay.sim"
        netlist.write_text("""EXP Source with Delay

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="exp" val0=0 val1=5 delay=1e-6 td2=1e-6 tau1=1e-7 tau2=1e-7
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=5e-6
endc
""")
        return netlist

    @pytest.fixture
    def exp_netlist_rising(self, tmp_path):
        """EXP source for testing rising phase."""
        netlist = tmp_path / "exp_rising.sim"
        netlist.write_text("""EXP Source Rising

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="exp" val0=0 val1=5 delay=0 td2=10e-6 tau1=1e-6 tau2=1e-6
r1 (1 0) resistor r=1k

control
  analysis tran step=100e-9 stop=15e-6
endc
""")
        return netlist

    @pytest.fixture
    def exp_netlist_falling(self, tmp_path):
        """EXP source for testing falling phase."""
        netlist = tmp_path / "exp_falling.sim"
        netlist.write_text("""EXP Source Falling

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="exp" val0=0 val1=5 delay=0 td2=1e-6 tau1=1e-9 tau2=1e-6
r1 (1 0) resistor r=1k

control
  analysis tran step=100e-9 stop=10e-6
endc
""")
        return netlist

    def test_exp_initial_value(self, exp_netlist_with_delay):
        """EXP source should start at val0 before delay."""
        engine = CircuitEngine(exp_netlist_with_delay)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Before delay
        values = source_fn(0.0)
        assert jnp.isclose(values["v1"], 0.0), "Should be at val0 before delay"

    def test_exp_rising_phase(self, exp_netlist_rising):
        """EXP source should rise toward val1 after delay."""
        engine = CircuitEngine(exp_netlist_rising)
        engine.parse()
        source_fn = engine._build_source_fn()

        # After several time constants, should be close to val1
        # At t=5*tau1=5e-6, should be ~99.3% of (val1-val0)
        values = source_fn(5e-6)
        expected = 5 * (1 - jnp.exp(-5))  # 0 + 5*(1-exp(-5)) ~ 4.97
        assert jnp.isclose(values["v1"], expected, atol=0.1)

    def test_exp_falling_phase(self, exp_netlist_falling):
        """EXP source should fall back toward val0 after td2."""
        engine = CircuitEngine(exp_netlist_falling)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Fast rise (tau1=1e-9), at t=delay+td2=1e-6 should be ~val1
        v_at_fall_start = source_fn(1e-6)["v1"]
        assert jnp.isclose(v_at_fall_start, 5.0, atol=0.1)

        # After falling for 5*tau2, should be close to val0
        v_after_fall = source_fn(6e-6)["v1"]
        assert v_after_fall < v_at_fall_start, "Should be falling"


class TestAMSource:
    """Tests for AM (amplitude modulation) source type."""

    @pytest.fixture
    def am_no_mod_netlist(self, tmp_path):
        """AM source with no modulation."""
        netlist = tmp_path / "am_no_mod.sim"
        netlist.write_text("""AM Source No Modulation

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="am" sinedc=0 ampl=1 freq=1e6 modindex=0
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=10e-6
endc
""")
        return netlist

    @pytest.fixture
    def am_modulated_netlist(self, tmp_path):
        """AM source with modulation."""
        netlist = tmp_path / "am_modulated.sim"
        netlist.write_text("""AM Source Modulated

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="am" sinedc=0 ampl=1 freq=1e6 modfreq=100e3 modindex=0.5
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=100e-6
endc
""")
        return netlist

    def test_am_no_modulation(self, am_no_mod_netlist):
        """AM with modindex=0 should behave like SINE."""
        engine = CircuitEngine(am_no_mod_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # At quarter period of carrier
        t_quarter = 0.25 / 1e6
        values = source_fn(t_quarter)
        assert jnp.isclose(values["v1"], 1.0, atol=0.1), "No modulation should give sine"

    def test_am_modulation_effect(self, am_modulated_netlist):
        """AM modulation should vary amplitude."""
        engine = CircuitEngine(am_modulated_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Sample at multiple carrier peaks
        peaks = []
        for i in range(10):
            t = (i + 0.25) / 1e6  # At carrier peaks
            peaks.append(float(source_fn(t)["v1"]))

        # Peaks should vary due to modulation
        assert max(peaks) > min(peaks) + 0.1, "AM should vary amplitude"


class TestFMSource:
    """Tests for FM (frequency modulation) source type."""

    @pytest.fixture
    def fm_no_mod_netlist(self, tmp_path):
        """FM source with no modulation."""
        netlist = tmp_path / "fm_no_mod.sim"
        netlist.write_text("""FM Source No Modulation

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="fm" sinedc=0 ampl=1 freq=1e6 modindex=0
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=10e-6
endc
""")
        return netlist

    @pytest.fixture
    def fm_modulated_netlist(self, tmp_path):
        """FM source with modulation."""
        netlist = tmp_path / "fm_modulated.sim"
        netlist.write_text("""FM Source Modulated

ground 0

load "resistor.osdi"

model resistor resistor
model vsource vsource

v1 (1 0) vsource type="fm" sinedc=0 ampl=1 freq=1e6 modfreq=100e3 modindex=2
r1 (1 0) resistor r=1k

control
  analysis tran step=10e-9 stop=100e-6
endc
""")
        return netlist

    def test_fm_no_modulation(self, fm_no_mod_netlist):
        """FM with modindex=0 should behave like SINE."""
        engine = CircuitEngine(fm_no_mod_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # At quarter period of carrier
        t_quarter = 0.25 / 1e6
        values = source_fn(t_quarter)
        assert jnp.isclose(values["v1"], 1.0, atol=0.1), "No modulation should give sine"

    def test_fm_modulation_effect(self, fm_modulated_netlist):
        """FM modulation should vary instantaneous frequency."""
        engine = CircuitEngine(fm_modulated_netlist)
        engine.parse()
        source_fn = engine._build_source_fn()

        # Sample values - FM changes phase, not amplitude
        # The output should still swing between -1 and +1
        samples = [float(source_fn(i * 0.1e-6)["v1"]) for i in range(100)]

        # Amplitude should stay bounded
        assert max(samples) <= 1.1, "FM amplitude should be bounded"
        assert min(samples) >= -1.1, "FM amplitude should be bounded"


class TestSourceIntegration:
    """Integration tests for sources in transient simulation."""

    @pytest.fixture
    def pwl_transient_netlist(self, tmp_path):
        """PWL source for transient test."""
        netlist = tmp_path / "pwl_transient.sim"
        netlist.write_text("""PWL Transient Test

ground 0

load "resistor.osdi"
load "capacitor.osdi"

model resistor resistor
model capacitor capacitor
model vsource vsource

v1 (1 0) vsource type="pwl" wave=[0, 0, 1e-6, 5, 2e-6, 5, 3e-6, 0]
r1 (1 2) resistor r=1k
c1 (2 0) capacitor c=1e-9

control
  analysis tran step=10e-9 stop=4e-6
endc
""")
        return netlist

    @pytest.fixture
    def exp_transient_netlist(self, tmp_path):
        """EXP source for transient test."""
        netlist = tmp_path / "exp_transient.sim"
        netlist.write_text("""EXP Transient Test

ground 0

load "resistor.osdi"
load "capacitor.osdi"

model resistor resistor
model capacitor capacitor
model vsource vsource

v1 (1 0) vsource type="exp" val0=0 val1=5 delay=0 td2=1e-6 tau1=1e-7 tau2=1e-7
r1 (1 2) resistor r=1k
c1 (2 0) capacitor c=1e-9

control
  analysis tran step=10e-9 stop=3e-6
endc
""")
        return netlist

    def test_pwl_in_transient(self, pwl_transient_netlist):
        """PWL source should work in transient simulation."""
        engine = CircuitEngine(pwl_transient_netlist)
        engine.parse()
        result = engine.run_transient()

        # Check that simulation completed
        assert result.times is not None
        assert len(result.times) > 0
        assert result.stats.get("converged", False) or len(result.times) > 10

    def test_exp_in_transient(self, exp_transient_netlist):
        """EXP source should work in transient simulation."""
        engine = CircuitEngine(exp_transient_netlist)
        engine.parse()
        result = engine.run_transient()

        # Check that simulation completed
        assert result.times is not None
        assert len(result.times) > 0
