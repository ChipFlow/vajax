"""Tests for Harmonic Balance analysis."""

import jax.numpy as jnp

from jax_spice.analysis.hb import (
    HBConfig,
    HBResult,
    build_apft_matrices,
    build_collocation_points,
    build_frequency_grid,
    build_frequency_grid_box,
    build_frequency_grid_diamond,
    complex_to_phasors,
    phasors_to_complex,
    run_hb_analysis,
)


class TestFrequencyGrid:
    """Test frequency grid generation."""

    def test_single_tone_box(self):
        """Test box truncation for single fundamental."""
        freqs, grid = build_frequency_grid_box([1e3], nharm=4)

        # Should have DC + 4 harmonics = 5 frequencies
        assert len(freqs) == 5

        # Check frequencies: DC, f0, 2f0, 3f0, 4f0
        expected = jnp.array([0.0, 1e3, 2e3, 3e3, 4e3])
        assert jnp.allclose(freqs, expected)

    def test_single_tone_diamond(self):
        """Test diamond truncation for single fundamental."""
        freqs, grid = build_frequency_grid_diamond([1e3], nharm=4)

        # Diamond with single tone is same as box
        assert len(freqs) == 5
        expected = jnp.array([0.0, 1e3, 2e3, 3e3, 4e3])
        assert jnp.allclose(freqs, expected)

    def test_two_tone_box(self):
        """Test box truncation for two fundamentals."""
        freqs, grid = build_frequency_grid_box([1e3, 1.5e3], nharm=2)

        # Box: k1 = 0..2, k2 = 0..2 -> 3*3 = 9 combinations
        # Filter removes duplicates (first nonzero must be positive)
        # Valid: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
        # Actually all are valid since we only consider non-negative k values
        assert len(freqs) == 9

    def test_two_tone_diamond(self):
        """Test diamond truncation for two fundamentals."""
        freqs, grid = build_frequency_grid_diamond([1e3, 1.5e3], nharm=2, immax=2)

        # Diamond: |k1| + |k2| <= 2
        # Valid: (0,0), (0,1), (0,2), (1,0), (1,1), (2,0)
        assert len(freqs) == 6

    def test_frequencies_sorted(self):
        """Test that frequencies are sorted."""
        freqs, grid = build_frequency_grid_box([1e3, 1.5e3], nharm=3)

        # Check sorted
        assert jnp.all(freqs[:-1] <= freqs[1:])

    def test_dc_included(self):
        """Test that DC is always included."""
        freqs, _ = build_frequency_grid_box([1e3], nharm=4)
        assert freqs[0] == 0.0

        freqs, _ = build_frequency_grid_diamond([1e3, 2e3], nharm=3)
        assert freqs[0] == 0.0


class TestCollocationPoints:
    """Test collocation point generation."""

    def test_number_of_points(self):
        """Test correct number of collocation points."""
        freqs = jnp.array([0.0, 1e3, 2e3, 3e3])  # nf = 4
        nf = len(freqs)
        nt_expected = 2 * nf - 1  # 7

        timepoints = build_collocation_points(freqs, [1e3])
        assert len(timepoints) == nt_expected

    def test_points_span_period(self):
        """Test that points span the specified periods."""
        freqs = jnp.array([0.0, 1e3, 2e3])
        n_periods = 3.0

        timepoints = build_collocation_points(freqs, [1e3], n_periods=n_periods)

        # Should span 3 periods of 1kHz (3ms)
        T = 1.0 / 1e3
        assert jnp.max(timepoints) < n_periods * T
        assert jnp.min(timepoints) >= 0.0

    def test_points_sorted(self):
        """Test that timepoints are sorted."""
        freqs = jnp.array([0.0, 1e3, 2e3, 3e3, 4e3])
        timepoints = build_collocation_points(freqs, [1e3])

        assert jnp.all(timepoints[:-1] <= timepoints[1:])


class TestAPFTMatrices:
    """Test APFT matrix construction."""

    def test_matrix_dimensions(self):
        """Test APFT matrices have correct dimensions."""
        freqs = jnp.array([0.0, 1e3, 2e3])  # nf = 3
        nf = len(freqs)
        nt = 2 * nf - 1  # 5

        timepoints = build_collocation_points(freqs, [1e3])
        APFT, IAPFT, DDT = build_apft_matrices(freqs, timepoints)

        # APFT: (2*nf-1, nt) = (5, 5)
        assert APFT.shape == (nt, nt)

        # IAPFT: (nt, 2*nf-1) = (5, 5)
        assert IAPFT.shape == (nt, nt)

        # DDT: (nt, nt) = (5, 5)
        assert DDT.shape == (nt, nt)

    def test_apft_iapft_inverse(self):
        """Test that APFT and IAPFT are approximately inverse."""
        freqs = jnp.array([0.0, 1e3, 2e3])
        timepoints = build_collocation_points(freqs, [1e3])
        APFT, IAPFT, DDT = build_apft_matrices(freqs, timepoints)

        # IAPFT @ APFT should be close to identity
        product = IAPFT @ APFT
        identity = jnp.eye(len(timepoints))

        assert jnp.allclose(product, identity, atol=1e-10)

    def test_dc_preserved(self):
        """Test that DC component is preserved through transform."""
        freqs = jnp.array([0.0, 1e3, 2e3])
        timepoints = build_collocation_points(freqs, [1e3])
        APFT, IAPFT, DDT = build_apft_matrices(freqs, timepoints)

        # DC signal (constant)
        dc_value = 2.5
        x_td = jnp.full(len(timepoints), dc_value)

        # Transform to FD
        x_fd = APFT @ x_td

        # DC component should match
        assert jnp.isclose(x_fd[0], dc_value, atol=1e-10)

        # Other components should be near zero
        assert jnp.allclose(x_fd[1:], 0.0, atol=1e-10)

    def test_ddt_dc_zero(self):
        """Test that DDT of DC signal is zero."""
        # Use more frequencies to avoid singular matrices
        freqs = jnp.array([0.0, 1e3, 2e3, 3e3])
        timepoints = build_collocation_points(freqs, [1e3])
        APFT, IAPFT, DDT = build_apft_matrices(freqs, timepoints)

        # DC signal
        x_dc = jnp.ones(len(timepoints))

        # Derivative should be zero (or very small)
        dx_dt = DDT @ x_dc
        assert jnp.allclose(dx_dt, 0.0, atol=1e-8)


class TestPhasorConversion:
    """Test phasor conversion utilities."""

    def test_roundtrip(self):
        """Test conversion roundtrip."""
        nf = 4
        phasors = jnp.array([1.0 + 0j, 0.5 - 0.5j, 0.2 + 0.1j, 0.1 + 0j])

        spectrum = complex_to_phasors(phasors)
        recovered = phasors_to_complex(spectrum, nf)

        assert jnp.allclose(phasors, recovered)

    def test_dc_real(self):
        """Test that DC is purely real."""
        spectrum = jnp.array([2.0, 0.5, 0.3, 0.1, 0.2])  # DC, Re1, Im1, Re2, Im2
        phasors = phasors_to_complex(spectrum, 3)

        # DC should be real
        assert jnp.imag(phasors[0]) == 0.0


class TestHBConfig:
    """Test HB configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HBConfig()

        assert config.freq == [1e3]
        assert config.nharm == 4
        assert config.truncation == "diamond"
        assert config.max_iterations == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = HBConfig(
            freq=[1e3, 2e3],
            nharm=6,
            truncation="box",
            max_iterations=50,
        )

        assert config.freq == [1e3, 2e3]
        assert config.nharm == 6
        assert config.truncation == "box"


class TestHBResult:
    """Test HB result structure."""

    def test_result_fields(self):
        """Test that result has expected fields."""
        result = HBResult(
            frequencies=jnp.array([0.0, 1e3]),
            phasors={"out": jnp.array([1.0 + 0j, 0.5 + 0.5j])},
            dc_voltages=jnp.array([1.0, 0.5]),
            converged=True,
            iterations=10,
            max_residual=1e-12,
        )

        assert len(result.frequencies) == 2
        assert "out" in result.phasors
        assert result.converged
        assert result.iterations == 10


class TestHBAnalysis:
    """Test HB analysis."""

    def test_simple_rc_hb(self):
        """Test HB on simple RC circuit."""
        # Create simple circuit matrices
        # RC circuit: R = 1k, C = 1uF, driven at 1kHz
        R = 1e3
        C = 1e-6

        # 1 node (excluding ground)
        Jr = jnp.array([[1 / R]])  # Resistive Jacobian
        Jc = jnp.array([[C]])  # Reactive Jacobian

        # Source at fundamental frequency
        sources = [
            {
                "type": "vsource",
                "pos_node": 1,
                "neg_node": 0,
                "params": {"ampl": 1.0, "freq": 1e3, "type": "sine"},
            }
        ]

        config = HBConfig(freq=[1e3], nharm=3)

        result = run_hb_analysis(
            Jr,
            Jc,
            sources,
            config,
            node_names={"out": 1},
            dc_voltages=jnp.array([0.0]),
        )

        assert result.converged
        assert len(result.frequencies) >= 3  # DC + harmonics

    def test_hb_frequency_grid_config(self):
        """Test that HB uses correct frequency grid."""
        config = HBConfig(freq=[1e3], nharm=4, truncation="box")
        freqs, _ = build_frequency_grid(config)

        # Box truncation: 0, f0, 2f0, 3f0, 4f0
        assert len(freqs) == 5
        assert jnp.allclose(freqs, jnp.array([0.0, 1e3, 2e3, 3e3, 4e3]))

    def test_hb_diamond_two_tone(self):
        """Test two-tone HB with diamond truncation."""
        config = HBConfig(freq=[1e3, 1.1e3], nharm=2, truncation="diamond")
        freqs, _ = build_frequency_grid(config)

        # Diamond with immax=2: DC, f1, f2, 2f1, 2f2, f1+f2 (if |k1|+|k2|<=2)
        assert len(freqs) >= 3  # At least DC, f1, f2
        assert freqs[0] == 0.0  # DC
