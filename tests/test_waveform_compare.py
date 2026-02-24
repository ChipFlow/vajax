"""Tests for waveform comparison utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vajax.utils import (
    WaveformComparison,
    compare_waveforms,
    find_vacask_binary,
)
from vajax.utils.rawfile import rawread


class TestCompareWaveforms:
    """Tests for compare_waveforms function."""

    def test_identical_waveforms(self):
        """Identical waveforms should have zero error."""
        data = np.linspace(0, 1, 100)
        result = compare_waveforms(data, data, "test")

        assert result.name == "test"
        assert result.max_abs_error == 0.0
        assert result.max_rel_error == 0.0
        assert result.rms_error == 0.0
        assert result.within_tolerance

    def test_small_difference(self):
        """Small differences should be within tolerance."""
        data1 = np.linspace(0, 1, 100)
        data2 = data1 + 1e-10  # Tiny offset

        result = compare_waveforms(data1, data2, "test", abs_tol=1e-9)

        assert result.max_abs_error < 1e-9
        assert result.within_tolerance

    def test_large_difference(self):
        """Large differences should exceed tolerance."""
        data1 = np.linspace(0, 1, 100)
        data2 = data1 + 0.1  # Significant offset

        result = compare_waveforms(data1, data2, "test", abs_tol=1e-9, rel_tol=1e-3)

        assert result.max_abs_error > 0.09
        assert not result.within_tolerance

    def test_different_lengths_downsamples(self):
        """Different length arrays should be compared by downsampling."""
        data1 = np.linspace(0, 1, 100)
        data2 = np.linspace(0, 1, 50)

        result = compare_waveforms(data1, data2, "test")

        # Should successfully compare
        assert result.points_compared == 50

    def test_jax_array_input(self):
        """Should handle JAX arrays as input."""
        import jax.numpy as jnp

        data1 = np.linspace(0, 1, 100)
        data2 = jnp.linspace(0, 1, 100)

        result = compare_waveforms(data1, data2, "test")

        # Should be nearly identical (floating point rounding)
        assert result.max_abs_error < 1e-10

    def test_relative_tolerance(self):
        """Should respect relative tolerance."""
        data1 = np.array([1.0, 10.0, 100.0])
        data2 = np.array([1.001, 10.01, 100.1])  # 0.1% difference

        result = compare_waveforms(data1, data2, "test", abs_tol=1e-9, rel_tol=0.01)

        # 0.1% is within 1% relative tolerance
        assert result.within_tolerance

    def test_str_representation(self):
        """String representation should be readable."""
        data = np.linspace(0, 1, 100)
        result = compare_waveforms(data, data, "voltage_out")

        s = str(result)
        assert "voltage_out" in s
        assert "✓" in s  # Pass indicator


class TestWaveformComparisonDataclass:
    """Tests for WaveformComparison dataclass."""

    def test_dataclass_fields(self):
        """Dataclass should have all expected fields."""
        wc = WaveformComparison(
            name="test",
            max_abs_error=1e-6,
            max_rel_error=0.001,
            rms_error=5e-7,
            points_compared=100,
            within_tolerance=True,
            abs_tol=1e-9,
            rel_tol=1e-3,
        )

        assert wc.name == "test"
        assert wc.max_abs_error == 1e-6
        assert wc.max_rel_error == 0.001
        assert wc.points_compared == 100


class TestFindVacaskBinary:
    """Tests for find_vacask_binary function."""

    def test_returns_path_or_none(self):
        """Should return Path or None."""
        result = find_vacask_binary()

        assert result is None or isinstance(result, Path)

    def test_found_binary_exists(self):
        """If found, binary should exist."""
        result = find_vacask_binary()

        if result is not None:
            assert result.exists()
            assert result.is_file()


class TestRawFileReader:
    """Tests for rawfile module."""

    def test_create_and_read_raw_file(self):
        """Test creating a minimal raw file and reading it."""
        # Create a minimal raw file in binary format
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            # Write header
            f.write(b"Title: Test Circuit\n")
            f.write(b"Date: Mon Jan 01 00:00:00 2025\n")
            f.write(b"Plotname: Transient Analysis\n")
            f.write(b"Flags: real\n")
            f.write(b"No. Variables: 2\n")
            f.write(b"No. Points: 5\n")
            f.write(b"Variables:\n")
            f.write(b"\t0\ttime\ttime\n")
            f.write(b"\t1\tv(1)\tvoltage\n")
            f.write(b"Binary:\n")

            # Write binary data: 5 points, 2 variables
            data = np.array(
                [
                    [0.0, 0.0],
                    [1e-9, 0.5],
                    [2e-9, 1.0],
                    [3e-9, 0.5],
                    [4e-9, 0.0],
                ],
                dtype=np.float64,
            )
            f.write(data.tobytes())
            f.write(b"\n")

            temp_path = Path(f.name)

        try:
            # Read the file
            raw = rawread(str(temp_path))
            rf = raw.get()

            # Check parsed data
            assert rf.title == "Test Circuit"
            assert rf.plotname == "Transient Analysis"
            assert rf.names == ["time", "v(1)"]
            assert rf.units == ["time", "voltage"]

            # Check data
            np.testing.assert_allclose(rf["time"], [0, 1e-9, 2e-9, 3e-9, 4e-9])
            np.testing.assert_allclose(rf["v(1)"], [0, 0.5, 1.0, 0.5, 0])

        finally:
            temp_path.unlink()

    def test_get_all_method(self):
        """Test get_all method returns dict of all vectors."""
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(b"Title: Test\n")
            f.write(b"Date: Test\n")
            f.write(b"Plotname: Test\n")
            f.write(b"Flags: real\n")
            f.write(b"No. Variables: 2\n")
            f.write(b"No. Points: 3\n")
            f.write(b"Variables:\n")
            f.write(b"\t0\ttime\ttime\n")
            f.write(b"\t1\tv(out)\tvoltage\n")
            f.write(b"Binary:\n")

            data = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
            f.write(data.tobytes())
            f.write(b"\n")

            temp_path = Path(f.name)

        try:
            rf = rawread(str(temp_path)).get()
            all_data = rf.get_all()

            assert "time" in all_data
            assert "v(out)" in all_data
            assert len(all_data) == 2

        finally:
            temp_path.unlink()


class TestVacaskIntegration:
    """Integration tests with actual VACASK (skipped if not available)."""

    @pytest.fixture
    def vacask_available(self):
        """Check if VACASK is available."""
        binary = find_vacask_binary()
        if binary is None:
            pytest.skip("VACASK binary not found")
        return binary

    @pytest.fixture
    def rc_sim_file(self):
        """Path to RC benchmark sim file."""
        path = Path(__file__).parent.parent / "vendor/VACASK/benchmark/rc/vacask/runme.sim"
        if not path.exists():
            pytest.skip("RC benchmark not found")
        return path

    def test_run_vacask_rc(self, vacask_available, rc_sim_file):
        """Test running VACASK on RC benchmark."""
        from vajax.utils import run_vacask

        raw_path, error = run_vacask(rc_sim_file, vacask_bin=vacask_available)

        assert error is None, f"VACASK failed: {error}"
        assert raw_path is not None
        assert raw_path.exists()

        # Parse the output
        rf = rawread(str(raw_path)).get()

        assert "time" in rf.names
        assert rf.data.shape[0] > 0  # Has data points

        # Cleanup
        raw_path.unlink()
