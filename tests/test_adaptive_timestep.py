"""Unit tests for adaptive timestep control.

Tests the predictor-corrector algorithm and LTE-based timestep adjustment.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vajax.analysis.integration import IntegrationMethod, compute_coefficients
from vajax.analysis.transient.predictor import (
    compute_new_timestep,
    compute_predictor_coeffs,
    estimate_lte,
    predict,
)


class TestIntegrationErrorCoeffs:
    """Test that integration methods return correct error coefficients."""

    def test_backward_euler_error_coeff(self):
        """BE error coefficient should be 1/2."""
        coeffs = compute_coefficients(IntegrationMethod.BACKWARD_EULER, dt=1e-9)
        assert coeffs.error_coeff == pytest.approx(0.5)

    def test_trapezoidal_error_coeff(self):
        """Trapezoidal error coefficient should be 1/12."""
        coeffs = compute_coefficients(IntegrationMethod.TRAPEZOIDAL, dt=1e-9)
        assert coeffs.error_coeff == pytest.approx(1.0 / 12.0)

    def test_gear2_error_coeff(self):
        """Gear2 error coefficient should be 2/9."""
        coeffs = compute_coefficients(IntegrationMethod.GEAR2, dt=1e-9)
        assert coeffs.error_coeff == pytest.approx(2.0 / 9.0)


class TestPredictorCoeffs:
    """Test polynomial extrapolation predictor coefficient computation."""

    def test_order_zero_fallback(self):
        """With no history, predictor should use constant extrapolation."""
        coeffs = compute_predictor_coeffs(past_dt=[], new_dt=1e-9, order=1)
        assert coeffs.order == 0
        assert len(coeffs.a) == 1
        assert coeffs.a[0] == pytest.approx(1.0)

    def test_linear_extrapolation_uniform_dt(self):
        """Linear extrapolation with uniform timestep.

        For uniform dt, linear extrapolation: x_{n+1} = 2*x_n - x_{n-1}
        So coefficients should be [2, -1].
        """
        dt = 1e-9
        coeffs = compute_predictor_coeffs(past_dt=[dt], new_dt=dt, order=1)

        assert coeffs.order == 1
        assert len(coeffs.a) == 2
        # For uniform dt: tau = [-1, -2], solving gives [2, -1]
        assert coeffs.a[0] == pytest.approx(2.0, rel=1e-10)
        assert coeffs.a[1] == pytest.approx(-1.0, rel=1e-10)

    def test_quadratic_extrapolation_uniform_dt(self):
        """Quadratic extrapolation with uniform timestep."""
        dt = 1e-9
        coeffs = compute_predictor_coeffs(past_dt=[dt, dt], new_dt=dt, order=2)

        assert coeffs.order == 2
        assert len(coeffs.a) == 3
        # For uniform dt: tau = [-1, -2, -3], solving gives [3, -3, 1]
        assert coeffs.a[0] == pytest.approx(3.0, rel=1e-10)
        assert coeffs.a[1] == pytest.approx(-3.0, rel=1e-10)
        assert coeffs.a[2] == pytest.approx(1.0, rel=1e-10)

    def test_variable_dt_coeffs(self):
        """Test coefficients with non-uniform timesteps."""
        # Past dt: [2, 1] means most recent step was 2x larger
        coeffs = compute_predictor_coeffs(past_dt=[2e-9, 1e-9], new_dt=1e-9, order=2)

        assert coeffs.order == 2
        assert len(coeffs.a) == 3
        # Sum of coefficients should be 1 (interpolates through t_{n+1})
        assert sum(coeffs.a) == pytest.approx(1.0, rel=1e-10)

    def test_order_limited_by_history(self):
        """If not enough history, order should be reduced."""
        coeffs = compute_predictor_coeffs(past_dt=[1e-9], new_dt=1e-9, order=5)
        # Only 1 past dt means 2 points, so max order is 1
        assert coeffs.order == 1


class TestPredict:
    """Test prediction from past solutions."""

    def test_linear_predict_constant(self):
        """Linear predictor on constant data should give same constant."""
        dt = 1e-9
        coeffs = compute_predictor_coeffs(past_dt=[dt], new_dt=dt, order=1)

        V_n = jnp.array([1.0, 2.0, 3.0])
        V_nm1 = jnp.array([1.0, 2.0, 3.0])  # Same as V_n

        V_pred = predict(coeffs, [V_n, V_nm1])

        # Constant data should predict same value
        np.testing.assert_allclose(V_pred, V_n, rtol=1e-10)

    def test_linear_predict_linear_trend(self):
        """Linear predictor on linear trend should extrapolate correctly."""
        dt = 1e-9
        coeffs = compute_predictor_coeffs(past_dt=[dt], new_dt=dt, order=1)

        # Linear trend: V(t) = 1.0 + t/dt
        V_n = jnp.array([2.0])  # at t=1*dt
        V_nm1 = jnp.array([1.0])  # at t=0

        V_pred = predict(coeffs, [V_n, V_nm1])

        # Should predict V_{n+1} = 3.0
        assert V_pred[0] == pytest.approx(3.0, rel=1e-10)

    def test_quadratic_predict_quadratic_trend(self):
        """Quadratic predictor on quadratic trend should extrapolate correctly."""
        dt = 1e-9
        coeffs = compute_predictor_coeffs(past_dt=[dt, dt], new_dt=dt, order=2)

        # Quadratic trend: V(t) = (t/dt)^2
        V_n = jnp.array([4.0])  # at t=2*dt: (2)^2 = 4
        V_nm1 = jnp.array([1.0])  # at t=1*dt: (1)^2 = 1
        V_nm2 = jnp.array([0.0])  # at t=0: (0)^2 = 0

        V_pred = predict(coeffs, [V_n, V_nm1, V_nm2])

        # Should predict V_{n+1} = (3)^2 = 9
        assert V_pred[0] == pytest.approx(9.0, rel=1e-10)


class TestLTEEstimation:
    """Test Local Truncation Error estimation."""

    def test_lte_scaling_factor(self):
        """Test the LTE scaling factor formula."""
        V_pred = jnp.array([1.0, 2.0])
        V_corr = jnp.array([1.1, 2.2])

        # LTE = factor * (corr - pred)
        # factor = err_int / (err_int - err_pred)
        err_int = 0.5  # BE
        err_pred = -0.25  # example predictor error

        lte = estimate_lte(V_pred, V_corr, err_pred, err_int)

        expected_factor = err_int / (err_int - err_pred)
        expected_lte = expected_factor * (V_corr - V_pred)

        np.testing.assert_allclose(lte, expected_lte, rtol=1e-10)

    def test_lte_zero_difference(self):
        """If predictor and corrector match, LTE should be zero."""
        V_pred = jnp.array([1.0, 2.0, 3.0])
        V_corr = jnp.array([1.0, 2.0, 3.0])

        lte = estimate_lte(V_pred, V_corr, -0.25, 0.5)

        np.testing.assert_allclose(lte, jnp.zeros(3), atol=1e-15)


class TestTimestepCalculation:
    """Test timestep adjustment based on LTE."""

    def test_new_timestep_formula(self):
        """Test h_new = h * ratio^(-1/(order+1))."""
        lte = jnp.array([1e-6])  # 1 uV error
        V_ref = jnp.array([1.0])  # 1V reference
        reltol = 1e-3
        abstol = 1e-6
        lte_ratio = 1.0
        current_dt = 1e-9
        order = 1

        dt_new, max_ratio = compute_new_timestep(
            lte, V_ref, reltol, abstol, lte_ratio, current_dt, order
        )

        # tol = max(1.0 * 1e-3, 1e-6) = 1e-3
        # lte_ratio_val = 1e-6 / (1e-3 * 1.0) = 1e-3
        # dt_new = 1e-9 * (1e-3)^(-1/2) = 1e-9 * ~31.6 = ~3.16e-8
        assert dt_new > current_dt  # Should increase dt
        assert max_ratio < 1.0  # LTE is well within tolerance

    def test_large_lte_reduces_timestep(self):
        """Large LTE should reduce timestep."""
        lte = jnp.array([1e-2])  # 10 mV error - large!
        V_ref = jnp.array([1.0])
        reltol = 1e-3
        abstol = 1e-6
        lte_ratio = 1.0
        current_dt = 1e-9
        order = 1

        dt_new, max_ratio = compute_new_timestep(
            lte, V_ref, reltol, abstol, lte_ratio, current_dt, order
        )

        # lte_ratio_val = 1e-2 / 1e-3 = 10
        # dt_new = 1e-9 * 10^(-1/2) = 1e-9 * 0.316 = 3.16e-10
        assert dt_new < current_dt  # Should decrease dt
        assert max_ratio > 1.0  # LTE exceeds tolerance

    def test_min_max_dt_bounds(self):
        """Timestep should respect min/max bounds."""
        lte = jnp.array([1e-15])  # Very small error
        V_ref = jnp.array([1.0])
        reltol = 1e-3
        abstol = 1e-6
        lte_ratio = 1.0
        current_dt = 1e-9
        order = 1
        min_dt = 1e-15
        max_dt = 1e-6

        dt_new, _ = compute_new_timestep(
            lte, V_ref, reltol, abstol, lte_ratio, current_dt, order, min_dt=min_dt, max_dt=max_dt
        )

        # Even though error suggests huge increase, should be capped
        assert dt_new <= max_dt
        assert dt_new >= min_dt

    def test_accept_reject_threshold(self):
        """Test the accept/reject decision based on dt ratio."""
        from vajax.analysis.transient import AdaptiveConfig

        config = AdaptiveConfig(redo_factor=2.5)

        # Simulate a step where LTE is too large
        lte = jnp.array([1e-1])  # Very large error
        V_ref = jnp.array([1.0])
        current_dt = 1e-9

        dt_new, max_ratio = compute_new_timestep(
            lte, V_ref, config.reltol, config.abstol, config.lte_ratio, current_dt, order=1
        )

        # Check if this would be rejected
        should_reject = (current_dt / dt_new) > config.redo_factor

        # With such large LTE, dt_new should be much smaller, causing rejection
        if max_ratio > config.redo_factor:
            assert should_reject


class TestAdaptiveConfig:
    """Test AdaptiveConfig defaults and validation."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        from vajax.analysis.transient import AdaptiveConfig

        config = AdaptiveConfig()

        assert config.lte_ratio > 0
        assert config.redo_factor > 1
        assert config.reltol > 0
        assert config.abstol > 0
        assert config.min_dt > 0
        assert config.warmup_steps >= 2

    def test_custom_config(self):
        """Custom config values should be stored correctly."""
        from vajax.analysis.transient import AdaptiveConfig

        config = AdaptiveConfig(
            lte_ratio=5.0,
            redo_factor=3.0,
            reltol=1e-4,
            abstol=1e-14,
        )

        assert config.lte_ratio == 5.0
        assert config.redo_factor == 3.0
        assert config.reltol == 1e-4
        assert config.abstol == 1e-14
