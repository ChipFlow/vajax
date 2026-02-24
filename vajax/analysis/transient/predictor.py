"""Polynomial extrapolation predictor for adaptive timestep control.

This module implements the predictor component of a predictor-corrector scheme
for adaptive timestep control. The predictor uses polynomial extrapolation
from past solutions to estimate the solution at the next timestep.

The predictor serves two purposes:
1. Provides a good initial guess for Newton-Raphson, improving convergence
2. Enables Local Truncation Error (LTE) estimation via comparison with corrector

Algorithm follows VACASK's polynomial extrapolation predictor (coretrancoef.cpp).
"""

from typing import List, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array


class PredictorCoeffs(NamedTuple):
    """Coefficients for polynomial extrapolation predictor.

    The predicted solution is: x_{n+1,pred} = sum_i a[i] * x_{n-i}
    where a[i] are the coefficients and x_{n-i} are past solutions.

    Attributes:
        a: Coefficients for past solutions [a_0, a_1, ..., a_{order-1}]
        error_coeff: Error coefficient C for LTE estimation
        order: Order of polynomial extrapolation (1=linear, 2=quadratic, etc.)
    """

    a: Array  # Coefficients for past solutions
    error_coeff: float  # Error coefficient for LTE
    order: int  # Order of extrapolation


def compute_predictor_coeffs(past_dt: List[float], new_dt: float, order: int) -> PredictorCoeffs:
    """Compute polynomial extrapolation coefficients for given timestep history.

    Given past timesteps [h_{n-1}, h_{n-2}, ...] (most recent first) and the
    proposed new timestep h_n, compute coefficients for predicting x_{n+1}.

    The polynomial extrapolation of order p uses p+1 past points to fit
    a polynomial and extrapolate to t_{n+1} = t_n + h_n.

    Args:
        past_dt: List of past timesteps [h_{n-1}, h_{n-2}, ...], most recent first
        new_dt: Proposed timestep h_n for next step
        order: Order of polynomial extrapolation (1=linear, 2=quadratic)

    Returns:
        PredictorCoeffs with coefficients and error coefficient
    """
    if order < 1:
        raise ValueError(f"Order must be >= 1, got {order}")

    n_history = len(past_dt)
    if n_history < order:
        # Not enough history - fall back to lower order
        order = n_history

    if order == 0:
        # No history at all - use identity (constant extrapolation)
        return PredictorCoeffs(
            a=jnp.array([1.0]),
            error_coeff=1.0,  # Large error - first step
            order=0,
        )

    # Compute normalized past timepoints relative to t_{n+1}
    # t_n is at -new_dt, t_{n-1} is at -(new_dt + past_dt[0]), etc.
    # Normalize by new_dt: tau_k = (t_{n-k} - t_{n+1}) / h_n
    tau = _compute_normalized_timepoints(past_dt, new_dt, order)

    # Solve for polynomial coefficients
    # We want: sum_i a_i * tau_i^j = delta_{j,0} for j=0..order
    # This ensures the polynomial passes through all points and
    # evaluates to 1 at t_{n+1} (tau=0)
    a = _solve_predictor_system(tau, order)

    # Compute error coefficient
    # C = -1 + sum_i a_i * tau_i^(order+1)
    error_coeff = _compute_error_coeff(a, tau, order)

    return PredictorCoeffs(
        a=jnp.array(a),
        error_coeff=error_coeff,
        order=order,
    )


def _compute_normalized_timepoints(past_dt: List[float], new_dt: float, order: int) -> np.ndarray:
    """Compute normalized past timepoints for predictor.

    Timepoints are normalized so that t_{n+1} is at tau=0 and
    distances are scaled by new_dt.

    tau_0 = (t_n - t_{n+1}) / h_n = -1
    tau_1 = (t_{n-1} - t_{n+1}) / h_n = -(1 + h_{n-1}/h_n)
    tau_2 = (t_{n-2} - t_{n+1}) / h_n = -(1 + h_{n-1}/h_n + h_{n-2}/h_n)
    ...

    Args:
        past_dt: Past timesteps [h_{n-1}, h_{n-2}, ...]
        new_dt: New timestep h_n
        order: Number of points to use

    Returns:
        Array of normalized timepoints [tau_0, tau_1, ..., tau_{order}]
    """
    tau = np.zeros(order + 1)
    cumsum = 0.0

    for i in range(order + 1):
        # tau_i = -(new_dt + sum(past_dt[0:i])) / new_dt
        tau[i] = -(1.0 + cumsum / new_dt)
        if i < len(past_dt):
            cumsum += past_dt[i]

    return tau


def _solve_predictor_system(tau: np.ndarray, order: int) -> np.ndarray:
    """Solve linear system for predictor coefficients.

    We want coefficients a_i such that:
    - sum_i a_i = 1  (polynomial evaluates to 1 at tau=0)
    - sum_i a_i * tau_i^j = 0 for j=1..order (derivatives vanish at tau=0)

    This is equivalent to Lagrange interpolation at tau=0.

    Args:
        tau: Normalized timepoints [tau_0, ..., tau_order]
        order: Order of polynomial

    Returns:
        Coefficients [a_0, a_1, ..., a_order]
    """
    n = order + 1

    # Build Vandermonde-like matrix
    # A[j, i] = tau[i]^j
    A = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            A[j, i] = tau[i] ** j

    # RHS: [1, 0, 0, ..., 0]
    b = np.zeros(n)
    b[0] = 1.0

    # Solve for coefficients
    try:
        a = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Singular matrix - fall back to constant extrapolation
        a = np.zeros(n)
        a[0] = 1.0

    return a


def _compute_error_coeff(a: np.ndarray, tau: np.ndarray, order: int) -> float:
    """Compute error coefficient for predictor.

    The error coefficient C appears in the LTE estimate:
        LTE ≈ C * h^(order+1) * x^(order+1)

    For polynomial extrapolation:
        C = -1 + sum_i a_i * tau_i^(order+1)

    Args:
        a: Predictor coefficients
        tau: Normalized timepoints
        order: Order of extrapolation

    Returns:
        Error coefficient C
    """
    # C = -1 + sum_i a_i * tau_i^(order+1)
    power = order + 1
    c = -1.0
    for i in range(len(a)):
        c += a[i] * (tau[i] ** power)
    return c


def predict(coeffs: PredictorCoeffs, V_history: List[Array]) -> Array:
    """Predict solution at t_{n+1} using polynomial extrapolation.

    Args:
        coeffs: Predictor coefficients from compute_predictor_coeffs()
        V_history: Past solutions [V_n, V_{n-1}, ...], most recent first

    Returns:
        Predicted solution V_{n+1,pred}
    """
    if len(V_history) == 0:
        raise ValueError("Need at least one past solution for prediction")

    # Ensure we have enough history for the coefficients
    n_coeffs = len(coeffs.a)
    if len(V_history) < n_coeffs:
        # Pad with oldest available solution
        V_history = list(V_history) + [V_history[-1]] * (n_coeffs - len(V_history))

    # V_pred = sum_i a_i * V_{n-i}
    V_pred = coeffs.a[0] * V_history[0]
    for i in range(1, n_coeffs):
        V_pred = V_pred + coeffs.a[i] * V_history[i]

    return V_pred


def estimate_lte(
    V_predicted: Array,
    V_corrected: Array,
    predictor_error_coeff: float,
    integrator_error_coeff: float,
) -> Array:
    """Estimate Local Truncation Error from predictor-corrector difference.

    The LTE is estimated using the difference between predictor and corrector:
        LTE = factor * (V_corrected - V_predicted)

    Where:
        factor = integrator_error / (integrator_error - predictor_error)

    This formula comes from the Milne device (comparing two different
    order methods on the same problem).

    Args:
        V_predicted: Predicted solution from polynomial extrapolation
        V_corrected: Corrected solution from Newton-Raphson
        predictor_error_coeff: Error coefficient C_p from predictor
        integrator_error_coeff: Error coefficient C_i from integration method

    Returns:
        Estimated LTE vector (same shape as V)
    """
    # Avoid division by zero
    denom = integrator_error_coeff - predictor_error_coeff
    if abs(denom) < 1e-15:
        # Coefficients are equal - can't estimate LTE this way
        # Return the difference directly as a conservative estimate
        return V_corrected - V_predicted

    factor = integrator_error_coeff / denom
    return factor * (V_corrected - V_predicted)


def compute_new_timestep(
    lte: Array,
    V_ref: Array,
    reltol: float,
    abstol: float,
    lte_ratio: float,
    current_dt: float,
    order: int,
    min_dt: float = 1e-18,
    max_dt: float = float("inf"),
) -> tuple[float, float]:
    """Compute new timestep from LTE and decide accept/reject.

    The new timestep is computed using:
        h_new = h * ratio^(-1/(order+1))

    Where ratio is the maximum LTE ratio across all unknowns.

    Args:
        lte: Local truncation error vector
        V_ref: Reference voltage for tolerance (typically V_corrected)
        reltol: Relative tolerance
        abstol: Absolute tolerance
        lte_ratio: LTE tolerance multiplier (tran_lteratio)
        current_dt: Current timestep
        order: Order of integration method
        min_dt: Minimum allowed timestep
        max_dt: Maximum allowed timestep

    Returns:
        Tuple of (new_dt, max_lte_ratio) where max_lte_ratio can be used
        for accept/reject decision
    """
    # Compute tolerance for each unknown
    # tol[i] = max(|V_ref[i]| * reltol, abstol)
    tol = jnp.maximum(jnp.abs(V_ref) * reltol, abstol)

    # Compute LTE ratio for each unknown
    # ratio[i] = |LTE[i]| / (tol[i] * lte_ratio)
    lte_ratios = jnp.abs(lte) / (tol * lte_ratio)

    # Find maximum ratio
    max_ratio = float(jnp.max(lte_ratios))

    # Compute new timestep using error scaling
    # h_new = h * ratio^(-1/(order+1))
    if max_ratio > 0:
        exponent = -1.0 / (order + 1)
        dt_new = current_dt * (max_ratio**exponent)
    else:
        # Perfect prediction - can increase timestep significantly
        dt_new = current_dt * 2.0

    # Clip to bounds
    dt_new = max(min_dt, min(max_dt, dt_new))

    return dt_new, max_ratio
