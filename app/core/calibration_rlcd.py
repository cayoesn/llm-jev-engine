"""RLCD Statistical Calibration Engine with vectorized Log-Sum-Exp, ECE, Brier, and NLL."""

from __future__ import annotations

import math
import threading
from typing import overload

import numpy as np

from app.api.schemas import (
    CalibrationFitRequest,
    CalibrationFitResponse,
    CalibrationMetricsReport,
    CalibrationPoint,
)

GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def apply_temperature_scaling(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable temperature-scaled softmax probabilities via Log-Sum-Exp.

    Args:
        logits: 1D (K,) or 2D (N, K) array of unnormalized neural logits.
        temperature: Strictly positive post-hoc scaling factor T > 0.

    Returns:
        Array of calibrated probabilities with identical shape summing to 1.0 along axis -1.

    Raises:
        ValueError: If temperature is <= 0.0.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive (got {temperature}).")
    arr = np.asarray(logits, dtype=np.float64)
    scaled = arr / float(temperature)
    max_logits = np.max(scaled, axis=-1, keepdims=True)
    shifted = scaled - max_logits
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals, axis=-1, keepdims=True)
    return exp_vals / np.maximum(denom, 1e-15)


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Alias for apply_temperature_scaling providing max-shifted Log-Sum-Exp softmax."""
    return apply_temperature_scaling(logits, temperature=temperature)


@overload
def apply_platt_scaling(logit_diff: float, a: float = 1.0, b: float = 0.0) -> float: ...


@overload
def apply_platt_scaling(
    logit_diff: np.ndarray, a: float = 1.0, b: float = 0.0
) -> np.ndarray: ...


def apply_platt_scaling(
    logit_diff: float | np.ndarray,
    a: float = 1.0,
    b: float = 0.0,
) -> float | np.ndarray:
    """Apply parametric logistic Platt scaling sigma(a * delta_z + b).

    Supports scalar logit margins, 1D margin arrays, or 2D (N, 2) binary logit matrices.
    """
    if isinstance(logit_diff, (int, float, np.floating)):
        exponent = float(np.clip(a * float(logit_diff) + b, -500.0, 500.0))
        return float(1.0 / (1.0 + math.exp(-exponent)))

    arr = np.asarray(logit_diff, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        delta = arr[:, 0] - arr[:, 1]
        clipped = np.clip(a * delta + b, -500.0, 500.0)
        p_pos = 1.0 / (1.0 + np.exp(-clipped))
        return np.stack([p_pos, 1.0 - p_pos], axis=-1)

    clipped = np.clip(a * arr + b, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compute_ece(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across uniform confidence bins in O(N + B).

    Args:
        probabilities: (N, K) calibrated probability matrix.
        targets: (N,) integer array of ground-truth class indices.
        n_bins: Number of reliability partition bins over [0.0, 1.0].

    Returns:
        Scalar Expected Calibration Error in [0.0, 1.0].
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]
    y_true = np.asarray(targets, dtype=np.int64)
    n_samples = probs.shape[0]
    if n_samples == 0:
        return 0.0

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece_total = 0.0
    for idx in range(n_bins):
        low = bin_boundaries[idx]
        high = bin_boundaries[idx + 1]
        if idx == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece_total += (count / n_samples) * abs(bin_acc - bin_conf)
    return float(np.clip(ece_total, 0.0, 1.0))


def compute_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Compute multiclass Brier Score in O(N * K) vectorized operations."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]
    y_true = np.asarray(targets, dtype=np.int64)
    n_samples, n_classes = probs.shape
    if n_samples == 0:
        return 0.0
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    squared_errors = np.sum((probs - one_hot) ** 2, axis=-1)
    return float(np.mean(squared_errors))


def compute_nll(probabilities: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> float:
    """Compute average Negative Log-Likelihood (cross-entropy loss) in O(N)."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]
    y_true = np.asarray(targets, dtype=np.int64)
    n_samples = probs.shape[0]
    if n_samples == 0:
        return 0.0
    target_probs = probs[np.arange(n_samples), y_true]
    clipped = np.clip(target_probs, eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def compute_wilson_interval(
    p_hat: float,
    n_eff: float = 64.0,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Compute closed-form 95% Wilson score credible interval for a probability estimate."""
    p = float(np.clip(p_hat, 0.0, 1.0))
    n = max(float(n_eff), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    low = float(np.clip(center - half_width, 0.0, 1.0))
    high = float(np.clip(center + half_width, 0.0, 1.0))
    return (round(low, 4), round(high, 4))


class RLCDCalibrator:
    """Thread-safe post-hoc statistical calibrator using Golden-Section NLL/ECE optimization.

    Maintains fitted temperature and Platt scaling parameters protected by a single RLock.
    """

    def __init__(
        self,
        default_temperature: float = 0.85,
        platt_a: float = 1.0,
        platt_b: float = 0.0,
        n_bins: int = 10,
    ) -> None:
        """Initialize the RLCDCalibrator with default scaling and reliability bin parameters."""
        if default_temperature <= 0.0:
            raise ValueError("default_temperature must be > 0.0.")
        self._lock = threading.RLock()
        self._temperature = float(default_temperature)
        self._platt_a = float(platt_a)
        self._platt_b = float(platt_b)
        self._n_bins = int(n_bins)
        self._metrics_history: list[CalibrationFitResponse] = []

    @property
    def temperature(self) -> float:
        """Return the current thread-safe post-hoc softmax temperature."""
        with self._lock:
            return self._temperature

    @property
    def platt_a(self) -> float:
        """Return the current Platt scaling slope coefficient."""
        with self._lock:
            return self._platt_a

    @property
    def platt_b(self) -> float:
        """Return the current Platt scaling intercept bias."""
        with self._lock:
            return self._platt_b

    @property
    def n_bins(self) -> int:
        """Return the configured number of ECE reliability bins."""
        with self._lock:
            return self._n_bins

    @property
    def metrics_history(self) -> list[CalibrationFitResponse]:
        """Return a shallow copy of historical calibration fit reports."""
        with self._lock:
            return list(self._metrics_history)

    def apply_temperature_scaling(
        self,
        logits: np.ndarray,
        temperature: float | None = None,
    ) -> np.ndarray:
        """Scale logits by either an explicit temperature or the calibrator's fitted temperature."""
        with self._lock:
            temp = self._temperature if temperature is None else float(temperature)
        return apply_temperature_scaling(logits, temperature=temp)

    def apply_platt_scaling(
        self,
        logit_diff: float | np.ndarray,
        a: float | None = None,
        b: float | None = None,
    ) -> float | np.ndarray:
        """Apply binary Platt scaling using explicit or fitted parameters."""
        with self._lock:
            slope = self._platt_a if a is None else float(a)
            intercept = self._platt_b if b is None else float(b)
        return apply_platt_scaling(logit_diff, a=slope, b=intercept)

    def evaluate_metrics(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        temperature: float | None = None,
        n_bins: int | None = None,
    ) -> CalibrationMetricsReport:
        """Evaluate ECE, Brier Score, NLL, mean confidence, and accuracy at a given temperature."""
        with self._lock:
            temp = self._temperature if temperature is None else float(temperature)
            bins = self._n_bins if n_bins is None else int(n_bins)

        logits_arr = np.asarray(logits, dtype=np.float64)
        targets_arr = np.asarray(targets, dtype=np.int64)
        probs = apply_temperature_scaling(logits_arr, temperature=temp)
        ece_val = compute_ece(probs, targets_arr, n_bins=bins)
        brier_val = compute_brier_score(probs, targets_arr)
        nll_val = compute_nll(probs, targets_arr)
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        mean_conf = float(np.mean(confidences)) if probs.shape[0] > 0 else 0.0
        emp_acc = float(np.mean(predictions == targets_arr)) if probs.shape[0] > 0 else 0.0

        return CalibrationMetricsReport(
            ece=round(ece_val, 6),
            brier_score=round(brier_val, 6),
            nll=round(nll_val, 6),
            mean_confidence=round(mean_conf, 6),
            empirical_accuracy=round(emp_acc, 6),
            num_samples=int(probs.shape[0]),
            num_bins=bins,
        )

    def fit_temperature(
        self,
        logits: np.ndarray | list[CalibrationPoint],
        targets: np.ndarray | None = None,
        n_bins: int | None = None,
        bounds: tuple[float, float] = (0.05, 10.0),
        tol: float = 1e-4,
    ) -> CalibrationFitResponse:
        """Optimize softmax temperature T via vectorized Golden-Section search in O(N * K)."""
        if isinstance(logits, list):
            logits_arr = np.array([pt.logits for pt in logits], dtype=np.float64)
            targets_arr = np.array([pt.target_index for pt in logits], dtype=np.int64)
        else:
            logits_arr = np.asarray(logits, dtype=np.float64)
            if targets is None:
                raise ValueError("targets must be provided when logits is a NumPy array.")
            targets_arr = np.asarray(targets, dtype=np.int64)

        with self._lock:
            bins = self._n_bins if n_bins is None else int(n_bins)

        pre_report = self.evaluate_metrics(
            logits_arr,
            targets_arr,
            temperature=1.0,
            n_bins=bins,
        )

        def objective(temp: float) -> float:
            probs = apply_temperature_scaling(logits_arr, temperature=temp)
            nll_val = compute_nll(probs, targets_arr)
            ece_val = compute_ece(probs, targets_arr, n_bins=bins)
            return nll_val + 0.35 * ece_val

        low, high = float(bounds[0]), float(bounds[1])
        c = high - GOLDEN_RATIO_CONJUGATE * (high - low)
        d = low + GOLDEN_RATIO_CONJUGATE * (high - low)
        fc = objective(c)
        fd = objective(d)

        while (high - low) > tol:
            if fc < fd:
                high = d
                d = c
                fd = fc
                c = high - GOLDEN_RATIO_CONJUGATE * (high - low)
                fc = objective(c)
            else:
                low = c
                c = d
                fc = fd
                d = low + GOLDEN_RATIO_CONJUGATE * (high - low)
                fd = objective(d)

        optimal_temp = round((low + high) / 2.0, 5)
        post_report = self.evaluate_metrics(
            logits_arr,
            targets_arr,
            temperature=optimal_temp,
            n_bins=bins,
        )

        pre_ece = max(pre_report.ece, 1e-9)
        pre_nll = max(pre_report.nll, 1e-9)
        ece_reduction = round(((pre_report.ece - post_report.ece) / pre_ece) * 100.0, 2)
        nll_reduction = round(((pre_report.nll - post_report.nll) / pre_nll) * 100.0, 2)
        platt_slope = round(1.0 / optimal_temp, 5)

        response = CalibrationFitResponse(
            optimal_temperature=optimal_temp,
            platt_scale_a=platt_slope,
            platt_bias_b=0.0,
            pre_calibration=pre_report,
            post_calibration=post_report,
            ece_reduction_pct=ece_reduction,
            nll_reduction_pct=nll_reduction,
        )

        with self._lock:
            self._temperature = optimal_temp
            self._platt_a = platt_slope
            self._platt_b = 0.0
            self._metrics_history.append(response)

        return response

    def fit_from_request(self, request: CalibrationFitRequest) -> CalibrationFitResponse:
        """Fit RLCD calibration parameters directly from a validated CalibrationFitRequest."""
        return self.fit_temperature(
            logits=request.samples,
            n_bins=request.num_bins,
        )
