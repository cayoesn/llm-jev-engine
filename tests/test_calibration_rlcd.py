"""Unit and concurrency tests for the RLCD Statistical Calibration Engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.api.schemas import CalibrationFitRequest, CalibrationPoint
from app.core.calibration_rlcd import (
    RLCDCalibrator,
    apply_platt_scaling,
    apply_temperature_scaling,
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_wilson_interval,
    stable_softmax,
)


def test_log_sum_exp_extreme_logits_numerical_stability() -> None:
    """Verify stable_softmax and apply_temperature_scaling prevent NaN/Inf on extreme logits."""
    extreme_1d = np.array([10000.0, 9999.0, -10000.0], dtype=np.float64)
    probs_1d = apply_temperature_scaling(extreme_1d, temperature=1.0)
    assert probs_1d.shape == (3,)
    assert not np.any(np.isnan(probs_1d))
    assert not np.any(np.isinf(probs_1d))
    assert np.sum(probs_1d) == pytest.approx(1.0, abs=1e-9)
    assert probs_1d[0] > probs_1d[1] > probs_1d[2]
    assert probs_1d[0] / probs_1d[1] == pytest.approx(np.e, rel=1e-5)

    extreme_2d = np.array(
        [
            [10000.0, 9999.0, -10000.0],
            [-50000.0, -50000.0, -50000.0],
        ],
        dtype=np.float64,
    )
    probs_2d = stable_softmax(extreme_2d, temperature=0.85)
    assert probs_2d.shape == (2, 3)
    assert np.allclose(np.sum(probs_2d, axis=-1), [1.0, 1.0])
    assert np.allclose(probs_2d[1], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    with pytest.raises(ValueError, match="temperature must be strictly positive"):
        apply_temperature_scaling(extreme_1d, temperature=0.0)

    with pytest.raises(ValueError, match="temperature must be strictly positive"):
        apply_temperature_scaling(extreme_1d, temperature=-1.5)


def test_platt_scaling_scalar_and_vectorized() -> None:
    """Verify binary Platt scaling for scalar logit differences, 1D arrays, and 2D matrices."""
    p_zero = apply_platt_scaling(0.0, a=1.5, b=0.0)
    assert isinstance(p_zero, float)
    assert p_zero == pytest.approx(0.5)

    p_extreme_pos = apply_platt_scaling(10000.0, a=2.0, b=0.5)
    p_extreme_neg = apply_platt_scaling(-10000.0, a=2.0, b=-0.5)
    assert p_extreme_pos == pytest.approx(1.0)
    assert p_extreme_neg == pytest.approx(0.0)

    diffs_1d = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    probs_1d = apply_platt_scaling(diffs_1d, a=1.0, b=0.0)
    assert isinstance(probs_1d, np.ndarray)
    assert probs_1d.shape == (3,)
    assert probs_1d[0] < 0.5 < probs_1d[2]
    assert probs_1d[1] == pytest.approx(0.5)

    binary_logits_2d = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    probs_2d = apply_platt_scaling(binary_logits_2d, a=1.0, b=0.0)
    assert isinstance(probs_2d, np.ndarray)
    assert probs_2d.shape == (2, 2)
    assert np.allclose(np.sum(probs_2d, axis=-1), [1.0, 1.0])
    assert probs_2d[0, 0] > 0.85
    assert probs_2d[1, 1] > 0.85


def test_ece_brier_nll_and_wilson_interval_invariants() -> None:
    """Verify ECE, Brier Score, NLL, and Wilson credible interval mathematical invariants."""
    perfect_probs = np.array(
        [
            [0.999, 0.0005, 0.0005],
            [0.0005, 0.999, 0.0005],
            [0.0005, 0.0005, 0.999],
        ],
        dtype=np.float64,
    )
    targets = np.array([0, 1, 2], dtype=np.int64)

    ece_perfect = compute_ece(perfect_probs, targets, n_bins=10)
    brier_perfect = compute_brier_score(perfect_probs, targets)
    nll_perfect = compute_nll(perfect_probs, targets)

    assert 0.0 <= ece_perfect < 0.01
    assert 0.0 <= brier_perfect < 0.001
    assert 0.0 <= nll_perfect < 0.002

    single_prob_1d = np.array([0.8, 0.2], dtype=np.float64)
    single_target = np.array([0], dtype=np.int64)
    assert compute_ece(single_prob_1d, single_target, n_bins=5) >= 0.0
    assert compute_brier_score(single_prob_1d, single_target) >= 0.0
    assert compute_nll(single_prob_1d, single_target) >= 0.0

    empty_probs = np.zeros((0, 2), dtype=np.float64)
    empty_targets = np.zeros((0,), dtype=np.int64)
    assert compute_ece(empty_probs, empty_targets) == 0.0
    assert compute_brier_score(empty_probs, empty_targets) == 0.0
    assert compute_nll(empty_probs, empty_targets) == 0.0

    overconfident_wrong = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.90, 0.05, 0.05],
            [0.02, 0.96, 0.02],
        ],
        dtype=np.float64,
    )
    wrong_targets = np.array([1, 2, 0], dtype=np.int64)
    ece_wrong = compute_ece(overconfident_wrong, wrong_targets, n_bins=10)
    brier_wrong = compute_brier_score(overconfident_wrong, wrong_targets)
    nll_wrong = compute_nll(overconfident_wrong, wrong_targets)

    assert ece_wrong > 0.85
    assert brier_wrong > 1.5
    assert nll_wrong > 2.5

    low, high = compute_wilson_interval(0.85, n_eff=100.0)
    assert 0.75 < low < 0.85 < high < 0.92


def test_fit_temperature_reduces_ece_and_nll_on_overconfident_logits() -> None:
    """Verify Golden-Section temperature optimization strictly reduces NLL and ECE."""
    rng = np.random.default_rng(42)
    n_samples = 120
    n_classes = 4
    base_logits = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_classes))
    true_probs = stable_softmax(base_logits, temperature=1.0)
    targets = np.array(
        [rng.choice(n_classes, p=true_probs[i]) for i in range(n_samples)],
        dtype=np.int64,
    )

    overconfident_logits = base_logits * 4.5
    calibrator = RLCDCalibrator(default_temperature=1.0, n_bins=10)
    fit_response = calibrator.fit_temperature(overconfident_logits, targets)

    assert fit_response.optimal_temperature > 1.5
    assert fit_response.post_calibration.nll < fit_response.pre_calibration.nll
    assert fit_response.post_calibration.ece < fit_response.pre_calibration.ece
    assert fit_response.post_calibration.brier_score <= fit_response.pre_calibration.brier_score
    assert fit_response.ece_reduction_pct > 20.0
    assert fit_response.nll_reduction_pct > 20.0
    assert calibrator.temperature == pytest.approx(fit_response.optimal_temperature)
    assert len(calibrator.metrics_history) == 1


def test_calibrator_with_calibration_points_and_platt_method() -> None:
    """Verify RLCDCalibrator accepts CalibrationPoint lists, properties, and error guards."""
    with pytest.raises(ValueError, match="default_temperature must be > 0.0"):
        RLCDCalibrator(default_temperature=0.0)

    points = [
        CalibrationPoint(logits=[6.0, -2.0], target_index=0),
        CalibrationPoint(logits=[5.5, -1.5], target_index=1),
        CalibrationPoint(logits=[-5.0, 4.5], target_index=1),
        CalibrationPoint(logits=[-4.5, 4.0], target_index=0),
        CalibrationPoint(logits=[4.0, -1.0], target_index=0),
        CalibrationPoint(logits=[-3.5, 3.5], target_index=1),
    ]
    calibrator = RLCDCalibrator(default_temperature=1.0, platt_a=1.2, platt_b=0.1, n_bins=5)
    assert calibrator.platt_a == pytest.approx(1.2)
    assert calibrator.platt_b == pytest.approx(0.1)
    assert calibrator.n_bins == 5
    assert calibrator.apply_platt_scaling(0.0) == pytest.approx(
        apply_platt_scaling(0.0, a=1.2, b=0.1)
    )

    request = CalibrationFitRequest(samples=points, num_bins=5, method="platt")
    response = calibrator.fit_from_request(request)

    assert response.optimal_temperature > 0.0
    assert response.post_calibration.nll <= response.pre_calibration.nll
    assert calibrator.platt_a > 0.0

    with pytest.raises(ValueError, match="targets must be provided"):
        calibrator.fit_temperature(np.array([[1.0, 2.0]]), targets=None)


def test_calibrator_thread_safety_under_concurrent_reads_and_fits() -> None:
    """Verify RLCDCalibrator RLock prevents race conditions under concurrent threads."""
    calibrator = RLCDCalibrator(default_temperature=0.9, n_bins=8)
    logits = np.array(
        [
            [5.0, 1.0, -2.0],
            [4.5, 2.0, -1.0],
            [-2.0, 5.0, 0.5],
            [-1.0, 3.5, 2.5],
            [0.5, -1.0, 4.5],
            [2.0, 2.5, -1.0],
        ],
        dtype=np.float64,
    )
    targets = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)

    def worker(idx: int) -> float:
        if idx % 3 == 0:
            res = calibrator.fit_temperature(logits * (1.0 + 0.1 * (idx % 5)), targets)
            return res.optimal_temperature
        probs = calibrator.apply_temperature_scaling(logits)
        report = calibrator.evaluate_metrics(logits, targets)
        return float(np.sum(probs)) + report.ece

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(24)))

    assert len(results) == 24
    assert all(r > 0.0 for r in results)
    assert len(calibrator.metrics_history) == 8
