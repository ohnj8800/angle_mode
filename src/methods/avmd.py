from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.common.metrics import (
    calculate_mode_metrics,
    calculate_reconstruction_metrics,
    calculate_spectral_overlap_matrix,
)
from src.common.postprocessing import evaluate_welch_support
from src.common.preprocessing import calculate_welch_psd
from src.common.vmd import variational_mode_decomposition


def create_bandwise_initial_centers(
    signal: np.ndarray,
    sampling_rate: float,
    mode_count: int,
    minimum_frequency_hz: float = 0.5,
    maximum_frequency_hz: float | None = None,
) -> np.ndarray:
    """
    將有效頻率範圍切成 K 個頻帶，並在每個頻帶內找出
    Welch PSD 最大值，作為該模態的初始中心頻率。

    這和 IOVMD 的 LOWESS 全域峰值選模態數不同：
    AVMD 的 K 由候選搜尋決定，PSD 只用於初始化中心頻率。
    """
    signal = np.asarray(signal, dtype=float).reshape(-1)

    if signal.size < 8:
        raise ValueError("Signal is too short for AVMD initialization.")

    if mode_count < 2:
        raise ValueError("mode_count must be at least 2.")

    nyquist_frequency = sampling_rate / 2.0

    if maximum_frequency_hz is None:
        maximum_frequency_hz = nyquist_frequency - sampling_rate / signal.size

    maximum_frequency_hz = min(
        float(maximum_frequency_hz),
        nyquist_frequency - np.finfo(float).eps,
    )

    if minimum_frequency_hz <= 0:
        raise ValueError("minimum_frequency_hz must be greater than zero.")

    if maximum_frequency_hz <= minimum_frequency_hz:
        raise ValueError(
            "maximum_frequency_hz must be greater than minimum_frequency_hz."
        )

    frequency_hz, psd = calculate_welch_psd(
        signal,
        sampling_rate=sampling_rate,
    )

    band_edges = np.linspace(
        minimum_frequency_hz,
        maximum_frequency_hz,
        mode_count + 1,
    )

    initial_centers = []

    for band_index in range(mode_count):
        left_edge = band_edges[band_index]
        right_edge = band_edges[band_index + 1]

        if band_index == mode_count - 1:
            band_mask = (
                (frequency_hz >= left_edge)
                & (frequency_hz <= right_edge)
            )
        else:
            band_mask = (
                (frequency_hz >= left_edge)
                & (frequency_hz < right_edge)
            )

        band_frequency = frequency_hz[band_mask]
        band_psd = psd[band_mask]

        if band_frequency.size == 0:
            center_frequency = (left_edge + right_edge) / 2.0
        else:
            center_frequency = band_frequency[np.argmax(band_psd)]

        initial_centers.append(center_frequency)

    initial_centers = np.asarray(initial_centers, dtype=float)

    return np.sort(initial_centers)


def _mean_off_diagonal(
    matrix: np.ndarray,
    absolute: bool = False,
) -> float:
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix.")

    if matrix.shape[0] < 2:
        return 0.0

    values = matrix[np.triu_indices(matrix.shape[0], k=1)]

    if absolute:
        values = np.abs(values)

    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.mean(values))


def evaluate_avmd_candidate(
    signal: np.ndarray,
    sampling_rate: float,
    mode_count: int,
    alpha: float,
    tau: float = 0.0,
    tolerance: float = 1e-7,
    maximum_iterations: int = 500,
    welch_tolerance_hz: float = 1.0,
) -> dict:
    """
    執行一組 K、alpha 候選參數，並計算選模所需品質指標。
    """
    signal = np.asarray(signal, dtype=float).reshape(-1)

    initial_centers = create_bandwise_initial_centers(
        signal=signal,
        sampling_rate=sampling_rate,
        mode_count=mode_count,
    )

    vmd_result = variational_mode_decomposition(
        signal=signal,
        sampling_rate=sampling_rate,
        initial_center_frequencies_hz=initial_centers,
        alpha=alpha,
        tau=tau,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )

    modes = np.asarray(vmd_result["modes"], dtype=float)

    mode_metrics = calculate_mode_metrics(
        modes=modes,
        sampling_rate=sampling_rate,
        algorithm_center_frequencies_hz=vmd_result[
            "center_frequencies_hz"
        ],
    )

    reconstruction_metrics = calculate_reconstruction_metrics(
        original_signal=signal,
        reconstructed_signal=vmd_result["reconstructed_signal"],
    )

    spectral_overlap_matrix = calculate_spectral_overlap_matrix(
        modes=modes,
    )

    mean_spectral_overlap = _mean_off_diagonal(
        spectral_overlap_matrix,
        absolute=False,
    )

    waveform_correlation_matrix = np.corrcoef(modes)

    mean_waveform_correlation = _mean_off_diagonal(
        waveform_correlation_matrix,
        absolute=True,
    )

    mode_metrics, _ = evaluate_welch_support(
        mode_metrics=mode_metrics,
        reference_signal=signal,
        sampling_rate=sampling_rate,
        frequency_tolerance_hz=welch_tolerance_hz,
    )

    welch_support_ratio = float(
        mode_metrics["welch_psd_supported"]
        .astype(float)
        .mean()
    )

    return {
        "mode_count": int(mode_count),
        "alpha": float(alpha),
        "initial_center_frequencies_hz": initial_centers.tolist(),
        "final_center_frequencies_hz": np.asarray(
            vmd_result["center_frequencies_hz"],
            dtype=float,
        ).tolist(),
        "reconstruction_correlation": float(
            reconstruction_metrics[
                "reconstruction_correlation"
            ]
        ),
        "relative_reconstruction_error": float(
            reconstruction_metrics[
                "relative_reconstruction_error"
            ]
        ),
        "mean_spectral_overlap": mean_spectral_overlap,
        "mean_waveform_correlation": mean_waveform_correlation,
        "welch_support_ratio": welch_support_ratio,
        "converged": bool(vmd_result["converged"]),
        "iterations": int(vmd_result["iterations"]),
        "valid": True,
        "error_message": "",
    }


def _minmax_normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)

    minimum = values.min()
    maximum = values.max()

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return pd.Series(
            np.ones(len(values)),
            index=values.index,
            dtype=float,
        )

    if np.isclose(maximum, minimum):
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=float,
        )

    return (values - minimum) / (maximum - minimum)


def select_avmd_parameters(
    reference_signal: np.ndarray,
    sampling_rate: float = 200.0,
    candidate_mode_counts: Iterable[int] = range(3, 11),
    candidate_alphas: Iterable[float] = (
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
    ),
    tau: float = 0.0,
    tolerance: float = 1e-7,
    maximum_iterations: int = 500,
    welch_tolerance_hz: float = 1.0,
) -> dict:
    """
    使用正常運轉參考訊號，自動選擇 AVMD 的 K 與 alpha。

    選擇分數越低越好，包含：
    35% 重建誤差
    25% 模態頻譜重疊
    15% 模態波形相關
    15% Welch PSD 未支持比例
    10% 模態數複雜度
    """
    reference_signal = np.asarray(
        reference_signal,
        dtype=float,
    ).reshape(-1)

    candidate_mode_counts = sorted(
        {int(value) for value in candidate_mode_counts}
    )
    candidate_alphas = sorted(
        {float(value) for value in candidate_alphas}
    )

    if not candidate_mode_counts:
        raise ValueError("No candidate mode counts were provided.")

    if not candidate_alphas:
        raise ValueError("No candidate alpha values were provided.")

    trial_records = []

    for mode_count in candidate_mode_counts:
        for alpha in candidate_alphas:
            try:
                record = evaluate_avmd_candidate(
                    signal=reference_signal,
                    sampling_rate=sampling_rate,
                    mode_count=mode_count,
                    alpha=alpha,
                    tau=tau,
                    tolerance=tolerance,
                    maximum_iterations=maximum_iterations,
                    welch_tolerance_hz=welch_tolerance_hz,
                )
            except Exception as error:
                record = {
                    "mode_count": int(mode_count),
                    "alpha": float(alpha),
                    "initial_center_frequencies_hz": [],
                    "final_center_frequencies_hz": [],
                    "reconstruction_correlation": np.nan,
                    "relative_reconstruction_error": np.nan,
                    "mean_spectral_overlap": np.nan,
                    "mean_waveform_correlation": np.nan,
                    "welch_support_ratio": np.nan,
                    "converged": False,
                    "iterations": 0,
                    "valid": False,
                    "error_message": str(error),
                }

            trial_records.append(record)

    trials = pd.DataFrame(trial_records)

    successful_mask = trials["valid"] & trials["converged"]

    if not successful_mask.any():
        successful_mask = trials["valid"]

    successful_trials = trials.loc[successful_mask].copy()

    if successful_trials.empty:
        error_messages = trials["error_message"].dropna().tolist()
        raise RuntimeError(
            "All AVMD parameter candidates failed. "
            f"Errors: {error_messages[:3]}"
        )

    minimum_k = min(candidate_mode_counts)
    maximum_k = max(candidate_mode_counts)

    successful_trials["normalized_reconstruction_error"] = (
        _minmax_normalize(
            successful_trials["relative_reconstruction_error"]
        )
    )

    successful_trials["normalized_spectral_overlap"] = (
        _minmax_normalize(
            successful_trials["mean_spectral_overlap"]
        )
    )

    successful_trials["normalized_waveform_correlation"] = (
        _minmax_normalize(
            successful_trials["mean_waveform_correlation"]
        )
    )

    successful_trials["unsupported_welch_ratio"] = (
        1.0 - successful_trials["welch_support_ratio"]
    )

    if maximum_k == minimum_k:
        successful_trials["normalized_complexity"] = 0.0
    else:
        successful_trials["normalized_complexity"] = (
            successful_trials["mode_count"] - minimum_k
        ) / (maximum_k - minimum_k)

    successful_trials["selection_score"] = (
        0.35
        * successful_trials["normalized_reconstruction_error"]
        + 0.25
        * successful_trials["normalized_spectral_overlap"]
        + 0.15
        * successful_trials["normalized_waveform_correlation"]
        + 0.15
        * successful_trials["unsupported_welch_ratio"]
        + 0.10
        * successful_trials["normalized_complexity"]
    )

    best_index = successful_trials["selection_score"].idxmin()
    best_record = successful_trials.loc[best_index]

    trials["selection_score"] = np.nan
    trials.loc[
        successful_trials.index,
        "selection_score",
    ] = successful_trials["selection_score"]

    best_mode_count = int(best_record["mode_count"])
    best_alpha = float(best_record["alpha"])

    best_initial_centers = create_bandwise_initial_centers(
        signal=reference_signal,
        sampling_rate=sampling_rate,
        mode_count=best_mode_count,
    )

    best_vmd_result = variational_mode_decomposition(
        signal=reference_signal,
        sampling_rate=sampling_rate,
        initial_center_frequencies_hz=best_initial_centers,
        alpha=best_alpha,
        tau=tau,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )

    trials = trials.sort_values(
        by=["selection_score", "mode_count", "alpha"],
        na_position="last",
    ).reset_index(drop=True)

    return {
        "best_mode_count": best_mode_count,
        "best_alpha": best_alpha,
        "best_initial_center_frequencies_hz": (
            best_initial_centers
        ),
        "best_selection_score": float(
            best_record["selection_score"]
        ),
        "trials": trials,
        "reference_vmd_result": best_vmd_result,
    }