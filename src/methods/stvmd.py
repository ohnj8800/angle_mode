from __future__ import annotations

import numpy as np

from src.common.vmd import variational_mode_decomposition


def short_time_variational_mode_decomposition(
    signal: np.ndarray,
    sampling_rate: float,
    mode_count: int = 9,
    window_duration_s: float = 8.0,
    hop_duration_s: float = 2.0,
    minimum_frequency_hz: float = 0.5,
    maximum_frequency_hz: float | None = None,
    alpha: float = 2000.0,
    tolerance: float = 1e-6,
    maximum_iterations: int = 200,
) -> dict:
    """以重疊短時窗、中心頻率追蹤及overlap-add執行動態VMD。

    這是適合本專案單通道資料的可重現工程實作：每個短時窗執行VMD，
    下一窗沿用上一窗中心頻率，最後以Hann權重重建時間連續的模態。
    它採用STVMD的短時、時變中心頻率概念，但不是作者MATLAB程式的逐行移植。
    """
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < 40:
        raise ValueError("STVMD輸入訊號太短。")
    if not np.all(np.isfinite(values)):
        raise ValueError("STVMD輸入訊號包含NaN或無限值。")
    if sampling_rate <= 0 or mode_count < 2:
        raise ValueError("sampling_rate必須大於0，mode_count至少為2。")

    nyquist_hz = sampling_rate / 2.0
    upper_hz = (
        nyquist_hz * 0.999
        if maximum_frequency_hz is None
        else min(float(maximum_frequency_hz), nyquist_hz * 0.999)
    )
    if not 0.0 < minimum_frequency_hz < upper_hz:
        raise ValueError("STVMD分析頻率上下限設定錯誤。")

    window_samples = int(round(window_duration_s * sampling_rate))
    hop_samples = int(round(hop_duration_s * sampling_rate))
    if window_samples < 40 or hop_samples < 1 or hop_samples > window_samples:
        raise ValueError("STVMD時間窗或hop設定錯誤。")
    if window_samples % 2:
        window_samples += 1

    half_window = window_samples // 2
    padded = np.pad(values, (half_window, half_window), mode="reflect")
    last_start = padded.size - window_samples
    starts = list(range(0, last_start + 1, hop_samples))
    if starts[-1] != last_start:
        starts.append(last_start)

    synthesis_window = np.hanning(window_samples)
    synthesis_window = np.maximum(synthesis_window, np.finfo(float).eps)
    accumulated = np.zeros((mode_count, padded.size), dtype=float)
    accumulated_weight = np.zeros(padded.size, dtype=float)
    center_tracks = np.zeros((len(starts), mode_count), dtype=float)
    convergence = np.zeros(len(starts), dtype=bool)
    iterations = np.zeros(len(starts), dtype=int)
    centers = np.linspace(minimum_frequency_hz, upper_hz, mode_count)

    for frame_index, start in enumerate(starts):
        frame = padded[start : start + window_samples]
        result = variational_mode_decomposition(
            signal=frame,
            sampling_rate=sampling_rate,
            initial_center_frequencies_hz=centers,
            alpha=alpha,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        frame_centers = np.asarray(result["center_frequencies_hz"], dtype=float)
        order = np.argsort(frame_centers)
        frame_centers = frame_centers[order]
        frame_modes = np.asarray(result["modes"], dtype=float)[order]

        accumulated[:, start : start + window_samples] += (
            frame_modes * synthesis_window
        )
        accumulated_weight[start : start + window_samples] += synthesis_window
        center_tracks[frame_index] = frame_centers
        convergence[frame_index] = bool(result["converged"])
        iterations[frame_index] = int(result["iterations"])

        # 平滑中心頻率初始化，避免單一短窗突發值讓下一窗模態跳動。
        centers = np.clip(
            0.7 * frame_centers + 0.3 * centers,
            minimum_frequency_hz,
            upper_hz,
        )
        centers = np.maximum.accumulate(centers)

    accumulated /= accumulated_weight[np.newaxis, :]
    modes = accumulated[:, half_window : half_window + values.size]
    mode_energy = np.mean(modes**2, axis=1)
    frame_weights = np.mean(
        np.stack(
            [
                padded[start : start + window_samples] ** 2
                for start in starts
            ]
        ),
        axis=1,
    )
    frame_weights = frame_weights / (
        np.sum(frame_weights) + np.finfo(float).eps
    )
    representative_centers_hz = np.sum(
        center_tracks * frame_weights[:, np.newaxis], axis=0
    )

    return {
        "modes": modes,
        "center_frequencies_hz": representative_centers_hz,
        "center_frequency_tracks_hz": center_tracks,
        "frame_center_time_s": (
            np.asarray(starts, dtype=float) - half_window + window_samples / 2.0
        )
        / sampling_rate,
        "mode_energy": mode_energy,
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "frame_count": len(starts),
        "converged_frame_fraction": float(np.mean(convergence)),
        "median_iterations": float(np.median(iterations)),
    }
