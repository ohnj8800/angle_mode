from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import hilbert


def _validate_inputs(
    time_s: np.ndarray,
    modes: np.ndarray,
    state_labels: np.ndarray,
    estimated_angle_deg: np.ndarray,
    angle_velocity_deg_s: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    modes = np.asarray(modes, dtype=float)
    state_labels = np.asarray(state_labels, dtype=str).reshape(-1)
    estimated_angle_deg = np.asarray(estimated_angle_deg, dtype=float).reshape(-1)
    angle_velocity_deg_s = np.asarray(angle_velocity_deg_s, dtype=float).reshape(-1)

    if modes.ndim != 2:
        raise ValueError("modes必須為(mode_count, sample_count)。")

    sample_count = modes.shape[1]
    lengths = {
        sample_count,
        len(time_s),
        len(state_labels),
        len(estimated_angle_deg),
        len(angle_velocity_deg_s),
    }

    if len(lengths) != 1:
        raise ValueError("時間、模態與角度時間軸的樣本數必須相同。")

    if sample_count < 20:
        raise ValueError("訊號太短，無法計算滑動視窗特徵。")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate必須大於0。")

    if not np.all(np.isfinite(modes)):
        raise ValueError("modes包含NaN或無限值。")

    return (
        time_s,
        modes,
        state_labels,
        estimated_angle_deg,
        angle_velocity_deg_s,
    )


def _dominant_label(labels: np.ndarray) -> str:
    counts = Counter(labels.tolist())
    return str(counts.most_common(1)[0][0])


def _local_frequency_features(
    values: np.ndarray,
    sampling_rate: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> tuple[float, float]:
    centered = values - np.mean(values)
    window = np.hanning(len(centered))
    frequency_hz = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate)
    power = np.abs(np.fft.rfft(centered * window)) ** 2
    mask = (
        (frequency_hz >= minimum_frequency_hz)
        & (frequency_hz <= maximum_frequency_hz)
    )

    if not np.any(mask):
        return float("nan"), float("nan")

    selected_frequency = frequency_hz[mask]
    selected_power = power[mask]
    power_sum = float(np.sum(selected_power))

    if power_sum <= np.finfo(float).eps:
        return float("nan"), float("nan")

    peak_frequency_hz = float(
        selected_frequency[int(np.argmax(selected_power))]
    )
    spectral_centroid_hz = float(
        np.sum(selected_frequency * selected_power) / power_sum
    )
    return peak_frequency_hz, spectral_centroid_hz


def extract_mode_window_features(
    *,
    method: str,
    time_s: np.ndarray,
    modes: np.ndarray,
    state_labels: np.ndarray,
    estimated_angle_deg: np.ndarray,
    angle_velocity_deg_s: np.ndarray,
    global_peak_frequencies_hz: np.ndarray,
    physical_roles: np.ndarray,
    sampling_rate: float = 200.0,
    window_duration_s: float = 2.0,
    step_duration_s: float = 0.5,
    minimum_dynamic_frequency_hz: float = 0.5,
    maximum_frequency_hz: float | None = None,
) -> pd.DataFrame:
    """逐一計算每個IMF在連續時間視窗中的頻率與能量特徵。"""
    (
        time_s,
        modes,
        state_labels,
        estimated_angle_deg,
        angle_velocity_deg_s,
    ) = _validate_inputs(
        time_s,
        modes,
        state_labels,
        estimated_angle_deg,
        angle_velocity_deg_s,
        sampling_rate,
    )
    global_peak_frequencies_hz = np.asarray(
        global_peak_frequencies_hz, dtype=float
    ).reshape(-1)
    physical_roles = np.asarray(physical_roles, dtype=str).reshape(-1)

    if len(global_peak_frequencies_hz) != len(modes):
        raise ValueError("全域峰值頻率數量必須與模態數相同。")

    if len(physical_roles) != len(modes):
        raise ValueError("模態角色數量必須與模態數相同。")

    if window_duration_s <= 0 or step_duration_s <= 0:
        raise ValueError("視窗長度與移動間隔必須大於0。")

    maximum_frequency_hz = (
        sampling_rate / 2.0
        if maximum_frequency_hz is None
        else float(maximum_frequency_hz)
    )
    window_samples = int(round(window_duration_s * sampling_rate))
    step_samples = int(round(step_duration_s * sampling_rate))

    if window_samples > modes.shape[1]:
        raise ValueError("視窗長度大於輸入訊號長度。")

    if window_samples < 20 or step_samples < 1:
        raise ValueError("視窗或移動間隔換算後的樣本數不合理。")

    envelopes = np.abs(hilbert(modes, axis=1))
    rows: list[dict] = []
    window_index = 0

    for start_sample in range(
        0,
        modes.shape[1] - window_samples + 1,
        step_samples,
    ):
        end_sample = start_sample + window_samples
        center_sample = start_sample + window_samples // 2
        window_labels = state_labels[start_sample:end_sample]
        moving_fraction = float(
            np.mean(np.char.startswith(window_labels, "moving_"))
        )
        dominant_state = _dominant_label(window_labels)

        for mode_index, mode_values in enumerate(modes):
            segment = mode_values[start_sample:end_sample]
            envelope_segment = envelopes[mode_index, start_sample:end_sample]
            global_peak_hz = float(global_peak_frequencies_hz[mode_index])

            if global_peak_hz >= minimum_dynamic_frequency_hz:
                local_peak_hz, local_centroid_hz = _local_frequency_features(
                    segment,
                    sampling_rate,
                    minimum_dynamic_frequency_hz,
                    maximum_frequency_hz,
                )
            else:
                local_peak_hz = float("nan")
                local_centroid_hz = float("nan")

            rows.append(
                {
                    "method": method,
                    "mode": f"IMF{mode_index + 1}",
                    "physical_role_candidate": physical_roles[mode_index],
                    "global_peak_frequency_hz": global_peak_hz,
                    "window_index": window_index,
                    "start_sample": start_sample,
                    "end_sample": end_sample - 1,
                    "start_time_s": float(time_s[start_sample]),
                    "end_time_s": float(time_s[end_sample - 1]),
                    "center_time_s": float(time_s[center_sample]),
                    "angle_state_label": str(state_labels[center_sample]),
                    "dominant_state_label": dominant_state,
                    "moving_fraction": moving_fraction,
                    "estimated_angle_mean_deg": float(
                        np.mean(estimated_angle_deg[start_sample:end_sample])
                    ),
                    "angle_velocity_rms_deg_s": float(
                        np.sqrt(
                            np.mean(
                                angle_velocity_deg_s[start_sample:end_sample]
                                ** 2
                            )
                        )
                    ),
                    "local_peak_frequency_hz": local_peak_hz,
                    "local_spectral_centroid_hz": local_centroid_hz,
                    "rms_nm": float(np.sqrt(np.mean(segment**2))),
                    "energy_nm2": float(np.sum(segment**2)),
                    "mean_square_nm2": float(np.mean(segment**2)),
                    "envelope_mean_nm": float(np.mean(envelope_segment)),
                    "envelope_std_nm": float(np.std(envelope_segment)),
                }
            )

        window_index += 1

    return pd.DataFrame(rows)
