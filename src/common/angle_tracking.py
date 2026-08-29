from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.preprocessing import zero_phase_filter


def _validate_series(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size < 20:
        raise ValueError(f"{name}至少需要20個樣本。")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}包含NaN或無限值。")

    return values


def extract_angle_component(
    raw_nm: np.ndarray,
    sampling_rate: float = 200.0,
    cutoff_hz: float = 0.1,
    filter_order: int = 4,
) -> np.ndarray:
    """保留FBG訊號中的DC與低頻角度成分。"""
    raw_nm = _validate_series(raw_nm, "raw_nm")

    return zero_phase_filter(
        signal=raw_nm,
        sampling_rate=sampling_rate,
        cutoff_hz=cutoff_hz,
        filter_type="lowpass",
        order=filter_order,
    )


def _two_cluster_centers(values: np.ndarray) -> tuple[float, float]:
    """以一維兩群迭代估計兩個穩定波長平台。"""
    values = _validate_series(values, "stable_values")
    centers = np.quantile(values, [0.2, 0.8]).astype(float)

    for _ in range(100):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()

        for cluster_index in range(2):
            cluster_values = values[labels == cluster_index]

            if cluster_values.size:
                updated[cluster_index] = np.median(cluster_values)

        updated = np.sort(updated)

        if np.allclose(updated, centers, rtol=0.0, atol=1e-10):
            centers = updated
            break

        centers = updated

    if np.isclose(centers[0], centers[1]):
        raise ValueError("無法從低頻訊號辨識兩個不同的角度平台。")

    return float(centers[0]), float(centers[1])


def estimate_angle_from_component(
    angle_component_nm: np.ndarray,
    sampling_rate: float = 200.0,
    lower_angle_deg: float = 10.0,
    upper_angle_deg: float = 43.0,
    stable_derivative_quantile: float = 0.65,
    edge_guard_s: float = 3.0,
    inverse_wavelength_response: bool = True,
) -> dict:
    """
    由兩個低頻波長平台建立10度與43度的資料內部校正。

    此校正只能驗證本資料中的兩個已知角度，不代表已完成任意
    角度的外部校正。
    """
    angle_component_nm = _validate_series(
        angle_component_nm,
        "angle_component_nm",
    )

    if sampling_rate <= 0:
        raise ValueError("sampling_rate必須大於0。")

    if not 0.0 < stable_derivative_quantile < 1.0:
        raise ValueError("stable_derivative_quantile必須介於0與1之間。")

    if upper_angle_deg <= lower_angle_deg:
        raise ValueError("upper_angle_deg必須大於lower_angle_deg。")

    derivative_nm_s = np.gradient(
        angle_component_nm,
        1.0 / sampling_rate,
    )
    absolute_derivative = np.abs(derivative_nm_s)
    derivative_limit = float(
        np.quantile(absolute_derivative, stable_derivative_quantile)
    )
    stable_mask = absolute_derivative <= derivative_limit

    edge_samples = int(round(edge_guard_s * sampling_rate))

    if edge_samples > 0 and 2 * edge_samples < stable_mask.size:
        stable_mask[:edge_samples] = False
        stable_mask[-edge_samples:] = False

    if int(stable_mask.sum()) < 20:
        raise ValueError("穩定樣本不足，無法估計角度平台。")

    lower_wavelength_nm, upper_wavelength_nm = _two_cluster_centers(
        angle_component_nm[stable_mask]
    )

    if inverse_wavelength_response:
        wavelength_at_lower_angle_nm = upper_wavelength_nm
        wavelength_at_upper_angle_nm = lower_wavelength_nm
    else:
        wavelength_at_lower_angle_nm = lower_wavelength_nm
        wavelength_at_upper_angle_nm = upper_wavelength_nm

    slope_deg_per_nm = (
        (upper_angle_deg - lower_angle_deg)
        / (
            wavelength_at_upper_angle_nm
            - wavelength_at_lower_angle_nm
        )
    )
    estimated_angle_deg = (
        lower_angle_deg
        + (
            angle_component_nm
            - wavelength_at_lower_angle_nm
        )
        * slope_deg_per_nm
    )
    estimated_angle_deg = np.clip(
        estimated_angle_deg,
        lower_angle_deg,
        upper_angle_deg,
    )
    angle_velocity_deg_s = np.gradient(
        estimated_angle_deg,
        1.0 / sampling_rate,
    )

    centered_component = angle_component_nm - np.mean(angle_component_nm)
    window = np.hanning(centered_component.size)
    frequency_hz = np.fft.rfftfreq(
        centered_component.size,
        d=1.0 / sampling_rate,
    )
    amplitude = np.abs(np.fft.rfft(centered_component * window))
    minimum_frequency_hz = sampling_rate / centered_component.size
    frequency_mask = (
        (frequency_hz >= minimum_frequency_hz)
        & (frequency_hz <= 0.5)
    )

    if frequency_mask.any():
        recurrence_frequency_hz = float(
            frequency_hz[frequency_mask][
                np.argmax(amplitude[frequency_mask])
            ]
        )
    else:
        recurrence_frequency_hz = float("nan")

    return {
        "estimated_angle_deg": estimated_angle_deg,
        "angle_velocity_deg_s": angle_velocity_deg_s,
        "wavelength_at_lower_angle_nm": float(
            wavelength_at_lower_angle_nm
        ),
        "wavelength_at_upper_angle_nm": float(
            wavelength_at_upper_angle_nm
        ),
        "slope_deg_per_nm": float(slope_deg_per_nm),
        "stable_derivative_limit_nm_s": derivative_limit,
        "stable_sample_count": int(stable_mask.sum()),
        "low_frequency_recurrence_candidate_hz": recurrence_frequency_hz,
        "calibration_scope": "two known angles in this recording only",
    }


def detect_angle_transitions(
    time_s: np.ndarray,
    estimated_angle_deg: np.ndarray,
    lower_angle_deg: float = 10.0,
    upper_angle_deg: float = 43.0,
    stable_margin_fraction: float = 0.10,
    minimum_duration_s: float = 1.0,
) -> pd.DataFrame:
    """以遲滯狀態機偵測角度離開平台及進入下一平台的時間。"""
    time_s = _validate_series(time_s, "time_s")
    estimated_angle_deg = _validate_series(
        estimated_angle_deg,
        "estimated_angle_deg",
    )

    if time_s.shape != estimated_angle_deg.shape:
        raise ValueError("time_s與estimated_angle_deg長度不同。")

    if not 0.0 < stable_margin_fraction < 0.5:
        raise ValueError("stable_margin_fraction必須介於0與0.5之間。")

    angle_range = upper_angle_deg - lower_angle_deg
    lower_stable_limit = (
        lower_angle_deg + stable_margin_fraction * angle_range
    )
    upper_stable_limit = (
        upper_angle_deg - stable_margin_fraction * angle_range
    )
    initial_count = min(
        estimated_angle_deg.size,
        max(20, int(round(10.0 / np.median(np.diff(time_s))))),
    )
    initial_angle = float(np.median(estimated_angle_deg[:initial_count]))
    midpoint = (lower_angle_deg + upper_angle_deg) / 2.0
    state = "lower" if initial_angle <= midpoint else "upper"
    armed = False
    start_index: int | None = None
    rows: list[dict] = []

    for sample_index, angle_deg in enumerate(estimated_angle_deg):
        if state == "lower":
            if angle_deg <= lower_stable_limit:
                armed = True
            elif armed and start_index is None:
                start_index = sample_index

            if start_index is not None and angle_deg >= upper_stable_limit:
                direction = "10_to_43"
                next_state = "upper"
            else:
                continue
        else:
            if angle_deg >= upper_stable_limit:
                armed = True
            elif armed and start_index is None:
                start_index = sample_index

            if start_index is not None and angle_deg <= lower_stable_limit:
                direction = "43_to_10"
                next_state = "lower"
            else:
                continue

        end_index = sample_index
        duration_s = float(time_s[end_index] - time_s[start_index])

        if duration_s >= minimum_duration_s:
            rows.append(
                {
                    "event_id": len(rows) + 1,
                    "direction": direction,
                    "start_sample": int(start_index),
                    "end_sample": int(end_index),
                    "start_time_s": float(time_s[start_index]),
                    "end_time_s": float(time_s[end_index]),
                    "duration_s": duration_s,
                    "from_angle_deg": (
                        lower_angle_deg
                        if direction == "10_to_43"
                        else upper_angle_deg
                    ),
                    "to_angle_deg": (
                        upper_angle_deg
                        if direction == "10_to_43"
                        else lower_angle_deg
                    ),
                }
            )

        state = next_state
        armed = False
        start_index = None

    transitions = pd.DataFrame(rows)

    if transitions.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "direction",
                "start_sample",
                "end_sample",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "from_angle_deg",
                "to_angle_deg",
                "same_direction_interval_s",
                "recurrence_frequency_hz",
            ]
        )

    transitions["same_direction_interval_s"] = (
        transitions.groupby("direction")["start_time_s"].diff()
    )
    transitions["recurrence_frequency_hz"] = np.divide(
        1.0,
        transitions["same_direction_interval_s"],
        out=np.full(len(transitions), np.nan, dtype=float),
        where=(
            transitions["same_direction_interval_s"].to_numpy(dtype=float)
            > 0.0
        ),
    )

    return transitions


def assign_angle_state_labels(
    estimated_angle_deg: np.ndarray,
    transitions: pd.DataFrame,
    lower_angle_deg: float = 10.0,
    upper_angle_deg: float = 43.0,
) -> np.ndarray:
    """建立每個樣本的穩定角度或移動方向標籤。"""
    estimated_angle_deg = _validate_series(
        estimated_angle_deg,
        "estimated_angle_deg",
    )
    midpoint = (lower_angle_deg + upper_angle_deg) / 2.0
    labels = np.where(
        estimated_angle_deg <= midpoint,
        "stable_10deg",
        "stable_43deg",
    ).astype(object)

    for row in transitions.itertuples(index=False):
        labels[row.start_sample : row.end_sample + 1] = (
            f"moving_{row.direction}"
        )

    return labels.astype(str)


def summarize_angle_states(
    angle_component_nm: np.ndarray,
    estimated_angle_deg: np.ndarray,
    state_labels: np.ndarray,
) -> pd.DataFrame:
    """彙整穩定10度、穩定43度與移動區段的量測統計。"""
    dataframe = pd.DataFrame(
        {
            "angle_component_nm": angle_component_nm,
            "estimated_angle_deg": estimated_angle_deg,
            "state_label": state_labels,
        }
    )

    return (
        dataframe.groupby("state_label", sort=False)
        .agg(
            sample_count=("estimated_angle_deg", "size"),
            mean_wavelength_nm=("angle_component_nm", "mean"),
            std_wavelength_nm=("angle_component_nm", "std"),
            mean_estimated_angle_deg=("estimated_angle_deg", "mean"),
            std_estimated_angle_deg=("estimated_angle_deg", "std"),
        )
        .reset_index()
    )
