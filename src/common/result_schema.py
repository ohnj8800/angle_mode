from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "method",
    "dataset",
    "sampling_rate_hz",
    "analysis_low_hz",
    "analysis_high_hz",
    "n_samples",
    "duration_s",
    "mean_signal",
    "std_signal",
    "mode_count",
    "total_mode_energy_nm2",
    "mode_energy_entropy",
    "dominant_mode",
    "dominant_center_frequency_hz",
    "dominant_peak_frequency_hz",
    "dominant_energy_ratio",
    "nearest_welch_peak_hz",
    "welch_frequency_difference_hz",
    "welch_supported",
    "reconstruction_correlation",
    "relative_reconstruction_error",
    "reconstruction_rmse_nm",
    "angle_component_cutoff_hz",
    "detected_transition_count",
    "motion_sensitive_mode_count",
]


WINDOW_COLUMNS = [
    "method",
    "dataset",
    "window_index",
    "start_time_s",
    "end_time_s",
    "center_time_s",
    "angle_state_label",
    "estimated_angle_deg",
    "angle_velocity_deg_s",
    "angle_component_mean_nm",
    "angle_component_std_nm",
    "dynamic_rms_nm",
    "dominant_mode",
    "dominant_mode_energy_ratio",
    "energy_weighted_frequency_hz",
    "mode_energy_entropy",
    "residual_rms_nm",
    "angle_movement_flag",
]


def _rows_to_dataframe(
    rows: Iterable[Mapping[str, Any]],
    columns: list[str],
) -> pd.DataFrame:

    dataframe = pd.DataFrame(list(rows))

    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

    return dataframe.loc[:, columns]


def save_method_summary(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> pd.DataFrame:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = _rows_to_dataframe(rows, SUMMARY_COLUMNS)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    return dataframe


def save_window_features(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> pd.DataFrame:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = _rows_to_dataframe(rows, WINDOW_COLUMNS)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    return dataframe
