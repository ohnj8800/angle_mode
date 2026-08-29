from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.metrics import calculate_mode_metrics


def calculate_transition_mode_response(
    modes: np.ndarray,
    sampling_rate: float,
    state_labels: np.ndarray,
    transitions: pd.DataFrame,
    algorithm_center_frequencies_hz: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比較每個IMF在穩定角度與角度移動期間的RMS響應。"""
    modes = np.asarray(modes, dtype=float)
    state_labels = np.asarray(state_labels, dtype=str)

    if modes.ndim != 2:
        raise ValueError("modes必須為(mode_count, sample_count)。")

    if modes.shape[1] != state_labels.size:
        raise ValueError("modes與state_labels樣本數不同。")

    metrics = calculate_mode_metrics(
        modes=modes,
        sampling_rate=sampling_rate,
        algorithm_center_frequencies_hz=algorithm_center_frequencies_hz,
    )
    stable_mask = np.char.startswith(state_labels, "stable_")

    if not stable_mask.any():
        raise ValueError("沒有穩定角度樣本，無法建立模態參考值。")

    stable_10_mask = state_labels == "stable_10deg"
    stable_43_mask = state_labels == "stable_43deg"
    event_rows: list[dict] = []

    for mode_index, mode in enumerate(modes):
        for event in transitions.itertuples(index=False):
            event_slice = slice(event.start_sample, event.end_sample + 1)
            event_signal = mode[event_slice]
            event_rows.append(
                {
                    "mode": f"IMF{mode_index + 1}",
                    "event_id": int(event.event_id),
                    "direction": event.direction,
                    "start_time_s": float(event.start_time_s),
                    "end_time_s": float(event.end_time_s),
                    "duration_s": float(event.duration_s),
                    "event_rms_nm": float(
                        np.sqrt(np.mean(event_signal**2))
                    ),
                    "event_energy_nm2": float(np.sum(event_signal**2)),
                }
            )

    event_response = pd.DataFrame(event_rows)
    aggregate_rows: list[dict] = []

    for mode_index, mode in enumerate(modes):
        mode_name = f"IMF{mode_index + 1}"
        mode_events = event_response.loc[event_response["mode"] == mode_name]
        stable_rms = float(np.sqrt(np.mean(mode[stable_mask] ** 2)))
        stable_10_rms = (
            float(np.sqrt(np.mean(mode[stable_10_mask] ** 2)))
            if stable_10_mask.any()
            else float("nan")
        )
        stable_43_rms = (
            float(np.sqrt(np.mean(mode[stable_43_mask] ** 2)))
            if stable_43_mask.any()
            else float("nan")
        )
        mean_transition_rms = float(mode_events["event_rms_nm"].mean())
        transition_rms_std = float(mode_events["event_rms_nm"].std(ddof=1))
        response_ratio = mean_transition_rms / (
            stable_rms + np.finfo(float).eps
        )
        response_cv = transition_rms_std / (
            mean_transition_rms + np.finfo(float).eps
        )
        metric_row = metrics.iloc[mode_index]
        aggregate_rows.append(
            {
                **metric_row.to_dict(),
                "stable_rms_nm": stable_rms,
                "stable_10deg_rms_nm": stable_10_rms,
                "stable_43deg_rms_nm": stable_43_rms,
                "mean_transition_rms_nm": mean_transition_rms,
                "transition_rms_std_nm": transition_rms_std,
                "transition_to_stable_rms_ratio": float(response_ratio),
                "transition_response_cv": float(response_cv),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["motion_sensitivity_rank"] = (
        aggregate["transition_to_stable_rms_ratio"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    aggregate["motion_sensitive_candidate"] = (
        aggregate["transition_to_stable_rms_ratio"] >= 1.10
    )
    aggregate = aggregate.sort_values(
        ["motion_sensitivity_rank", "peak_frequency_hz"]
    ).reset_index(drop=True)

    return aggregate, event_response
