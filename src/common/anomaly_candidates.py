from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_FEATURES = {
    "local_peak_frequency_hz": "frequency_deviation_score",
    "rms_nm": "rms_deviation_score",
    "envelope_mean_nm": "envelope_deviation_score",
}
STABLE_STATES = ("stable_10deg", "stable_43deg")


def _feature_floor(feature: str, median: pd.Series) -> np.ndarray:
    if feature == "local_peak_frequency_hz":
        return np.full(len(median), 0.25)
    return np.maximum(np.abs(median.to_numpy(dtype=float)) * 0.01, 1e-12)


def score_stable_mode_windows(
    window_features: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    candidate_threshold: float = 3.5,
) -> pd.DataFrame:
    """依相同方法、模態與角度狀態的正常基準計算穩健偏離分數。"""
    if candidate_threshold <= 0:
        raise ValueError("candidate_threshold必須大於0。")

    required_window_columns = {
        "method",
        "mode",
        "physical_role_candidate",
        "global_peak_frequency_hz",
        "angle_state_label",
        "dominant_state_label",
        "moving_fraction",
        *SCORE_FEATURES,
    }
    required_baseline_columns = {
        "method",
        "mode",
        "angle_state_label",
        "feature",
        "median",
        "robust_sigma",
        "standard_deviation",
    }
    missing_windows = sorted(required_window_columns - set(window_features.columns))
    missing_baseline = sorted(required_baseline_columns - set(baseline.columns))
    if missing_windows:
        raise ValueError(f"滑動視窗資料缺少欄位：{missing_windows}")
    if missing_baseline:
        raise ValueError(f"正常基準資料缺少欄位：{missing_baseline}")

    eligible = window_features.loc[
        window_features["angle_state_label"].isin(STABLE_STATES)
        & (
            window_features["angle_state_label"]
            == window_features["dominant_state_label"]
        )
        & np.isclose(window_features["moving_fraction"], 0.0)
        & (window_features["global_peak_frequency_hz"] >= 0.5)
        & (window_features["physical_role_candidate"] != "angle_position_mode")
    ].copy()

    keys = ["method", "mode", "angle_state_label"]
    score_columns: list[str] = []
    for feature, score_column in SCORE_FEATURES.items():
        reference = baseline.loc[
            baseline["feature"] == feature,
            keys + ["median", "robust_sigma", "standard_deviation"],
        ].rename(
            columns={
                "median": f"{feature}_baseline_median",
                "robust_sigma": f"{feature}_baseline_robust_sigma",
                "standard_deviation": f"{feature}_baseline_std",
            }
        )
        eligible = eligible.merge(reference, on=keys, how="left", validate="many_to_one")

        median_column = f"{feature}_baseline_median"
        robust_column = f"{feature}_baseline_robust_sigma"
        std_column = f"{feature}_baseline_std"
        floor = _feature_floor(feature, eligible[median_column])
        robust = eligible[robust_column].to_numpy(dtype=float)
        standard = eligible[std_column].to_numpy(dtype=float)
        scale = np.where(
            np.isfinite(robust) & (robust >= floor),
            robust,
            np.where(np.isfinite(standard) & (standard >= floor), standard, floor),
        )
        eligible[f"{feature}_score_scale"] = scale
        eligible[score_column] = (
            np.abs(eligible[feature] - eligible[median_column]) / scale
        )
        score_columns.append(score_column)

    eligible["anomaly_score"] = eligible[score_columns].max(axis=1, skipna=True)
    eligible["dominant_deviation_feature"] = (
        eligible[score_columns]
        .idxmax(axis=1)
        .map({value: key for key, value in SCORE_FEATURES.items()})
    )
    eligible["anomaly_candidate"] = eligible["anomaly_score"] >= candidate_threshold
    eligible["candidate_threshold"] = candidate_threshold
    return eligible.sort_values(["method", "mode", "center_time_s"]).reset_index(drop=True)


def merge_consecutive_candidates(
    scored_windows: pd.DataFrame,
    *,
    minimum_consecutive_windows: int = 2,
    maximum_center_gap_s: float | None = None,
) -> pd.DataFrame:
    """將同一模態連續超標的重疊視窗合併成候選事件。"""
    if minimum_consecutive_windows < 1:
        raise ValueError("minimum_consecutive_windows至少為1。")

    if maximum_center_gap_s is None:
        centers = np.sort(scored_windows["center_time_s"].dropna().unique())
        steps = np.diff(centers)
        positive_steps = steps[steps > 0]
        typical_step = float(np.median(positive_steps)) if len(positive_steps) else 0.5
        maximum_center_gap_s = typical_step * 1.5

    candidates = scored_windows.loc[scored_windows["anomaly_candidate"]].copy()
    rows: list[dict] = []
    group_columns = ["method", "mode", "angle_state_label"]

    for (method, mode, state), group in candidates.groupby(group_columns, sort=False):
        group = group.sort_values("center_time_s")
        event_id = group["center_time_s"].diff().gt(maximum_center_gap_s).cumsum()
        for _, event in group.groupby(event_id, sort=False):
            if len(event) < minimum_consecutive_windows:
                continue
            peak = event.loc[event["anomaly_score"].idxmax()]
            rows.append(
                {
                    "method": method,
                    "mode": mode,
                    "angle_state_label": state,
                    "physical_role_candidate": peak["physical_role_candidate"],
                    "event_start_time_s": float(event["start_time_s"].min()),
                    "event_end_time_s": float(event["end_time_s"].max()),
                    "event_duration_s": float(
                        event["end_time_s"].max() - event["start_time_s"].min()
                    ),
                    "peak_time_s": float(peak["center_time_s"]),
                    "candidate_window_count": int(len(event)),
                    "peak_anomaly_score": float(peak["anomaly_score"]),
                    "mean_anomaly_score": float(event["anomaly_score"].mean()),
                    "dominant_deviation_feature": peak["dominant_deviation_feature"],
                    "global_peak_frequency_hz": float(peak["global_peak_frequency_hz"]),
                    "representative_local_frequency_hz": float(
                        event["local_peak_frequency_hz"].median()
                    ),
                    "local_frequency_low_hz": float(event["local_peak_frequency_hz"].min()),
                    "local_frequency_high_hz": float(event["local_peak_frequency_hz"].max()),
                    "interpretation": "anomaly_candidate_not_confirmed_fault",
                }
            )

    columns = [
        "method", "mode", "angle_state_label", "physical_role_candidate",
        "event_start_time_s", "event_end_time_s", "event_duration_s", "peak_time_s",
        "candidate_window_count", "peak_anomaly_score", "mean_anomaly_score",
        "dominant_deviation_feature", "global_peak_frequency_hz",
        "representative_local_frequency_hz", "local_frequency_low_hz",
        "local_frequency_high_hz", "interpretation",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["event_start_time_s", "method", "global_peak_frequency_hz"]
    ).reset_index(drop=True)
