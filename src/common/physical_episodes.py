from __future__ import annotations

import numpy as np
import pandas as pd


def build_physical_episodes(
    clusters: pd.DataFrame,
    angle_timeline: pd.DataFrame,
    *,
    minimum_method_count: int = 2,
    maximum_cluster_gap_s: float = 1.0,
    angle_context_padding_s: float = 3.0,
    angle_range_threshold_deg: float = 0.5,
    peak_velocity_threshold_deg_s: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """合併同時發生的跨頻帶候選，並判斷是否碰到角度移動區段。

    角度關聯以 angle tracker 的 ``state_label == "moving"`` 為主。事件本身
    若與 moving 重疊，或事件的代表時間落在 moving 前後的指定緩衝範圍內，
    才標為 angle-associated。角度範圍與速度仍保留為診斷欄位，但不再因為
    前後繪圖脈絡中的小幅漂移而直接改變分類。
    """
    selected = clusters.loc[clusters["method_count"] >= minimum_method_count].copy()
    selected = selected.sort_values("cluster_start_time_s").reset_index(drop=True)
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    episode_numbers: list[int] = []
    episode_number = 1
    current_end = float(selected.loc[0, "cluster_end_time_s"])
    for row in selected.itertuples(index=False):
        if (
            episode_numbers
            and float(row.cluster_start_time_s) > current_end + maximum_cluster_gap_s
        ):
            episode_number += 1
            current_end = float(row.cluster_end_time_s)
        else:
            current_end = max(current_end, float(row.cluster_end_time_s))
        episode_numbers.append(episode_number)

    selected["physical_episode_id"] = [
        f"E{number:03d}" for number in episode_numbers
    ]
    time_s = angle_timeline["time_s"].to_numpy(dtype=float)
    angle_deg = angle_timeline["estimated_angle_deg"].to_numpy(dtype=float)
    velocity = angle_timeline["angle_velocity_deg_s"].to_numpy(dtype=float)
    state_labels = (
        angle_timeline["state_label"].astype(str).to_numpy()
        if "state_label" in angle_timeline.columns
        else None
    )
    rows: list[dict] = []

    for episode_id, group in selected.groupby("physical_episode_id", sort=False):
        start = float(group["cluster_start_time_s"].min())
        end = float(group["cluster_end_time_s"].max())
        context_start = max(float(time_s[0]), start - angle_context_padding_s)
        context_end = min(float(time_s[-1]), end + angle_context_padding_s)
        mask = (time_s >= context_start) & (time_s <= context_end)
        local_angle = angle_deg[mask]
        local_velocity = velocity[mask]
        angle_range = float(np.ptp(local_angle))
        peak_velocity = float(np.max(np.abs(local_velocity)))
        representative_time = float(group["representative_time_s"].median())

        episode_mask = (time_s >= start) & (time_s <= end)
        representative_context_mask = (
            (time_s >= representative_time - angle_context_padding_s)
            & (time_s <= representative_time + angle_context_padding_s)
        )
        if state_labels is not None:
            moving = np.char.startswith(state_labels.astype(str), "moving")
            episode_overlaps_moving = bool(np.any(moving & episode_mask))
            representative_near_moving = bool(
                np.any(moving & representative_context_mask)
            )
            angle_associated = (
                episode_overlaps_moving or representative_near_moving
            )
            classification_basis = (
                "episode_overlaps_moving_state"
                if episode_overlaps_moving
                else (
                    "representative_time_near_moving_state"
                    if representative_near_moving
                    else "representative_time_and_episode_are_stable"
                )
            )
        else:
            # 向下相容舊測試或沒有 state_label 的外部資料。
            episode_overlaps_moving = False
            representative_near_moving = False
            angle_associated = (
                angle_range >= angle_range_threshold_deg
                or peak_velocity >= peak_velocity_threshold_deg_s
            )
            classification_basis = "legacy_angle_range_or_velocity_threshold"
        frequencies = sorted(group["representative_frequency_hz"].round(2).unique())
        methods = sorted(
            {
                method
                for value in group["supporting_methods"]
                for method in str(value).split(",")
            }
        )
        rows.append(
            {
                "physical_episode_id": episode_id,
                "cluster_count": int(len(group)),
                "consensus_cluster_ids": ",".join(group["consensus_cluster_id"]),
                "episode_start_time_s": start,
                "episode_end_time_s": end,
                "episode_duration_s": end - start,
                "representative_time_s": representative_time,
                "angle_state_labels": ",".join(sorted(group["angle_state_label"].unique())),
                "response_frequency_bands_hz": ",".join(f"{value:.2f}" for value in frequencies),
                "supporting_methods": ",".join(methods),
                "maximum_method_count": int(group["method_count"].max()),
                "maximum_anomaly_score": float(group["maximum_anomaly_score"].max()),
                "angle_context_start_time_s": context_start,
                "angle_context_end_time_s": context_end,
                "angle_start_deg": float(local_angle[0]),
                "angle_end_deg": float(local_angle[-1]),
                "angle_minimum_deg": float(np.min(local_angle)),
                "angle_maximum_deg": float(np.max(local_angle)),
                "angle_range_deg": angle_range,
                "peak_absolute_angle_velocity_deg_s": peak_velocity,
                "episode_overlaps_moving_state": episode_overlaps_moving,
                "representative_time_near_moving_state": representative_near_moving,
                "classification_basis": classification_basis,
                "episode_classification": (
                    "angle_associated_dynamic_response"
                    if angle_associated
                    else "stable_state_anomaly_candidate"
                ),
                "interpretation": (
                    "modal change overlaps or is adjacent to tracked angle movement; not an independent fault"
                    if angle_associated
                    else "modal change occurs within a tracked stable-angle interval; requires independent validation"
                ),
            }
        )

    episodes = pd.DataFrame(rows)
    mapping = selected[
        [
            "physical_episode_id",
            "consensus_cluster_id",
            "representative_time_s",
            "representative_frequency_hz",
            "method_count",
            "supporting_methods",
            "consensus_level",
        ]
    ].copy()
    return episodes, mapping
