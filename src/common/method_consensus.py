from __future__ import annotations

import numpy as np
import pandas as pd


def _event_frequency(row: pd.Series) -> float:
    local = row.get("representative_local_frequency_hz", np.nan)
    if pd.notna(local):
        return float(local)
    return float(row["global_peak_frequency_hz"])


def _time_gap_seconds(left: pd.Series, right: pd.Series) -> float:
    return max(
        0.0,
        float(left["event_start_time_s"] - right["event_end_time_s"]),
        float(right["event_start_time_s"] - left["event_end_time_s"]),
    )


def compare_method_candidates(
    events: pd.DataFrame,
    *,
    maximum_time_gap_s: float = 1.0,
    minimum_frequency_tolerance_hz: float = 1.0,
    relative_frequency_tolerance: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """依時間與頻率接近程度，建立跨VMD方法的候選事件群集。"""
    required = {
        "method",
        "mode",
        "angle_state_label",
        "event_start_time_s",
        "event_end_time_s",
        "peak_time_s",
        "peak_anomaly_score",
        "global_peak_frequency_hz",
        "representative_local_frequency_hz",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"異常候選事件缺少欄位：{missing}")
    if maximum_time_gap_s < 0:
        raise ValueError("maximum_time_gap_s不可小於0。")
    if minimum_frequency_tolerance_hz <= 0 or relative_frequency_tolerance < 0:
        raise ValueError("頻率容許值設定錯誤。")

    source = events.reset_index(drop=True).copy()
    source["source_event_index"] = np.arange(len(source))
    source["comparison_frequency_hz"] = source.apply(_event_frequency, axis=1)
    parent = list(range(len(source)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pair_rows: list[dict] = []
    for left_index in range(len(source)):
        left = source.iloc[left_index]
        for right_index in range(left_index + 1, len(source)):
            right = source.iloc[right_index]
            if left["method"] == right["method"]:
                continue
            if left["angle_state_label"] != right["angle_state_label"]:
                continue

            time_gap = _time_gap_seconds(left, right)
            frequency_gap = abs(
                float(left["comparison_frequency_hz"])
                - float(right["comparison_frequency_hz"])
            )
            mean_frequency = np.mean(
                [left["comparison_frequency_hz"], right["comparison_frequency_hz"]]
            )
            frequency_tolerance = max(
                minimum_frequency_tolerance_hz,
                relative_frequency_tolerance * float(mean_frequency),
            )
            if time_gap <= maximum_time_gap_s and frequency_gap <= frequency_tolerance:
                union(left_index, right_index)
                pair_rows.append(
                    {
                        "left_event_index": left_index,
                        "right_event_index": right_index,
                        "left_method": left["method"],
                        "right_method": right["method"],
                        "angle_state_label": left["angle_state_label"],
                        "time_gap_s": time_gap,
                        "frequency_gap_hz": frequency_gap,
                        "frequency_tolerance_hz": frequency_tolerance,
                    }
                )

    source["component"] = [find(index) for index in range(len(source))]
    component_order = (
        source.groupby("component")["event_start_time_s"].min().sort_values().index
    )
    cluster_ids = {
        component: f"C{position:03d}"
        for position, component in enumerate(component_order, start=1)
    }
    source["consensus_cluster_id"] = source["component"].map(cluster_ids)

    cluster_rows: list[dict] = []
    for cluster_id, group in source.groupby("consensus_cluster_id", sort=False):
        methods = sorted(group["method"].unique())
        method_count = len(methods)
        level = {
            5: "five_method_consensus",
            4: "four_method_consensus",
            3: "three_method_consensus",
            2: "two_method_consensus",
            1: "single_method_only",
        }.get(method_count, "multi_method_consensus")
        cluster_rows.append(
            {
                "consensus_cluster_id": cluster_id,
                "angle_state_label": group["angle_state_label"].iloc[0],
                "consensus_level": level,
                "method_count": method_count,
                "supporting_methods": ",".join(methods),
                "member_event_count": int(len(group)),
                "cluster_start_time_s": float(group["event_start_time_s"].min()),
                "cluster_end_time_s": float(group["event_end_time_s"].max()),
                "representative_time_s": float(group["peak_time_s"].median()),
                "representative_frequency_hz": float(
                    group["comparison_frequency_hz"].median()
                ),
                "frequency_low_hz": float(group["comparison_frequency_hz"].min()),
                "frequency_high_hz": float(group["comparison_frequency_hz"].max()),
                "maximum_anomaly_score": float(group["peak_anomaly_score"].max()),
                "interpretation": (
                    "cross_method_anomaly_candidate"
                    if method_count >= 2
                    else "single_method_candidate_requires_review"
                ),
            }
        )

    clusters = pd.DataFrame(cluster_rows).sort_values(
        ["cluster_start_time_s", "representative_frequency_hz"]
    ).reset_index(drop=True)
    membership = source.drop(columns="component").merge(
        clusters[
            ["consensus_cluster_id", "consensus_level", "method_count", "supporting_methods"]
        ],
        on="consensus_cluster_id",
        how="left",
    )
    pair_matches = pd.DataFrame(pair_rows)
    return clusters, membership, pair_matches
