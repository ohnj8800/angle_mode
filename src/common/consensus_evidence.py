from __future__ import annotations

import numpy as np
import pandas as pd


def select_consensus_evidence(
    clusters: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    minimum_method_count: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """選出跨方法候選群集及其組成事件，供逐事件檢查。"""
    selected_clusters = clusters.loc[
        clusters["method_count"] >= minimum_method_count
    ].copy()
    selected_membership = membership.loc[
        membership["consensus_cluster_id"].isin(
            selected_clusters["consensus_cluster_id"]
        )
    ].copy()
    return (
        selected_clusters.sort_values("cluster_start_time_s").reset_index(drop=True),
        selected_membership.sort_values(
            ["consensus_cluster_id", "method", "mode"]
        ).reset_index(drop=True),
    )


def time_window_mask(
    time_s: np.ndarray,
    event_start_time_s: float,
    event_end_time_s: float,
    *,
    padding_s: float = 3.0,
) -> np.ndarray:
    """建立事件前後留有背景時間的布林遮罩。"""
    if padding_s < 0:
        raise ValueError("padding_s不可小於0。")
    start = max(float(time_s[0]), event_start_time_s - padding_s)
    end = min(float(time_s[-1]), event_end_time_s + padding_s)
    return (time_s >= start) & (time_s <= end)
