from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_FEATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "window_analysis"
    / "all_method_mode_window_features.csv"
)
SEGMENT_PATH = (
    PROJECT_ROOT
    / "results"
    / "stable_cycle_comparison"
    / "stable_segment_boundaries.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "mode_behavior_atlas"

MINIMUM_DYNAMIC_FREQUENCY_HZ = 0.5
FREQUENCY_CLUSTER_TOLERANCE_HZ = 2.0
RATIO_THRESHOLD = 1.25
METHOD_COUNT = 5


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + np.finfo(float).eps))


def _ratio_flag(ratio: float) -> bool:
    return bool(
        np.isfinite(ratio)
        and (ratio >= RATIO_THRESHOLD or ratio <= 1.0 / RATIO_THRESHOLD)
    )


def _attach_stable_segment(
    features: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    result = features.copy()
    result["stable_segment_id"] = pd.NA
    for segment in segments.itertuples(index=False):
        mask = (
            result["center_time_s"].between(
                float(segment.analysis_start_time_s),
                float(segment.analysis_end_time_s),
            )
            & (result["moving_fraction"] == 0.0)
        )
        result.loc[mask, "stable_segment_id"] = segment.segment_id
    return result


def _build_mode_atlas(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method", "mode", "global_peak_frequency_hz"]
    for (method, mode, frequency_hz), group in features.groupby(keys, sort=True):
        stable_10 = group.loc[
            group["angle_state_label"] == "stable_10deg", "rms_nm"
        ]
        stable_43 = group.loc[
            group["angle_state_label"] == "stable_43deg", "rms_nm"
        ]
        stable_all = group.loc[
            group["angle_state_label"].isin(["stable_10deg", "stable_43deg"]),
            "rms_nm",
        ]
        moving = group.loc[group["moving_fraction"] > 0.0, "rms_nm"]

        cycle_medians = (
            group.dropna(subset=["stable_segment_id"])
            .groupby("stable_segment_id")["rms_nm"]
            .median()
        )
        previous_10 = cycle_medians.reindex(["10deg_1", "10deg_2"]).dropna()
        late_10 = cycle_medians.get("10deg_3", np.nan)

        rms_10 = float(stable_10.median()) if not stable_10.empty else np.nan
        rms_43 = float(stable_43.median()) if not stable_43.empty else np.nan
        rms_stable = (
            float(stable_all.median()) if not stable_all.empty else np.nan
        )
        rms_moving = float(moving.median()) if not moving.empty else np.nan
        previous_10_median = (
            float(previous_10.median()) if not previous_10.empty else np.nan
        )
        late_10_median = float(late_10) if np.isfinite(late_10) else np.nan

        state_ratio = _safe_ratio(rms_43, rms_10)
        movement_ratio = _safe_ratio(rms_moving, rms_stable)
        late_cycle_ratio = _safe_ratio(late_10_median, previous_10_median)

        angle_state_sensitive = _ratio_flag(state_ratio)
        movement_sensitive = _ratio_flag(movement_ratio)
        late_cycle_shift = _ratio_flag(late_cycle_ratio)

        behaviors = []
        if movement_sensitive:
            behaviors.append("angle_transition_sensitive")
        if angle_state_sensitive:
            behaviors.append("angle_state_sensitive")
        if late_cycle_shift:
            behaviors.append("late_cycle_shift")
        if not behaviors:
            behaviors.append("relatively_stable_mode")

        rows.append(
            {
                "method": method,
                "mode": mode,
                "global_peak_frequency_hz": float(frequency_hz),
                "stable_10deg_median_rms_nm": rms_10,
                "stable_43deg_median_rms_nm": rms_43,
                "moving_median_rms_nm": rms_moving,
                "stable_43_to_10_rms_ratio": state_ratio,
                "moving_to_stable_rms_ratio": movement_ratio,
                "previous_10deg_median_rms_nm": previous_10_median,
                "late_10deg_median_rms_nm": late_10_median,
                "late_to_previous_10deg_rms_ratio": late_cycle_ratio,
                "angle_transition_sensitive": movement_sensitive,
                "angle_state_sensitive": angle_state_sensitive,
                "late_cycle_shift": late_cycle_shift,
                "behavior_candidates": ",".join(behaviors),
                "interpretation_limit": (
                    "behavioral association only; not a component or fault label"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["global_peak_frequency_hz", "method", "mode"]
    ).reset_index(drop=True)


def _cluster_frequencies(atlas: pd.DataFrame) -> pd.DataFrame:
    ordered = atlas.sort_values("global_peak_frequency_hz").reset_index(drop=True)
    clusters: list[list[int]] = []
    centers: list[float] = []
    for index, row in ordered.iterrows():
        frequency = float(row["global_peak_frequency_hz"])
        candidates = [
            cluster_index
            for cluster_index, center in enumerate(centers)
            if abs(frequency - center) <= FREQUENCY_CLUSTER_TOLERANCE_HZ
        ]
        if not candidates:
            clusters.append([index])
            centers.append(frequency)
            continue
        selected = min(candidates, key=lambda item: abs(frequency - centers[item]))
        clusters[selected].append(index)
        centers[selected] = float(
            ordered.loc[clusters[selected], "global_peak_frequency_hz"].median()
        )

    rows = []
    for number, members in enumerate(clusters, start=1):
        group = ordered.loc[members]
        methods = sorted(group["method"].unique())
        behavior_sets = {
            behavior
            for value in group["behavior_candidates"]
            for behavior in str(value).split(",")
        }
        rows.append(
            {
                "frequency_cluster_id": f"F{number:03d}",
                "representative_frequency_hz": float(
                    group["global_peak_frequency_hz"].median()
                ),
                "minimum_frequency_hz": float(
                    group["global_peak_frequency_hz"].min()
                ),
                "maximum_frequency_hz": float(
                    group["global_peak_frequency_hz"].max()
                ),
                "supporting_method_count": len(methods),
                "supporting_methods": ",".join(methods),
                "member_modes": ";".join(
                    f"{row.method}:{row.mode}({row.global_peak_frequency_hz:.2f}Hz)"
                    for row in group.itertuples(index=False)
                ),
                "angle_transition_method_count": int(
                    group.loc[group["angle_transition_sensitive"], "method"].nunique()
                ),
                "angle_state_method_count": int(
                    group.loc[group["angle_state_sensitive"], "method"].nunique()
                ),
                "late_cycle_shift_method_count": int(
                    group.loc[group["late_cycle_shift"], "method"].nunique()
                ),
                "behavior_candidates": ",".join(sorted(behavior_sets)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["supporting_method_count", "representative_frequency_hz"],
        ascending=[False, True],
    ).reset_index(drop=True)


def main() -> None:
    for path in (WINDOW_FEATURE_PATH, SEGMENT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"缺少必要檔案：{path}")

    features = pd.read_csv(WINDOW_FEATURE_PATH, encoding="utf-8-sig")
    segments = pd.read_csv(SEGMENT_PATH, encoding="utf-8-sig")
    dynamic = features.loc[
        features["global_peak_frequency_hz"] >= MINIMUM_DYNAMIC_FREQUENCY_HZ
    ].copy()
    dynamic = _attach_stable_segment(dynamic, segments)

    atlas = _build_mode_atlas(dynamic)
    frequency_groups = _cluster_frequencies(atlas)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(
        OUTPUT_ROOT / "all_method_mode_behavior_atlas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    frequency_groups.to_csv(
        OUTPUT_ROOT / "cross_method_frequency_behavior_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 70)
    print(f"{METHOD_COUNT}種VMD模態行為表建立完成")
    print(f"動態模態數：{len(atlas)}")
    print(f"跨方法相近頻率群數：{len(frequency_groups)}")
    print(f"{METHOD_COUNT}種方法共同涵蓋的頻率群：")
    common = frequency_groups.loc[
        frequency_groups["supporting_method_count"] == METHOD_COUNT,
        [
            "frequency_cluster_id",
            "representative_frequency_hz",
            "supporting_methods",
            "angle_transition_method_count",
            "angle_state_method_count",
            "late_cycle_shift_method_count",
        ],
    ]
    if common.empty:
        print("無")
    else:
        print(common.to_string(index=False))
    print(f"結果位置：{OUTPUT_ROOT}")
    print("行為分類只表示與角度或循環的關聯，不代表零件名稱或故障類型。")


if __name__ == "__main__":
    main()
