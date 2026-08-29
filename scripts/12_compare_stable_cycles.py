from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_fbg_csv
from src.common.preprocessing import preprocess_fbg_signal


SAMPLING_RATE = 200.0
TARGET_FREQUENCIES_HZ = (38.5, 60.0)
WINDOW_DURATION_S = 2.0
STEP_DURATION_S = 0.5
EDGE_GUARD_S = 5.0
BAND_HALF_WIDTH_HZ = 1.5
PEAK_SEARCH_HALF_WIDTH_HZ = 5.0
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "angle_10deg_43deg_repeat.csv"
ANGLE_PATH = PROJECT_ROOT / "results" / "angle_tracking" / "angle_timeline.csv"
VMD_WINDOW_PATH = (
    PROJECT_ROOT
    / "results"
    / "window_analysis"
    / "all_method_mode_window_features.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "stable_cycle_comparison"


def _find_stable_segments(timeline: pd.DataFrame) -> pd.DataFrame:
    labels = timeline["state_label"].astype(str)
    run_id = labels.ne(labels.shift()).cumsum()
    rows = []
    state_counts: dict[str, int] = {}
    for _, group in timeline.assign(run_id=run_id).groupby("run_id", sort=True):
        state = str(group["state_label"].iloc[0])
        if state not in {"stable_10deg", "stable_43deg"}:
            continue
        start = float(group["time_s"].iloc[0])
        end = float(group["time_s"].iloc[-1])
        duration = end - start
        if duration <= 2.0 * EDGE_GUARD_S + WINDOW_DURATION_S:
            continue
        state_counts[state] = state_counts.get(state, 0) + 1
        short_state = "10deg" if state == "stable_10deg" else "43deg"
        rows.append(
            {
                "segment_id": f"{short_state}_{state_counts[state]}",
                "angle_state_label": state,
                "cycle_index_within_state": state_counts[state],
                "segment_start_time_s": start,
                "segment_end_time_s": end,
                "segment_duration_s": duration,
                "analysis_start_time_s": start + EDGE_GUARD_S,
                "analysis_end_time_s": end - EDGE_GUARD_S,
            }
        )
    return pd.DataFrame(rows)


def _raw_window_features(
    time_s: np.ndarray,
    signal: np.ndarray,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    window_samples = int(round(WINDOW_DURATION_S * SAMPLING_RATE))
    step_samples = int(round(STEP_DURATION_S * SAMPLING_RATE))
    rows = []
    for segment in segments.itertuples(index=False):
        starts = np.flatnonzero(time_s >= segment.analysis_start_time_s)
        ends = np.flatnonzero(time_s <= segment.analysis_end_time_s)
        if starts.size == 0 or ends.size == 0:
            continue
        first = int(starts[0])
        last_exclusive = int(ends[-1]) + 1
        for start in range(first, last_exclusive - window_samples + 1, step_samples):
            end = start + window_samples
            values = signal[start:end]
            frequency_hz, psd = welch(
                values,
                fs=SAMPLING_RATE,
                window="hann",
                nperseg=min(256, len(values)),
                noverlap=min(128, len(values) // 2),
                detrend="constant",
                scaling="density",
            )
            for target in TARGET_FREQUENCIES_HZ:
                band = (
                    (frequency_hz >= target - BAND_HALF_WIDTH_HZ)
                    & (frequency_hz <= target + BAND_HALF_WIDTH_HZ)
                )
                search = (
                    (frequency_hz >= target - PEAK_SEARCH_HALF_WIDTH_HZ)
                    & (frequency_hz <= target + PEAK_SEARCH_HALF_WIDTH_HZ)
                )
                band_power = float(np.trapezoid(psd[band], frequency_hz[band]))
                search_frequency = frequency_hz[search]
                search_psd = psd[search]
                rows.append(
                    {
                        "segment_id": segment.segment_id,
                        "angle_state_label": segment.angle_state_label,
                        "cycle_index_within_state": segment.cycle_index_within_state,
                        "target_frequency_hz": target,
                        "window_start_time_s": float(time_s[start]),
                        "window_end_time_s": float(time_s[end - 1]),
                        "center_time_s": float((time_s[start] + time_s[end - 1]) / 2.0),
                        "band_power_nm2": band_power,
                        "band_rms_nm": float(np.sqrt(max(band_power, 0.0))),
                        "local_peak_frequency_hz": float(
                            search_frequency[np.argmax(search_psd)]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _summarize_raw(features: pd.DataFrame) -> pd.DataFrame:
    summary = (
        features.groupby(
            [
                "segment_id",
                "angle_state_label",
                "cycle_index_within_state",
                "target_frequency_hz",
            ],
            as_index=False,
        )
        .agg(
            window_count=("center_time_s", "size"),
            median_band_rms_nm=("band_rms_nm", "median"),
            q25_band_rms_nm=("band_rms_nm", lambda x: x.quantile(0.25)),
            q75_band_rms_nm=("band_rms_nm", lambda x: x.quantile(0.75)),
            median_local_peak_frequency_hz=("local_peak_frequency_hz", "median"),
            q25_local_peak_frequency_hz=(
                "local_peak_frequency_hz", lambda x: x.quantile(0.25)
            ),
            q75_local_peak_frequency_hz=(
                "local_peak_frequency_hz", lambda x: x.quantile(0.75)
            ),
        )
    )
    return summary.sort_values(
        ["angle_state_label", "cycle_index_within_state", "target_frequency_hz"]
    ).reset_index(drop=True)


def _build_late_cycle_comparison(raw_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stable_10 = raw_summary.loc[
        raw_summary["angle_state_label"] == "stable_10deg"
    ]
    for target, group in stable_10.groupby("target_frequency_hz"):
        group = group.sort_values("cycle_index_within_state")
        if len(group) < 2:
            continue
        late = group.iloc[-1]
        previous = group.iloc[:-1]
        previous_rms = float(previous["median_band_rms_nm"].median())
        previous_peak = float(previous["median_local_peak_frequency_hz"].median())
        rows.append(
            {
                "target_frequency_hz": float(target),
                "late_segment_id": late["segment_id"],
                "previous_segment_count": int(len(previous)),
                "previous_median_band_rms_nm": previous_rms,
                "late_median_band_rms_nm": float(late["median_band_rms_nm"]),
                "late_to_previous_rms_ratio": float(
                    late["median_band_rms_nm"]
                    / (previous_rms + np.finfo(float).eps)
                ),
                "previous_median_local_peak_frequency_hz": previous_peak,
                "late_median_local_peak_frequency_hz": float(
                    late["median_local_peak_frequency_hz"]
                ),
                "late_peak_frequency_shift_hz": float(
                    late["median_local_peak_frequency_hz"] - previous_peak
                ),
                "interpretation": "descriptive cycle difference; not a fault label",
            }
        )
    return pd.DataFrame(rows)


def _summarize_vmd_cycles(
    window_features: pd.DataFrame,
    segments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dynamic = window_features.loc[
        window_features["physical_role_candidate"] != "angle_position_mode"
    ].copy()
    mapping_rows = []
    summary_rows = []
    for method, method_data in dynamic.groupby("method"):
        modes = method_data[
            ["mode", "global_peak_frequency_hz"]
        ].drop_duplicates()
        for target in TARGET_FREQUENCIES_HZ:
            selected = modes.iloc[
                (modes["global_peak_frequency_hz"] - target).abs().argmin()
            ]
            mapping_rows.append(
                {
                    "method": method,
                    "target_frequency_hz": target,
                    "selected_mode": selected["mode"],
                    "selected_global_peak_frequency_hz": float(
                        selected["global_peak_frequency_hz"]
                    ),
                    "frequency_difference_hz": float(
                        abs(selected["global_peak_frequency_hz"] - target)
                    ),
                }
            )
            selected_windows = method_data.loc[
                method_data["mode"] == selected["mode"]
            ]
            for segment in segments.itertuples(index=False):
                local = selected_windows.loc[
                    selected_windows["center_time_s"].between(
                        segment.analysis_start_time_s,
                        segment.analysis_end_time_s,
                    )
                    & (selected_windows["moving_fraction"] == 0.0)
                ]
                if local.empty:
                    continue
                summary_rows.append(
                    {
                        "method": method,
                        "target_frequency_hz": target,
                        "selected_mode": selected["mode"],
                        "selected_global_peak_frequency_hz": float(
                            selected["global_peak_frequency_hz"]
                        ),
                        "segment_id": segment.segment_id,
                        "angle_state_label": segment.angle_state_label,
                        "cycle_index_within_state": segment.cycle_index_within_state,
                        "window_count": int(len(local)),
                        "median_mode_rms_nm": float(local["rms_nm"].median()),
                        "median_mode_envelope_nm": float(
                            local["envelope_mean_nm"].median()
                        ),
                        "median_local_peak_frequency_hz": float(
                            local["local_peak_frequency_hz"].median()
                        ),
                    }
                )
    return pd.DataFrame(mapping_rows), pd.DataFrame(summary_rows)


def _plot_raw_cycles(summary: pd.DataFrame, output_path: Path) -> None:
    segment_order = summary[
        ["segment_id", "angle_state_label", "cycle_index_within_state"]
    ].drop_duplicates()
    segment_order["state_order"] = segment_order["angle_state_label"].map(
        {"stable_10deg": 0, "stable_43deg": 1}
    )
    segment_order = segment_order.sort_values(
        ["state_order", "cycle_index_within_state"]
    )
    labels = segment_order["segment_id"].tolist()
    x = np.arange(len(labels))
    figure, axes = plt.subplots(
        len(TARGET_FREQUENCIES_HZ),
        1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, target in zip(axes, TARGET_FREQUENCIES_HZ):
        subset = summary.loc[summary["target_frequency_hz"] == target].set_index(
            "segment_id"
        ).reindex(labels)
        y = subset["median_band_rms_nm"].to_numpy(dtype=float)
        lower = y - subset["q25_band_rms_nm"].to_numpy(dtype=float)
        upper = subset["q75_band_rms_nm"].to_numpy(dtype=float) - y
        axis.errorbar(
            x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=4,
            color="tab:blue", linewidth=1.2,
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel("Band RMS (nm)")
        axis.set_title(f"Raw sliding-Welch band RMS near {target:.1f} Hz")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Stable angle segment")
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_vmd_cycles(summary: pd.DataFrame, output_path: Path) -> None:
    methods = sorted(summary["method"].unique())
    figure, axes = plt.subplots(
        len(TARGET_FREQUENCIES_HZ),
        1,
        figsize=(13, 9),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, target in zip(axes, TARGET_FREQUENCIES_HZ):
        subset = summary.loc[
            (summary["target_frequency_hz"] == target)
            & (summary["angle_state_label"] == "stable_10deg")
        ]
        segment_order = (
            subset[["segment_id", "cycle_index_within_state"]]
            .drop_duplicates()
            .sort_values("cycle_index_within_state")
        )
        labels = segment_order["segment_id"].tolist()
        x = np.arange(len(labels))
        for method in methods:
            values = subset.loc[subset["method"] == method].set_index(
                "segment_id"
            ).reindex(labels)
            axis.plot(
                x,
                values["median_mode_rms_nm"],
                marker="o",
                linewidth=1.1,
                label=method,
            )
        axis.set_xticks(x, labels)
        axis.set_ylabel("Median modal RMS (nm)")
        axis.set_title(f"Closest VMD mode to {target:.1f} Hz across stable 10-degree cycles")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Stable 10-degree segment")
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for path in (RAW_FILE, ANGLE_PATH, VMD_WINDOW_PATH):
        if not path.exists():
            raise FileNotFoundError(f"缺少必要檔案：{path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timeline = pd.read_csv(ANGLE_PATH, encoding="utf-8-sig")
    segments = _find_stable_segments(timeline)
    if segments.empty:
        raise ValueError("未找到足夠長的穩定角度區段。")
    data, _ = load_fbg_csv(RAW_FILE, sampling_rate=SAMPLING_RATE)
    length = min(len(data), len(timeline))
    time_s = data["time_s"].to_numpy(dtype=float)[:length]
    raw_nm = data["ch1_nm"].to_numpy(dtype=float)[:length]
    dynamic_nm = preprocess_fbg_signal(raw_nm, SAMPLING_RATE)["dynamic_nm"]
    raw_features = _raw_window_features(time_s, dynamic_nm, segments)
    raw_summary = _summarize_raw(raw_features)
    late_comparison = _build_late_cycle_comparison(raw_summary)
    vmd_windows = pd.read_csv(VMD_WINDOW_PATH, encoding="utf-8-sig")
    mode_mapping, vmd_summary = _summarize_vmd_cycles(vmd_windows, segments)

    segments.to_csv(
        OUTPUT_ROOT / "stable_segment_boundaries.csv", index=False, encoding="utf-8-sig"
    )
    raw_features.to_csv(
        OUTPUT_ROOT / "stable_cycle_raw_window_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_summary.to_csv(
        OUTPUT_ROOT / "stable_cycle_raw_summary.csv", index=False, encoding="utf-8-sig"
    )
    late_comparison.to_csv(
        OUTPUT_ROOT / "late_10deg_cycle_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mode_mapping.to_csv(
        OUTPUT_ROOT / "target_frequency_mode_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vmd_summary.to_csv(
        OUTPUT_ROOT / "stable_cycle_vmd_summary.csv", index=False, encoding="utf-8-sig"
    )
    _plot_raw_cycles(raw_summary, OUTPUT_ROOT / "01_raw_stable_cycle_comparison.png")
    _plot_vmd_cycles(vmd_summary, OUTPUT_ROOT / "02_vmd_stable_cycle_comparison.png")

    print("=" * 70)
    print("穩定角度循環分段比較完成")
    print(f"穩定區段數：{len(segments)}")
    print(segments[[
        "segment_id", "angle_state_label", "segment_start_time_s",
        "segment_end_time_s", "segment_duration_s",
    ]].to_string(index=False))
    print("-" * 70)
    print("最後一次10度相對先前10度區段：")
    print(late_comparison[[
        "target_frequency_hz", "late_segment_id",
        "late_to_previous_rms_ratio", "late_peak_frequency_shift_hz",
    ]].to_string(index=False))
    print(f"結果位置：{OUTPUT_ROOT}")
    print("本步驟只描述循環差異，不將差異直接判定為故障。")


if __name__ == "__main__":
    main()
