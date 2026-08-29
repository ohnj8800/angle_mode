from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ANGLE_ROOT = PROJECT_ROOT / "results" / "angle_tracking"
MODE_ROOT = PROJECT_ROOT / "results" / "mode_analysis"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "event_analysis"
METHODS = ("VMD", "IOVMD", "AVMD", "SVMD", "STVMD")
BASELINE_WINDOW_S = 5.0
BASELINE_GUARD_S = 1.0
EVENT_ENHANCEMENT_THRESHOLD = 1.05
FREQUENCY_MATCH_TOLERANCE_HZ = 5.0


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return float("nan")

    return float(np.sqrt(np.mean(values**2)))


def calculate_corrected_event_response(
    method: str,
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_root = MODE_ROOT / method.lower()
    decomposition = np.load(method_root / "decomposition.npz")
    time_s = decomposition["time_s"]
    modes = decomposition["modes"]
    mode_summary = pd.read_csv(
        method_root / "mode_response_summary.csv",
        encoding="utf-8-sig",
    ).set_index("mode")
    rows = []

    for event in transitions.itertuples(index=False):
        event_mask = (
            (time_s >= event.start_time_s)
            & (time_s <= event.end_time_s)
        )
        pre_mask = (
            (time_s >= event.start_time_s - BASELINE_GUARD_S - BASELINE_WINDOW_S)
            & (time_s < event.start_time_s - BASELINE_GUARD_S)
        )
        post_mask = (
            (time_s > event.end_time_s + BASELINE_GUARD_S)
            & (time_s <= event.end_time_s + BASELINE_GUARD_S + BASELINE_WINDOW_S)
        )

        for mode_index, mode_values in enumerate(modes):
            mode_name = f"IMF{mode_index + 1}"
            pre_rms_nm = _rms(mode_values[pre_mask])
            post_rms_nm = _rms(mode_values[post_mask])
            event_rms_nm = _rms(mode_values[event_mask])
            expected_baseline_rms_nm = float(
                np.sqrt((pre_rms_nm**2 + post_rms_nm**2) / 2.0)
            )
            corrected_response_ratio = float(
                event_rms_nm / expected_baseline_rms_nm
            )
            rows.append(
                {
                    "method": method,
                    "mode": mode_name,
                    "event_id": int(event.event_id),
                    "direction": event.direction,
                    "start_time_s": float(event.start_time_s),
                    "end_time_s": float(event.end_time_s),
                    "duration_s": float(event.duration_s),
                    "peak_frequency_hz": float(
                        mode_summary.loc[mode_name, "peak_frequency_hz"]
                    ),
                    "pre_event_rms_nm": pre_rms_nm,
                    "post_event_rms_nm": post_rms_nm,
                    "expected_baseline_rms_nm": expected_baseline_rms_nm,
                    "event_rms_nm": event_rms_nm,
                    "corrected_response_ratio": corrected_response_ratio,
                    "excess_percent": 100.0 * (corrected_response_ratio - 1.0),
                }
            )

    event_table = pd.DataFrame(rows)
    aggregate_rows = []

    for mode_name, group in event_table.groupby("mode", sort=False):
        source = mode_summary.loc[mode_name]
        stable_10_rms = float(source["stable_10deg_rms_nm"])
        stable_43_rms = float(source["stable_43deg_rms_nm"])
        state_ratio = stable_43_rms / stable_10_rms
        corrected = group["corrected_response_ratio"].to_numpy(dtype=float)
        direction_means = group.groupby("direction")[
            "corrected_response_ratio"
        ].mean()
        aggregate_rows.append(
            {
                "method": method,
                "mode": mode_name,
                "peak_frequency_hz": float(source["peak_frequency_hz"]),
                "angle_component_correlation": float(
                    source["angle_component_correlation"]
                ),
                "stable_10deg_rms_nm": stable_10_rms,
                "stable_43deg_rms_nm": stable_43_rms,
                "state_ratio_43_to_10": state_ratio,
                "state_contrast_ratio": max(state_ratio, 1.0 / state_ratio),
                "mean_corrected_response_ratio": float(np.mean(corrected)),
                "minimum_corrected_response_ratio": float(np.min(corrected)),
                "maximum_corrected_response_ratio": float(np.max(corrected)),
                "corrected_response_cv": float(
                    np.std(corrected) / (np.mean(corrected) + 1e-12)
                ),
                "enhanced_event_count": int(
                    np.sum(corrected >= EVENT_ENHANCEMENT_THRESHOLD)
                ),
                "event_count": int(len(corrected)),
                "mean_10_to_43_response": float(
                    direction_means.get("10_to_43", np.nan)
                ),
                "mean_43_to_10_response": float(
                    direction_means.get("43_to_10", np.nan)
                ),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["role_after_state_correction"] = "background_dynamic_mode"
    aggregate.loc[
        aggregate["state_contrast_ratio"] >= 1.30,
        "role_after_state_correction",
    ] = "angle_state_sensitive_mode"
    aggregate.loc[
        (
            aggregate["mean_corrected_response_ratio"]
            >= EVENT_ENHANCEMENT_THRESHOLD
        )
        & (aggregate["enhanced_event_count"] >= 3),
        "role_after_state_correction",
    ] = "angle_movement_response_candidate"
    angle_mode = aggregate["angle_component_correlation"].abs().idxmax()
    aggregate.loc[angle_mode, "role_after_state_correction"] = (
        "angle_position_mode"
    )
    return event_table, aggregate


def find_cross_method_bands(aggregate: pd.DataFrame) -> pd.DataFrame:
    candidates = aggregate.loc[
        aggregate["role_after_state_correction"]
        == "angle_movement_response_candidate"
    ].sort_values("peak_frequency_hz")
    groups: list[list[pd.Series]] = []

    for _, row in candidates.iterrows():
        if not groups:
            groups.append([row])
            continue

        group_frequency = float(
            np.mean([item["peak_frequency_hz"] for item in groups[-1]])
        )

        if abs(float(row["peak_frequency_hz"]) - group_frequency) <= (
            FREQUENCY_MATCH_TOLERANCE_HZ
        ):
            groups[-1].append(row)
        else:
            groups.append([row])

    rows = []

    for band_id, group in enumerate(groups, start=1):
        methods = sorted({str(item["method"]) for item in group})

        if len(methods) < 2:
            continue

        frequencies = np.array(
            [float(item["peak_frequency_hz"]) for item in group]
        )
        responses = np.array(
            [float(item["mean_corrected_response_ratio"]) for item in group]
        )
        rows.append(
            {
                "band_id": band_id,
                "frequency_low_hz": float(np.min(frequencies)),
                "frequency_high_hz": float(np.max(frequencies)),
                "representative_frequency_hz": float(np.median(frequencies)),
                "supporting_methods": ", ".join(methods),
                "supporting_method_count": len(methods),
                "mean_corrected_response_ratio": float(np.mean(responses)),
                "interpretation": "cross_method_angle_movement_response_band",
            }
        )

    return pd.DataFrame(rows)


def plot_event_response_heatmap(
    event_table: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(METHODS),
        1,
        figsize=(13, 10),
        constrained_layout=True,
    )

    for axis, method in zip(axes, METHODS):
        method_aggregate = aggregate.loc[
            (aggregate["method"] == method)
            & (aggregate["peak_frequency_hz"] >= 0.5)
        ].sort_values("peak_frequency_hz")
        mode_order = method_aggregate["mode"].tolist()
        pivot = (
            event_table.loc[event_table["method"] == method]
            .pivot(index="mode", columns="event_id", values="corrected_response_ratio")
            .reindex(mode_order)
        )
        image = axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            cmap="RdBu_r",
            vmin=0.8,
            vmax=1.2,
        )
        labels = [
            f"{row.mode}\n{row.peak_frequency_hz:.1f} Hz"
            for row in method_aggregate.itertuples(index=False)
        ]
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        axis.set_xticks(
            np.arange(len(pivot.columns)),
            labels=[f"E{event_id}" for event_id in pivot.columns],
        )
        axis.set_title(method)
        axis.set_xlabel("Angle-change event")
        axis.set_ylabel("Mode")

        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    figure.colorbar(
        image,
        ax=axes,
        label="Event RMS / local pre-post baseline RMS",
        shrink=0.8,
    )
    figure.suptitle("Event-by-event modal response after angle-state correction")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_state_and_movement_roles(
    aggregate: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = {
        "VMD": "tab:purple",
        "IOVMD": "tab:blue",
        "AVMD": "tab:green",
        "SVMD": "tab:red",
        "STVMD": "tab:brown",
    }

    for method in METHODS:
        subset = aggregate.loc[
            (aggregate["method"] == method)
            & (aggregate["peak_frequency_hz"] >= 0.5)
        ]
        axis.scatter(
            subset["state_contrast_ratio"],
            subset["mean_corrected_response_ratio"],
            label=method,
            color=colors[method],
            alpha=0.8,
        )

        for row in subset.itertuples(index=False):
            if (
                row.state_contrast_ratio >= 1.30
                or row.mean_corrected_response_ratio >= EVENT_ENHANCEMENT_THRESHOLD
            ):
                axis.annotate(
                    f"{row.peak_frequency_hz:.1f} Hz",
                    (row.state_contrast_ratio, row.mean_corrected_response_ratio),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

    axis.axvline(1.30, color="0.3", linestyle="--", linewidth=1.0)
    axis.axhline(
        EVENT_ENHANCEMENT_THRESHOLD,
        color="0.3",
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlabel("Stable-angle contrast ratio")
    axis.set_ylabel("Mean event response after local baseline correction")
    axis.set_title("Separate angle-state modes from angle-movement modes")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    transitions: pd.DataFrame,
    aggregate: pd.DataFrame,
    common_bands: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# 角度切換逐事件模態分析",
        "",
        f"共分析{len(transitions)}次角度切換。每次事件皆以切換前後各"
        f"{BASELINE_WINDOW_S:.0f}秒的局部RMS建立基準，以降低10°與43°"
        "本身能量差異造成的誤判。",
        "",
        "## 角度狀態相關模態",
        "",
    ]
    state_modes = aggregate.loc[
        aggregate["role_after_state_correction"]
        == "angle_state_sensitive_mode"
    ].sort_values(["peak_frequency_hz", "method"])
    lines.extend(
        [
            "| 方法 | 模態 | 頻率 (Hz) | 43°/10° RMS |",
            "|---|---:|---:|---:|",
        ]
    )

    for row in state_modes.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.mode} | {row.peak_frequency_hz:.3f} | "
            f"{row.state_ratio_43_to_10:.3f} |"
        )

    lines.extend(["", "## 跨方法角度移動響應頻帶", ""])

    if common_bands.empty:
        lines.append("目前沒有獲得至少兩種方法支持的移動響應頻帶。")
    else:
        lines.extend(
            [
                "| 頻率範圍 (Hz) | 代表頻率 (Hz) | 支持方法 | 平均校正響應 |",
                "|---:|---:|---|---:|",
            ]
        )

        for row in common_bands.itertuples(index=False):
            lines.append(
                f"| {row.frequency_low_hz:.3f}–{row.frequency_high_hz:.3f} | "
                f"{row.representative_frequency_hz:.3f} | "
                f"{row.supporting_methods} | "
                f"{row.mean_corrected_response_ratio:.3f} |"
            )

    lines.extend(
        [
            "",
            "## 判讀原則",
            "",
            "- 角度位置模態：與低頻角度曲線高度相關。",
            "- 角度狀態模態：10°與43°穩定期間的RMS不同。",
            "- 角度移動模態：扣除事件前後局部狀態基準後仍增強。",
            "- 目前僅有四次切換，結果仍應稱為候選頻帶，不應稱為異常或故障。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    transitions = pd.read_csv(
        ANGLE_ROOT / "angle_transitions.csv",
        encoding="utf-8-sig",
    )
    event_tables = []
    aggregate_tables = []

    for method in METHODS:
        event_table, aggregate = calculate_corrected_event_response(
            method,
            transitions,
        )
        event_tables.append(event_table)
        aggregate_tables.append(aggregate)

    all_events = pd.concat(event_tables, ignore_index=True)
    all_aggregate = pd.concat(aggregate_tables, ignore_index=True)
    common_bands = find_cross_method_bands(all_aggregate)
    all_events.to_csv(
        OUTPUT_ROOT / "event_corrected_mode_response.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_aggregate.to_csv(
        OUTPUT_ROOT / "mode_role_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    common_bands.to_csv(
        OUTPUT_ROOT / "cross_method_movement_bands.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_event_response_heatmap(
        all_events,
        all_aggregate,
        OUTPUT_ROOT / "01_event_response_heatmap.png",
    )
    plot_state_and_movement_roles(
        all_aggregate,
        OUTPUT_ROOT / "02_state_vs_movement_modes.png",
    )
    write_summary(
        transitions,
        all_aggregate,
        common_bands,
        OUTPUT_ROOT / "EVENT_ANALYSIS_SUMMARY.md",
    )
    print("=" * 70)
    print("逐事件模態分析完成")
    print(common_bands.to_string(index=False))
    print(f"結果位置：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
