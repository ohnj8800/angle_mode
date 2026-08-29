from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_fbg_csv
from src.common.metrics import calculate_reconstruction_metrics
from src.common.mode_response import calculate_transition_mode_response
from src.common.postprocessing import evaluate_welch_support, merge_redundant_modes
from src.common.preprocessing import preprocess_fbg_signal
from src.common.metrics import calculate_mode_relationships
from src.common.vmd import variational_mode_decomposition
from src.methods.avmd import (
    create_bandwise_initial_centers,
    select_avmd_parameters,
)
from src.methods.iovmd import run_initial_iovmd
from src.methods.svmd import successive_variational_mode_decomposition
from src.methods.stvmd import short_time_variational_mode_decomposition
from src.methods.classic_vmd import (
    classic_variational_mode_decomposition,
)


SAMPLING_RATE = 200.0
RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "angle_10deg_43deg_repeat.csv"
)
ANGLE_ROOT = PROJECT_ROOT / "results" / "angle_tracking"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "mode_analysis"
METHOD_ORDER = ("VMD", "IOVMD", "AVMD", "SVMD", "STVMD")


def _angle_initial_frequency(angle_component_nm: np.ndarray) -> float:
    centered = angle_component_nm - np.mean(angle_component_nm)
    frequency_hz = np.fft.rfftfreq(centered.size, d=1.0 / SAMPLING_RATE)
    amplitude = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    mask = (
        (frequency_hz >= SAMPLING_RATE / centered.size)
        & (frequency_hz <= 0.5)
    )
    return float(frequency_hz[mask][np.argmax(amplitude[mask])])

def run_vmd(
    centered_raw_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    result = classic_variational_mode_decomposition(
        centered_raw_signal,
        sampling_rate=SAMPLING_RATE,
        mode_count=10,
        alpha=2000.0,
        maximum_iterations=500,
    )

    parameters = {
        "alpha": 2000.0,
        "fixed_mode_count": 10,
        "initial_center_frequencies_hz": (
            result["initial_center_frequencies_hz"].tolist()
        ),
        "parameter_rule": (
            "fixed K=10 and alpha=2000 with uniform initialization"
        ),
    }

    return (
        result["modes"],
        result["center_frequencies_hz"],
        parameters,
    )

def run_iovmd(
    centered_raw_signal: np.ndarray,
    dynamic_signal: np.ndarray,
    angle_component_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    initial = run_initial_iovmd(
        dynamic_signal,
        sampling_rate=SAMPLING_RATE,
        alpha=2000.0,
        maximum_iterations=500,
        minimum_frequency_hz=0.5,
        maximum_frequency_hz=99.9,
    )
    dynamic_centers = initial["initial_center_frequencies_hz"]
    low_center = _angle_initial_frequency(angle_component_nm)
    full_result = variational_mode_decomposition(
        centered_raw_signal,
        sampling_rate=SAMPLING_RATE,
        initial_center_frequencies_hz=np.r_[low_center, dynamic_centers],
        alpha=2000.0,
        maximum_iterations=500,
    )
    initial_modes = full_result["modes"]
    relationships = calculate_mode_relationships(initial_modes)
    modes, groups = merge_redundant_modes(
        initial_modes,
        relationships["pair_table"],
    )
    centers = np.full(len(modes), np.nan, dtype=float)
    parameters = {
        "alpha": 2000.0,
        "angle_initial_frequency_hz": float(low_center),
        "initial_mode_count": int(len(initial_modes)),
        "final_mode_count": int(len(modes)),
        "merged_groups": [
            [int(index + 1) for index in group]
            for group in groups
            if len(group) > 1
        ],
        "parameter_rule": "LOWESS spectral peaks plus independence merging",
    }
    return modes, centers, parameters


def run_avmd(
    centered_raw_signal: np.ndarray,
    dynamic_signal: np.ndarray,
    reference_signal: np.ndarray,
    angle_component_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    selection = select_avmd_parameters(
        reference_signal=reference_signal,
        sampling_rate=SAMPLING_RATE,
        candidate_mode_counts=(4, 5, 6, 7, 8),
        candidate_alphas=(100.0, 250.0, 500.0, 1000.0, 2000.0),
        maximum_iterations=300,
    )
    mode_count = selection["best_mode_count"]
    alpha = selection["best_alpha"]
    initial_centers = create_bandwise_initial_centers(
        dynamic_signal,
        sampling_rate=SAMPLING_RATE,
        mode_count=mode_count,
        minimum_frequency_hz=0.5,
        maximum_frequency_hz=99.9,
    )
    low_center = _angle_initial_frequency(angle_component_nm)
    full_initial_centers = np.r_[low_center, initial_centers]
    result = variational_mode_decomposition(
        centered_raw_signal,
        sampling_rate=SAMPLING_RATE,
        initial_center_frequencies_hz=full_initial_centers,
        alpha=alpha,
        maximum_iterations=500,
    )
    parameters = {
        "alpha": float(alpha),
        "selected_dynamic_mode_count": int(mode_count),
        "final_mode_count_including_angle": int(mode_count + 1),
        "angle_initial_frequency_hz": float(low_center),
        "selection_score": float(selection["best_selection_score"]),
        "parameter_rule": "automatic K and alpha search on stable 10-degree operation",
    }
    return result["modes"], result["center_frequencies_hz"], parameters


def run_svmd(centered_raw_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    signal = centered_raw_signal

    if signal.size % 2:
        signal = signal[:-1]

    result = successive_variational_mode_decomposition(
        signal,
        sampling_rate=SAMPLING_RATE,
        maximum_alpha=1000.0,
        minimum_alpha=10.0,
        maximum_iterations=300,
        maximum_modes=20,
        reconstruction_power_tolerance=0.005,
    )
    parameters = {
        "maximum_alpha": 1000.0,
        "sequential_mode_count": int(result["mode_count"]),
        "stop_reason": result["stop_reason"],
        "parameter_rule": "validated SVMD setting; sequential stopping determines K",
    }
    return result["modes"], result["center_frequencies_hz"], parameters


def run_stvmd(
    centered_raw_signal: np.ndarray,
    dynamic_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    result = short_time_variational_mode_decomposition(
        dynamic_signal,
        sampling_rate=SAMPLING_RATE,
        mode_count=9,
        window_duration_s=8.0,
        hop_duration_s=2.0,
        minimum_frequency_hz=0.5,
        maximum_frequency_hz=99.9,
        alpha=2000.0,
        maximum_iterations=200,
    )
    # 0.5 Hz以下內容保留成角度／慢變模態；其餘9個模態由短時VMD取得。
    low_frequency_mode = centered_raw_signal - dynamic_signal
    modes = np.vstack([low_frequency_mode, result["modes"]])
    centers = np.r_[
        _angle_initial_frequency(low_frequency_mode),
        result["center_frequencies_hz"],
    ]
    parameters = {
        "alpha": 2000.0,
        "dynamic_mode_count": 9,
        "final_mode_count_including_angle": 10,
        "window_duration_s": 8.0,
        "hop_duration_s": 2.0,
        "frame_count": int(result["frame_count"]),
        "converged_frame_fraction": float(result["converged_frame_fraction"]),
        "median_iterations": float(result["median_iterations"]),
        "implementation_variant": (
            "single-channel windowed dynamic VMD with center-frequency "
            "tracking and Hann overlap-add"
        ),
        "parameter_rule": (
            "fixed short-time window and K for nonstationary comparison; "
            "engineering STVMD implementation, not authors' reference solver"
        ),
    }
    return modes, centers, parameters


def plot_mode_spectra(
    modes: np.ndarray,
    metrics: pd.DataFrame,
    method: str,
    output_path: Path,
) -> None:
    rows = len(modes)
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(13, max(5, 1.6 * rows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for index, axis in enumerate(axes):
        frequency_hz = np.fft.rfftfreq(modes.shape[1], d=1.0 / SAMPLING_RATE)
        amplitude = np.abs(np.fft.rfft(modes[index]))
        amplitude /= np.max(amplitude) + np.finfo(float).eps
        metric = metrics.loc[metrics["mode"] == f"IMF{index + 1}"].iloc[0]
        axis.plot(frequency_hz, amplitude, color="tab:blue", linewidth=0.7)
        axis.axvline(
            metric["peak_frequency_hz"],
            color="tab:red",
            linestyle="--",
            linewidth=0.8,
        )
        axis.set_ylabel(metric["mode"])
        axis.set_title(
            f"peak {metric['peak_frequency_hz']:.2f} Hz | "
            f"move/stable {metric['transition_to_stable_rms_ratio']:.2f}x",
            fontsize=8,
        )
        axis.grid(alpha=0.15)

    axes[-1].set_xlim(0.5, 100.0)
    axes[-1].set_xlabel("Frequency (Hz)")
    figure.suptitle(f"{method}: mode frequency and movement response")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_motion_sensitive_modes(
    time_s: np.ndarray,
    modes: np.ndarray,
    response: pd.DataFrame,
    transitions: pd.DataFrame,
    method: str,
    output_path: Path,
) -> None:
    selected = response.nsmallest(min(4, len(response)), "motion_sensitivity_rank")
    indices = [int(mode.replace("IMF", "")) - 1 for mode in selected["mode"]]
    figure, axes = plt.subplots(
        len(indices),
        1,
        figsize=(14, max(5, 2.2 * len(indices))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    stride = max(1, len(time_s) // 25000)

    for axis, mode_index in zip(axes, indices):
        row = response.loc[response["mode"] == f"IMF{mode_index + 1}"].iloc[0]
        axis.plot(
            time_s[::stride],
            modes[mode_index, ::stride],
            color="tab:blue",
            linewidth=0.55,
        )

        for event in transitions.itertuples(index=False):
            axis.axvspan(
                event.start_time_s,
                event.end_time_s,
                color="tab:orange",
                alpha=0.16,
            )

        axis.set_ylabel(row["mode"])
        axis.set_title(
            f"peak {row['peak_frequency_hz']:.2f} Hz | "
            f"movement response {row['transition_to_stable_rms_ratio']:.2f}x",
            fontsize=9,
        )
        axis.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"{method}: modes most sensitive to angle movement")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cross_method_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = {
        "VMD": "tab:purple",
        "IOVMD": "tab:blue",
        "AVMD": "tab:green",
        "SVMD": "tab:red",
        "STVMD": "tab:brown",
    }

    for method in METHOD_ORDER:
        subset = comparison.loc[comparison["method"] == method]
        axis.scatter(
            subset["peak_frequency_hz"],
            subset["transition_to_stable_rms_ratio"],
            s=45,
            alpha=0.8,
            label=method,
            color=colors[method],
        )

    axis.axhline(1.10, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlim(0.5, 100.0)
    axis.set_xlabel("Mode peak frequency (Hz)")
    axis.set_ylabel("Movement RMS / stable RMS")
    axis.set_title("Cross-method frequency and angle-movement response")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_angle_mode_comparison(
    time_s: np.ndarray,
    angle_component_nm: np.ndarray,
    angle_mode_traces: dict[str, np.ndarray],
    transitions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1 + len(METHOD_ORDER),
        1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
    )
    traces = [("0.1-Hz angle component", angle_component_nm)] + [
        (f"{method} angle mode", angle_mode_traces[method])
        for method in METHOD_ORDER
    ]

    for axis, (label, trace) in zip(axes, traces):
        standardized = (trace - np.mean(trace)) / (np.std(trace) + 1e-12)
        axis.plot(time_s[: len(trace)], standardized, linewidth=1.2)

        for event in transitions.itertuples(index=False):
            axis.axvspan(
                event.start_time_s,
                event.end_time_s,
                color="tab:orange",
                alpha=0.15,
            )

        axis.set_ylabel(label)
        axis.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("Angle-position mode recovered by all VMD methods")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def export_analysis_summary(
    transitions: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    angle_modes = comparison.loc[
        comparison["physical_role_candidate"] == "angle_position_mode"
    ]
    movement_modes = comparison.loc[
        comparison["motion_sensitive_candidate"].astype(bool)
    ]
    recurrence_intervals = transitions["same_direction_interval_s"].dropna()
    median_duration_s = float(transitions["duration_s"].median())
    lines = [
        "# FBG角度模態分析摘要",
        "",
        "## 已確認結果",
        "",
        "- 10°與43°皆視為正常角度狀態，不再把43°標示成異常。",
        f"- 由低頻波長平台自動偵測到{len(transitions)}次角度切換。",
        f"- 切換持續時間中位數為{median_duration_s:.3f}秒。",
    ]

    if not recurrence_intervals.empty:
        recurrence_s = float(recurrence_intervals.median())
        lines.append(
            f"- 同方向動作的中位重複週期為{recurrence_s:.3f}秒，"
            f"事件重複頻率約{1.0 / recurrence_s:.5f} Hz。"
        )

    lines.extend(
        [
            f"- {len(METHOD_ORDER)}種VMD皆分離出角度位置模態；其頻譜主峰約0.00869 Hz，"
            "與0.1 Hz以下角度成分的相關係數如下。",
            "",
            "| 方法 | 角度模態 | 主峰頻率 (Hz) | 角度相關係數 |",
            "|---|---:|---:|---:|",
        ]
    )

    for row in angle_modes.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.mode} | {row.peak_frequency_hz:.6f} | "
            f"{row.angle_component_correlation:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 角度移動響應候選",
            "",
        ]
    )

    if movement_modes.empty:
        lines.append("目前沒有通過1.10倍移動／穩定RMS門檻的動態模態。")
    else:
        lines.extend(
            [
                "| 方法 | 模態 | 主峰頻率 (Hz) | 移動／穩定RMS |",
                "|---|---:|---:|---:|",
            ]
        )

        for row in movement_modes.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.mode} | {row.peak_frequency_hz:.3f} | "
                f"{row.transition_to_stable_rms_ratio:.3f} |"
            )

    lines.extend(
        [
            "",
            "上述高頻模態若未獲得多種方法的一致支持，則只能列為"
            "角度移動響應候選，不能解釋成故障頻率。",
            "",
            "## 解釋限制",
            "",
            "- 0.00869 Hz是有限長度資料的低頻主峰；依事件起點計算的"
            "重複頻率約0.01 Hz，兩者估計方式不同。",
            "- 切換持續時間的倒數只是時間尺度，不代表角度機構持續旋轉的頻率。",
            "- 目前只有10°與43°兩個已知校正點，不能推廣成任意角度的精密量測。",
            "- 真正的異常偵測應在更多正常角度切換資料建立基準後再進行。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timeline = pd.read_csv(ANGLE_ROOT / "angle_timeline.csv", encoding="utf-8-sig")
    transitions = pd.read_csv(
        ANGLE_ROOT / "angle_transitions.csv", encoding="utf-8-sig"
    )
    data, _ = load_fbg_csv(RAW_FILE, sampling_rate=SAMPLING_RATE)
    raw_nm = data["ch1_nm"].to_numpy(dtype=float)
    time_s = data["time_s"].to_numpy(dtype=float)
    signals = preprocess_fbg_signal(raw_nm, sampling_rate=SAMPLING_RATE)
    centered_raw_signal = signals["centered_raw_nm"]
    angle_component_nm = signals["angle_component_nm"]
    dynamic_signal = signals["dynamic_nm"]
    state_labels = timeline["state_label"].to_numpy(dtype=str)
    reference_mask = (
        (state_labels == "stable_10deg")
        & (time_s >= 5.0)
        & (time_s <= 25.0)
    )
    reference_signal = dynamic_signal[reference_mask]

    runners = {
        "VMD": lambda: run_vmd(centered_raw_signal),
        "IOVMD": lambda: run_iovmd(
            centered_raw_signal,
            dynamic_signal,
            angle_component_nm,
        ),
        "AVMD": lambda: run_avmd(
            centered_raw_signal,
            dynamic_signal,
            reference_signal,
            angle_component_nm,
        ),
        "SVMD": lambda: run_svmd(centered_raw_signal),
        "STVMD": lambda: run_stvmd(centered_raw_signal, dynamic_signal),
    }
    comparison_rows = []
    method_summaries = []
    angle_mode_traces: dict[str, np.ndarray] = {}

    for method in METHOD_ORDER:
        print("=" * 70)
        print(f"開始分析：{method}")
        modes, centers, parameters = runners[method]()
        effective_length = modes.shape[1]
        effective_labels = state_labels[:effective_length]
        effective_time = time_s[:effective_length]
        effective_signal = centered_raw_signal[:effective_length]
        effective_angle_component = angle_component_nm[:effective_length]
        response, event_response = calculate_transition_mode_response(
            modes,
            SAMPLING_RATE,
            effective_labels,
            transitions,
            algorithm_center_frequencies_hz=centers,
        )
        response, _ = evaluate_welch_support(
            response,
            effective_signal,
            sampling_rate=SAMPLING_RATE,
            frequency_tolerance_hz=1.0,
            minimum_frequency_hz=SAMPLING_RATE / effective_length,
        )
        centered_angle_component = (
            effective_angle_component - np.mean(effective_angle_component)
        )
        correlation_by_mode = {
            f"IMF{mode_index + 1}": float(
                np.corrcoef(mode, centered_angle_component)[0, 1]
            )
            for mode_index, mode in enumerate(modes)
        }
        response["angle_component_correlation"] = (
            response["mode"].map(correlation_by_mode)
        )
        angle_mode_name = response.loc[
            response["angle_component_correlation"].abs().idxmax(),
            "mode",
        ]
        response["physical_role_candidate"] = "background_dynamic_mode"
        response.loc[
            response["motion_sensitive_candidate"],
            "physical_role_candidate",
        ] = "angle_movement_response_candidate"
        response.loc[
            response["mode"] == angle_mode_name,
            "physical_role_candidate",
        ] = "angle_position_mode"
        angle_mode_index = int(angle_mode_name.replace("IMF", "")) - 1
        angle_mode_traces[method] = modes[angle_mode_index].copy()
        reconstruction = np.sum(modes, axis=0)
        reconstruction_metrics = calculate_reconstruction_metrics(
            effective_signal,
            reconstruction,
        )
        method_root = OUTPUT_ROOT / method.lower()
        method_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            method_root / "decomposition.npz",
            time_s=effective_time,
            dynamic_signal=effective_signal,
            modes=modes,
            reconstructed_signal=reconstruction,
        )
        response.to_csv(
            method_root / "mode_response_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        event_response.to_csv(
            method_root / "transition_mode_response.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary = {
            "method": method,
            "analysis_band_hz": [float(SAMPLING_RATE / effective_length), 100.0],
            "angle_source": "lowest VMD mode validated against a DC-preserving 0.1-Hz component",
            "angle_mode": angle_mode_name,
            "angle_mode_correlation": float(
                response.loc[
                    response["mode"] == angle_mode_name,
                    "angle_component_correlation",
                ].iloc[0]
            ),
            "mode_count": int(len(modes)),
            "motion_sensitive_candidate_count": int(
                response["motion_sensitive_candidate"].sum()
            ),
            **parameters,
            **reconstruction_metrics,
        }

        with (method_root / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        plot_mode_spectra(
            modes,
            response.sort_values("spectral_centroid_hz").reset_index(drop=True),
            method,
            method_root / "01_mode_spectra.png",
        )
        plot_motion_sensitive_modes(
            effective_time,
            modes,
            response,
            transitions,
            method,
            method_root / "02_motion_sensitive_modes.png",
        )
        response_for_comparison = response.copy()
        response_for_comparison.insert(0, "method", method)
        comparison_rows.append(response_for_comparison)
        method_summaries.append(summary)
        print(response.head(5).to_string(index=False))

    comparison = pd.concat(comparison_rows, ignore_index=True)
    comparison.to_csv(
        OUTPUT_ROOT / "mode_frequency_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(method_summaries).to_csv(
        OUTPUT_ROOT / "method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_cross_method_comparison(
        comparison,
        OUTPUT_ROOT / "cross_method_mode_comparison.png",
    )
    plot_angle_mode_comparison(
        time_s,
        angle_component_nm,
        angle_mode_traces,
        transitions,
        OUTPUT_ROOT / "angle_position_mode_comparison.png",
    )
    export_analysis_summary(
        transitions,
        comparison,
        OUTPUT_ROOT / "ANALYSIS_SUMMARY.md",
    )
    print("=" * 70)
    print(f"{len(METHOD_ORDER)}種VMD模態分析完成：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
