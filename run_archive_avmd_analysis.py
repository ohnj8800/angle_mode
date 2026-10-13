from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_batch_static_analysis import METHOD_ORDER, decompose_signal
from src.common.archive_metadata import discover_archive_csvs
from src.common.io import load_fbg_csv
from src.common.metrics import calculate_mode_metrics, calculate_reconstruction_metrics
from src.common.postprocessing import evaluate_welch_support
from src.common.preprocessing import calculate_welch_psd, preprocess_fbg_signal
from src.methods.avmd import select_adaptive_avmd_k, select_avmd_parameters


CONDITION_LABELS = {
    "無負載": "No load",
    "有負載": "Load",
    "偏心": "Eccentric",
    "無負載開關": "No-load switching",
    "260607": "260607",
    "其他": "Other",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "遞迴分析260720整包FBG資料；保留五種方法，但預設只執行AVMD。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "260720",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "archive_avmd",
    )
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=["AVMD"],
        help="預設只執行AVMD；其他方法仍保留並可手動指定。",
    )
    parser.add_argument(
        "--avmd-k",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--avmd-alpha",
        nargs="+",
        type=float,
        default=[100.0, 250.0, 500.0, 1000.0, 2000.0],
    )
    parser.add_argument(
        "--dynamic-lowpass-hz",
        type=float,
        default=40.0,
        help="AVMD動態訊號上限頻率；預設40 Hz。",
    )
    parser.add_argument("--adaptive-k-min", type=int, default=3)
    parser.add_argument("--adaptive-k-max", type=int, default=9)
    parser.add_argument("--adaptive-k-steps", type=int, default=3)
    parser.add_argument(
        "--disable-adaptive-k",
        action="store_true",
        help="停用逐檔adaptive-K，改用參考檔選出的共用K。",
    )
    parser.add_argument(
        "--skip-per-file-plots",
        action="store_true",
        help="略過每個CSV的前處理與模態圖，只保留表格及總覽圖。",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="只執行資料品質檢查、前處理及前後對照，暫不執行AVMD。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="只供快速測試；正式分析請勿設定。",
    )
    return parser.parse_args()


def _dominant_frequency(
    frequency_hz: np.ndarray,
    psd: np.ndarray,
    minimum_hz: float = 0.5,
    maximum_hz: float = 99.0,
) -> float:
    mask = (frequency_hz >= minimum_hz) & (frequency_hz <= maximum_hz)
    if not np.any(mask):
        return float("nan")
    selected_frequency = frequency_hz[mask]
    selected_psd = psd[mask]
    return float(selected_frequency[int(np.argmax(selected_psd))])


def _band_power(
    frequency_hz: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (frequency_hz >= low_hz) & (frequency_hz <= high_hz)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapezoid(psd[mask], frequency_hz[mask]))


def _safe_output_folder(output_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    folder = output_root / "per_file" / relative.parent / relative.stem
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def plot_preprocessing_comparison(
    data: pd.DataFrame,
    preprocessing: dict[str, np.ndarray],
    raw_frequency_hz: np.ndarray,
    raw_psd: np.ndarray,
    processed_frequency_hz: np.ndarray,
    processed_psd: np.ndarray,
    title: str,
    output_path: Path,
    dynamic_lowpass_hz: float,
) -> None:
    time_s = data["time_s"].to_numpy()
    raw = preprocessing["raw_nm"]
    centered = preprocessing["centered_raw_nm"]
    angle_component = preprocessing["angle_component_nm"]
    dynamic = preprocessing["dynamic_nm"]

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(time_s, raw, color="0.15", linewidth=0.65, label="Cleaned raw")
    axes[0].plot(
        time_s,
        angle_component,
        color="tab:orange",
        linewidth=1.0,
        label="Low-frequency component",
    )
    axes[0].set_ylabel("Bragg wavelength (nm)")
    axes[0].set_title(f"{title} | raw wavelength and low-frequency component")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(time_s, centered, color="0.65", linewidth=0.6, label="Before")
    axes[1].plot(
        time_s,
        dynamic,
        color="0.05",
        linewidth=0.65,
        label=f"After: 0.5–{dynamic_lowpass_hz:g} Hz",
    )
    axes[1].set_ylabel("Centered wavelength (nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Before and after dynamic-signal preprocessing")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    positive_raw = raw_frequency_hz > 0
    positive_processed = processed_frequency_hz > 0
    axes[2].semilogy(
        raw_frequency_hz[positive_raw],
        np.maximum(raw_psd[positive_raw], np.finfo(float).tiny),
        color="0.65",
        linewidth=0.8,
        label="Before",
    )
    axes[2].semilogy(
        processed_frequency_hz[positive_processed],
        np.maximum(processed_psd[positive_processed], np.finfo(float).tiny),
        color="0.05",
        linewidth=0.9,
        label="After",
    )
    axes[2].axvline(0.5, color="tab:red", linestyle="--", linewidth=1.0, label="0.5 Hz")
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Welch PSD (nm²/Hz)")
    axes[2].set_title("Welch PSD before and after preprocessing")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.2)

    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def analyze_preprocessing(
    inventory: list[dict],
    sampling_rate: float,
    output_root: Path,
    make_per_file_plots: bool,
    dynamic_lowpass_hz: float,
) -> pd.DataFrame:
    rows = []
    for index, metadata in enumerate(inventory, start=1):
        print(f"前處理 {index}/{len(inventory)} | {metadata['relative_path']}", flush=True)
        try:
            data, information = load_fbg_csv(
                metadata["file_path"],
                sampling_rate=sampling_rate,
                remove_exact_duplicates=True,
            )
            preprocessing = preprocess_fbg_signal(
                data["ch1_nm"].to_numpy(),
                sampling_rate=sampling_rate,
                dynamic_lowpass_cutoff_hz=dynamic_lowpass_hz,
            )
            centered = preprocessing["centered_raw_nm"]
            dynamic = preprocessing["dynamic_nm"]
            raw_frequency_hz, raw_psd = calculate_welch_psd(
                centered, sampling_rate=sampling_rate
            )
            processed_frequency_hz, processed_psd = calculate_welch_psd(
                dynamic, sampling_rate=sampling_rate
            )

            row = {
                **{key: value for key, value in metadata.items() if key != "file_path"},
                **information,
                "status": "ok",
                "error_message": "",
                "raw_ac_rms_nm": float(np.sqrt(np.mean(centered**2))),
                "processed_dynamic_rms_nm": float(np.sqrt(np.mean(dynamic**2))),
                "raw_peak_to_peak_nm": float(np.ptp(preprocessing["raw_nm"])),
                "processed_peak_to_peak_nm": float(np.ptp(dynamic)),
                "raw_dominant_welch_frequency_hz": _dominant_frequency(
                    raw_frequency_hz, raw_psd
                ),
                "processed_dominant_welch_frequency_hz": _dominant_frequency(
                    processed_frequency_hz, processed_psd
                ),
                "raw_power_below_0_5_hz": _band_power(
                    raw_frequency_hz, raw_psd, 0.0, 0.5
                ),
                "processed_power_below_0_5_hz": _band_power(
                    processed_frequency_hz, processed_psd, 0.0, 0.5
                ),
            }
            rows.append(row)

            if make_per_file_plots:
                folder = _safe_output_folder(output_root, metadata["relative_path"])
                plot_preprocessing_comparison(
                    data,
                    preprocessing,
                    raw_frequency_hz,
                    raw_psd,
                    processed_frequency_hz,
                    processed_psd,
                    metadata["relative_path"],
                    folder / "01_preprocessing_comparison.png",
                    dynamic_lowpass_hz,
                )
        except Exception as error:
            rows.append(
                {
                    **{key: value for key, value in metadata.items() if key != "file_path"},
                    "status": "failed",
                    "error_message": str(error),
                }
            )
    return pd.DataFrame(rows)


def choose_avmd_reference(
    preprocessing_summary: pd.DataFrame,
    inventory_by_relative_path: dict[str, dict],
    sampling_rate: float,
    dynamic_lowpass_hz: float,
) -> tuple[str, np.ndarray]:
    valid = preprocessing_summary.loc[preprocessing_summary["status"] == "ok"].copy()
    preferred = valid.loc[
        valid["condition"].eq("無負載")
        & valid["analysis_type"].eq("fixed_angle")
        & valid["angle_deg"].eq(10)
    ].copy()
    if preferred.empty:
        preferred = valid.loc[valid["analysis_type"].eq("fixed_angle")].copy()
    if preferred.empty:
        preferred = valid.copy()
    median_rms = float(preferred["processed_dynamic_rms_nm"].median())
    chosen = preferred.iloc[
        (preferred["processed_dynamic_rms_nm"] - median_rms).abs().argmin()
    ]
    relative_path = str(chosen["relative_path"])
    metadata = inventory_by_relative_path[relative_path]
    data, _ = load_fbg_csv(
        metadata["file_path"], sampling_rate=sampling_rate,
        remove_exact_duplicates=True,
    )
    preprocessing = preprocess_fbg_signal(
        data["ch1_nm"].to_numpy(),
        sampling_rate=sampling_rate,
        dynamic_lowpass_cutoff_hz=dynamic_lowpass_hz,
    )
    return relative_path, preprocessing["dynamic_nm"]


def prepare_avmd_settings(
    preprocessing_summary: pd.DataFrame,
    inventory_by_relative_path: dict[str, dict],
    sampling_rate: float,
    candidate_k: list[int],
    candidate_alpha: list[float],
    output_root: Path,
    dynamic_lowpass_hz: float,
) -> dict:
    reference_path, reference_signal = choose_avmd_reference(
        preprocessing_summary,
        inventory_by_relative_path,
        sampling_rate,
        dynamic_lowpass_hz,
    )
    selection = select_avmd_parameters(
        reference_signal=reference_signal,
        sampling_rate=sampling_rate,
        candidate_mode_counts=candidate_k,
        candidate_alphas=candidate_alpha,
        maximum_iterations=300,
        maximum_frequency_hz=dynamic_lowpass_hz,
    )
    selection["trials"].to_csv(
        output_root / "avmd_parameter_trials.csv", index=False
    )
    return {
        "reference_file": reference_path,
        "mode_count": int(selection["best_mode_count"]),
        "alpha": float(selection["best_alpha"]),
        "selection_score": float(selection["best_selection_score"]),
        "maximum_frequency_hz": float(dynamic_lowpass_hz),
    }


def plot_method_modes(
    signal: np.ndarray,
    modes: np.ndarray,
    metrics: pd.DataFrame,
    sampling_rate: float,
    method: str,
    title: str,
    output_path: Path,
) -> None:
    mode_count = len(modes)
    duration_to_show = min(8.0, len(signal) / sampling_rate)
    show_count = max(2, min(len(signal), int(duration_to_show * sampling_rate)))
    time_s = np.arange(show_count) / sampling_rate
    figure, axes = plt.subplots(
        mode_count + 1,
        2,
        figsize=(15, 2.2 * (mode_count + 1)),
        constrained_layout=True,
    )

    axes[0, 0].plot(time_s, signal[:show_count], color="0.1", linewidth=0.7)
    axes[0, 0].set_title(f"{title} | preprocessed signal")
    axes[0, 0].set_ylabel("nm")
    signal_frequency, signal_psd = calculate_welch_psd(signal, sampling_rate)
    axes[0, 1].semilogy(
        signal_frequency[1:],
        np.maximum(signal_psd[1:], np.finfo(float).tiny),
        color="0.1",
        linewidth=0.8,
    )
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_title(f"{method} input Welch PSD")

    for mode_index, mode in enumerate(modes):
        row = metrics.iloc[mode_index]
        axes[mode_index + 1, 0].plot(
            time_s, mode[:show_count], color="0.15", linewidth=0.65
        )
        axes[mode_index + 1, 0].set_ylabel(str(row["mode"]))
        frequency, psd = calculate_welch_psd(mode, sampling_rate)
        axes[mode_index + 1, 1].semilogy(
            frequency[1:],
            np.maximum(psd[1:], np.finfo(float).tiny),
            color="0.15",
            linewidth=0.75,
        )
        axes[mode_index + 1, 1].axvline(
            float(row["peak_frequency_hz"]), color="tab:red", linewidth=0.8
        )
        axes[mode_index + 1, 1].set_xlim(0, 100)
        axes[mode_index + 1, 1].set_title(
            f"peak={row['peak_frequency_hz']:.2f} Hz, energy={row['energy_ratio']:.1%}"
        )
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_mode_analysis(
    inventory: list[dict],
    valid_relative_paths: set[str],
    methods: list[str],
    sampling_rate: float,
    avmd_settings: dict | None,
    output_root: Path,
    make_per_file_plots: bool,
    dynamic_lowpass_hz: float,
    adaptive_k_enabled: bool,
    adaptive_k_min: int,
    adaptive_k_max: int,
    adaptive_k_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    summaries = []
    metric_tables = []
    parameters = {}
    adaptive_trial_tables = []
    for index, metadata in enumerate(inventory, start=1):
        relative_path = metadata["relative_path"]
        if relative_path not in valid_relative_paths:
            continue
        data, _ = load_fbg_csv(
            metadata["file_path"], sampling_rate=sampling_rate,
            remove_exact_duplicates=True,
        )
        preprocessing = preprocess_fbg_signal(
            data["ch1_nm"].to_numpy(),
            sampling_rate=sampling_rate,
            dynamic_lowpass_cutoff_hz=dynamic_lowpass_hz,
        )
        signal = preprocessing["dynamic_nm"]
        for method in methods:
            print(
                f"模態分析 {index}/{len(inventory)} | {method} | {relative_path}",
                flush=True,
            )
            started = time.perf_counter()
            try:
                file_avmd_settings = avmd_settings
                adaptive_selection = None
                if method == "AVMD" and adaptive_k_enabled:
                    if avmd_settings is None:
                        raise ValueError("AVMD settings are required.")
                    adaptive_selection = select_adaptive_avmd_k(
                        signal=signal,
                        sampling_rate=sampling_rate,
                        initial_mode_count=int(avmd_settings["mode_count"]),
                        alpha=float(avmd_settings["alpha"]),
                        minimum_mode_count=adaptive_k_min,
                        maximum_mode_count=adaptive_k_max,
                        maximum_steps=adaptive_k_steps,
                        maximum_iterations=300,
                        maximum_frequency_hz=dynamic_lowpass_hz,
                    )
                    file_avmd_settings = {
                        **avmd_settings,
                        "mode_count": int(adaptive_selection["best_mode_count"]),
                        "adaptive_k": True,
                        "adaptive_k_score": float(adaptive_selection["best_score"]),
                        "adaptive_k_stop_reason": adaptive_selection["stop_reason"],
                        "adaptive_k_search_path": adaptive_selection["search_path"],
                    }
                    trials = adaptive_selection["trials"].copy()
                    trials.insert(0, "relative_path", relative_path)
                    adaptive_trial_tables.append(trials)
                modes, centers, method_parameters = decompose_signal(
                    method, signal, sampling_rate, file_avmd_settings
                )
                metrics = calculate_mode_metrics(modes, sampling_rate, centers)
                metrics, _ = evaluate_welch_support(
                    metrics,
                    signal,
                    sampling_rate=sampling_rate,
                    frequency_tolerance_hz=1.0,
                    minimum_frequency_hz=0.5,
                )
                reconstruction = calculate_reconstruction_metrics(
                    signal, np.sum(modes, axis=0)
                )
                dominant = metrics.loc[metrics["energy_ratio"].idxmax()]
                elapsed = time.perf_counter() - started
                metrics.insert(0, "relative_path", relative_path)
                metrics.insert(1, "file_name", metadata["file_name"])
                metrics.insert(2, "condition", metadata["condition"])
                metrics.insert(3, "analysis_type", metadata["analysis_type"])
                metrics.insert(4, "angle_deg", metadata["angle_deg"])
                metrics.insert(5, "repeat", metadata["repeat"])
                metrics.insert(6, "method", method)
                metric_tables.append(metrics)
                summaries.append(
                    {
                        **{key: value for key, value in metadata.items() if key != "file_path"},
                        "method": method,
                        "status": "ok",
                        "error_message": "",
                        "mode_count": len(modes),
                        "shared_initial_k": (
                            int(avmd_settings["mode_count"])
                            if method == "AVMD" and avmd_settings is not None
                            else np.nan
                        ),
                        "adaptive_k_used": bool(
                            method == "AVMD" and adaptive_selection is not None
                        ),
                        "adaptive_k_score": (
                            float(adaptive_selection["best_score"])
                            if adaptive_selection is not None else np.nan
                        ),
                        "adaptive_k_stop_reason": (
                            adaptive_selection["stop_reason"]
                            if adaptive_selection is not None else ""
                        ),
                        "adaptive_k_candidates_evaluated": (
                            len(adaptive_selection["evaluated_mode_counts"])
                            if adaptive_selection is not None else 0
                        ),
                        "dominant_mode": str(dominant["mode"]),
                        "dominant_mode_peak_frequency_hz": float(
                            dominant["peak_frequency_hz"]
                        ),
                        "dominant_mode_energy_ratio": float(dominant["energy_ratio"]),
                        "welch_support_ratio": float(
                            metrics["welch_psd_supported"].astype(float).mean()
                        ),
                        "elapsed_s": elapsed,
                        **reconstruction,
                    }
                )
                parameters[f"{relative_path}|{method}"] = method_parameters
                if make_per_file_plots:
                    folder = _safe_output_folder(output_root, relative_path)
                    plot_method_modes(
                        signal,
                        modes,
                        metrics,
                        sampling_rate,
                        method,
                        relative_path,
                        folder / f"02_{method}_modes.png",
                    )
            except Exception as error:
                summaries.append(
                    {
                        **{key: value for key, value in metadata.items() if key != "file_path"},
                        "method": method,
                        "status": "failed",
                        "error_message": str(error),
                        "elapsed_s": time.perf_counter() - started,
                    }
                )
    summary = pd.DataFrame(summaries)
    all_metrics = (
        pd.concat(metric_tables, ignore_index=True)
        if metric_tables
        else pd.DataFrame()
    )
    adaptive_trials = (
        pd.concat(adaptive_trial_tables, ignore_index=True)
        if adaptive_trial_tables else pd.DataFrame()
    )
    return summary, all_metrics, parameters, adaptive_trials


def _plot_condition_lines(
    table: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for condition, group in table.groupby("condition"):
        ordered = group.sort_values("angle_deg")
        axis.plot(
            ordered["angle_deg"],
            ordered[value_column],
            marker="o",
            linewidth=1.8,
            label=CONDITION_LABELS.get(condition, condition),
        )
    axis.set_xlabel("Angle (degree)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_data_quality_overview(
    preprocessing_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    successful = preprocessing_summary.loc[
        preprocessing_summary["status"].eq("ok")
    ].copy()
    issue_counts = {
        "Exact duplicates removed": int(
            successful["removed_exact_duplicate_row_count"].fillna(0).sum()
        ),
        "Invalid CH1 removed": int(
            successful["invalid_ch1_count"].fillna(0).sum()
        ),
        "Missing cells recorded": int(
            successful["missing_value_count"].fillna(0).sum()
        ),
        "Failed files": int(preprocessing_summary["status"].eq("failed").sum()),
    }
    type_counts = preprocessing_summary["analysis_type"].value_counts().sort_index()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].bar(issue_counts.keys(), issue_counts.values(), color="0.25")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Data-quality actions")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(issue_counts.values()):
        axes[0].text(index, value, str(value), ha="center", va="bottom")

    axes[1].bar(type_counts.index, type_counts.values, color="0.45")
    axes[1].set_ylabel("CSV files")
    axes[1].set_title("Archive file types")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(type_counts.values):
        axes[1].text(index, value, str(value), ha="center", va="bottom")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_aggregate_outputs(
    preprocessing_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    output_root: Path,
) -> None:
    fixed = preprocessing_summary.loc[
        preprocessing_summary["status"].eq("ok")
        & preprocessing_summary["angle_deg"].notna()
    ].copy()
    if not fixed.empty:
        angle_summary = (
            fixed.groupby(["condition", "angle_deg"], dropna=False)
            .agg(
                file_count=("relative_path", "size"),
                median_mean_wavelength_nm=("ch1_mean_nm", "median"),
                median_raw_ac_rms_nm=("raw_ac_rms_nm", "median"),
                median_processed_dynamic_rms_nm=("processed_dynamic_rms_nm", "median"),
                median_raw_dominant_frequency_hz=(
                    "raw_dominant_welch_frequency_hz", "median"
                ),
                median_processed_dominant_frequency_hz=(
                    "processed_dominant_welch_frequency_hz", "median"
                ),
            )
            .reset_index()
        )
        angle_summary.to_csv(output_root / "angle_preprocessing_summary.csv", index=False)
        _plot_condition_lines(
            angle_summary,
            "median_mean_wavelength_nm",
            "Median mean Bragg wavelength (nm)",
            "Angle versus mean Bragg wavelength after row cleaning",
            output_root / "02_angle_vs_mean_wavelength.png",
        )

        figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for condition, group in angle_summary.groupby("condition"):
            ordered = group.sort_values("angle_deg")
            label = CONDITION_LABELS.get(condition, condition)
            axes[0].plot(
                ordered["angle_deg"], ordered["median_raw_ac_rms_nm"],
                marker="o", label=label,
            )
            axes[1].plot(
                ordered["angle_deg"], ordered["median_processed_dynamic_rms_nm"],
                marker="o", label=label,
            )
        axes[0].set_title("Before: centered raw AC RMS")
        axes[1].set_title("After: 0.5–40 Hz dynamic RMS")
        for axis in axes:
            axis.set_xlabel("Angle (degree)")
            axis.set_ylabel("RMS (nm)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.savefig(
            output_root / "03_angle_vs_rms_before_after.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for condition, group in angle_summary.groupby("condition"):
            ordered = group.sort_values("angle_deg")
            label = CONDITION_LABELS.get(condition, condition)
            axes[0].plot(
                ordered["angle_deg"], ordered["median_raw_dominant_frequency_hz"],
                marker="o", label=label,
            )
            axes[1].plot(
                ordered["angle_deg"],
                ordered["median_processed_dominant_frequency_hz"],
                marker="o", label=label,
            )
        axes[0].set_title("Before preprocessing")
        axes[1].set_title("After preprocessing")
        for axis in axes:
            axis.set_xlabel("Angle (degree)")
            axis.set_ylabel("Dominant Welch frequency (Hz)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.savefig(
            output_root / "04_angle_vs_frequency_before_after.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    if method_summary.empty:
        return
    successful = method_summary.loc[method_summary["status"].eq("ok")].copy()
    fixed_modes = successful.loc[successful["angle_deg"].notna()].copy()
    if fixed_modes.empty:
        return
    avmd = fixed_modes.loc[fixed_modes["method"].eq("AVMD")].copy()
    if not avmd.empty:
        avmd_angle = (
            avmd.groupby(["condition", "angle_deg"])
            .agg(
                file_count=("relative_path", "size"),
                median_avmd_dominant_frequency_hz=(
                    "dominant_mode_peak_frequency_hz", "median"
                ),
                median_avmd_dominant_energy_ratio=(
                    "dominant_mode_energy_ratio", "median"
                ),
                median_reconstruction_correlation=(
                    "reconstruction_correlation", "median"
                ),
            )
            .reset_index()
        )
        avmd_angle.to_csv(output_root / "avmd_angle_summary.csv", index=False)
        _plot_condition_lines(
            avmd_angle,
            "median_avmd_dominant_frequency_hz",
            "AVMD dominant-mode frequency (Hz)",
            "Angle versus AVMD dominant-mode frequency",
            output_root / "05_angle_vs_avmd_dominant_frequency.png",
        )
        _plot_condition_lines(
            avmd_angle,
            "median_avmd_dominant_energy_ratio",
            "AVMD dominant-mode energy ratio",
            "Angle versus AVMD dominant-mode energy ratio",
            output_root / "06_angle_vs_avmd_energy_ratio.png",
        )

        figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        labels = []
        values = []
        for condition, group in avmd.groupby("condition"):
            labels.append(CONDITION_LABELS.get(condition, condition))
            values.append(group["reconstruction_correlation"].dropna().to_numpy())
        if values:
            axis.boxplot(values, tick_labels=labels, showfliers=True)
            axis.axhline(0.90, color="tab:red", linestyle="--", label="0.90 threshold")
            axis.set_ylabel("Reconstruction correlation")
            axis.set_title("AVMD reconstruction quality across the complete archive")
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
            figure.savefig(
                output_root / "07_avmd_reconstruction_quality.png",
                dpi=180,
                bbox_inches="tight",
            )
        plt.close(figure)

        if "adaptive_k_used" in avmd.columns and avmd["adaptive_k_used"].any():
            adaptive_summary = (
                avmd.groupby(["condition", "angle_deg"])
                .agg(
                    file_count=("relative_path", "size"),
                    median_selected_k=("mode_count", "median"),
                    minimum_selected_k=("mode_count", "min"),
                    maximum_selected_k=("mode_count", "max"),
                    median_adaptive_k_score=("adaptive_k_score", "median"),
                )
                .reset_index()
            )
            adaptive_summary.to_csv(
                output_root / "adaptive_k_angle_summary.csv", index=False
            )
            _plot_condition_lines(
                adaptive_summary,
                "median_selected_k",
                "Median selected K",
                "Per-file adaptive-K selected by angle and condition",
                output_root / "08_adaptive_k_by_angle.png",
            )


def write_summary(
    preprocessing_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    avmd_settings: dict | None,
    output_path: Path,
) -> None:
    successful_preprocessing = int(preprocessing_summary["status"].eq("ok").sum())
    failed_preprocessing = int(preprocessing_summary["status"].eq("failed").sum())
    duplicates = int(
        preprocessing_summary.get(
            "removed_exact_duplicate_row_count", pd.Series(dtype=float)
        ).fillna(0).sum()
    )
    lines = [
        "# 260720整包資料前處理與AVMD分析摘要",
        "",
        f"- CSV總數：{len(preprocessing_summary)}",
        f"- 前處理成功：{successful_preprocessing}",
        f"- 前處理失敗：{failed_preprocessing}",
        f"- 移除完全重複列：{duplicates}",
        "- 缺值或非數字列只在資料品質表中記錄，不會把偏心或高振動當成錯誤刪除。",
        "- 原始波長保留供角度與平均波長比較；0.5–40 Hz動態訊號供AVMD分析。",
    ]
    if avmd_settings is not None:
        lines.extend(
            [
                "",
                "## AVMD初始共用設定",
                "",
                f"- 參考檔：{avmd_settings['reference_file']}",
                f"- 初始K：{avmd_settings['mode_count']}",
                f"- alpha：{avmd_settings['alpha']}",
                f"- 選擇分數：{avmd_settings['selection_score']:.6f}",
            ]
        )
        if avmd_settings.get("adaptive_k_enabled", False):
            lines.append(
                "- 每個CSV再由adaptive-K迴圈依重建、模態重疊、Welch支持度與複雜度選擇K。"
            )
        else:
            lines.append("- adaptive-K已停用，全部CSV沿用參考檔的K。")
    if not method_summary.empty:
        completed = int(method_summary["status"].eq("ok").sum())
        failed = int(method_summary["status"].eq("failed").sum())
        lines.extend(
            [
                "",
                "## 模態分析執行情形",
                "",
                f"- 成功：{completed}",
                f"- 失敗：{failed}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    if arguments.sampling_rate <= 0:
        raise ValueError("sampling-rate必須大於0。")
    if not 0.5 < arguments.dynamic_lowpass_hz < arguments.sampling_rate / 2.0:
        raise ValueError("dynamic-lowpass-hz必須介於0.5 Hz與Nyquist頻率之間。")
    if arguments.adaptive_k_min < 2:
        raise ValueError("adaptive-k-min至少必須為2。")
    if arguments.adaptive_k_max < arguments.adaptive_k_min:
        raise ValueError("adaptive-k-max不得小於adaptive-k-min。")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    inventory = discover_archive_csvs(arguments.input_dir)
    if arguments.max_files is not None:
        inventory = inventory[: arguments.max_files]
    print(f"找到 {len(inventory)} 個CSV。", flush=True)
    pd.DataFrame(
        [{key: value for key, value in item.items() if key != "file_path"} for item in inventory]
    ).to_csv(arguments.output_dir / "all_files_inventory.csv", index=False)

    preprocessing_summary = analyze_preprocessing(
        inventory,
        arguments.sampling_rate,
        arguments.output_dir,
        make_per_file_plots=not arguments.skip_per_file_plots,
        dynamic_lowpass_hz=arguments.dynamic_lowpass_hz,
    )
    preprocessing_summary.to_csv(
        arguments.output_dir / "preprocessing_before_after.csv", index=False
    )
    plot_data_quality_overview(
        preprocessing_summary,
        arguments.output_dir / "01_data_quality_overview.png",
    )

    if arguments.preprocess_only:
        create_aggregate_outputs(
            preprocessing_summary, pd.DataFrame(), arguments.output_dir
        )
        write_summary(
            preprocessing_summary,
            pd.DataFrame(),
            None,
            arguments.output_dir / "ANALYSIS_SUMMARY.md",
        )
        print(f"整包前處理完成：{arguments.output_dir}", flush=True)
        return

    inventory_by_relative_path = {
        item["relative_path"]: item for item in inventory
    }
    avmd_settings = None
    if "AVMD" in arguments.methods:
        print("從正常無負載10度代表檔選擇共用AVMD參數...", flush=True)
        avmd_settings = prepare_avmd_settings(
            preprocessing_summary,
            inventory_by_relative_path,
            arguments.sampling_rate,
            arguments.avmd_k,
            arguments.avmd_alpha,
            arguments.output_dir,
            arguments.dynamic_lowpass_hz,
        )
        avmd_settings["adaptive_k_enabled"] = not arguments.disable_adaptive_k

    valid_paths = set(
        preprocessing_summary.loc[
            preprocessing_summary["status"].eq("ok"), "relative_path"
        ].astype(str)
    )
    method_summary, mode_metrics, parameters, adaptive_k_trials = run_mode_analysis(
        inventory,
        valid_paths,
        arguments.methods,
        arguments.sampling_rate,
        avmd_settings,
        arguments.output_dir,
        make_per_file_plots=not arguments.skip_per_file_plots,
        dynamic_lowpass_hz=arguments.dynamic_lowpass_hz,
        adaptive_k_enabled=not arguments.disable_adaptive_k,
        adaptive_k_min=arguments.adaptive_k_min,
        adaptive_k_max=arguments.adaptive_k_max,
        adaptive_k_steps=arguments.adaptive_k_steps,
    )
    method_summary.to_csv(arguments.output_dir / "method_file_summary.csv", index=False)
    mode_metrics.to_csv(arguments.output_dir / "all_mode_metrics.csv", index=False)
    adaptive_k_trials.to_csv(
        arguments.output_dir / "adaptive_k_trials.csv", index=False
    )

    if avmd_settings is not None:
        parameters["AVMD_SHARED_SELECTION"] = avmd_settings
    (arguments.output_dir / "all_method_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    create_aggregate_outputs(
        preprocessing_summary, method_summary, arguments.output_dir
    )
    settings = {
        "input_dir": str(arguments.input_dir),
        "output_dir": str(arguments.output_dir),
        "sampling_rate_hz": arguments.sampling_rate,
        "methods": arguments.methods,
        "file_count": len(inventory),
        "remove_exact_duplicates": True,
        "dynamic_highpass_hz": 0.5,
        "dynamic_lowpass_hz": arguments.dynamic_lowpass_hz,
        "angle_lowpass_hz": 0.1,
        "avmd_candidate_k": arguments.avmd_k,
        "avmd_candidate_alpha": arguments.avmd_alpha,
        "adaptive_k_enabled": not arguments.disable_adaptive_k,
        "adaptive_k_min": arguments.adaptive_k_min,
        "adaptive_k_max": arguments.adaptive_k_max,
        "adaptive_k_steps": arguments.adaptive_k_steps,
    }
    (arguments.output_dir / "analysis_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(
        preprocessing_summary,
        method_summary,
        avmd_settings,
        arguments.output_dir / "ANALYSIS_SUMMARY.md",
    )
    print(f"整包分析完成：{arguments.output_dir}", flush=True)


if __name__ == "__main__":
    main()
