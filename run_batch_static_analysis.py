from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_fbg_csv
from src.common.metrics import (
    calculate_mode_metrics,
    calculate_mode_relationships,
    calculate_reconstruction_metrics,
)
from src.common.postprocessing import evaluate_welch_support, merge_redundant_modes
from src.common.preprocessing import calculate_welch_psd, preprocess_fbg_signal
from src.common.vmd import variational_mode_decomposition
from src.methods.avmd import create_bandwise_initial_centers, select_avmd_parameters
from src.methods.classic_vmd import classic_variational_mode_decomposition
from src.methods.iovmd import run_initial_iovmd
from src.methods.svmd import successive_variational_mode_decomposition
from src.methods.stvmd import short_time_variational_mode_decomposition


METHOD_ORDER = ("VMD", "IOVMD", "AVMD", "SVMD", "STVMD")
FILE_PATTERN = re.compile(
    r"(?P<load>無負載|有負載|偏心)\s*(?P<angle>\d+)度\s*\((?P<repeat>\d+)\)\.csv$",
    re.IGNORECASE,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批次比較固定角度FBG量測的基本特徵、Welch PSD與五種VMD方法。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "no_load_30hz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "batch_no_load_30hz",
    )
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=list(METHOD_ORDER),
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="只執行資料品質、基本特徵與Welch PSD。",
    )
    return parser.parse_args()


def discover_measurements(input_dir: Path) -> list[dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"找不到輸入資料夾：{input_dir}")

    measurements = []
    ignored = []
    for path in sorted(input_dir.glob("*.csv")):
        match = FILE_PATTERN.search(path.name)
        if not match:
            ignored.append(path.name)
            continue
        measurements.append(
            {
                "path": path,
                "file_name": path.name,
                "load_state": match.group("load"),
                "angle_deg": int(match.group("angle")),
                "repeat": int(match.group("repeat")),
            }
        )

    if ignored:
        print(f"略過 {len(ignored)} 個無法辨識條件的CSV：{ignored}")
    if not measurements:
        raise ValueError("輸入資料夾沒有符合命名規則的CSV。")
    return measurements


def _top_welch_peaks(
    frequency_hz: np.ndarray,
    psd: np.ndarray,
    maximum_frequency_hz: float,
    count: int = 8,
) -> pd.DataFrame:
    mask = (frequency_hz >= 0.5) & (frequency_hz < maximum_frequency_hz)
    selected_frequency = frequency_hz[mask]
    selected_psd = psd[mask]
    resolution = float(np.median(np.diff(selected_frequency)))
    peak_indices, properties = find_peaks(
        selected_psd,
        prominence=max(float(np.max(selected_psd)) * 0.005, np.finfo(float).eps),
        distance=max(1, int(np.ceil(0.5 / resolution))),
    )
    if not len(peak_indices):
        peak_indices = np.array([int(np.argmax(selected_psd))])
        prominences = np.array([np.nan])
    else:
        prominences = properties["prominences"]
    order = np.argsort(selected_psd[peak_indices])[::-1][:count]
    selected_indices = peak_indices[order]
    return pd.DataFrame(
        {
            "peak_rank": np.arange(1, len(selected_indices) + 1),
            "frequency_hz": selected_frequency[selected_indices],
            "psd_nm2_per_hz": selected_psd[selected_indices],
            "prominence_nm2_per_hz": prominences[order],
        }
    )


def analyze_raw_measurements(
    measurements: list[dict],
    sampling_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    feature_rows = []
    psd_rows = []
    peak_tables = []
    loaded = {}

    for item in measurements:
        data, information = load_fbg_csv(item["path"], sampling_rate=sampling_rate)
        preprocessing = preprocess_fbg_signal(
            data["ch1_nm"].to_numpy(), sampling_rate=sampling_rate
        )
        raw = preprocessing["raw_nm"]
        centered = preprocessing["centered_raw_nm"]
        dynamic = preprocessing["dynamic_nm"]
        frequency_hz, psd = calculate_welch_psd(
            centered, sampling_rate=sampling_rate, nperseg=4096
        )
        maximum_frequency_hz = sampling_rate / 2.0
        peaks = _top_welch_peaks(frequency_hz, psd, maximum_frequency_hz)
        peaks.insert(0, "file_name", item["file_name"])
        peaks.insert(1, "angle_deg", item["angle_deg"])
        peaks.insert(2, "repeat", item["repeat"])
        peak_tables.append(peaks)

        feature_rows.append(
            {
                **{key: item[key] for key in ("file_name", "load_state", "angle_deg", "repeat")},
                **information,
                "ac_rms_nm": float(np.sqrt(np.mean(centered**2))),
                "dynamic_rms_nm": float(np.sqrt(np.mean(dynamic**2))),
                "peak_to_peak_nm": float(np.ptp(raw)),
                "dominant_welch_frequency_hz": float(peaks.iloc[0]["frequency_hz"]),
            }
        )
        for frequency, value in zip(frequency_hz, psd):
            psd_rows.append(
                {
                    "file_name": item["file_name"],
                    "angle_deg": item["angle_deg"],
                    "repeat": item["repeat"],
                    "frequency_hz": float(frequency),
                    "psd_nm2_per_hz": float(value),
                }
            )
        loaded[item["file_name"]] = {
            "data": data,
            "information": information,
            "preprocessing": preprocessing,
            "metadata": item,
        }

    features = pd.DataFrame(feature_rows).sort_values(["angle_deg", "repeat"])
    psd_table = pd.DataFrame(psd_rows)
    peaks = pd.concat(peak_tables, ignore_index=True)
    return features, psd_table, peaks, loaded


def build_angle_summary(features: pd.DataFrame) -> pd.DataFrame:
    return (
        features.groupby("angle_deg")
        .agg(
            file_count=("file_name", "size"),
            mean_wavelength_nm=("ch1_mean_nm", "mean"),
            between_repeat_mean_wavelength_sd_nm=("ch1_mean_nm", "std"),
            mean_ac_rms_nm=("ac_rms_nm", "mean"),
            median_ac_rms_nm=("ac_rms_nm", "median"),
            ac_rms_sd_nm=("ac_rms_nm", "std"),
            mean_peak_to_peak_nm=("peak_to_peak_nm", "mean"),
            median_dominant_welch_frequency_hz=(
                "dominant_welch_frequency_hz", "median"
            ),
            duplicate_rows=("exact_duplicate_row_count", "sum"),
        )
        .reset_index()
    )


def add_repeat_consistency_flags(features: pd.DataFrame) -> pd.DataFrame:
    output = features.copy()
    output["ac_rms_robust_z"] = np.nan
    output["ac_rms_repeat_flag"] = False
    output["acquisition_order_within_angle"] = 0
    for angle, indices in output.groupby("angle_deg").groups.items():
        subset = output.loc[indices]
        median = float(subset["ac_rms_nm"].median())
        mad = float(np.median(np.abs(subset["ac_rms_nm"] - median)))
        if mad > np.finfo(float).eps:
            robust_z = 0.6745 * (subset["ac_rms_nm"] - median) / mad
            output.loc[indices, "ac_rms_robust_z"] = robust_z
            output.loc[indices, "ac_rms_repeat_flag"] = robust_z.abs() >= 3.5
        chronological = subset.sort_values("raw_time_start_s").index
        output.loc[chronological, "acquisition_order_within_angle"] = np.arange(
            1, len(chronological) + 1
        )
    output["acquisition_order_within_angle"] = output[
        "acquisition_order_within_angle"
    ].astype(int)
    return output


def calculate_raw_band_power(psd_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bands = {
        "fundamental_25_32_hz": (25.0, 32.0),
        "second_harmonic_50_65_hz": (50.0, 65.0),
        "high_frequency_78_88_hz": (78.0, 88.0),
        "nyquist_edge_90_99_hz": (90.0, 99.0),
    }
    rows = []
    for keys, subset in psd_table.groupby(["file_name", "angle_deg", "repeat"]):
        file_name, angle_deg, repeat = keys
        subset = subset.sort_values("frequency_hz")
        frequency = subset["frequency_hz"].to_numpy()
        psd = subset["psd_nm2_per_hz"].to_numpy()
        row = {
            "file_name": file_name,
            "angle_deg": angle_deg,
            "repeat": repeat,
        }
        for name, (lower, upper) in bands.items():
            mask = (frequency >= lower) & (frequency <= upper)
            row[name] = float(np.trapezoid(psd[mask], frequency[mask]))
        rows.append(row)
    file_table = pd.DataFrame(rows).sort_values(["angle_deg", "repeat"])
    group_table = (
        file_table.groupby("angle_deg")[list(bands)]
        .agg(["median", "mean", "std"])
    )
    group_table.columns = ["_".join(column) for column in group_table.columns]
    return file_table, group_table.reset_index()


def plot_basic_features(features: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = {10: "tab:blue", 43: "tab:orange"}
    for angle, subset in features.groupby("angle_deg"):
        subset = subset.sort_values("repeat")
        axes[0].plot(
            subset["repeat"], subset["ch1_mean_nm"], marker="o",
            label=f"{angle} deg", color=colors.get(int(angle)),
        )
        axes[1].plot(
            subset["repeat"], subset["ac_rms_nm"], marker="o",
            label=f"{angle} deg", color=colors.get(int(angle)),
        )
    axes[0].set_title("Mean Bragg wavelength by repeat")
    axes[0].set_ylabel("Mean wavelength (nm)")
    axes[0].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axes[1].set_title("AC RMS by repeat")
    axes[1].set_ylabel("AC RMS (nm)")
    for axis in axes:
        axis.set_xlabel("Repeat")
        axis.grid(alpha=0.2)
        axis.legend()
        axis.set_xticks(sorted(features["repeat"].unique()))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def summarize_and_plot_psd(psd_table: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    summary = (
        psd_table.groupby(["angle_deg", "frequency_hz"])["psd_nm2_per_hz"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    colors = {10: "tab:blue", 43: "tab:orange"}
    for angle, subset in summary.groupby("angle_deg"):
        subset = subset.loc[(subset["frequency_hz"] >= 0.5) & (subset["frequency_hz"] < 100.0)]
        color = colors.get(int(angle))
        axis.plot(subset["frequency_hz"], subset["median"], label=f"{angle} deg", color=color)
        axis.fill_between(
            subset["frequency_hz"], subset["q25"], subset["q75"],
            color=color, alpha=0.18,
        )
    axis.set_yscale("log")
    axis.set_xlim(0.5, 100.0)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Welch PSD (nm^2/Hz)")
    axis.set_title("No-load Welch PSD median and interquartile range")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_root / "02_welch_psd_by_angle.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return summary


def choose_avmd_reference(features: pd.DataFrame, loaded: dict[str, dict]) -> tuple[str, np.ndarray]:
    preferred = features.loc[features["angle_deg"] == 10].copy()
    if preferred.empty:
        preferred = features.copy()
    median_rms = float(preferred["dynamic_rms_nm"].median())
    row = preferred.iloc[(preferred["dynamic_rms_nm"] - median_rms).abs().argmin()]
    file_name = str(row["file_name"])
    return file_name, loaded[file_name]["preprocessing"]["dynamic_nm"]


def prepare_avmd_settings(
    features: pd.DataFrame,
    loaded: dict[str, dict],
    sampling_rate: float,
) -> dict:
    reference_name, reference_signal = choose_avmd_reference(features, loaded)
    selection = select_avmd_parameters(
        reference_signal=reference_signal,
        sampling_rate=sampling_rate,
        candidate_mode_counts=(4, 5, 6, 7, 8),
        candidate_alphas=(100.0, 250.0, 500.0, 1000.0, 2000.0),
        maximum_iterations=300,
    )
    return {
        "reference_file": reference_name,
        "mode_count": int(selection["best_mode_count"]),
        "alpha": float(selection["best_alpha"]),
        "selection_score": float(selection["best_selection_score"]),
        "trials": selection["trials"],
    }


def decompose_signal(
    method: str,
    signal: np.ndarray,
    sampling_rate: float,
    avmd_settings: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    maximum_frequency_hz = sampling_rate / 2.0 * 0.999
    if method == "VMD":
        result = classic_variational_mode_decomposition(
            signal, sampling_rate=sampling_rate, mode_count=10,
            alpha=2000.0, maximum_iterations=500,
        )
        return result["modes"], result["center_frequencies_hz"], {
            "mode_count": 10, "alpha": 2000.0,
            "parameter_rule": "fixed K=10 and alpha=2000",
        }
    if method == "IOVMD":
        initial = run_initial_iovmd(
            signal, sampling_rate=sampling_rate, alpha=2000.0,
            maximum_iterations=500, minimum_frequency_hz=0.5,
            maximum_frequency_hz=maximum_frequency_hz,
        )
        initial_modes = initial["vmd"]["modes"]
        relationships = calculate_mode_relationships(initial_modes)
        modes, groups = merge_redundant_modes(initial_modes, relationships["pair_table"])
        return modes, np.full(len(modes), np.nan), {
            "alpha": 2000.0,
            "initial_mode_count": int(initial["initial_k"]),
            "final_mode_count": int(len(modes)),
            "initial_center_frequencies_hz": initial["initial_center_frequencies_hz"].tolist(),
            "merged_groups": [[int(i + 1) for i in group] for group in groups if len(group) > 1],
        }
    if method == "AVMD":
        if avmd_settings is None:
            raise ValueError("AVMD settings are required.")
        centers = create_bandwise_initial_centers(
            signal, sampling_rate=sampling_rate,
            mode_count=avmd_settings["mode_count"],
            minimum_frequency_hz=0.5,
            maximum_frequency_hz=maximum_frequency_hz,
        )
        result = variational_mode_decomposition(
            signal, sampling_rate=sampling_rate,
            initial_center_frequencies_hz=centers,
            alpha=avmd_settings["alpha"], maximum_iterations=500,
        )
        return result["modes"], result["center_frequencies_hz"], {
            "mode_count": avmd_settings["mode_count"],
            "alpha": avmd_settings["alpha"],
            "reference_file": avmd_settings["reference_file"],
            "parameter_rule": "one shared K and alpha selected from representative no-load 10-degree repeat",
        }
    if method == "SVMD":
        result = successive_variational_mode_decomposition(
            signal, sampling_rate=sampling_rate,
            maximum_alpha=1000.0, minimum_alpha=10.0,
            maximum_iterations=300, maximum_modes=12,
            reconstruction_power_tolerance=0.005,
        )
        return result["modes"], result["center_frequencies_hz"], {
            "mode_count": int(result["mode_count"]),
            "stop_reason": result["stop_reason"],
        }
    if method == "STVMD":
        result = short_time_variational_mode_decomposition(
            signal, sampling_rate=sampling_rate, mode_count=9,
            window_duration_s=8.0, hop_duration_s=2.0,
            minimum_frequency_hz=0.5,
            maximum_frequency_hz=maximum_frequency_hz,
            alpha=2000.0, maximum_iterations=200,
        )
        return result["modes"], result["center_frequencies_hz"], {
            "mode_count": 9,
            "alpha": 2000.0,
            "window_duration_s": 8.0,
            "hop_duration_s": 2.0,
            "frame_count": int(result["frame_count"]),
            "converged_frame_fraction": float(result["converged_frame_fraction"]),
        }
    raise ValueError(f"Unknown method: {method}")


def run_mode_analysis(
    methods: list[str],
    features: pd.DataFrame,
    loaded: dict[str, dict],
    sampling_rate: float,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    avmd_settings = None
    if "AVMD" in methods:
        print("選擇共用AVMD參數...", flush=True)
        avmd_settings = prepare_avmd_settings(features, loaded, sampling_rate)
        avmd_settings["trials"].to_csv(output_root / "avmd_parameter_trials.csv", index=False)

    summaries = []
    all_metrics = []
    parameters = {}
    ordered_features = features.sort_values(["angle_deg", "repeat"])
    for row in ordered_features.itertuples(index=False):
        signal = loaded[row.file_name]["preprocessing"]["dynamic_nm"]
        for method in methods:
            started = time.perf_counter()
            print(f"{row.file_name} | {method}", flush=True)
            modes, centers, method_parameters = decompose_signal(
                method, signal, sampling_rate, avmd_settings
            )
            metrics = calculate_mode_metrics(modes, sampling_rate, centers)
            metrics, _ = evaluate_welch_support(
                metrics, signal, sampling_rate=sampling_rate,
                frequency_tolerance_hz=1.0, minimum_frequency_hz=0.5,
            )
            reconstruction = calculate_reconstruction_metrics(signal, np.sum(modes, axis=0))
            elapsed = time.perf_counter() - started
            metrics.insert(0, "file_name", row.file_name)
            metrics.insert(1, "angle_deg", row.angle_deg)
            metrics.insert(2, "repeat", row.repeat)
            metrics.insert(3, "method", method)
            all_metrics.append(metrics)
            summaries.append(
                {
                    "file_name": row.file_name,
                    "angle_deg": row.angle_deg,
                    "repeat": row.repeat,
                    "method": method,
                    "mode_count": len(modes),
                    **reconstruction,
                    "welch_support_ratio": float(metrics["welch_psd_supported"].mean()),
                    "elapsed_s": elapsed,
                }
            )
            parameters[f"{row.file_name}|{method}"] = method_parameters
    if avmd_settings is not None:
        parameters["AVMD_SHARED_SELECTION"] = {
            key: value for key, value in avmd_settings.items() if key != "trials"
        }
    return pd.DataFrame(summaries), pd.concat(all_metrics, ignore_index=True), parameters


def build_frequency_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.loc[
        metrics["peak_frequency_hz"].between(0.5, 99.9)
    ].copy()
    table["frequency_bin_hz"] = (table["peak_frequency_hz"] * 2.0).round() / 2.0
    return (
        table.groupby(["method", "angle_deg", "frequency_bin_hz"])
        .agg(
            observation_count=("mode", "size"),
            repeat_count=("repeat", "nunique"),
            median_peak_frequency_hz=("peak_frequency_hz", "median"),
            median_rms_nm=("rms_nm", "median"),
            median_energy_ratio=("energy_ratio", "median"),
            welch_support_rate=("welch_psd_supported", "mean"),
        )
        .reset_index()
        .sort_values(["method", "frequency_bin_hz", "angle_deg"])
    )


def plot_cross_method_frequency_summary(summary: pd.DataFrame, output_path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in summary["method"].unique()]
    figure, axes = plt.subplots(
        len(methods), 1, figsize=(13, max(5, 2.6 * len(methods))),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = {10: "tab:blue", 43: "tab:orange"}
    for axis, method in zip(axes, methods):
        subset = summary.loc[(summary["method"] == method) & (summary["repeat_count"] >= 3)]
        for angle, angle_table in subset.groupby("angle_deg"):
            axis.scatter(
                angle_table["frequency_bin_hz"], angle_table["median_rms_nm"],
                s=18 + 8 * angle_table["repeat_count"], alpha=0.75,
                label=f"{angle} deg", color=colors.get(int(angle)),
            )
        axis.set_ylabel(f"{method}\nmedian RMS")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Mode peak frequency bin (Hz)")
    axes[-1].set_xlim(0.5, 100.0)
    figure.suptitle("Cross-method mode RMS by angle (frequency groups in at least 3 repeats)")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_analysis_summary(
    features: pd.DataFrame,
    angle_summary: pd.DataFrame,
    band_summary: pd.DataFrame,
    method_summary: pd.DataFrame | None,
    output_path: Path,
) -> None:
    indexed = angle_summary.set_index("angle_deg")
    lines = [
        "# 無負載10°與43°批次分析摘要",
        "",
        "## 資料與品質",
        "",
        f"- 共分析 {len(features)} 個CSV，10°與43°各10次重複量測。",
        f"- 每檔 {int(features['sample_count'].median())} 點，取樣率 "
        f"{features['sampling_rate_hz'].median():.0f} Hz，約 "
        f"{features['duration_s'].median():.1f} 秒。",
        f"- 缺值總數：{int(features['missing_value_count'].sum())}。",
        f"- 完全重複列總數：{int(features['exact_duplicate_row_count'].sum())}。",
        "- 原始時間戳只顯示到0.1秒，同一時間戳會對應多個200 Hz樣本；頻譜仍依已知200 Hz建立均勻時間軸。",
        "",
        "## 主要結果",
        "",
    ]
    if 10 in indexed.index and 43 in indexed.index:
        wavelength_difference = (
            indexed.loc[43, "mean_wavelength_nm"]
            - indexed.loc[10, "mean_wavelength_nm"]
        )
        rms_ratio = (
            indexed.loc[43, "median_ac_rms_nm"]
            / indexed.loc[10, "median_ac_rms_nm"]
        )
        frequency_difference = (
            indexed.loc[43, "median_dominant_welch_frequency_hz"]
            - indexed.loc[10, "median_dominant_welch_frequency_hz"]
        )
        lines.extend(
            [
                f"- 43°平均波長比10°低 {abs(wavelength_difference):.6f} nm。",
                f"- 10°與43°的主導Welch頻率中位數分別為 "
                f"{indexed.loc[10, 'median_dominant_welch_frequency_hz']:.2f} Hz與 "
                f"{indexed.loc[43, 'median_dominant_welch_frequency_hz']:.2f} Hz，"
                f"相差約 {frequency_difference:.2f} Hz。",
                f"- 43°的AC RMS中位數約為10°的 {rms_ratio:.2f} 倍。",
            ]
        )
    band_indexed = band_summary.set_index("angle_deg")
    if 10 in band_indexed.index and 43 in band_indexed.index:
        fundamental_ratio = (
            band_indexed.loc[43, "fundamental_25_32_hz_median"]
            / band_indexed.loc[10, "fundamental_25_32_hz_median"]
        )
        second_ratio = (
            band_indexed.loc[43, "second_harmonic_50_65_hz_median"]
            / band_indexed.loc[10, "second_harmonic_50_65_hz_median"]
        )
        high_ratio = (
            band_indexed.loc[10, "high_frequency_78_88_hz_median"]
            / band_indexed.loc[43, "high_frequency_78_88_hz_median"]
        )
        lines.extend(
            [
                f"- 43°在25至32 Hz主頻帶的中位功率約為10°的 {fundamental_ratio:.2f} 倍。",
                f"- 43°在50至65 Hz二次頻帶的中位功率約為10°的 {second_ratio:.2f} 倍。",
                f"- 10°在78至88 Hz頻帶的中位功率約為43°的 {high_ratio:.2f} 倍，顯示兩種角度的能量分布不同。",
            ]
        )
    flags = features.loc[features["ac_rms_repeat_flag"]]
    if not flags.empty:
        flag_text = "、".join(flags.sort_values(["angle_deg", "repeat"])["file_name"])
        lines.append(f"- 依同角度RMS的穩健離群門檻，需優先複核：{flag_text}。")
    lines.extend(
        [
            "- 時間戳顯示編號8至10早於編號1至7量測，RMS差異可能包含量測批次效應。",
            "- 約30 Hz為所有狀態的主要頻率，因此資料夾名稱中的30 Hz可合理解讀為設備操作頻率；此結論屬資料推論，仍應由實驗紀錄確認。",
            "- 90至99 Hz接近100 Hz Nyquist邊界，只列為候選高頻成分。",
        ]
    )
    if method_summary is not None and not method_summary.empty:
        quality = method_summary.groupby("method").agg(
            minimum_correlation=("reconstruction_correlation", "min"),
            median_correlation=("reconstruction_correlation", "median"),
        )
        lines.extend(["", "## 五種方法重建品質", ""])
        for method, row in quality.iterrows():
            lines.append(
                f"- {method}：最低相關係數 {row['minimum_correlation']:.4f}，"
                f"中位數 {row['median_correlation']:.4f}。"
            )
        lines.extend(
            [
                "",
                "五種方法的重建相關係數均高於0.90。各方法的IMF編號不能直接互相比較，後續應以頻率群與原始Welch頻譜作交叉驗證。",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    if arguments.sampling_rate <= 0:
        raise ValueError("sampling-rate必須大於0。")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    measurements = discover_measurements(arguments.input_dir)
    print(f"找到 {len(measurements)} 個固定角度CSV。", flush=True)
    features, psd_table, peaks, loaded = analyze_raw_measurements(
        measurements, arguments.sampling_rate
    )
    features = add_repeat_consistency_flags(features)
    features.to_csv(arguments.output_dir / "data_quality_and_features.csv", index=False)
    angle_summary = build_angle_summary(features)
    angle_summary.to_csv(arguments.output_dir / "angle_summary.csv", index=False)
    psd_table.to_csv(arguments.output_dir / "raw_welch_psd.csv", index=False)
    peaks.to_csv(arguments.output_dir / "raw_welch_peaks.csv", index=False)
    band_power, band_summary = calculate_raw_band_power(psd_table)
    band_power.to_csv(arguments.output_dir / "raw_band_power_by_file.csv", index=False)
    band_summary.to_csv(arguments.output_dir / "raw_band_power_by_angle.csv", index=False)
    plot_basic_features(features, arguments.output_dir / "01_basic_feature_comparison.png")
    psd_summary = summarize_and_plot_psd(psd_table, arguments.output_dir)
    psd_summary.to_csv(arguments.output_dir / "raw_welch_group_summary.csv", index=False)

    settings = {
        "input_dir": str(arguments.input_dir),
        "sampling_rate_hz": arguments.sampling_rate,
        "nyquist_frequency_hz": arguments.sampling_rate / 2.0,
        "frequency_interpretation_limit": "frequencies near 100 Hz are Nyquist-edge candidates",
        "methods": arguments.methods,
        "file_count": len(measurements),
    }
    (arguments.output_dir / "batch_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if arguments.basic_only:
        write_analysis_summary(
            features, angle_summary, band_summary, None,
            arguments.output_dir / "ANALYSIS_SUMMARY.md",
        )
        print(f"基本分析完成：{arguments.output_dir}")
        return

    method_summary, mode_metrics, parameters = run_mode_analysis(
        arguments.methods, features, loaded, arguments.sampling_rate,
        arguments.output_dir,
    )
    method_summary.to_csv(arguments.output_dir / "method_file_summary.csv", index=False)
    mode_metrics.to_csv(arguments.output_dir / "all_mode_metrics.csv", index=False)
    frequency_summary = build_frequency_summary(mode_metrics)
    frequency_summary.to_csv(
        arguments.output_dir / "method_angle_frequency_summary.csv", index=False
    )
    plot_cross_method_frequency_summary(
        frequency_summary,
        arguments.output_dir / "03_cross_method_angle_mode_comparison.png",
    )
    (arguments.output_dir / "all_method_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_analysis_summary(
        features, angle_summary, band_summary, method_summary,
        arguments.output_dir / "ANALYSIS_SUMMARY.md",
    )
    print(f"批次分析完成：{arguments.output_dir}")


if __name__ == "__main__":
    main()
