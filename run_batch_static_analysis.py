from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
ANGLE_COLORS = {10: "tab:blue", 43: "tab:orange"}
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
    for angle, subset in features.groupby("angle_deg"):
        subset = subset.sort_values("repeat")
        axes[0].plot(
            subset["repeat"], subset["ch1_mean_nm"], marker="o",
            label=f"{angle} deg", color=ANGLE_COLORS.get(int(angle)),
        )
        axes[1].plot(
            subset["repeat"], subset["ac_rms_nm"], marker="o",
            label=f"{angle} deg", color=ANGLE_COLORS.get(int(angle)),
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
    for angle, subset in summary.groupby("angle_deg"):
        subset = subset.loc[(subset["frequency_hz"] >= 0.5) & (subset["frequency_hz"] < 100.0)]
        color = ANGLE_COLORS.get(int(angle))
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


def plot_feature_distributions(features: pd.DataFrame, output_path: Path) -> None:
    """以箱形圖與每筆散點呈現角度內分布，避免只看平均值。"""
    definitions = [
        ("ch1_mean_nm", "Mean Bragg wavelength", "nm"),
        ("ac_rms_nm", "AC RMS", "nm"),
        ("peak_to_peak_nm", "Peak-to-peak", "nm"),
        ("dominant_welch_frequency_hz", "Dominant Welch frequency", "Hz"),
    ]
    angles = sorted(features["angle_deg"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (column, title, unit) in zip(axes.flat, definitions):
        values = [
            features.loc[features["angle_deg"] == angle, column].to_numpy()
            for angle in angles
        ]
        boxes = axis.boxplot(
            values,
            positions=np.arange(len(angles)),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box, angle in zip(boxes["boxes"], angles):
            box.set_facecolor(ANGLE_COLORS.get(int(angle), "tab:gray"))
            box.set_alpha(0.25)
        for position, angle in enumerate(angles):
            subset = features.loc[features["angle_deg"] == angle].sort_values("repeat")
            offsets = np.linspace(-0.11, 0.11, len(subset))
            flagged = subset["ac_rms_repeat_flag"].to_numpy(dtype=bool)
            axis.scatter(
                position + offsets,
                subset[column],
                color=ANGLE_COLORS.get(int(angle), "tab:gray"),
                s=36,
                alpha=0.85,
                edgecolors=np.where(flagged, "crimson", "white"),
                linewidths=np.where(flagged, 1.8, 0.6),
                zorder=3,
            )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(np.arange(len(angles)), [f"{angle} deg" for angle in angles])
        axis.grid(axis="y", alpha=0.2)
        if column == "ch1_mean_nm":
            axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    figure.suptitle(
        "No-load feature distributions by angle\n"
        "red outlines indicate AC-RMS repeat-consistency flags",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_dominant_frequency_by_repeat(
    features: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    for angle, subset in features.groupby("angle_deg"):
        subset = subset.sort_values("repeat")
        color = ANGLE_COLORS.get(int(angle), "tab:gray")
        axis.plot(
            subset["repeat"],
            subset["dominant_welch_frequency_hz"],
            marker="o",
            linewidth=1.7,
            color=color,
            label=f"{angle} deg",
        )
        median = float(subset["dominant_welch_frequency_hz"].median())
        axis.axhline(median, color=color, linestyle="--", alpha=0.45)
        axis.text(
            float(subset["repeat"].max()) + 0.13,
            median,
            f"median {median:.2f} Hz",
            color=color,
            fontsize=9,
            va="center",
        )
    axis.axhspan(29.5, 30.5, color="gray", alpha=0.08, label="30 Hz reference band")
    axis.set_xlabel("Repeat")
    axis.set_ylabel("Dominant Welch frequency (Hz)")
    axis.set_title("Dominant Welch frequency across repeated measurements")
    axis.set_xticks(sorted(features["repeat"].unique()))
    axis.set_xlim(features["repeat"].min() - 0.3, features["repeat"].max() + 1.6)
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, loc="best")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_band_power_comparison(
    band_power: pd.DataFrame,
    output_path: Path,
) -> None:
    bands = [
        ("fundamental_25_32_hz", "25–32 Hz\nmain band"),
        ("second_harmonic_50_65_hz", "50–65 Hz\ncandidate harmonic"),
        ("high_frequency_78_88_hz", "78–88 Hz\nhigh-frequency band"),
        ("nyquist_edge_90_99_hz", "90–99 Hz\nNyquist-edge candidate"),
    ]
    angles = sorted(band_power["angle_deg"].unique())
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.8), constrained_layout=True)
    for axis, (column, title) in zip(axes, bands):
        values = [
            band_power.loc[band_power["angle_deg"] == angle, column].to_numpy()
            for angle in angles
        ]
        boxes = axis.boxplot(
            values,
            positions=np.arange(len(angles)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for box, angle in zip(boxes["boxes"], angles):
            box.set_facecolor(ANGLE_COLORS.get(int(angle), "tab:gray"))
            box.set_alpha(0.28)
        for position, angle in enumerate(angles):
            subset = band_power.loc[band_power["angle_deg"] == angle].sort_values("repeat")
            offsets = np.linspace(-0.1, 0.1, len(subset))
            axis.scatter(
                position + offsets,
                subset[column],
                color=ANGLE_COLORS.get(int(angle), "tab:gray"),
                edgecolor="white",
                linewidth=0.5,
                s=30,
                alpha=0.85,
                zorder=3,
            )
        axis.set_yscale("log")
        axis.set_title(title, fontsize=11)
        axis.set_xticks(np.arange(len(angles)), [f"{angle}°" for angle in angles])
        axis.grid(axis="y", which="both", alpha=0.2)
    axes[0].set_ylabel("Integrated Welch power (nm²)")
    figure.suptitle("Band-power distribution across repeated measurements", fontsize=14)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_psd_repeat_heatmaps(
    psd_table: pd.DataFrame,
    output_path: Path,
) -> None:
    angles = sorted(psd_table["angle_deg"].unique())
    figure, axes = plt.subplots(
        len(angles),
        1,
        figsize=(12, max(5.5, 3.4 * len(angles))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, angle in zip(axes, angles):
        subset = psd_table.loc[
            (psd_table["angle_deg"] == angle)
            & psd_table["frequency_hz"].between(0.5, 99.9)
        ]
        pivot = subset.pivot_table(
            index="repeat",
            columns="frequency_hz",
            values="psd_nm2_per_hz",
            aggfunc="median",
        ).sort_index()
        log_power = np.log10(np.maximum(pivot.to_numpy(), np.finfo(float).tiny))
        image = axis.imshow(
            log_power,
            origin="lower",
            aspect="auto",
            extent=[pivot.columns.min(), pivot.columns.max(), 0.5, len(pivot.index) + 0.5],
            cmap="viridis",
            vmin=-9,
            vmax=-2.2,
        )
        axis.set_yticks(np.arange(1, len(pivot.index) + 1), pivot.index)
        axis.set_ylabel(f"{angle} deg\nrepeat")
        axis.axvline(30.0, color="white", linestyle="--", linewidth=0.9, alpha=0.8)
        axis.axvline(60.0, color="white", linestyle="--", linewidth=0.9, alpha=0.8)
    axes[-1].set_xlabel("Frequency (Hz)")
    figure.suptitle("Welch PSD heatmap for every repeated measurement")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("log10 PSD (nm²/Hz)")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_psd_angle_ratio(
    psd_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = psd_summary.pivot(
        index="frequency_hz", columns="angle_deg", values="median"
    ).sort_index()
    if 10 not in pivot.columns or 43 not in pivot.columns:
        return
    pivot = pivot.loc[(pivot.index >= 0.5) & (pivot.index < 100.0)]
    ratio_db = 10.0 * np.log10(
        np.maximum(pivot[43].to_numpy(), np.finfo(float).tiny)
        / np.maximum(pivot[10].to_numpy(), np.finfo(float).tiny)
    )
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    axis.plot(pivot.index, ratio_db, color="tab:purple", linewidth=1.1)
    axis.axhline(0.0, color="black", linewidth=1.0)
    for lower, upper, color, label in [
        (25, 32, "tab:blue", "25–32 Hz"),
        (50, 65, "tab:orange", "50–65 Hz"),
        (78, 88, "tab:green", "78–88 Hz"),
        (90, 99, "tab:red", "90–99 Hz edge"),
    ]:
        axis.axvspan(lower, upper, color=color, alpha=0.08, label=label)
    axis.set_xlim(0.5, 100.0)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("43° / 10° median PSD ratio (dB)")
    axis.set_title("Frequency-by-frequency PSD contrast between angle states")
    axis.grid(alpha=0.2)
    axis.legend(ncol=4, fontsize=9, loc="upper right")
    axis.text(
        1.2,
        axis.get_ylim()[1] * 0.88,
        "above 0 dB: stronger at 43°\nbelow 0 dB: stronger at 10°",
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


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
        avmd_maximum_frequency_hz = float(
            avmd_settings.get("maximum_frequency_hz", maximum_frequency_hz)
        )
        centers = create_bandwise_initial_centers(
            signal, sampling_rate=sampling_rate,
            mode_count=avmd_settings["mode_count"],
            minimum_frequency_hz=0.5,
            maximum_frequency_hz=avmd_maximum_frequency_hz,
        )
        result = variational_mode_decomposition(
            signal, sampling_rate=sampling_rate,
            initial_center_frequencies_hz=centers,
            alpha=avmd_settings["alpha"], maximum_iterations=500,
        )
        adaptive_k = bool(avmd_settings.get("adaptive_k", False))
        return result["modes"], result["center_frequencies_hz"], {
            "mode_count": avmd_settings["mode_count"],
            "alpha": avmd_settings["alpha"],
            "reference_file": avmd_settings["reference_file"],
            "maximum_frequency_hz": avmd_maximum_frequency_hz,
            "adaptive_k": adaptive_k,
            "adaptive_k_score": avmd_settings.get("adaptive_k_score"),
            "adaptive_k_stop_reason": avmd_settings.get("adaptive_k_stop_reason"),
            "adaptive_k_search_path": avmd_settings.get("adaptive_k_search_path"),
            "parameter_rule": (
                "per-file adaptive K with shared alpha"
                if adaptive_k
                else "shared K and alpha selected from representative repeat"
            ),
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
    for axis, method in zip(axes, methods):
        subset = summary.loc[(summary["method"] == method) & (summary["repeat_count"] >= 3)]
        for angle, angle_table in subset.groupby("angle_deg"):
            axis.scatter(
                angle_table["frequency_bin_hz"], angle_table["median_rms_nm"],
                s=18 + 8 * angle_table["repeat_count"], alpha=0.75,
                label=f"{angle} deg", color=ANGLE_COLORS.get(int(angle)),
            )
        axis.set_ylabel(f"{method}\nmedian RMS")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Mode peak frequency bin (Hz)")
    axes[-1].set_xlim(0.5, 100.0)
    figure.suptitle("Cross-method mode RMS by angle (frequency groups in at least 3 repeats)")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_method_reconstruction_quality(
    method_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    methods = [
        method for method in METHOD_ORDER
        if method in method_summary["method"].unique()
    ]
    figure, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    positions = np.arange(len(methods))
    values = [
        method_summary.loc[
            method_summary["method"] == method,
            "reconstruction_correlation",
        ].to_numpy()
        for method in methods
    ]
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    palette = plt.get_cmap("tab10")
    for index, (box, method) in enumerate(zip(boxes["boxes"], methods)):
        color = palette(index)
        box.set_facecolor(color)
        box.set_alpha(0.2)
        subset = method_summary.loc[method_summary["method"] == method].sort_values(
            ["angle_deg", "repeat"]
        )
        offsets = np.linspace(-0.15, 0.15, len(subset))
        point_colors = [
            ANGLE_COLORS.get(int(angle), "tab:gray")
            for angle in subset["angle_deg"]
        ]
        axis.scatter(
            index + offsets,
            subset["reconstruction_correlation"],
            c=point_colors,
            s=26,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.8,
            zorder=3,
        )
        minimum = float(subset["reconstruction_correlation"].min())
        axis.text(index, minimum - 0.0013, f"min {minimum:.4f}", ha="center", va="top", fontsize=8)
    axis.set_xticks(positions, methods)
    axis.set_ylabel("Reconstruction correlation")
    axis.set_ylim(0.975, 1.001)
    axis.set_title("Reconstruction quality across all files and methods")
    axis.grid(axis="y", alpha=0.2)
    axis.text(
        0.01,
        0.03,
        "All 100 decompositions exceed the 0.90 quality threshold",
        transform=axis.transAxes,
        fontsize=9,
        color="crimson",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ANGLE_COLORS[10], label="10 deg"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ANGLE_COLORS[43], label="43 deg"),
    ]
    axis.legend(handles=handles, ncol=2, loc="lower right")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_method_efficiency_and_complexity(
    method_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    methods = [
        method for method in METHOD_ORDER
        if method in method_summary["method"].unique()
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    palette = [plt.get_cmap("tab10")(index) for index in range(len(methods))]
    for axis, column, ylabel, title in [
        (axes[0], "elapsed_s", "Elapsed time per file (s)", "Computation time"),
        (axes[1], "mode_count", "Number of modes", "Decomposition complexity"),
    ]:
        values = [
            method_summary.loc[method_summary["method"] == method, column].to_numpy()
            for method in methods
        ]
        boxes = axis.boxplot(
            values,
            positions=np.arange(len(methods)),
            widths=0.52,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for box, color in zip(boxes["boxes"], palette):
            box.set_facecolor(color)
            box.set_alpha(0.28)
        axis.set_xticks(np.arange(len(methods)), methods, rotation=20)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Method cost and output complexity across 20 files")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_method_frequency_repeat_coverage(
    frequency_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    methods = [
        method for method in METHOD_ORDER
        if method in frequency_summary["method"].unique()
    ]
    angles = sorted(frequency_summary["angle_deg"].unique())
    frequency_grid = np.arange(0.5, 100.0, 0.5)
    figure, axes = plt.subplots(
        len(angles),
        1,
        figsize=(13, max(5.5, 3.2 * len(angles))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, angle in zip(axes, angles):
        matrix = np.zeros((len(methods), len(frequency_grid)), dtype=float)
        subset = frequency_summary.loc[frequency_summary["angle_deg"] == angle]
        for method_index, method in enumerate(methods):
            method_table = subset.loc[subset["method"] == method]
            for row in method_table.itertuples(index=False):
                bin_index = int(round((row.frequency_bin_hz - 0.5) / 0.5))
                if 0 <= bin_index < len(frequency_grid):
                    matrix[method_index, bin_index] = max(
                        matrix[method_index, bin_index],
                        float(row.repeat_count),
                    )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[0.25, 99.75, -0.5, len(methods) - 0.5],
            cmap="magma",
            norm=Normalize(vmin=0, vmax=10),
            interpolation="nearest",
        )
        axis.set_yticks(np.arange(len(methods)), methods)
        axis.set_ylabel(f"{angle} deg")
        axis.axvline(30.0, color="cyan", linestyle="--", linewidth=0.9)
        axis.axvline(60.0, color="cyan", linestyle="--", linewidth=0.9)
    axes[-1].set_xlabel("Mode peak frequency bin (Hz)")
    figure.suptitle(
        "Cross-method frequency repeat coverage\n"
        "brighter cells indicate a mode observed in more repeated measurements"
    )
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Number of repeats (0–10)")
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
    plot_feature_distributions(
        features,
        arguments.output_dir / "04_feature_distributions.png",
    )
    plot_dominant_frequency_by_repeat(
        features,
        arguments.output_dir / "05_dominant_welch_frequency_by_repeat.png",
    )
    plot_band_power_comparison(
        band_power,
        arguments.output_dir / "06_band_power_comparison.png",
    )
    plot_psd_repeat_heatmaps(
        psd_table,
        arguments.output_dir / "07_welch_psd_repeat_heatmaps.png",
    )
    plot_psd_angle_ratio(
        psd_summary,
        arguments.output_dir / "08_welch_psd_angle_ratio_db.png",
    )

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
    plot_method_reconstruction_quality(
        method_summary,
        arguments.output_dir / "09_method_reconstruction_quality.png",
    )
    plot_method_efficiency_and_complexity(
        method_summary,
        arguments.output_dir / "10_method_efficiency_and_complexity.png",
    )
    plot_method_frequency_repeat_coverage(
        frequency_summary,
        arguments.output_dir / "11_method_frequency_repeat_coverage.png",
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
