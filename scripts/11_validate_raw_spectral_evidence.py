from pathlib import Path
import json
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
WINDOW_DURATION_S = 2.0
STEP_DURATION_S = 0.5
WELCH_SEGMENT_SAMPLES = 256
WELCH_OVERLAP_SAMPLES = 128
BAND_HALF_WIDTH_HZ = 1.5
PEAK_SEARCH_HALF_WIDTH_HZ = 5.0
SCORE_THRESHOLD = 3.5
MINIMUM_CONSECUTIVE_WINDOWS = 2
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "angle_10deg_43deg_repeat.csv"
ANGLE_PATH = PROJECT_ROOT / "results" / "angle_tracking" / "angle_timeline.csv"
EPISODE_PATH = (
    PROJECT_ROOT
    / "results"
    / "physical_episodes"
    / "physical_episode_summary.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "raw_spectral_validation"


def _remove_previous_validation_outputs() -> int:
    """清除本步驟會依事件編號變動的舊圖與舊視窗明細。"""
    removed = 0
    for path in OUTPUT_ROOT.glob("E*_raw_spectral_validation.png"):
        if path.is_file():
            path.unlink()
            removed += 1
    feature_path = OUTPUT_ROOT / "raw_band_window_features.csv"
    if feature_path.exists():
        feature_path.unlink()
    return removed


def _parse_frequencies(text: str) -> list[float]:
    return [float(value) for value in str(text).split(",") if str(value).strip()]


def _robust_location_scale(values: np.ndarray, floor: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, floor)
    return median, robust_sigma


def _maximum_consecutive_true(values: np.ndarray) -> int:
    maximum = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _window_features(
    time_s: np.ndarray,
    signal: np.ndarray,
    state_labels: np.ndarray,
    target_frequency_hz: float,
) -> pd.DataFrame:
    window_samples = int(round(WINDOW_DURATION_S * SAMPLING_RATE))
    step_samples = int(round(STEP_DURATION_S * SAMPLING_RATE))
    rows = []

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        end = start + window_samples
        segment = signal[start:end]
        frequency_hz, psd = welch(
            segment,
            fs=SAMPLING_RATE,
            window="hann",
            nperseg=min(WELCH_SEGMENT_SAMPLES, len(segment)),
            noverlap=min(WELCH_OVERLAP_SAMPLES, len(segment) // 2),
            detrend="constant",
            scaling="density",
        )
        band_mask = (
            (frequency_hz >= target_frequency_hz - BAND_HALF_WIDTH_HZ)
            & (frequency_hz <= target_frequency_hz + BAND_HALF_WIDTH_HZ)
        )
        search_mask = (
            (frequency_hz >= target_frequency_hz - PEAK_SEARCH_HALF_WIDTH_HZ)
            & (frequency_hz <= target_frequency_hz + PEAK_SEARCH_HALF_WIDTH_HZ)
        )
        band_power = float(np.trapezoid(psd[band_mask], frequency_hz[band_mask]))
        search_frequency = frequency_hz[search_mask]
        search_psd = psd[search_mask]
        local_peak = float(search_frequency[np.argmax(search_psd)])
        labels = state_labels[start:end]
        unique, counts = np.unique(labels, return_counts=True)
        dominant_state = str(unique[np.argmax(counts)])
        rows.append(
            {
                "target_frequency_hz": target_frequency_hz,
                "window_start_time_s": float(time_s[start]),
                "window_end_time_s": float(time_s[end - 1]),
                "center_time_s": float((time_s[start] + time_s[end - 1]) / 2.0),
                "dominant_state_label": dominant_state,
                "stable_10deg_fraction": float(np.mean(labels == "stable_10deg")),
                "stable_43deg_fraction": float(np.mean(labels == "stable_43deg")),
                "band_power_nm2": band_power,
                "band_rms_nm": float(np.sqrt(max(band_power, 0.0))),
                "log_band_power": float(np.log10(max(band_power, np.finfo(float).tiny))),
                "local_peak_frequency_hz": local_peak,
            }
        )
    return pd.DataFrame(rows)


def _score_episode_band(
    features: pd.DataFrame,
    episode: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    start = float(episode["episode_start_time_s"])
    end = float(episode["episode_end_time_s"])
    episode_mask = features["center_time_s"].between(start, end)
    guard_mask = features["center_time_s"].between(start - 3.0, end + 3.0)
    stable_states = [
        value.strip()
        for value in str(episode["angle_state_labels"]).split(",")
        if value.strip().startswith("stable_")
    ]
    stable_states = sorted(set(stable_states))
    if len(stable_states) != 1:
        raise ValueError(
            f"{episode['physical_episode_id']}無法決定唯一穩定角度基準："
            f"{episode['angle_state_labels']}"
        )
    baseline_state = stable_states[0]
    baseline_fraction_column = f"{baseline_state}_fraction"
    if baseline_fraction_column not in features.columns:
        raise KeyError(f"缺少狀態基準欄位：{baseline_fraction_column}")
    baseline_mask = (
        (features[baseline_fraction_column] >= 1.0 - 1e-12)
        & ~guard_mask
    )
    baseline = features.loc[baseline_mask]
    if len(baseline) < 20:
        raise ValueError(
            f"可用的{baseline_state}原始頻譜基準視窗不足20個。"
        )

    power_median, power_sigma = _robust_location_scale(
        baseline["log_band_power"].to_numpy(),
        floor=0.02,
    )
    frequency_resolution_hz = SAMPLING_RATE / WELCH_SEGMENT_SAMPLES
    frequency_median, frequency_sigma = _robust_location_scale(
        baseline["local_peak_frequency_hz"].to_numpy(),
        floor=frequency_resolution_hz,
    )
    scored = features.copy()
    scored["power_deviation_score"] = (
        np.abs(scored["log_band_power"] - power_median) / power_sigma
    )
    scored["frequency_deviation_score"] = (
        np.abs(scored["local_peak_frequency_hz"] - frequency_median)
        / frequency_sigma
    )
    scored["combined_raw_score"] = scored[
        ["power_deviation_score", "frequency_deviation_score"]
    ].max(axis=1)
    scored["inside_episode"] = episode_mask
    scored["raw_candidate_window"] = (
        scored["inside_episode"]
        & (scored["combined_raw_score"] >= SCORE_THRESHOLD)
    )
    local = scored.loc[episode_mask].sort_values("center_time_s")
    if local.empty:
        raise ValueError(
            f"{episode['physical_episode_id']}在指定事件範圍內沒有可評分視窗。"
        )
    consecutive = _maximum_consecutive_true(local["raw_candidate_window"].to_numpy())
    dominant = (
        "band_power"
        if float(local["power_deviation_score"].max())
        >= float(local["frequency_deviation_score"].max())
        else "local_peak_frequency"
    )
    result = {
        "physical_episode_id": episode["physical_episode_id"],
        "target_frequency_hz": float(features["target_frequency_hz"].iloc[0]),
        "episode_start_time_s": start,
        "episode_end_time_s": end,
        "baseline_angle_state": baseline_state,
        "baseline_window_count": int(baseline_mask.sum()),
        "episode_window_count": int(episode_mask.sum()),
        "baseline_median_band_rms_nm": float(baseline["band_rms_nm"].median()),
        "episode_median_band_rms_nm": float(local["band_rms_nm"].median()),
        "episode_to_baseline_rms_ratio": float(
            local["band_rms_nm"].median()
            / (baseline["band_rms_nm"].median() + np.finfo(float).eps)
        ),
        "maximum_power_deviation_score": float(local["power_deviation_score"].max()),
        "maximum_frequency_deviation_score": float(
            local["frequency_deviation_score"].max()
        ),
        "maximum_combined_raw_score": float(local["combined_raw_score"].max()),
        "maximum_consecutive_candidate_windows": int(consecutive),
        "dominant_raw_deviation": dominant,
        "raw_spectrum_supported": bool(consecutive >= MINIMUM_CONSECUTIVE_WINDOWS),
    }
    return scored, result


def _plot_episode(
    episode: pd.Series,
    time_s: np.ndarray,
    signal: np.ndarray,
    angle_timeline: pd.DataFrame,
    scored_by_frequency: dict[float, pd.DataFrame],
    output_path: Path,
) -> None:
    frequencies = sorted(scored_by_frequency)
    figure, axes = plt.subplots(
        2 + len(frequencies),
        1,
        figsize=(15, 5 + 2.5 * len(frequencies)),
        sharex=True,
        constrained_layout=True,
    )
    plot_start = max(float(time_s[0]), float(episode["episode_start_time_s"]) - 12.0)
    plot_end = min(float(time_s[-1]), float(episode["episode_end_time_s"]) + 4.0)
    raw_mask = (time_s >= plot_start) & (time_s <= plot_end)
    centered = signal - np.median(signal[raw_mask])
    axes[0].plot(time_s[raw_mask], centered[raw_mask], color="black", linewidth=0.6)
    axes[0].set_ylabel("CH1 centered\n(nm)")
    axes[0].set_title("Original FBG signal (no VMD)")

    angle_time = angle_timeline["time_s"].to_numpy(dtype=float)
    angle_mask = (angle_time >= plot_start) & (angle_time <= plot_end)
    axes[1].plot(
        angle_time[angle_mask],
        angle_timeline.loc[angle_mask, "estimated_angle_deg"],
        color="tab:purple",
        linewidth=1.0,
    )
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title("Estimated angle")

    for axis, frequency in zip(axes[2:], frequencies):
        scored = scored_by_frequency[frequency]
        local = scored.loc[scored["center_time_s"].between(plot_start, plot_end)]
        axis.plot(
            local["center_time_s"],
            local["combined_raw_score"],
            color="tab:blue",
            marker="o",
            markersize=2.5,
            linewidth=0.9,
            label="raw spectral deviation score",
        )
        axis.axhline(SCORE_THRESHOLD, color="black", linestyle="--", linewidth=0.9)
        axis.set_ylabel(f"{frequency:.1f} Hz\nscore")
        axis.set_title("Sliding Welch band-power / local-frequency deviation")
        axis.legend(loc="upper left", fontsize=8)

    for axis in axes:
        axis.axvspan(
            float(episode["episode_start_time_s"]),
            float(episode["episode_end_time_s"]),
            color="tab:red",
            alpha=0.12,
        )
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"{episode['physical_episode_id']}: independent raw-spectrum validation"
    )
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for path in (RAW_FILE, ANGLE_PATH, EPISODE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"缺少必要檔案：{path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    removed_plot_count = _remove_previous_validation_outputs()
    data, _ = load_fbg_csv(RAW_FILE, sampling_rate=SAMPLING_RATE)
    angle_timeline = pd.read_csv(ANGLE_PATH, encoding="utf-8-sig")
    episodes = pd.read_csv(EPISODE_PATH, encoding="utf-8-sig")
    targets = episodes.loc[
        episodes["episode_classification"] == "stable_state_anomaly_candidate"
    ].copy()
    if targets.empty:
        pd.DataFrame().to_csv(
            OUTPUT_ROOT / "raw_band_window_features.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            columns=[
                "physical_episode_id",
                "target_frequency_hz",
                "raw_spectrum_supported",
                "skip_reason",
            ]
        ).to_csv(
            OUTPUT_ROOT / "raw_spectral_episode_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (OUTPUT_ROOT / "raw_spectral_validation_settings.json").write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "no stable_state_anomaly_candidate",
                    "interpretation": (
                        "all current cross-method episodes overlap or are close "
                        "to estimated angle variation"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("=" * 70)
        print("非VMD原始頻譜驗證略過")
        print("目前沒有stable_state_anomaly_candidate可供獨立頻譜驗證。")
        print(f"已清除上次驗證圖：{removed_plot_count}")
        print(f"結果位置：{OUTPUT_ROOT}")
        return

    length = min(len(data), len(angle_timeline))
    time_s = data["time_s"].to_numpy(dtype=float)[:length]
    raw_nm = data["ch1_nm"].to_numpy(dtype=float)[:length]
    dynamic_nm = preprocess_fbg_signal(raw_nm, SAMPLING_RATE)["dynamic_nm"]
    state_labels = angle_timeline["state_label"].to_numpy(dtype=str)[:length]
    feature_tables = []
    validation_rows = []

    for episode in targets.itertuples(index=False):
        episode_series = pd.Series(episode._asdict())
        scored_by_frequency = {}
        for frequency in _parse_frequencies(episode.response_frequency_bands_hz):
            features = _window_features(time_s, dynamic_nm, state_labels, frequency)
            scored, validation = _score_episode_band(features, episode_series)
            scored.insert(0, "physical_episode_id", episode.physical_episode_id)
            feature_tables.append(scored)
            validation_rows.append(validation)
            scored_by_frequency[frequency] = scored
        _plot_episode(
            episode_series,
            time_s,
            raw_nm,
            angle_timeline.iloc[:length].reset_index(drop=True),
            scored_by_frequency,
            OUTPUT_ROOT / f"{episode.physical_episode_id}_raw_spectral_validation.png",
        )

    all_features = pd.concat(feature_tables, ignore_index=True)
    validation = pd.DataFrame(validation_rows)
    all_features.to_csv(
        OUTPUT_ROOT / "raw_band_window_features.csv", index=False, encoding="utf-8-sig"
    )
    validation.to_csv(
        OUTPUT_ROOT / "raw_spectral_episode_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    settings = {
        "method": "sliding Welch PSD on raw-derived 0.5-Hz high-pass signal; no VMD",
        "sampling_rate_hz": SAMPLING_RATE,
        "window_duration_s": WINDOW_DURATION_S,
        "step_duration_s": STEP_DURATION_S,
        "welch_segment_samples": WELCH_SEGMENT_SAMPLES,
        "welch_overlap_samples": WELCH_OVERLAP_SAMPLES,
        "band_half_width_hz": BAND_HALF_WIDTH_HZ,
        "peak_search_half_width_hz": PEAK_SEARCH_HALF_WIDTH_HZ,
        "score_threshold": SCORE_THRESHOLD,
        "minimum_consecutive_windows": MINIMUM_CONSECUTIVE_WINDOWS,
        "baseline": (
            "all other fully stable windows from the same angle state as each "
            "episode, with a 3-second event guard"
        ),
    }
    (OUTPUT_ROOT / "raw_spectral_validation_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("=" * 70)
    print("非VMD原始頻譜驗證完成")
    print(
        validation[
            [
                "physical_episode_id",
                "target_frequency_hz",
                "episode_to_baseline_rms_ratio",
                "maximum_power_deviation_score",
                "maximum_frequency_deviation_score",
                "maximum_consecutive_candidate_windows",
                "raw_spectrum_supported",
            ]
        ].to_string(index=False)
    )
    print(f"結果位置：{OUTPUT_ROOT}")
    print(f"已清除上次驗證圖：{removed_plot_count}")
    print("raw_spectrum_supported=True只代表原始頻譜也支持候選，不等同故障。")


if __name__ == "__main__":
    main()
