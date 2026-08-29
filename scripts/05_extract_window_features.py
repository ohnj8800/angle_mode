from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.window_features import extract_mode_window_features


SAMPLING_RATE = 200.0
WINDOW_DURATION_S = 2.0
STEP_DURATION_S = 0.5
MINIMUM_DYNAMIC_FREQUENCY_HZ = 0.5
METHODS = ("VMD", "IOVMD", "AVMD", "SVMD", "STVMD")
ANGLE_ROOT = PROJECT_ROOT / "results" / "angle_tracking"
MODE_ROOT = PROJECT_ROOT / "results" / "mode_analysis"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "window_analysis"


def _load_method_inputs(
    method: str,
    timeline: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    method_root = MODE_ROOT / method.lower()
    decomposition_path = method_root / "decomposition.npz"
    summary_path = method_root / "mode_response_summary.csv"

    if not decomposition_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"缺少{method}分解結果，請先執行03_analyze_vmd_modes.py。"
        )

    decomposition = np.load(decomposition_path)
    time_s = decomposition["time_s"]
    modes = decomposition["modes"]
    mode_summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    effective_timeline = timeline.iloc[: modes.shape[1]].reset_index(drop=True)

    if len(effective_timeline) != modes.shape[1]:
        raise ValueError(f"{method}模態長度超過角度時間軸長度。")

    expected_modes = [f"IMF{index + 1}" for index in range(len(modes))]
    mode_summary = (
        mode_summary.set_index("mode").loc[expected_modes].reset_index()
    )
    return time_s, modes, mode_summary, effective_timeline


def _plot_method_feature(
    features: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    dynamic = features.loc[
        features["global_peak_frequency_hz"] >= MINIMUM_DYNAMIC_FREQUENCY_HZ
    ]
    mode_order = (
        dynamic[["mode", "global_peak_frequency_hz"]]
        .drop_duplicates()
        .sort_values("global_peak_frequency_hz")
    )
    figure, axes = plt.subplots(
        len(mode_order),
        1,
        figsize=(14, max(5, 1.8 * len(mode_order))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, row in zip(axes, mode_order.itertuples(index=False)):
        subset = dynamic.loc[dynamic["mode"] == row.mode]
        axis.plot(
            subset["center_time_s"],
            subset[value_column],
            color="tab:blue",
            linewidth=0.9,
        )
        moving = subset["moving_fraction"] > 0.0
        axis.scatter(
            subset.loc[moving, "center_time_s"],
            subset.loc[moving, value_column],
            color="tab:orange",
            s=8,
            label="window overlaps movement",
        )
        axis.set_ylabel(f"{row.mode}\n{row.global_peak_frequency_hz:.1f} Hz")
        axis.grid(alpha=0.2)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    figure.supylabel(ylabel)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timeline_path = ANGLE_ROOT / "angle_timeline.csv"

    if not timeline_path.exists():
        raise FileNotFoundError(
            "缺少angle_timeline.csv，請先執行02_track_angle.py。"
        )

    timeline = pd.read_csv(timeline_path, encoding="utf-8-sig")
    all_features = []

    for method in METHODS:
        print("=" * 70)
        print(f"開始計算{method}滑動視窗特徵")
        time_s, modes, mode_summary, effective_timeline = _load_method_inputs(
            method, timeline
        )
        features = extract_mode_window_features(
            method=method,
            time_s=time_s,
            modes=modes,
            state_labels=effective_timeline["state_label"].to_numpy(dtype=str),
            estimated_angle_deg=effective_timeline[
                "estimated_angle_deg"
            ].to_numpy(dtype=float),
            angle_velocity_deg_s=effective_timeline[
                "angle_velocity_deg_s"
            ].to_numpy(dtype=float),
            global_peak_frequencies_hz=mode_summary[
                "peak_frequency_hz"
            ].to_numpy(dtype=float),
            physical_roles=mode_summary[
                "physical_role_candidate"
            ].to_numpy(dtype=str),
            sampling_rate=SAMPLING_RATE,
            window_duration_s=WINDOW_DURATION_S,
            step_duration_s=STEP_DURATION_S,
            minimum_dynamic_frequency_hz=MINIMUM_DYNAMIC_FREQUENCY_HZ,
            maximum_frequency_hz=SAMPLING_RATE / 2.0,
        )
        method_root = OUTPUT_ROOT / method.lower()
        method_root.mkdir(parents=True, exist_ok=True)
        features.to_csv(
            method_root / "mode_window_features.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _plot_method_feature(
            features,
            value_column="rms_nm",
            ylabel="Window RMS (nm)",
            title=f"{method}: modal RMS over time",
            output_path=method_root / "01_mode_rms_timeline.png",
        )
        _plot_method_feature(
            features,
            value_column="local_peak_frequency_hz",
            ylabel="Local peak frequency (Hz)",
            title=f"{method}: local modal peak frequency over time",
            output_path=method_root / "02_mode_frequency_timeline.png",
        )
        all_features.append(features)
        print(
            f"{method}: {features['window_index'].nunique()}個視窗，"
            f"{features['mode'].nunique()}個模態，{len(features)}列特徵"
        )

    combined = pd.concat(all_features, ignore_index=True)
    combined.to_csv(
        OUTPUT_ROOT / "all_method_mode_window_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    settings = pd.DataFrame(
        [
            {
                "sampling_rate_hz": SAMPLING_RATE,
                "window_duration_s": WINDOW_DURATION_S,
                "step_duration_s": STEP_DURATION_S,
                "window_samples": int(WINDOW_DURATION_S * SAMPLING_RATE),
                "step_samples": int(STEP_DURATION_S * SAMPLING_RATE),
                "frequency_resolution_hz": 1.0 / WINDOW_DURATION_S,
                "minimum_dynamic_frequency_hz": (
                    MINIMUM_DYNAMIC_FREQUENCY_HZ
                ),
                "maximum_frequency_hz": SAMPLING_RATE / 2.0,
            }
        ]
    )
    settings.to_csv(
        OUTPUT_ROOT / "window_analysis_settings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("=" * 70)
    print(f"{len(METHODS)}種VMD滑動視窗特徵完成")
    print(f"合併結果：{OUTPUT_ROOT / 'all_method_mode_window_features.csv'}")


if __name__ == "__main__":
    main()
