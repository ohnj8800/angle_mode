from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.angle_tracking import (
    assign_angle_state_labels,
    detect_angle_transitions,
    estimate_angle_from_component,
    summarize_angle_states,
)
from src.common.io import load_fbg_csv
from src.common.preprocessing import preprocess_fbg_signal


SAMPLING_RATE = 200.0
RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "angle_10deg_43deg_repeat.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "angle_tracking"


def _shade_transitions(axis, transitions: pd.DataFrame) -> None:
    for event in transitions.itertuples(index=False):
        color = "tab:orange" if event.direction == "10_to_43" else "tab:green"
        axis.axvspan(
            event.start_time_s,
            event.end_time_s,
            color=color,
            alpha=0.18,
        )


def plot_angle_component(
    time_s: np.ndarray,
    raw_nm: np.ndarray,
    angle_component_nm: np.ndarray,
    transitions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.plot(time_s, raw_nm, color="0.75", linewidth=0.45, label="Raw CH1")
    axis.plot(
        time_s,
        angle_component_nm,
        color="tab:blue",
        linewidth=2.0,
        label="Low-frequency angle component",
    )
    _shade_transitions(axis, transitions)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Bragg wavelength (nm)")
    axis.set_title("FBG angle component and detected movements")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_estimated_angle(
    time_s: np.ndarray,
    estimated_angle_deg: np.ndarray,
    transitions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.plot(
        time_s,
        estimated_angle_deg,
        color="tab:purple",
        linewidth=1.8,
    )
    axis.axhline(10.0, color="tab:blue", linestyle="--", linewidth=1.0)
    axis.axhline(43.0, color="tab:red", linestyle="--", linewidth=1.0)
    _shade_transitions(axis, transitions)

    for event in transitions.itertuples(index=False):
        center_s = (event.start_time_s + event.end_time_s) / 2.0
        axis.text(
            center_s,
            26.5,
            f"E{event.event_id}\n{event.direction.replace('_', ' ')}",
            ha="center",
            va="center",
            fontsize=8,
        )

    axis.set_ylim(7.0, 46.0)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Estimated angle (degree)")
    axis.set_title("Estimated 10-degree / 43-degree timeline")
    axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_angle_velocity(
    time_s: np.ndarray,
    angle_velocity_deg_s: np.ndarray,
    transitions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    axis.plot(
        time_s,
        angle_velocity_deg_s,
        color="tab:brown",
        linewidth=1.0,
    )
    _shade_transitions(axis, transitions)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Estimated angular velocity (degree/s)")
    axis.set_title("Angle-change timing and direction")
    axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data, information = load_fbg_csv(RAW_FILE, sampling_rate=SAMPLING_RATE)
    time_s = data["time_s"].to_numpy(dtype=float)
    raw_nm = data["ch1_nm"].to_numpy(dtype=float)
    signals = preprocess_fbg_signal(
        raw_nm,
        sampling_rate=SAMPLING_RATE,
        lowpass_cutoff_hz=0.1,
        highpass_cutoff_hz=0.5,
    )
    angle_component_nm = signals["angle_component_nm"]
    calibration = estimate_angle_from_component(
        angle_component_nm,
        sampling_rate=SAMPLING_RATE,
        lower_angle_deg=10.0,
        upper_angle_deg=43.0,
    )
    estimated_angle_deg = calibration.pop("estimated_angle_deg")
    angle_velocity_deg_s = calibration.pop("angle_velocity_deg_s")
    transitions = detect_angle_transitions(
        time_s,
        estimated_angle_deg,
        lower_angle_deg=10.0,
        upper_angle_deg=43.0,
    )
    state_labels = assign_angle_state_labels(
        estimated_angle_deg,
        transitions,
    )
    state_summary = summarize_angle_states(
        angle_component_nm,
        estimated_angle_deg,
        state_labels,
    )
    timeline = pd.DataFrame(
        {
            "time_s": time_s,
            "raw_ch1_nm": raw_nm,
            "angle_component_nm": angle_component_nm,
            "estimated_angle_deg": estimated_angle_deg,
            "angle_velocity_deg_s": angle_velocity_deg_s,
            "state_label": state_labels,
        }
    )
    timeline.to_csv(
        OUTPUT_ROOT / "angle_timeline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transitions.to_csv(
        OUTPUT_ROOT / "angle_transitions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    state_summary.to_csv(
        OUTPUT_ROOT / "angle_state_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration_output = {
        "source_file": RAW_FILE.name,
        "sampling_rate_hz": SAMPLING_RATE,
        "sample_count": information["sample_count"],
        "duration_s": information["duration_s"],
        "angle_component_cutoff_hz": 0.1,
        "known_angles_deg": [10.0, 43.0],
        "detected_transition_count": int(len(transitions)),
        **calibration,
    }

    with (OUTPUT_ROOT / "angle_calibration.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(calibration_output, file, indent=2, ensure_ascii=False)

    plot_angle_component(
        time_s,
        raw_nm,
        angle_component_nm,
        transitions,
        OUTPUT_ROOT / "01_angle_component.png",
    )
    plot_estimated_angle(
        time_s,
        estimated_angle_deg,
        transitions,
        OUTPUT_ROOT / "02_estimated_angle_timeline.png",
    )
    plot_angle_velocity(
        time_s,
        angle_velocity_deg_s,
        transitions,
        OUTPUT_ROOT / "03_angle_transition_detection.png",
    )

    print("=" * 70)
    print("角度模態與切換事件分析完成")
    print(f"10度平台波長：{calibration['wavelength_at_lower_angle_nm']:.6f} nm")
    print(f"43度平台波長：{calibration['wavelength_at_upper_angle_nm']:.6f} nm")
    print(f"偵測到的切換事件：{len(transitions)}")
    print(transitions.to_string(index=False))
    print(f"結果位置：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
