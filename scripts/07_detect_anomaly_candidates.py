from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.anomaly_candidates import (
    merge_consecutive_candidates,
    score_stable_mode_windows,
)


WINDOW_PATH = PROJECT_ROOT / "results" / "window_analysis" / "all_method_mode_window_features.csv"
BASELINE_PATH = PROJECT_ROOT / "results" / "normal_baseline" / "normal_baseline_by_state.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anomaly_candidates"
METHODS = ("VMD", "IOVMD", "AVMD", "SVMD", "STVMD")
CANDIDATE_THRESHOLD = 3.5
MINIMUM_CONSECUTIVE_WINDOWS = 2
DISPLAY_SCORE_LIMIT = 10.0


def _plot_score_timeline(scored: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(len(METHODS), 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for axis, method in zip(axes, METHODS):
        subset = scored.loc[scored["method"] == method]
        timeline = subset.groupby("center_time_s", as_index=False)["anomaly_score"].max()
        display_score = timeline["anomaly_score"].clip(upper=DISPLAY_SCORE_LIMIT)
        axis.plot(timeline["center_time_s"], display_score, color="tab:blue", linewidth=1)
        candidate = timeline["anomaly_score"] >= CANDIDATE_THRESHOLD
        axis.scatter(
            timeline.loc[candidate, "center_time_s"],
            display_score.loc[candidate],
            color="tab:red", s=12, label="candidate window",
        )
        axis.axhline(CANDIDATE_THRESHOLD, color="black", linestyle="--", linewidth=1)
        axis.set_ylabel("Max score")
        axis.set_title(method)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s); gaps are angle-movement windows excluded from scoring")
    figure.suptitle(f"Maximum stable-state modal deviation score (display capped at {DISPLAY_SCORE_LIMIT:g})")
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_candidate_frequency(events: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    colors = {
        "VMD": "tab:purple",
        "IOVMD": "tab:blue",
        "AVMD": "tab:orange",
        "SVMD": "tab:green",
        "STVMD": "tab:brown",
    }
    for method in METHODS:
        subset = events.loc[events["method"] == method]
        axis.scatter(
            subset["peak_time_s"], subset["representative_local_frequency_hz"],
            s=25 + 8 * subset["peak_anomaly_score"].clip(upper=10),
            color=colors[method], alpha=0.75, label=method,
        )
    if events.empty:
        axis.text(0.5, 0.5, "No persistent candidate events", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("Candidate peak time (s)")
    axis.set_ylabel("Representative local frequency (Hz)")
    axis.set_title("Persistent modal anomaly candidates in stable angle periods")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not WINDOW_PATH.exists():
        raise FileNotFoundError("缺少滑動視窗特徵，請先執行05_extract_window_features.py。")
    if not BASELINE_PATH.exists():
        raise FileNotFoundError("缺少正常模態基準，請先執行06_build_normal_baseline.py。")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    window_features = pd.read_csv(WINDOW_PATH, encoding="utf-8-sig")
    baseline = pd.read_csv(BASELINE_PATH, encoding="utf-8-sig")
    scored = score_stable_mode_windows(
        window_features, baseline, candidate_threshold=CANDIDATE_THRESHOLD
    )
    events = merge_consecutive_candidates(
        scored, minimum_consecutive_windows=MINIMUM_CONSECUTIVE_WINDOWS
    )

    scored.to_csv(OUTPUT_ROOT / "window_deviation_scores.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUTPUT_ROOT / "modal_anomaly_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "candidate_threshold_robust_deviations": CANDIDATE_THRESHOLD,
        "minimum_consecutive_windows": MINIMUM_CONSECUTIVE_WINDOWS,
        "scored_features": "local_peak_frequency_hz, rms_nm, envelope_mean_nm",
        "combined_score": "maximum feature deviation score",
        "excluded_periods": "angle movement and mixed-state windows",
        "excluded_mode": "angle_position_mode",
        "interpretation": "candidate only; not a confirmed fault",
        "baseline_limitation": "internal stable windows from the same recording",
    }]).to_csv(OUTPUT_ROOT / "anomaly_detection_settings.csv", index=False, encoding="utf-8-sig")
    _plot_score_timeline(scored, OUTPUT_ROOT / "01_max_anomaly_score_timeline.png")
    _plot_candidate_frequency(events, OUTPUT_ROOT / "02_candidate_time_frequency.png")

    print("=" * 70)
    print("穩定角度期間的模態異常候選偵測完成")
    print(f"評分視窗列數：{len(scored)}")
    print(f"單一視窗超標列數：{int(scored['anomaly_candidate'].sum())}")
    if events.empty:
        print("連續異常候選事件：0")
    else:
        print("連續異常候選事件：")
        print(events.groupby("method").size().reindex(METHODS, fill_value=0).to_string())
    print(f"結果位置：{OUTPUT_ROOT}")
    print("注意：輸出是偏離正常模式的候選，不代表已確認故障。")


if __name__ == "__main__":
    main()
