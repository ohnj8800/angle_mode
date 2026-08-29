from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.physical_episodes import build_physical_episodes


CLUSTER_PATH = PROJECT_ROOT / "results" / "method_consensus" / "all_candidate_clusters.csv"
ANGLE_PATH = PROJECT_ROOT / "results" / "angle_tracking" / "angle_timeline.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "physical_episodes"
MAXIMUM_CLUSTER_GAP_S = 1.0
ANGLE_CONTEXT_PADDING_S = 3.0
ANGLE_RANGE_THRESHOLD_DEG = 0.5
PEAK_VELOCITY_THRESHOLD_DEG_S = 0.2


def _plot_episode_timeline(
    angle_timeline: pd.DataFrame,
    episodes: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
    axis.plot(
        angle_timeline["time_s"],
        angle_timeline["estimated_angle_deg"],
        color="tab:purple",
        linewidth=1.0,
        label="estimated angle",
    )
    for episode in episodes.itertuples(index=False):
        color = (
            "tab:orange"
            if episode.episode_classification == "angle_associated_dynamic_response"
            else "tab:red"
        )
        axis.axvspan(
            episode.episode_start_time_s,
            episode.episode_end_time_s,
            color=color,
            alpha=0.22,
        )
        axis.text(
            (episode.episode_start_time_s + episode.episode_end_time_s) / 2,
            episode.angle_maximum_deg + 1.0,
            f"{episode.physical_episode_id}\n{episode.response_frequency_bands_hz} Hz",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Estimated angle (deg)")
    axis.set_title("Cross-method modal candidates grouped into physical episodes")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for path in (CLUSTER_PATH, ANGLE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"缺少前一步結果：{path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    clusters = pd.read_csv(CLUSTER_PATH, encoding="utf-8-sig")
    angle_timeline = pd.read_csv(ANGLE_PATH, encoding="utf-8-sig")
    episodes, mapping = build_physical_episodes(
        clusters,
        angle_timeline,
        maximum_cluster_gap_s=MAXIMUM_CLUSTER_GAP_S,
        angle_context_padding_s=ANGLE_CONTEXT_PADDING_S,
        angle_range_threshold_deg=ANGLE_RANGE_THRESHOLD_DEG,
        peak_velocity_threshold_deg_s=PEAK_VELOCITY_THRESHOLD_DEG_S,
    )
    episodes.to_csv(
        OUTPUT_ROOT / "physical_episode_summary.csv", index=False, encoding="utf-8-sig"
    )
    mapping.to_csv(
        OUTPUT_ROOT / "cluster_to_episode_mapping.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{
        "maximum_cluster_gap_s": MAXIMUM_CLUSTER_GAP_S,
        "angle_context_padding_s": ANGLE_CONTEXT_PADDING_S,
        "angle_range_threshold_deg": ANGLE_RANGE_THRESHOLD_DEG,
        "peak_velocity_threshold_deg_s": PEAK_VELOCITY_THRESHOLD_DEG_S,
        "classification_rule": (
            "angle-associated only when the episode overlaps moving state or "
            "its representative time is within angle_context_padding_s of moving"
        ),
        "angle_range_and_velocity_role": "diagnostic columns only when state_label exists",
        "classification_scope": "cause association only; not fault confirmation",
    }]).to_csv(
        OUTPUT_ROOT / "physical_episode_settings.csv", index=False, encoding="utf-8-sig"
    )
    _plot_episode_timeline(
        angle_timeline,
        episodes,
        OUTPUT_ROOT / "01_physical_episode_timeline.png",
    )
    print("=" * 70)
    print("跨頻帶候選合併與角度關聯判斷完成")
    print(f"跨方法候選群集：{len(mapping)}")
    print(f"合併後物理事件：{len(episodes)}")
    print(episodes[[
        "physical_episode_id", "episode_start_time_s", "episode_end_time_s",
        "response_frequency_bands_hz", "episode_classification",
    ]].to_string(index=False))
    print(f"結果位置：{OUTPUT_ROOT}")
    print("注意：angle_associated表示事件重疊或鄰近角度移動，不代表設備故障。")
    print("stable_state_anomaly_candidate也只代表穩定角度候選，仍需第11步驗證。")


if __name__ == "__main__":
    main()
