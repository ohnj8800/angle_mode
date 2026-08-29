from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.consensus_evidence import select_consensus_evidence, time_window_mask


CONSENSUS_ROOT = PROJECT_ROOT / "results" / "method_consensus"
MODE_ROOT = PROJECT_ROOT / "results" / "mode_analysis"
ANGLE_PATH = PROJECT_ROOT / "results" / "angle_tracking" / "angle_timeline.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "consensus_event_evidence"
METHOD_FOLDER = {
    "VMD": "vmd",
    "IOVMD": "iovmd",
    "AVMD": "avmd",
    "SVMD": "svmd",
    "STVMD": "stvmd",
}
PADDING_S = 3.0


def _remove_previous_event_plots() -> int:
    """刪除本步驟上次產生、但本次未必會同名覆寫的事件圖。"""
    removed = 0
    for path in OUTPUT_ROOT.glob("C*.png"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def _load_decompositions() -> dict[str, dict[str, np.ndarray]]:
    decompositions = {}
    for method, folder in METHOD_FOLDER.items():
        path = MODE_ROOT / folder / "decomposition.npz"
        if not path.exists():
            raise FileNotFoundError(f"缺少{method}分解結果：{path}")
        with np.load(path) as data:
            decompositions[method] = {
                "time_s": data["time_s"].copy(),
                "modes": data["modes"].copy(),
            }
    return decompositions


def _plot_event(
    cluster: pd.Series,
    members: pd.DataFrame,
    angle_timeline: pd.DataFrame,
    decompositions: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    row_count = 2 + len(members)
    figure, axes = plt.subplots(
        row_count,
        1,
        figsize=(15, max(8, 2.0 * row_count)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    raw_time = angle_timeline["time_s"].to_numpy(dtype=float)
    raw_mask = time_window_mask(
        raw_time,
        cluster["cluster_start_time_s"],
        cluster["cluster_end_time_s"],
        padding_s=PADDING_S,
    )
    local_raw = angle_timeline.loc[raw_mask, "raw_ch1_nm"].to_numpy(dtype=float)
    axes[0].plot(
        raw_time[raw_mask],
        local_raw - np.median(local_raw),
        color="black",
        linewidth=0.65,
    )
    axes[0].set_ylabel("CH1\ncentered (nm)")
    axes[0].set_title("Raw FBG signal")

    axes[1].plot(
        raw_time[raw_mask],
        angle_timeline.loc[raw_mask, "estimated_angle_deg"],
        color="tab:purple",
        linewidth=1.0,
    )
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title(
        f"Estimated angle; expected state: {cluster['angle_state_label']}"
    )

    colors = {
        "VMD": "tab:purple",
        "IOVMD": "tab:blue",
        "AVMD": "tab:orange",
        "SVMD": "tab:green",
        "STVMD": "tab:brown",
    }
    for axis, member in zip(axes[2:], members.itertuples(index=False)):
        method_data = decompositions[member.method]
        time_s = method_data["time_s"]
        mode_index = int(str(member.mode).replace("IMF", "")) - 1
        if mode_index < 0 or mode_index >= len(method_data["modes"]):
            raise IndexError(f"{member.method} {member.mode}不存在於分解結果。")
        mask = time_window_mask(
            time_s,
            cluster["cluster_start_time_s"],
            cluster["cluster_end_time_s"],
            padding_s=PADDING_S,
        )
        axis.plot(
            time_s[mask],
            method_data["modes"][mode_index, mask],
            color=colors[member.method],
            linewidth=0.65,
        )
        axis.axvspan(
            member.event_start_time_s,
            member.event_end_time_s,
            color="tab:red",
            alpha=0.12,
        )
        axis.set_ylabel(f"{member.method}\n{member.mode} (nm)")
        axis.set_title(
            f"local {member.representative_local_frequency_hz:.2f} Hz | "
            f"score {member.peak_anomaly_score:.2f} | "
            f"driver: {member.dominant_deviation_feature}",
            fontsize=9,
        )

    for axis in axes:
        axis.axvspan(
            cluster["cluster_start_time_s"],
            cluster["cluster_end_time_s"],
            color="tab:red",
            alpha=0.06,
        )
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"{cluster['consensus_cluster_id']} | {cluster['consensus_level']} | "
        f"{cluster['representative_frequency_hz']:.2f} Hz | "
        f"methods: {cluster['supporting_methods']}"
    )
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    cluster_path = CONSENSUS_ROOT / "all_candidate_clusters.csv"
    membership_path = CONSENSUS_ROOT / "candidate_event_membership.csv"
    for path in (cluster_path, membership_path, ANGLE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"缺少前一步結果：{path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    removed_plot_count = _remove_previous_event_plots()
    clusters = pd.read_csv(cluster_path, encoding="utf-8-sig")
    membership = pd.read_csv(membership_path, encoding="utf-8-sig")
    angle_timeline = pd.read_csv(ANGLE_PATH, encoding="utf-8-sig")
    selected_clusters, evidence = select_consensus_evidence(clusters, membership)
    decompositions = _load_decompositions()

    evidence_columns = [
        "consensus_cluster_id", "consensus_level", "method_count",
        "supporting_methods", "method", "mode", "angle_state_label",
        "event_start_time_s", "event_end_time_s", "peak_time_s",
        "peak_anomaly_score", "dominant_deviation_feature",
        "global_peak_frequency_hz", "representative_local_frequency_hz",
        "local_frequency_low_hz", "local_frequency_high_hz",
        "physical_role_candidate", "interpretation",
    ]
    evidence[evidence_columns].to_csv(
        OUTPUT_ROOT / "consensus_event_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for cluster in selected_clusters.itertuples(index=False):
        cluster_series = pd.Series(cluster._asdict())
        members = evidence.loc[
            evidence["consensus_cluster_id"] == cluster.consensus_cluster_id
        ].sort_values(["method", "mode"])
        safe_name = (
            f"{cluster.consensus_cluster_id}_"
            f"{cluster.representative_time_s:.2f}s_"
            f"{cluster.representative_frequency_hz:.2f}Hz.png"
        )
        _plot_event(
            cluster_series,
            members,
            angle_timeline,
            decompositions,
            OUTPUT_ROOT / safe_name,
        )

    pd.DataFrame([{
        "minimum_method_count": 2,
        "event_context_padding_s": PADDING_S,
        "red_background": "consensus cluster time range",
        "red_mode_band": "individual method event range",
        "review_goal": "verify raw signal and matched IMFs change at the same time",
    }]).to_csv(
        OUTPUT_ROOT / "evidence_plot_settings.csv", index=False, encoding="utf-8-sig"
    )
    print("=" * 70)
    print("跨方法共識候選證據圖完成")
    print(f"共識候選群集：{len(selected_clusters)}")
    print(f"證據成員列數：{len(evidence)}")
    print(f"事件圖數量：{len(selected_clusters)}")
    print(f"已清除上次事件圖：{removed_plot_count}")
    print(f"結果位置：{OUTPUT_ROOT}")
    print("下一步：逐張檢查原始訊號與各方法模態是否同步出現可見變化。")


if __name__ == "__main__":
    main()
