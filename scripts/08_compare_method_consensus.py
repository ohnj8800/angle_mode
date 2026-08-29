from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.method_consensus import compare_method_candidates


INPUT_PATH = PROJECT_ROOT / "results" / "anomaly_candidates" / "modal_anomaly_candidates.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "method_consensus"
MAXIMUM_TIME_GAP_S = 1.0
MINIMUM_FREQUENCY_TOLERANCE_HZ = 1.0
RELATIVE_FREQUENCY_TOLERANCE = 0.05


def _plot_consensus_map(clusters: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    styles = {
        "five_method_consensus": (
            "darkred", "*", "5-method consensus"
        ),
        "four_method_consensus": (
            "tab:red", "P", "4-method consensus"
        ),
        "three_method_consensus": (
            "tab:orange", "X", "3-method consensus"
        ),
        "two_method_consensus": (
            "goldenrod", "D", "2-method consensus"
        ),
        "single_method_only": (
            "0.65", "o", "single method only"
        ),
    }
    for level, (color, marker, label) in styles.items():
        subset = clusters.loc[clusters["consensus_level"] == level]
        axis.scatter(
            subset["representative_time_s"],
            subset["representative_frequency_hz"],
            s=35 + 10 * subset["maximum_anomaly_score"].clip(upper=10),
            color=color,
            marker=marker,
            alpha=0.8,
            label=label,
        )
    axis.set_xlabel("Representative candidate time (s)")
    axis.set_ylabel("Representative frequency (Hz)")
    axis.set_title("Agreement of anomaly candidates across VMD methods")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_agreement_counts(clusters: pd.DataFrame, output_path: Path) -> None:
    order = [
        "five_method_consensus",
        "four_method_consensus",
        "three_method_consensus",
        "two_method_consensus",
        "single_method_only",
    ]
    counts = clusters["consensus_level"].value_counts().reindex(order, fill_value=0)
    labels = ["5 methods", "4 methods", "3 methods", "2 methods", "1 method"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar(
        labels,
        counts.values,
        color=["darkred", "tab:red", "tab:orange", "goldenrod", "0.65"],
    )
    axis.bar_label(bars)
    axis.set_ylabel("Candidate cluster count")
    axis.set_title("Cross-method agreement level")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError("缺少異常候選事件，請先執行07_detect_anomaly_candidates.py。")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    clusters, membership, pair_matches = compare_method_candidates(
        events,
        maximum_time_gap_s=MAXIMUM_TIME_GAP_S,
        minimum_frequency_tolerance_hz=MINIMUM_FREQUENCY_TOLERANCE_HZ,
        relative_frequency_tolerance=RELATIVE_FREQUENCY_TOLERANCE,
    )
    consensus = clusters.loc[clusters["method_count"] >= 2].copy()
    clusters.to_csv(
        OUTPUT_ROOT / "all_candidate_clusters.csv", index=False, encoding="utf-8-sig"
    )
    consensus.to_csv(
        OUTPUT_ROOT / "cross_method_consensus_candidates.csv", index=False, encoding="utf-8-sig"
    )
    membership.to_csv(
        OUTPUT_ROOT / "candidate_event_membership.csv", index=False, encoding="utf-8-sig"
    )
    pair_matches.to_csv(
        OUTPUT_ROOT / "matched_event_pairs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{
        "maximum_time_gap_s": MAXIMUM_TIME_GAP_S,
        "minimum_frequency_tolerance_hz": MINIMUM_FREQUENCY_TOLERANCE_HZ,
        "relative_frequency_tolerance": RELATIVE_FREQUENCY_TOLERANCE,
        "matching_rule": "same angle state; different method; close in time and frequency",
        "five_method_meaning": "strongest repeatability; still not a confirmed fault",
        "four_method_meaning": "high repeatability; still not a confirmed fault",
        "three_method_meaning": "moderate cross-method support",
        "two_method_meaning": "partial cross-method support",
        "single_method_meaning": "method-sensitive candidate requiring review",
    }]).to_csv(
        OUTPUT_ROOT / "method_consensus_settings.csv", index=False, encoding="utf-8-sig"
    )
    _plot_consensus_map(clusters, OUTPUT_ROOT / "01_cross_method_consensus_map.png")
    _plot_agreement_counts(clusters, OUTPUT_ROOT / "02_agreement_level_counts.png")

    print("=" * 70)
    print("五種VMD異常候選共識比較完成")
    print(clusters["consensus_level"].value_counts().to_string())
    print(f"跨方法共識候選群集：{len(consensus)}")
    if not consensus.empty:
        print("共識候選摘要：")
        print(
            consensus[
                [
                    "consensus_cluster_id", "representative_time_s",
                    "representative_frequency_hz", "supporting_methods",
                    "consensus_level",
                ]
            ].to_string(index=False)
        )
    print(f"結果位置：{OUTPUT_ROOT}")
    print("注意：跨方法一致可提高候選可信度，但仍不能直接等同實際故障。")


if __name__ == "__main__":
    main()
