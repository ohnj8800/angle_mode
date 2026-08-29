import pandas as pd

from src.common.physical_episodes import build_physical_episodes


def test_merges_overlapping_frequency_clusters_into_one_angle_episode():
    clusters = pd.DataFrame(
        [
            {
                "consensus_cluster_id": "C001", "method_count": 3,
                "cluster_start_time_s": 10.0, "cluster_end_time_s": 14.0,
                "representative_time_s": 12.0, "representative_frequency_hz": 20.0,
                "supporting_methods": "AVMD,IOVMD,SVMD", "consensus_level": "three_method_consensus",
                "angle_state_label": "stable_10deg", "maximum_anomaly_score": 6.0,
            },
            {
                "consensus_cluster_id": "C002", "method_count": 2,
                "cluster_start_time_s": 13.0, "cluster_end_time_s": 15.0,
                "representative_time_s": 14.0, "representative_frequency_hz": 50.0,
                "supporting_methods": "AVMD,SVMD", "consensus_level": "two_method_consensus",
                "angle_state_label": "stable_10deg", "maximum_anomaly_score": 5.0,
            },
        ]
    )
    angle = pd.DataFrame(
        {
            "time_s": [0.0, 10.0, 12.0, 14.0, 18.0],
            "estimated_angle_deg": [10.0, 10.0, 10.4, 10.8, 10.8],
            "angle_velocity_deg_s": [0.0, 0.0, 0.2, 0.3, 0.0],
        }
    )
    episodes, mapping = build_physical_episodes(clusters, angle)

    assert len(episodes) == 1
    assert len(mapping) == 2
    assert episodes.loc[0, "response_frequency_bands_hz"] == "20.00,50.00"
    assert episodes.loc[0, "episode_classification"] == "angle_associated_dynamic_response"
