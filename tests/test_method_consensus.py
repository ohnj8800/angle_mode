import pandas as pd

from src.common.method_consensus import compare_method_candidates


def test_groups_events_with_close_time_and_frequency_across_methods():
    events = pd.DataFrame(
        [
            {
                "method": "IOVMD", "mode": "IMF2", "angle_state_label": "stable_10deg",
                "event_start_time_s": 10.0, "event_end_time_s": 13.0, "peak_time_s": 11.0,
                "peak_anomaly_score": 5.0, "global_peak_frequency_hz": 20.0,
                "representative_local_frequency_hz": 20.0,
            },
            {
                "method": "AVMD", "mode": "IMF3", "angle_state_label": "stable_10deg",
                "event_start_time_s": 12.0, "event_end_time_s": 14.0, "peak_time_s": 13.0,
                "peak_anomaly_score": 6.0, "global_peak_frequency_hz": 20.5,
                "representative_local_frequency_hz": 20.5,
            },
            {
                "method": "SVMD", "mode": "IMF4", "angle_state_label": "stable_10deg",
                "event_start_time_s": 30.0, "event_end_time_s": 32.0, "peak_time_s": 31.0,
                "peak_anomaly_score": 4.0, "global_peak_frequency_hz": 50.0,
                "representative_local_frequency_hz": 50.0,
            },
        ]
    )

    clusters, membership, pairs = compare_method_candidates(events)

    assert sorted(clusters["method_count"].tolist()) == [1, 2]
    assert len(clusters.loc[clusters["consensus_level"] == "two_method_consensus"]) == 1
    assert len(membership) == 3
    assert len(pairs) == 1
