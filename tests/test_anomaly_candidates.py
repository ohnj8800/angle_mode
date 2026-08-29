import pandas as pd

from src.common.anomaly_candidates import (
    merge_consecutive_candidates,
    score_stable_mode_windows,
)


def test_scores_and_merges_two_consecutive_candidate_windows():
    rows = []
    for index, rms in enumerate([1.0, 1.1, 2.0, 2.2, 1.0]):
        rows.append({
            "method": "TEST", "mode": "IMF2",
            "physical_role_candidate": "background_dynamic_mode",
            "global_peak_frequency_hz": 10.0,
            "window_index": index, "start_time_s": index * 0.5,
            "end_time_s": index * 0.5 + 2.0,
            "center_time_s": index * 0.5 + 1.0,
            "angle_state_label": "stable_10deg",
            "dominant_state_label": "stable_10deg", "moving_fraction": 0.0,
            "local_peak_frequency_hz": 10.0,
            "rms_nm": rms, "envelope_mean_nm": rms,
        })
    features = pd.DataFrame(rows)
    baseline_rows = []
    for feature in ("local_peak_frequency_hz", "rms_nm", "envelope_mean_nm"):
        baseline_rows.append({
            "method": "TEST", "mode": "IMF2", "angle_state_label": "stable_10deg",
            "feature": feature, "median": 10.0 if feature.startswith("local") else 1.0,
            "robust_sigma": 0.2, "standard_deviation": 0.2,
        })
    baseline = pd.DataFrame(baseline_rows)

    scored = score_stable_mode_windows(features, baseline, candidate_threshold=3.5)
    events = merge_consecutive_candidates(scored, minimum_consecutive_windows=2)

    assert scored["anomaly_candidate"].tolist() == [False, False, True, True, False]
    assert len(events) == 1
    assert events.loc[0, "candidate_window_count"] == 2
    assert events.loc[0, "interpretation"] == "anomaly_candidate_not_confirmed_fault"
