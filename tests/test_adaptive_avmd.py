import numpy as np

from src.methods.avmd import select_adaptive_avmd_k


def test_adaptive_k_stays_within_bounds_and_records_trials() -> None:
    sampling_rate = 200.0
    time_s = np.arange(0.0, 4.0, 1.0 / sampling_rate)
    signal = (
        np.sin(2.0 * np.pi * 8.0 * time_s)
        + 0.5 * np.sin(2.0 * np.pi * 22.0 * time_s)
    )

    result = select_adaptive_avmd_k(
        signal=signal,
        sampling_rate=sampling_rate,
        initial_mode_count=4,
        alpha=500.0,
        minimum_mode_count=3,
        maximum_mode_count=5,
        maximum_steps=2,
        maximum_iterations=150,
        maximum_frequency_hz=40.0,
    )

    assert 3 <= result["best_mode_count"] <= 5
    assert len(result["trials"]) >= 2
    assert result["evaluated_mode_counts"] == [3, 4, 5]
    assert result["search_strategy"] == "global"
    assert np.isfinite(result["best_score"])
