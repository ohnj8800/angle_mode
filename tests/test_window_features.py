import numpy as np

from src.common.window_features import extract_mode_window_features


def test_window_features_track_frequency_and_energy_change() -> None:
    sampling_rate = 200.0
    time_s = np.arange(0.0, 6.0, 1.0 / sampling_rate)
    first_half = time_s < 3.0
    mode = np.where(
        first_half,
        np.sin(2.0 * np.pi * 10.0 * time_s),
        2.0 * np.sin(2.0 * np.pi * 20.0 * time_s),
    )
    labels = np.where(first_half, "stable_10deg", "stable_43deg")
    features = extract_mode_window_features(
        method="TEST",
        time_s=time_s,
        modes=mode[None, :],
        state_labels=labels,
        estimated_angle_deg=np.where(first_half, 10.0, 43.0),
        angle_velocity_deg_s=np.zeros_like(time_s),
        global_peak_frequencies_hz=np.array([20.0]),
        physical_roles=np.array(["background_dynamic_mode"]),
        sampling_rate=sampling_rate,
        window_duration_s=2.0,
        step_duration_s=1.0,
    )
    early = features.loc[features["center_time_s"] <= 2.0]
    late = features.loc[features["center_time_s"] >= 4.0]
    assert np.allclose(early["local_peak_frequency_hz"], 10.0, atol=0.5)
    assert np.allclose(late["local_peak_frequency_hz"], 20.0, atol=0.5)
    assert late["rms_nm"].mean() > 1.8 * early["rms_nm"].mean()
