import numpy as np

from src.common.angle_tracking import (
    assign_angle_state_labels,
    detect_angle_transitions,
    estimate_angle_from_component,
)


def create_synthetic_angle_signal(sampling_rate: float = 20.0):
    segments = [
        np.full(100, 1549.91),
        np.linspace(1549.91, 1549.67, 40),
        np.full(100, 1549.67),
        np.linspace(1549.67, 1549.91, 40),
        np.full(100, 1549.91),
    ]
    wavelength = np.concatenate(segments)
    time_s = np.arange(wavelength.size) / sampling_rate
    return time_s, wavelength


def test_two_angle_calibration_maps_platforms():
    _, wavelength = create_synthetic_angle_signal()
    result = estimate_angle_from_component(
        wavelength,
        sampling_rate=20.0,
        edge_guard_s=0.0,
    )
    angle = result["estimated_angle_deg"]
    assert np.isclose(np.median(angle[10:90]), 10.0, atol=0.2)
    assert np.isclose(np.median(angle[150:230]), 43.0, atol=0.2)
    assert result["slope_deg_per_nm"] < 0.0


def test_transition_detector_finds_both_directions():
    time_s, wavelength = create_synthetic_angle_signal()
    calibration = estimate_angle_from_component(
        wavelength,
        sampling_rate=20.0,
        edge_guard_s=0.0,
    )
    transitions = detect_angle_transitions(
        time_s,
        calibration["estimated_angle_deg"],
        minimum_duration_s=0.2,
    )
    assert transitions["direction"].tolist() == ["10_to_43", "43_to_10"]
    labels = assign_angle_state_labels(
        calibration["estimated_angle_deg"],
        transitions,
    )
    assert np.any(labels == "moving_10_to_43")
    assert np.any(labels == "moving_43_to_10")
