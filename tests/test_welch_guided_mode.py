import numpy as np
import pandas as pd

from src.common.postprocessing import select_welch_guided_mode


def test_welch_guided_mode_ignores_high_energy_low_frequency_mode() -> None:
    metrics = pd.DataFrame(
        {
            "mode": ["IMF1", "IMF2", "IMF3"],
            "peak_frequency_hz": [0.8, 29.7, 61.0],
            "energy_ratio": [0.70, 0.20, 0.10],
        }
    )
    frequency = np.linspace(0.5, 100.0, 200)
    psd = np.ones_like(frequency)
    psd[np.argmin(np.abs(frequency - 30.0))] = 100.0

    selected = select_welch_guided_mode(
        metrics,
        {"frequency_hz": frequency, "psd": psd},
        minimum_frequency_hz=20.0,
        maximum_frequency_hz=40.0,
        maximum_frequency_difference_hz=3.0,
    )

    assert selected["mode"] == "IMF2"
    assert selected["frequency_agreement"]
    assert selected["frequency_difference_hz"] < 1.0


def test_welch_guided_mode_flags_missing_motor_band_mode() -> None:
    metrics = pd.DataFrame(
        {
            "mode": ["IMF1", "IMF2"],
            "peak_frequency_hz": [0.8, 60.0],
            "energy_ratio": [0.8, 0.2],
        }
    )
    frequency = np.linspace(0.5, 100.0, 200)
    psd = np.ones_like(frequency)

    selected = select_welch_guided_mode(
        metrics,
        {"frequency_hz": frequency, "psd": psd},
        minimum_frequency_hz=20.0,
        maximum_frequency_hz=40.0,
    )

    assert selected["mode"] == "IMF1"
    assert not selected["frequency_agreement"]
    assert selected["selection_reason"] == "energy_fallback_no_mode_in_tracking_band"
