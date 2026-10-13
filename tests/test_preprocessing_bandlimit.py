import numpy as np

from src.common.preprocessing import preprocess_fbg_signal


def _fft_amplitude(signal: np.ndarray, sampling_rate: float, target_hz: float) -> float:
    frequency = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    amplitude = np.abs(np.fft.rfft(signal))
    return float(amplitude[np.argmin(np.abs(frequency - target_hz))])


def test_optional_dynamic_lowpass_suppresses_out_of_band_component() -> None:
    sampling_rate = 200.0
    time_s = np.arange(0.0, 5.0, 1.0 / sampling_rate)
    raw = (
        1550.0
        + np.sin(2.0 * np.pi * 10.0 * time_s)
        + 0.5 * np.sin(2.0 * np.pi * 70.0 * time_s)
    )

    result = preprocess_fbg_signal(
        raw,
        sampling_rate=sampling_rate,
        dynamic_lowpass_cutoff_hz=40.0,
    )
    dynamic = result["dynamic_nm"]

    assert _fft_amplitude(dynamic, sampling_rate, 10.0) > 100.0
    assert _fft_amplitude(dynamic, sampling_rate, 70.0) < 1.0
