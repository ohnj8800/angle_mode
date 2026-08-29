import numpy as np
import pandas as pd
from scipy.signal import find_peaks, get_window, hilbert


def calculate_envelope_spectrum(
    mode: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    計算單一IMF的Hilbert包絡與包絡振幅頻譜。
    """
    mode = np.asarray(mode, dtype=float)

    if mode.ndim != 1:
        raise ValueError("mode必須是一維訊號。")

    analytic_signal = hilbert(mode)
    envelope = np.abs(analytic_signal)

    # 移除包絡平均值，否則0 Hz會非常大。
    centered_envelope = envelope - np.mean(envelope)

    sample_count = len(centered_envelope)

    window = get_window(
        "hann",
        sample_count,
        fftbins=True,
    )

    windowed_envelope = (
        centered_envelope * window
    )

    spectrum = np.fft.rfft(
        windowed_envelope
    )

    frequency_hz = np.fft.rfftfreq(
        sample_count,
        d=1.0 / sampling_rate,
    )

    amplitude = (
        2.0
        * np.abs(spectrum)
        / np.sum(window)
    )

    amplitude[0] = amplitude[0] / 2.0

    if sample_count % 2 == 0:
        amplitude[-1] = amplitude[-1] / 2.0

    return frequency_hz, amplitude, envelope


def find_envelope_peaks(
    modes: np.ndarray,
    sampling_rate: float,
    minimum_frequency_hz: float = 0.05,
    maximum_frequency_hz: float = 5.0,
    minimum_prominence_ratio: float = 0.05,
    minimum_peak_distance_hz: float = 0.10,
    maximum_peaks_per_mode: int = 5,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    找出每個IMF包絡頻譜中最明顯的候選調變頻率。
    """
    modes = np.asarray(modes, dtype=float)

    peak_rows = []
    spectrum_results = []

    for mode_index, mode in enumerate(modes):
        frequency_hz, amplitude, envelope = (
            calculate_envelope_spectrum(
                mode=mode,
                sampling_rate=sampling_rate,
            )
        )

        frequency_mask = (
            (frequency_hz >= minimum_frequency_hz)
            & (frequency_hz <= maximum_frequency_hz)
        )

        selected_frequency = (
            frequency_hz[frequency_mask]
        )

        selected_amplitude = (
            amplitude[frequency_mask]
        )

        maximum_amplitude = float(
            np.max(selected_amplitude)
        )

        if maximum_amplitude > 0:
            normalized_amplitude = (
                selected_amplitude
                / maximum_amplitude
            )
        else:
            normalized_amplitude = (
                np.zeros_like(selected_amplitude)
            )

        frequency_resolution = float(
            np.median(
                np.diff(selected_frequency)
            )
        )

        distance_bins = max(
            1,
            int(
                np.ceil(
                    minimum_peak_distance_hz
                    / frequency_resolution
                )
            ),
        )

        peak_indices, peak_properties = (
            find_peaks(
                normalized_amplitude,
                prominence=(
                    minimum_prominence_ratio
                ),
                distance=distance_bins,
            )
        )

        # 先依prominence由大到小，最多保留5個。
        if len(peak_indices) > 0:
            prominence_order = np.argsort(
                peak_properties["prominences"]
            )[::-1]

            retained_order = prominence_order[
                :maximum_peaks_per_mode
            ]

            retained_peak_indices = (
                peak_indices[retained_order]
            )

            retained_prominences = (
                peak_properties["prominences"][
                    retained_order
                ]
            )

            # 輸出時再依頻率排序。
            frequency_order = np.argsort(
                selected_frequency[
                    retained_peak_indices
                ]
            )

            retained_peak_indices = (
                retained_peak_indices[
                    frequency_order
                ]
            )

            retained_prominences = (
                retained_prominences[
                    frequency_order
                ]
            )

            for peak_number, (
                peak_index,
                prominence,
            ) in enumerate(
                zip(
                    retained_peak_indices,
                    retained_prominences,
                ),
                start=1,
            ):
                peak_rows.append(
                    {
                        "mode": (
                            f"IMF{mode_index + 1}"
                        ),
                        "peak_number": peak_number,
                        "envelope_frequency_hz": float(
                            selected_frequency[
                                peak_index
                            ]
                        ),
                        "envelope_amplitude_nm": float(
                            selected_amplitude[
                                peak_index
                            ]
                        ),
                        "normalized_amplitude": float(
                            normalized_amplitude[
                                peak_index
                            ]
                        ),
                        "prominence": float(
                            prominence
                        ),
                    }
                )

        spectrum_results.append(
            {
                "mode": f"IMF{mode_index + 1}",
                "frequency_hz": (
                    selected_frequency
                ),
                "amplitude_nm": (
                    selected_amplitude
                ),
                "normalized_amplitude": (
                    normalized_amplitude
                ),
                "envelope_nm": envelope,
            }
        )

    peak_table = pd.DataFrame(peak_rows)

    return peak_table, spectrum_results