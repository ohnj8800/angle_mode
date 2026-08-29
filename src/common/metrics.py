import numpy as np
import pandas as pd
from scipy.signal import hilbert

def calculate_mode_metrics(
    modes: np.ndarray,
    sampling_rate: float,
    algorithm_center_frequencies_hz: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    計算每個IMF的中心頻率、峰值頻率、能量與能量占比。
    """
    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 2:
        raise ValueError(
            "modes形狀必須為(mode_count, sample_count)。"
        )

    mode_count, sample_count = modes.shape

    if algorithm_center_frequencies_hz is not None:
        algorithm_centers = np.asarray(
            algorithm_center_frequencies_hz,
            dtype=float,
        )

        if len(algorithm_centers) != mode_count:
            raise ValueError(
                "algorithm_center_frequencies_hz數量"
                "必須與模態數相同。"
            )
    else:
        algorithm_centers = np.full(
            mode_count,
            np.nan,
        )

    energies = np.sum(modes ** 2, axis=1)
    total_mode_energy = float(np.sum(energies))

    rows = []

    for mode_index, mode in enumerate(modes):
        frequency_hz = np.fft.rfftfreq(
            sample_count,
            d=1.0 / sampling_rate,
        )

        spectrum = np.fft.rfft(mode)
        spectral_energy = np.abs(spectrum) ** 2

        # 排除0 Hz。
        positive_mask = frequency_hz > 0

        positive_frequency = frequency_hz[positive_mask]
        positive_energy = spectral_energy[positive_mask]

        energy_sum = float(np.sum(positive_energy))

        if energy_sum > np.finfo(float).eps:
            spectral_centroid_hz = float(
                np.sum(
                    positive_frequency
                    * positive_energy
                )
                / energy_sum
            )

            peak_frequency_hz = float(
                positive_frequency[
                    np.argmax(positive_energy)
                ]
            )
        else:
            spectral_centroid_hz = float("nan")
            peak_frequency_hz = float("nan")

        energy = float(energies[mode_index])

        if total_mode_energy > 0:
            energy_ratio = energy / total_mode_energy
        else:
            energy_ratio = float("nan")

        rows.append(
            {
                "mode": f"IMF{mode_index + 1}",
                "algorithm_center_frequency_hz": float(
                    algorithm_centers[mode_index]
                ),
                "spectral_centroid_hz": spectral_centroid_hz,
                "peak_frequency_hz": peak_frequency_hz,
                "energy_nm2": energy,
                "energy_ratio": energy_ratio,
                "rms_nm": float(
                    np.sqrt(np.mean(mode ** 2))
                ),
                "standard_deviation_nm": float(
                    np.std(mode)
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_reconstruction_metrics(
    original_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
) -> dict:
    """計算訊號重建品質。"""
    original_signal = np.asarray(
        original_signal,
        dtype=float,
    )

    reconstructed_signal = np.asarray(
        reconstructed_signal,
        dtype=float,
    )

    if original_signal.shape != reconstructed_signal.shape:
        raise ValueError("原始訊號與重建訊號長度不同。")

    residual = original_signal - reconstructed_signal

    correlation = float(
        np.corrcoef(
            original_signal,
            reconstructed_signal,
        )[0, 1]
    )

    relative_error = float(
        np.linalg.norm(residual)
        / (
            np.linalg.norm(original_signal)
            + np.finfo(float).eps
        )
    )

    rmse = float(
        np.sqrt(np.mean(residual ** 2))
    )

    return {
        "reconstruction_correlation": correlation,
        "relative_reconstruction_error": relative_error,
        "reconstruction_rmse_nm": rmse,
        "passes_correlation_threshold_0_90": (
            correlation >= 0.90
        ),
    }

def _safe_correlation_matrix(
    signals: np.ndarray,
) -> np.ndarray:
    """
    計算相關係數矩陣，並處理接近常數的訊號。
    """
    signals = np.asarray(signals, dtype=float)

    centered = (
        signals
        - np.mean(signals, axis=1, keepdims=True)
    )

    norms = np.linalg.norm(
        centered,
        axis=1,
    )

    denominator = np.outer(norms, norms)

    correlation = np.divide(
        centered @ centered.T,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=float,
        ),
        where=denominator > np.finfo(float).eps,
    )

    np.fill_diagonal(correlation, 1.0)

    return np.clip(correlation, -1.0, 1.0)


def calculate_spectral_overlap_matrix(
    modes: np.ndarray,
) -> np.ndarray:
    """
    計算模態間的頻譜重疊程度。

    0代表幾乎沒有重疊。
    1代表頻譜分布幾乎相同。
    """
    modes = np.asarray(modes, dtype=float)

    power_spectra = np.abs(
        np.fft.rfft(modes, axis=1)
    ) ** 2

    power_sum = np.sum(
        power_spectra,
        axis=1,
        keepdims=True,
    )

    normalized_power = np.divide(
        power_spectra,
        power_sum,
        out=np.zeros_like(power_spectra),
        where=power_sum > np.finfo(float).eps,
    )

    mode_count = len(modes)

    overlap_matrix = np.eye(
        mode_count,
        dtype=float,
    )

    for first_index in range(mode_count):
        for second_index in range(
            first_index + 1,
            mode_count,
        ):
            overlap = float(
                np.sum(
                    np.minimum(
                        normalized_power[first_index],
                        normalized_power[second_index],
                    )
                )
            )

            overlap_matrix[
                first_index,
                second_index,
            ] = overlap

            overlap_matrix[
                second_index,
                first_index,
            ] = overlap

    return overlap_matrix


def calculate_mode_relationships(
    modes: np.ndarray,
    waveform_correlation_threshold: float = 0.50,
    envelope_correlation_threshold: float = 0.50,
    spectral_overlap_threshold: float = 0.30,
) -> dict:
    """
    計算IOVMD模態間的獨立性。

    重複候選條件：
    1. 原波形絕對相關係數 >= 0.50；或
    2. 包絡相關 >= 0.50，而且頻譜重疊 >= 0.30。
    """
    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 2:
        raise ValueError(
            "modes必須是二維陣列。"
        )

    waveform_correlation = (
        _safe_correlation_matrix(modes)
    )

    envelopes = np.abs(
        hilbert(modes, axis=1)
    )

    envelope_correlation = (
        _safe_correlation_matrix(envelopes)
    )

    spectral_overlap = (
        calculate_spectral_overlap_matrix(modes)
    )

    pair_rows = []
    mode_count = len(modes)

    for first_index in range(mode_count):
        for second_index in range(
            first_index + 1,
            mode_count,
        ):
            waveform_value = float(
                abs(
                    waveform_correlation[
                        first_index,
                        second_index,
                    ]
                )
            )

            envelope_value = float(
                envelope_correlation[
                    first_index,
                    second_index,
                ]
            )

            overlap_value = float(
                spectral_overlap[
                    first_index,
                    second_index,
                ]
            )

            high_waveform_similarity = (
                waveform_value
                >= waveform_correlation_threshold
            )

            high_envelope_similarity = (
                envelope_value
                >= envelope_correlation_threshold
            )

            high_spectral_overlap = (
                overlap_value
                >= spectral_overlap_threshold
            )

            redundant_candidate = (
                high_waveform_similarity
                or (
                    high_envelope_similarity
                    and high_spectral_overlap
                )
            )

            shared_modulation_only = (
                high_envelope_similarity
                and not high_spectral_overlap
                and not high_waveform_similarity
            )

            if redundant_candidate:
                interpretation = (
                    "possible_redundant_modes"
                )
            elif shared_modulation_only:
                interpretation = (
                    "shared_modulation_not_redundant"
                )
            else:
                interpretation = "independent"

            pair_rows.append(
                {
                    "mode_1": f"IMF{first_index + 1}",
                    "mode_2": f"IMF{second_index + 1}",
                    "absolute_waveform_correlation": (
                        waveform_value
                    ),
                    "envelope_correlation": (
                        envelope_value
                    ),
                    "spectral_overlap": overlap_value,
                    "redundant_candidate": (
                        redundant_candidate
                    ),
                    "shared_modulation_only": (
                        shared_modulation_only
                    ),
                    "interpretation": interpretation,
                }
            )

    pair_table = pd.DataFrame(pair_rows)

    # 每個模態取與其他模態最大的波形相關程度。
    absolute_waveform = np.abs(
        waveform_correlation.copy()
    )

    np.fill_diagonal(
        absolute_waveform,
        0.0,
    )

    maximum_other_correlation = np.max(
        absolute_waveform,
        axis=1,
    )

    independence_table = pd.DataFrame(
        {
            "mode": [
                f"IMF{index + 1}"
                for index in range(mode_count)
            ],
            "maximum_absolute_correlation": (
                maximum_other_correlation
            ),
            "independence_score": (
                1.0 - maximum_other_correlation
            ),
        }
    )

    return {
        "waveform_correlation": waveform_correlation,
        "envelope_correlation": envelope_correlation,
        "spectral_overlap": spectral_overlap,
        "pair_table": pair_table,
        "independence_table": independence_table,
        "thresholds": {
            "waveform_correlation_threshold": (
                waveform_correlation_threshold
            ),
            "envelope_correlation_threshold": (
                envelope_correlation_threshold
            ),
            "spectral_overlap_threshold": (
                spectral_overlap_threshold
            ),
        },
    }