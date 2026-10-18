import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.common.preprocessing import calculate_welch_psd


def merge_redundant_modes(
    modes: np.ndarray,
    pair_table: pd.DataFrame,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    將被標記為redundant_candidate的模態合併。

    使用群組合併：
    若IMF1與IMF2重複，IMF2又與IMF3重複，
    三者會被視為同一組。
    """
    modes = np.asarray(modes, dtype=float)
    mode_count = len(modes)

    parent = list(range(mode_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)

        if first_root != second_root:
            parent[second_root] = first_root

    if len(pair_table) > 0:
        redundant_mask = (
            pair_table["redundant_candidate"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        redundant_pairs = pair_table.loc[
            redundant_mask
        ]

        for _, row in redundant_pairs.iterrows():
            first_index = (
                int(
                    str(row["mode_1"])
                    .replace("IMF", "")
                )
                - 1
            )

            second_index = (
                int(
                    str(row["mode_2"])
                    .replace("IMF", "")
                )
                - 1
            )

            union(first_index, second_index)

    groups_dictionary = {}

    for mode_index in range(mode_count):
        root = find(mode_index)

        groups_dictionary.setdefault(
            root,
            [],
        ).append(mode_index)

    groups = list(groups_dictionary.values())

    merged_modes = np.vstack(
        [
            np.sum(
                modes[group],
                axis=0,
            )
            for group in groups
        ]
    )

    return merged_modes, groups


def evaluate_welch_support(
    mode_metrics: pd.DataFrame,
    reference_signal: np.ndarray,
    sampling_rate: float = 200.0,
    frequency_tolerance_hz: float = 1.0,
    minimum_prominence_ratio: float = 0.01,
    minimum_peak_distance_hz: float = 0.5,
    minimum_frequency_hz: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """
    檢查IMF峰值頻率附近是否存在Welch PSD峰值。

    PSD支持的定義：
    IMF峰值頻率與有效Welch峰值相差不超過1 Hz。
    """
    frequency_hz, psd = calculate_welch_psd(
        signal=reference_signal,
        sampling_rate=sampling_rate,
        nperseg=4096,
    )

    frequency_resolution_hz = float(
        np.median(np.diff(frequency_hz))
    )

    frequency_mask = (
        (frequency_hz >= minimum_frequency_hz)
        & (frequency_hz < sampling_rate / 2.0)
    )

    selected_frequency = frequency_hz[frequency_mask]
    selected_psd = psd[frequency_mask]

    maximum_psd = float(
        np.max(selected_psd)
    )

    prominence_threshold = (
        maximum_psd
        * minimum_prominence_ratio
    )

    distance_bins = max(
        1,
        int(
            np.ceil(
                minimum_peak_distance_hz
                / frequency_resolution_hz
            )
        ),
    )

    peak_indices, properties = find_peaks(
        selected_psd,
        prominence=prominence_threshold,
        distance=distance_bins,
    )

    welch_peak_frequencies = (
        selected_frequency[peak_indices]
    )

    welch_peak_psd = selected_psd[peak_indices]

    output_metrics = mode_metrics.copy()

    nearest_frequencies = []
    nearest_psd_values = []
    frequency_differences = []
    support_results = []

    for mode_peak_frequency in output_metrics[
        "peak_frequency_hz"
    ]:
        if len(welch_peak_frequencies) == 0:
            nearest_frequencies.append(np.nan)
            nearest_psd_values.append(np.nan)
            frequency_differences.append(np.nan)
            support_results.append(False)
            continue

        difference = np.abs(
            welch_peak_frequencies
            - mode_peak_frequency
        )

        nearest_index = int(
            np.argmin(difference)
        )

        nearest_frequency = float(
            welch_peak_frequencies[nearest_index]
        )

        nearest_psd = float(
            welch_peak_psd[nearest_index]
        )

        nearest_difference = float(
            difference[nearest_index]
        )

        nearest_frequencies.append(
            nearest_frequency
        )

        nearest_psd_values.append(
            nearest_psd
        )

        frequency_differences.append(
            nearest_difference
        )

        support_results.append(
            nearest_difference
            <= frequency_tolerance_hz
        )

    output_metrics[
        "nearest_welch_peak_hz"
    ] = nearest_frequencies

    output_metrics[
        "welch_frequency_difference_hz"
    ] = frequency_differences

    output_metrics[
        "nearest_welch_psd_nm2_per_hz"
    ] = nearest_psd_values

    output_metrics[
        "welch_psd_supported"
    ] = support_results

    psd_information = {
        "frequency_hz": selected_frequency,
        "psd": selected_psd,
        "peak_frequencies_hz": (
            welch_peak_frequencies
        ),
        "peak_psd": welch_peak_psd,
        "frequency_resolution_hz": (
            frequency_resolution_hz
        ),
        "frequency_tolerance_hz": (
            frequency_tolerance_hz
        ),
        "minimum_prominence_ratio": (
            minimum_prominence_ratio
        ),
        "peak_prominences": (
            properties.get(
                "prominences",
                np.array([]),
            )
        ),
    }

    return output_metrics, psd_information


def select_welch_guided_mode(
    mode_metrics: pd.DataFrame,
    psd_information: dict,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    maximum_frequency_difference_hz: float = 3.0,
) -> dict:
    """Select the IMF that best represents vibration in a known motor band.

    The energy-dominant IMF is intentionally kept separate: slow structural
    motion can contain more energy than the rotating component.  This selector
    first finds the strongest Welch bin inside the requested motor band, then
    chooses the IMF whose peak is nearest to that reference frequency.
    """
    if maximum_frequency_hz <= minimum_frequency_hz:
        raise ValueError(
            "maximum_frequency_hz must be greater than minimum_frequency_hz."
        )
    if maximum_frequency_difference_hz <= 0:
        raise ValueError("maximum_frequency_difference_hz must be positive.")
    if mode_metrics.empty:
        raise ValueError("mode_metrics is empty.")

    frequency_hz = np.asarray(psd_information["frequency_hz"], dtype=float)
    psd = np.asarray(psd_information["psd"], dtype=float)
    band_mask = (
        np.isfinite(frequency_hz)
        & np.isfinite(psd)
        & (frequency_hz >= minimum_frequency_hz)
        & (frequency_hz <= maximum_frequency_hz)
    )

    candidates = mode_metrics.loc[
        mode_metrics["peak_frequency_hz"].notna()
        & mode_metrics["peak_frequency_hz"].between(
            minimum_frequency_hz, maximum_frequency_hz, inclusive="both"
        )
    ].copy()

    if not np.any(band_mask):
        fallback = mode_metrics.loc[mode_metrics["energy_ratio"].idxmax()]
        return {
            "mode": str(fallback["mode"]),
            "mode_peak_frequency_hz": float(fallback["peak_frequency_hz"]),
            "mode_energy_ratio": float(fallback["energy_ratio"]),
            "reference_welch_frequency_hz": float("nan"),
            "frequency_difference_hz": float("nan"),
            "frequency_agreement": False,
            "selection_reason": "energy_fallback_no_mode_in_tracking_band",
        }

    selected_frequency = frequency_hz[band_mask]
    selected_psd = psd[band_mask]
    reference_frequency = float(selected_frequency[int(np.argmax(selected_psd))])
    if candidates.empty:
        fallback = mode_metrics.loc[mode_metrics["energy_ratio"].idxmax()]
        fallback_frequency = float(fallback["peak_frequency_hz"])
        difference = abs(fallback_frequency - reference_frequency)
        return {
            "mode": str(fallback["mode"]),
            "mode_peak_frequency_hz": fallback_frequency,
            "mode_energy_ratio": float(fallback["energy_ratio"]),
            "reference_welch_frequency_hz": reference_frequency,
            "frequency_difference_hz": difference,
            "frequency_agreement": False,
            "selection_reason": "energy_fallback_no_mode_in_tracking_band",
        }
    candidates["tracking_difference_hz"] = (
        candidates["peak_frequency_hz"] - reference_frequency
    ).abs()
    # Frequency agreement is the primary criterion.  Energy only breaks ties
    # caused by the finite FFT resolution.
    selected = candidates.sort_values(
        ["tracking_difference_hz", "energy_ratio"],
        ascending=[True, False],
    ).iloc[0]
    difference = float(selected["tracking_difference_hz"])

    return {
        "mode": str(selected["mode"]),
        "mode_peak_frequency_hz": float(selected["peak_frequency_hz"]),
        "mode_energy_ratio": float(selected["energy_ratio"]),
        "reference_welch_frequency_hz": reference_frequency,
        "frequency_difference_hz": difference,
        "frequency_agreement": bool(
            difference <= maximum_frequency_difference_hz
        ),
        "selection_reason": "nearest_mode_to_motor_band_welch_peak",
    }
