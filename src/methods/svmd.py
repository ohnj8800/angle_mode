from __future__ import annotations

import numpy as np


def _mirror_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    鏡射延伸訊號，降低分解時的端點效應。
    """
    sample_count = len(signal)
    half = sample_count // 2

    return np.concatenate(
        (
            signal[:half][::-1],
            signal,
            signal[half:][::-1],
        )
    )


def _create_alpha_schedule(
    minimum_alpha: float,
    maximum_alpha: float,
) -> list[float]:
    """
    建立SVMD逐步提高alpha的排程。

    每次抽取模態時，先使用較寬鬆的alpha，
    再逐漸提高到maximum_alpha，使模態逐漸窄頻化。
    """
    if not 0 < minimum_alpha <= maximum_alpha:
        raise ValueError(
            "alpha必須符合0 < minimum_alpha <= maximum_alpha。"
        )

    values = []

    alpha = float(minimum_alpha)
    m = 0.0
    bit_flag = 0
    guard_count = 0

    while alpha < maximum_alpha + 1.0:
        values.append(
            float(
                min(
                    alpha,
                    maximum_alpha,
                )
            )
        )

        if np.isclose(
            values[-1],
            maximum_alpha,
        ):
            break

        if (
            abs(
                m
                - np.log(maximum_alpha)
            )
            > 1.0
        ):
            m += 1.0
        else:
            m += 0.05
            bit_flag += 1

        if bit_flag >= 2:
            alpha += 1.0

        if alpha <= maximum_alpha - 1.0:
            if bit_flag == 1:
                alpha = maximum_alpha - 1.0
            else:
                alpha = float(np.exp(m))

        guard_count += 1

        if guard_count > 10000:
            raise RuntimeError(
                "SVMD alpha排程未正常結束。"
            )

    return values


def successive_variational_mode_decomposition(
    signal: np.ndarray,
    sampling_rate: float,
    maximum_alpha: float = 1000.0,
    tau: float = 0.0,
    tolerance: float = 1e-7,
    minimum_alpha: float = 10.0,
    maximum_iterations: int = 300,
    maximum_modes: int = 12,
    reconstruction_power_tolerance: float = 0.005,
) -> dict:
    """
    Successive Variational Mode Decomposition。

    與一般VMD同時求解K個模態不同，SVMD會：

    1. 抽取一個窄頻模態。
    2. 建立排除濾波條件，避免再次抽取相同頻帶。
    3. 繼續抽取下一模態。
    4. 當頻域剩餘誤差低於門檻時停止。

    Parameters
    ----------
    signal:
        一維輸入訊號，樣本數必須為偶數。

    sampling_rate:
        取樣率，單位Hz。

    maximum_alpha:
        每個模態最終使用的窄頻限制參數。

    tau:
        對偶上升步長。高雜訊訊號通常設為0。

    tolerance:
        單一模態迭代收斂門檻。

    minimum_alpha:
        alpha排程起始值。

    maximum_iterations:
        每個alpha階段最大迭代次數。

    maximum_modes:
        安全模態數上限，避免無限抽取。

    reconstruction_power_tolerance:
        頻域剩餘能量比例停止門檻。
        原始精確重建條件使用0.005。

    Returns
    -------
    dict:
        modes、中心頻率、重建訊號、殘差及停止資訊。
    """
    signal = np.asarray(
        signal,
        dtype=float,
    ).reshape(-1)

    if len(signal) < 50:
        raise ValueError(
            "SVMD輸入訊號至少需要50個樣本。"
        )

    if len(signal) % 2 != 0:
        raise ValueError(
            "SVMD輸入訊號樣本數必須為偶數。"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "SVMD輸入訊號包含NaN或無限值。"
        )

    if sampling_rate <= 0:
        raise ValueError(
            "sampling_rate必須大於0。"
        )

    if maximum_iterations < 1:
        raise ValueError(
            "maximum_iterations必須至少為1。"
        )

    if maximum_modes < 1:
        raise ValueError(
            "maximum_modes必須至少為1。"
        )

    if not (
        0.0
        < reconstruction_power_tolerance
        < 1.0
    ):
        raise ValueError(
            "reconstruction_power_tolerance"
            "必須介於0與1之間。"
        )

    epsilon = np.finfo(float).eps

    mirrored_signal = _mirror_signal(
        signal
    )

    total_sample_count = len(
        mirrored_signal
    )

    frequency_grid = (
        np.arange(
            total_sample_count,
            dtype=float,
        )
        / total_sample_count
        - 0.5
    )

    signal_spectrum = np.fft.fftshift(
        np.fft.fft(
            mirrored_signal
        )
    )

    one_sided_spectrum = (
        signal_spectrum.copy()
    )

    one_sided_spectrum[
        : total_sample_count // 2
    ] = 0.0

    signal_spectrum_energy = float(
        np.linalg.norm(
            one_sided_spectrum
        )
        ** 2
    )

    alpha_schedule = (
        _create_alpha_schedule(
            minimum_alpha=minimum_alpha,
            maximum_alpha=maximum_alpha,
        )
    )

    extracted_spectra = []
    normalized_centers = []
    residual_power_ratios = []
    iterations_per_mode = []
    exclusion_rows = []

    stop_reason = "maximum_modes"

    for _mode_index in range(
        maximum_modes
    ):
        current_mode = np.zeros(
            total_sample_count,
            dtype=complex,
        )

        center_frequency = 0.0
        mode_iteration_count = 0

        if extracted_spectra:
            sum_previous_modes = np.sum(
                np.stack(
                    extracted_spectra
                ),
                axis=0,
            )
        else:
            sum_previous_modes = np.zeros(
                total_sample_count,
                dtype=complex,
            )

        if exclusion_rows:
            sum_exclusion = np.sum(
                np.stack(
                    exclusion_rows
                ),
                axis=0,
            )
        else:
            sum_exclusion = np.zeros(
                total_sample_count,
                dtype=float,
            )

        for alpha in alpha_schedule:
            dual_variable = np.zeros(
                total_sample_count,
                dtype=complex,
            )

            for _iteration in range(
                maximum_iterations
            ):
                previous_mode = (
                    current_mode.copy()
                )

                previous_center = (
                    center_frequency
                )

                distance_squared = (
                    frequency_grid
                    - previous_center
                ) ** 2

                frequency_term = (
                    alpha ** 2
                    * distance_squared ** 2
                )

                denominator = (
                    1.0
                    + frequency_term
                    * (
                        1.0
                        + 2.0
                        * alpha
                        * distance_squared
                    )
                    + sum_exclusion
                )

                numerator = (
                    one_sided_spectrum
                    + frequency_term
                    * previous_mode
                    + dual_variable / 2.0
                )

                current_mode = (
                    numerator
                    / denominator
                )

                positive_slice = slice(
                    total_sample_count // 2,
                    total_sample_count,
                )

                positive_power = np.abs(
                    current_mode[
                        positive_slice
                    ]
                ) ** 2

                positive_power_sum = float(
                    np.sum(
                        positive_power
                    )
                )

                if (
                    positive_power_sum
                    > epsilon
                ):
                    center_frequency = float(
                        np.dot(
                            frequency_grid[
                                positive_slice
                            ],
                            positive_power,
                        )
                        / positive_power_sum
                    )

                inner_numerator = (
                    frequency_term
                    * (
                        one_sided_spectrum
                        - current_mode
                        - sum_previous_modes
                        + dual_variable / 2.0
                    )
                    - sum_previous_modes
                )

                inner_fraction = (
                    inner_numerator
                    / (
                        1.0
                        + frequency_term
                    )
                )

                reconstructed_spectrum = (
                    current_mode
                    + inner_fraction
                    + sum_previous_modes
                )

                dual_variable = (
                    dual_variable
                    + tau
                    * (
                        one_sided_spectrum
                        - reconstructed_spectrum
                    )
                )

                difference = (
                    current_mode
                    - previous_mode
                )

                relative_change = float(
                    np.abs(
                        epsilon
                        + np.vdot(
                            difference,
                            difference,
                        )
                        / (
                            np.vdot(
                                previous_mode,
                                previous_mode,
                            )
                            + epsilon
                        )
                    )
                )

                mode_iteration_count += 1

                if (
                    relative_change
                    <= tolerance
                ):
                    break

        extracted_spectra.append(
            current_mode.copy()
        )

        normalized_centers.append(
            center_frequency
        )

        iterations_per_mode.append(
            mode_iteration_count
        )

        distance_fourth = (
            frequency_grid
            - center_frequency
        ) ** 4

        exclusion_filter = (
            1.0
            / (
                maximum_alpha ** 2
                * distance_fourth
                + epsilon
            )
        )

        exclusion_rows.append(
            exclusion_filter
        )

        summed_modes = np.sum(
            np.stack(
                extracted_spectra
            ),
            axis=0,
        )

        residual_power_ratio = float(
            np.linalg.norm(
                summed_modes
                - one_sided_spectrum
            )
            ** 2
            / (
                signal_spectrum_energy
                + epsilon
            )
        )

        residual_power_ratios.append(
            residual_power_ratio
        )

        if (
            residual_power_ratio
            < reconstruction_power_tolerance
        ):
            stop_reason = (
                "reconstruction_power_tolerance"
            )
            break

    mode_count = len(
        extracted_spectra
    )

    full_spectra = np.zeros(
        (
            total_sample_count,
            mode_count,
        ),
        dtype=complex,
    )

    positive_modes = np.stack(
        extracted_spectra,
        axis=1,
    )

    full_spectra[
        total_sample_count // 2 :,
        :,
    ] = positive_modes[
        total_sample_count // 2 :,
        :,
    ]

    full_spectra[
        1 : total_sample_count // 2,
        :,
    ] = np.conj(
        positive_modes[
            total_sample_count - 1
            : total_sample_count // 2
            : -1,
            :,
        ]
    )

    full_spectra[
        total_sample_count // 2,
        :,
    ] = np.real(
        full_spectra[
            total_sample_count // 2,
            :,
        ]
    )

    full_spectra[0, :] = np.conj(
        full_spectra[-1, :]
    )

    mirrored_modes = np.real(
        np.fft.ifft(
            np.fft.ifftshift(
                full_spectra,
                axes=0,
            ),
            axis=0,
        )
    ).T

    modes = mirrored_modes[
        :,
        total_sample_count // 4
        : 3 * total_sample_count // 4,
    ]

    center_frequencies_hz = (
        np.asarray(
            normalized_centers,
            dtype=float,
        )
        * sampling_rate
    )

    sort_order = np.argsort(
        center_frequencies_hz
    )

    modes = modes[
        sort_order
    ]

    center_frequencies_hz = (
        center_frequencies_hz[
            sort_order
        ]
    )

    reconstructed_signal = np.sum(
        modes,
        axis=0,
    )

    residual = (
        signal
        - reconstructed_signal
    )

    reconstruction_correlation = float(
        np.corrcoef(
            signal,
            reconstructed_signal,
        )[0, 1]
    )

    relative_reconstruction_error = float(
        np.linalg.norm(
            residual
        )
        / (
            np.linalg.norm(
                signal
            )
            + epsilon
        )
    )

    converged = (
        stop_reason
        == "reconstruction_power_tolerance"
    )

    return {
        "modes": modes,
        "center_frequencies_hz": (
            center_frequencies_hz
        ),
        "reconstructed_signal": (
            reconstructed_signal
        ),
        "residual": residual,
        "correlation": (
            reconstruction_correlation
        ),
        "relative_error": (
            relative_reconstruction_error
        ),
        "mode_count": int(
            mode_count
        ),
        "residual_power_ratios": (
            np.asarray(
                residual_power_ratios,
                dtype=float,
            )
        ),
        "iterations_per_mode": (
            np.asarray(
                iterations_per_mode,
                dtype=int,
            )
        ),
        "total_iterations": int(
            np.sum(
                iterations_per_mode
            )
        ),
        "maximum_alpha": float(
            maximum_alpha
        ),
        "minimum_alpha": float(
            minimum_alpha
        ),
        "alpha_schedule": np.asarray(
            alpha_schedule,
            dtype=float,
        ),
        "reconstruction_power_tolerance": float(
            reconstruction_power_tolerance
        ),
        "stop_reason": stop_reason,
        "converged": bool(
            converged
        ),
    }