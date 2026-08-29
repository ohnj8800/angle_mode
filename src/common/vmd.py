import numpy as np


def variational_mode_decomposition(
    signal: np.ndarray,
    sampling_rate: float,
    initial_center_frequencies_hz: np.ndarray | list[float],
    alpha: float = 2000.0,
    tau: float = 0.0,
    tolerance: float = 1e-7,
    maximum_iterations: int = 500,
) -> dict:
    """
    執行一維Variational Mode Decomposition。

    Parameters
    ----------
    signal:
        要分解的一維動態訊號。

    sampling_rate:
        取樣率，單位Hz。

    initial_center_frequencies_hz:
        各模態的初始中心頻率。
        IOVMD會使用LOWESS峰值搜尋結果。

    alpha:
        頻寬懲罰參數。數值越大，模態頻帶通常越窄。

    tau:
        拉格朗日乘子更新步長。
        tau=0代表允許部分雜訊不被模態重建。

    tolerance:
        收斂判斷門檻。

    maximum_iterations:
        最大迭代次數。
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal必須是一維陣列。")

    if len(signal) < 20:
        raise ValueError("訊號太短，無法執行VMD。")

    if not np.all(np.isfinite(signal)):
        raise ValueError("訊號包含NaN或無限值。")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate必須大於0。")

    if alpha <= 0:
        raise ValueError("alpha必須大於0。")

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations至少必須為1。")

    initial_centers = np.asarray(
        initial_center_frequencies_hz,
        dtype=float,
    )

    if initial_centers.ndim != 1 or len(initial_centers) == 0:
        raise ValueError("至少必須提供一個初始中心頻率。")

    nyquist_frequency = sampling_rate / 2.0

    if np.any(initial_centers <= 0):
        raise ValueError("初始中心頻率必須大於0 Hz。")

    if np.any(initial_centers >= nyquist_frequency):
        raise ValueError(
            "初始中心頻率必須小於Nyquist頻率。"
        )

    # 依頻率由低到高排列。
    initial_centers = np.sort(initial_centers)

    original_length = len(signal)
    working_signal = signal.copy()

    # FFT處理使用偶數資料長度。
    if len(working_signal) % 2 != 0:
        working_signal = np.append(
            working_signal,
            working_signal[-1],
        )

    working_length = len(working_signal)
    half_length = working_length // 2

    # 鏡射延伸，降低訊號頭尾的不連續問題。
    mirrored_signal = np.concatenate(
        (
            working_signal[:half_length][::-1],
            working_signal,
            working_signal[-half_length:][::-1],
        )
    )

    total_length = len(mirrored_signal)
    mode_count = len(initial_centers)

    # 正規化頻率範圍約為-0.5到+0.5 cycles/sample。
    normalized_frequency = (
        np.arange(total_length, dtype=float) / total_length
        - 0.5
    )

    signal_spectrum = np.fft.fftshift(
        np.fft.fft(mirrored_signal)
    )

    # 只保留解析訊號的非負頻率。
    positive_signal_spectrum = signal_spectrum.copy()
    positive_signal_spectrum[: total_length // 2] = 0.0

    previous_modes_spectrum = np.zeros(
        (total_length, mode_count),
        dtype=complex,
    )

    dual_variable = np.zeros(
        total_length,
        dtype=complex,
    )

    # Hz轉為cycles/sample。
    previous_center_frequencies = (
        initial_centers / sampling_rate
    )

    converged = False
    relative_change = float("inf")

    positive_slice = slice(
        total_length // 2,
        total_length,
    )

    for iteration in range(1, maximum_iterations + 1):
        current_modes_spectrum = np.zeros_like(
            previous_modes_spectrum
        )

        current_center_frequencies = (
            previous_center_frequencies.copy()
        )

        for mode_index in range(mode_count):
            # 已更新模態使用本次迭代值；
            # 尚未更新模態使用上一次迭代值。
            earlier_modes = np.sum(
                current_modes_spectrum[:, :mode_index],
                axis=1,
            )

            later_modes = np.sum(
                previous_modes_spectrum[
                    :, mode_index + 1:
                ],
                axis=1,
            )

            other_modes = earlier_modes + later_modes

            frequency_distance = (
                normalized_frequency
                - previous_center_frequencies[mode_index]
            )

            denominator = (
                1.0
                + alpha * frequency_distance ** 2
            )

            current_modes_spectrum[:, mode_index] = (
                positive_signal_spectrum
                - other_modes
                - dual_variable / 2.0
            ) / denominator

            positive_mode = current_modes_spectrum[
                positive_slice,
                mode_index,
            ]

            positive_frequency = normalized_frequency[
                positive_slice
            ]

            spectral_energy = np.abs(positive_mode) ** 2
            energy_sum = float(np.sum(spectral_energy))

            # 以頻譜能量加權平均更新中心頻率。
            if energy_sum > np.finfo(float).eps:
                current_center_frequencies[mode_index] = (
                    np.sum(
                        positive_frequency
                        * spectral_energy
                    )
                    / energy_sum
                )

        # 更新拉格朗日乘子。
        dual_variable = dual_variable + tau * (
            np.sum(current_modes_spectrum, axis=1)
            - positive_signal_spectrum
        )

        numerator = np.sum(
            np.abs(
                current_modes_spectrum
                - previous_modes_spectrum
            ) ** 2
        )

        denominator = (
            np.sum(
                np.abs(previous_modes_spectrum) ** 2
            )
            + np.finfo(float).eps
        )

        relative_change = float(
            numerator / denominator
        )

        previous_modes_spectrum = (
            current_modes_spectrum
        )

        previous_center_frequencies = (
            current_center_frequencies
        )

        if relative_change < tolerance:
            converged = True
            break

    # 將每個模態轉回時域。
    extended_modes = np.empty(
        (mode_count, total_length),
        dtype=float,
    )

    for mode_index in range(mode_count):
        positive_fft = np.zeros(
            total_length // 2 + 1,
            dtype=complex,
        )

        positive_fft[:-1] = previous_modes_spectrum[
            total_length // 2:,
            mode_index,
        ]

        extended_modes[mode_index] = np.fft.irfft(
            positive_fft,
            n=total_length,
        )

    # 移除前後鏡射延伸區域。
    crop_start = total_length // 4
    crop_end = crop_start + working_length

    modes = extended_modes[
        :,
        crop_start:crop_end,
    ]

    # 如果原訊號為奇數長度，移除補上的最後一點。
    modes = modes[:, :original_length]

    final_center_frequencies_hz = (
        previous_center_frequencies
        * sampling_rate
    )

    # 確保模態由低頻排到高頻。
    sort_order = np.argsort(
        final_center_frequencies_hz
    )

    modes = modes[sort_order]
    final_center_frequencies_hz = (
        final_center_frequencies_hz[sort_order]
    )

    reconstructed_signal = np.sum(
        modes,
        axis=0,
    )

    residual = signal - reconstructed_signal

    correlation = float(
        np.corrcoef(
            signal,
            reconstructed_signal,
        )[0, 1]
    )

    relative_error = float(
        np.linalg.norm(residual)
        / (
            np.linalg.norm(signal)
            + np.finfo(float).eps
        )
    )

    return {
        "modes": modes,
        "center_frequencies_hz": (
            final_center_frequencies_hz
        ),
        "reconstructed_signal": reconstructed_signal,
        "residual": residual,
        "correlation": correlation,
        "relative_error": relative_error,
        "converged": converged,
        "iterations": iteration,
        "final_relative_change": relative_change,
        "alpha": alpha,
        "tau": tau,
        "tolerance": tolerance,
    }