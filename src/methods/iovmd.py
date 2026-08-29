import numpy as np
import pandas as pd
from scipy.signal import find_peaks, get_window, savgol_filter

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    def lowess(
        endog,
        exog,
        frac,
        it=0,
        is_sorted=True,
        return_sorted=False,
    ):
        """缺少statsmodels時的本地一階平滑備援。"""
        del exog, it, is_sorted
        values = np.asarray(endog, dtype=float)
        window_length = max(5, int(round(frac * len(values))))

        if window_length % 2 == 0:
            window_length += 1

        window_length = min(
            window_length,
            len(values) - 1 if len(values) % 2 == 0 else len(values),
        )
        smoothed = savgol_filter(
            values,
            window_length=window_length,
            polyorder=1,
            mode="interp",
        )

        if return_sorted:
            return np.column_stack((np.arange(len(smoothed)), smoothed))

        return smoothed

from src.common.vmd import variational_mode_decomposition

def calculate_one_sided_spectrum(
    signal: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算單邊振幅頻譜。

    使用Hann窗降低有限長度訊號造成的頻譜洩漏。
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal必須是一維陣列。")

    if len(signal) < 20:
        raise ValueError("訊號太短，無法計算頻譜。")

    if not np.all(np.isfinite(signal)):
        raise ValueError("訊號包含NaN或無限值。")

    sample_count = len(signal)

    window = get_window(
        window="hann",
        Nx=sample_count,
        fftbins=True,
    )

    windowed_signal = signal * window

    spectrum = np.fft.rfft(windowed_signal)

    frequency_hz = np.fft.rfftfreq(
        sample_count,
        d=1.0 / sampling_rate,
    )

    # 以窗函數總和修正振幅。
    amplitude = 2.0 * np.abs(spectrum) / np.sum(window)

    # DC與Nyquist不應乘2。
    amplitude[0] = amplitude[0] / 2.0

    if sample_count % 2 == 0:
        amplitude[-1] = amplitude[-1] / 2.0

    return frequency_hz, amplitude


def search_initial_modes(
    signal: np.ndarray,
    sampling_rate: float = 200.0,
    minimum_frequency_hz: float = 0.5,
    maximum_frequency_hz: float | None = None,
    lowess_fraction: float = 0.03,
    minimum_prominence_ratio: float = 0.02,
    minimum_height_ratio: float = 0.01,
    minimum_peak_distance_hz: float = 1.0,
) -> dict:
    """
    以LOWESS平滑頻譜及局部峰值搜尋取得IOVMD初始K。

    回傳：
    - 原始頻譜
    - LOWESS平滑頻譜
    - 選出的峰值
    - 初始模態數K
    """
    if maximum_frequency_hz is None:
        maximum_frequency_hz = sampling_rate / 2.0

    nyquist_frequency = sampling_rate / 2.0

    if maximum_frequency_hz > nyquist_frequency:
        raise ValueError(
            f"maximum_frequency_hz不可高於Nyquist頻率"
            f"{nyquist_frequency} Hz。"
        )

    if not 0 < lowess_fraction <= 1:
        raise ValueError("lowess_fraction必須介於0與1之間。")

    frequency_hz, amplitude = calculate_one_sided_spectrum(
        signal=signal,
        sampling_rate=sampling_rate,
    )

    frequency_resolution_hz = float(
        np.median(np.diff(frequency_hz))
    )

    # 排除Nyquist端點，因為端點不能由find_peaks判定為局部峰值。
    effective_maximum = min(
        maximum_frequency_hz,
        nyquist_frequency - frequency_resolution_hz,
    )

    frequency_mask = (
        (frequency_hz >= minimum_frequency_hz)
        & (frequency_hz <= effective_maximum)
    )

    selected_frequency = frequency_hz[frequency_mask]
    selected_amplitude = amplitude[frequency_mask]

    if len(selected_frequency) < 10:
        raise ValueError("指定頻率範圍內的頻譜點太少。")

    # 3%一階LOWESS平滑。
    smoothed_amplitude = lowess(
        endog=selected_amplitude,
        exog=selected_frequency,
        frac=lowess_fraction,
        it=0,
        is_sorted=True,
        return_sorted=False,
    )

    # LOWESS可能產生極小負值，振幅不應小於0。
    smoothed_amplitude = np.clip(
        smoothed_amplitude,
        a_min=0.0,
        a_max=None,
    )

    maximum_smoothed_amplitude = float(
        np.max(smoothed_amplitude)
    )

    if maximum_smoothed_amplitude <= 0:
        raise ValueError("平滑後頻譜沒有有效振幅。")

    normalized_smoothed = (
        smoothed_amplitude / maximum_smoothed_amplitude
    )

    maximum_raw_amplitude = float(np.max(selected_amplitude))

    if maximum_raw_amplitude > 0:
        normalized_raw = (
            selected_amplitude / maximum_raw_amplitude
        )
    else:
        normalized_raw = np.zeros_like(selected_amplitude)

    minimum_distance_bins = max(
        1,
        int(
            np.ceil(
                minimum_peak_distance_hz
                / frequency_resolution_hz
            )
        ),
    )

    peak_indices, peak_properties = find_peaks(
        normalized_smoothed,
        prominence=minimum_prominence_ratio,
        height=minimum_height_ratio,
        distance=minimum_distance_bins,
    )

    if len(peak_indices) == 0:
        raise ValueError(
            "沒有找到有效頻譜峰。請先檢查訊號或峰值門檻，"
            "不要直接任意指定K。"
        )

    peak_table = pd.DataFrame(
        {
            "peak_number": np.arange(
                1,
                len(peak_indices) + 1,
            ),
            "frequency_hz": selected_frequency[peak_indices],
            "raw_amplitude_nm": selected_amplitude[peak_indices],
            "smoothed_amplitude_nm": (
                smoothed_amplitude[peak_indices]
            ),
            "normalized_height": peak_properties["peak_heights"],
            "prominence": peak_properties["prominences"],
        }
    )

    peak_table = peak_table.sort_values(
        by="frequency_hz"
    ).reset_index(drop=True)

    peak_table["peak_number"] = np.arange(
        1,
        len(peak_table) + 1,
    )

    initial_k = int(len(peak_table))

    return {
        "frequency_hz": selected_frequency,
        "raw_amplitude_nm": selected_amplitude,
        "normalized_raw_amplitude": normalized_raw,
        "smoothed_amplitude_nm": smoothed_amplitude,
        "normalized_smoothed_amplitude": normalized_smoothed,
        "peak_table": peak_table,
        "initial_k": initial_k,
        "frequency_resolution_hz": frequency_resolution_hz,
        "parameters": {
            "minimum_frequency_hz": minimum_frequency_hz,
            "maximum_frequency_hz": maximum_frequency_hz,
            "lowess_fraction": lowess_fraction,
            "minimum_prominence_ratio": minimum_prominence_ratio,
            "minimum_height_ratio": minimum_height_ratio,
            "minimum_peak_distance_hz": minimum_peak_distance_hz,
        },
    }

def run_initial_iovmd(
    signal: np.ndarray,
    sampling_rate: float = 200.0,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tolerance: float = 1e-7,
    maximum_iterations: int = 500,
    minimum_frequency_hz: float = 0.5,
    maximum_frequency_hz: float | None = None,
) -> dict:
    """
    執行IOVMD初始分解。

    1. LOWESS頻譜平滑
    2. 搜尋候選峰
    3. 候選峰數量作為初始K
    4. 候選峰頻率作為VMD初始中心頻率
    """
    if maximum_frequency_hz is None:
        maximum_frequency_hz = sampling_rate / 2.0

    peak_search_result = search_initial_modes(
        signal=signal,
        sampling_rate=sampling_rate,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        lowess_fraction=0.03,
        minimum_prominence_ratio=0.02,
        minimum_height_ratio=0.01,
        minimum_peak_distance_hz=1.0,
    )

    initial_center_frequencies_hz = (
        peak_search_result["peak_table"][
            "frequency_hz"
        ].to_numpy(dtype=float)
    )

    vmd_result = variational_mode_decomposition(
        signal=signal,
        sampling_rate=sampling_rate,
        initial_center_frequencies_hz=(
            initial_center_frequencies_hz
        ),
        alpha=alpha,
        tau=tau,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )

    return {
        "initial_k": (
            peak_search_result["initial_k"]
        ),
        "initial_center_frequencies_hz": (
            initial_center_frequencies_hz
        ),
        "peak_search": peak_search_result,
        "vmd": vmd_result,
    }
