import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def _validate_signal(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """檢查輸入訊號。"""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("輸入訊號必須是一維陣列。")

    if len(signal) < 20:
        raise ValueError("訊號太短，無法進行濾波與頻譜分析。")

    if not np.all(np.isfinite(signal)):
        raise ValueError("訊號包含NaN或無限值。")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate必須大於0。")

    return signal


def zero_phase_filter(
    signal: np.ndarray,
    sampling_rate: float,
    cutoff_hz: float,
    filter_type: str,
    order: int = 4,
) -> np.ndarray:
    """
    使用Butterworth零相位濾波器。

    零相位代表前後各濾波一次，避免訊號在時間上產生位移。
    """
    signal = _validate_signal(signal, sampling_rate)

    nyquist = sampling_rate / 2.0

    if not 0 < cutoff_hz < nyquist:
        raise ValueError(
            f"截止頻率必須介於0與Nyquist頻率{nyquist} Hz之間。"
        )

    sos = butter(
        N=order,
        Wn=cutoff_hz,
        btype=filter_type,
        fs=sampling_rate,
        output="sos",
    )

    return sosfiltfilt(sos, signal)


def preprocess_fbg_signal(
    ch1_nm: np.ndarray,
    sampling_rate: float = 200.0,
    lowpass_cutoff_hz: float = 0.1,
    highpass_cutoff_hz: float = 0.5,
    filter_order: int = 4,
) -> dict[str, np.ndarray]:
    """
    將FBG波長訊號分成低頻與動態訊號兩條分析路徑。

    angle_component_nm：
        保留DC與緩慢變化，用於估計10度、43度及切換時間。

    dynamic_nm：
        只供VMD分析振動模態。角度辨識不可使用此訊號。

    residual_without_angle_nm：
        原始訊號扣除角度成分，可用來檢查角度與振動是否分離。
    """
    raw_nm = _validate_signal(ch1_nm, sampling_rate)

    # 不對角度路徑去趨勢，避免刪除真正的角度平台與切換資訊。
    angle_component_nm = zero_phase_filter(
        signal=raw_nm,
        sampling_rate=sampling_rate,
        cutoff_hz=lowpass_cutoff_hz,
        filter_type="lowpass",
        order=filter_order,
    )

    residual_without_angle_nm = raw_nm - angle_component_nm

    # 0.5 Hz以上才交給VMD分析機械振動；此處不再用來估計角度。
    dynamic_nm = zero_phase_filter(
        signal=raw_nm,
        sampling_rate=sampling_rate,
        cutoff_hz=highpass_cutoff_hz,
        filter_type="highpass",
        order=filter_order,
    )

    # 消除濾波後可能留下的極小平均偏移。
    dynamic_nm = dynamic_nm - np.mean(dynamic_nm)

    return {
        "raw_nm": raw_nm,
        "centered_raw_nm": raw_nm - np.mean(raw_nm),
        "angle_component_nm": angle_component_nm,
        "residual_without_angle_nm": residual_without_angle_nm,
        "dynamic_nm": dynamic_nm,
    }


def calculate_welch_psd(
    signal: np.ndarray,
    sampling_rate: float = 200.0,
    nperseg: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算Welch功率頻譜密度。

    Welch方法會把訊號分成多個重疊小段，
    分別計算頻譜後再平均，使頻譜較穩定。
    """
    signal = _validate_signal(signal, sampling_rate)

    actual_nperseg = min(nperseg, len(signal))
    noverlap = actual_nperseg // 2

    frequency_hz, psd = welch(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=actual_nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    return frequency_hz, psd
