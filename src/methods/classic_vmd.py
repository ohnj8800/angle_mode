from __future__ import annotations

import numpy as np

from src.common.vmd import variational_mode_decomposition


def create_uniform_initial_centers(
    signal_length: int,
    sampling_rate: float,
    mode_count: int,
) -> np.ndarray:
    """建立傳統VMD使用的固定、均勻初始中心頻率。"""
    if signal_length < 20:
        raise ValueError("signal_length至少必須為20。")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate必須大於0。")
    if mode_count < 2:
        raise ValueError("mode_count至少必須為2。")

    frequency_resolution_hz = sampling_rate / signal_length
    nyquist_frequency_hz = sampling_rate / 2.0

    # 原始VMD常用均勻頻率初始化。第一個中心從可解析的最低正頻率開始，
    # 以保留角度位置低頻模態；其餘中心均勻分布到Nyquist頻率以下。
    return np.linspace(
        frequency_resolution_hz,
        nyquist_frequency_hz * (mode_count - 0.5) / mode_count,
        mode_count,
    )


def classic_variational_mode_decomposition(
    signal: np.ndarray,
    sampling_rate: float,
    mode_count: int = 10,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tolerance: float = 1e-7,
    maximum_iterations: int = 500,
) -> dict:
    """以固定K、固定alpha及均勻初始化執行Classic VMD基準組。"""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    initial_centers = create_uniform_initial_centers(
        signal_length=signal.size,
        sampling_rate=sampling_rate,
        mode_count=mode_count,
    )
    result = variational_mode_decomposition(
        signal=signal,
        sampling_rate=sampling_rate,
        initial_center_frequencies_hz=initial_centers,
        alpha=alpha,
        tau=tau,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )
    result["initial_center_frequencies_hz"] = initial_centers
    result["mode_count"] = int(mode_count)
    result["parameter_rule"] = (
        "fixed K and alpha with uniformly spaced initial center frequencies"
    )
    return result
