from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_BASELINE_FEATURES = (
    "local_peak_frequency_hz",
    "rms_nm",
    "energy_nm2",
    "envelope_mean_nm",
)
STABLE_STATES = ("stable_10deg", "stable_43deg")


def _robust_statistics(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().to_numpy(dtype=float)

    if clean.size == 0:
        return {
            "sample_count": 0,
            "mean": float("nan"),
            "standard_deviation": float("nan"),
            "median": float("nan"),
            "mad": float("nan"),
            "robust_sigma": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }

    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    return {
        "sample_count": int(clean.size),
        "mean": float(np.mean(clean)),
        "standard_deviation": float(np.std(clean, ddof=1))
        if clean.size > 1
        else 0.0,
        "median": median,
        "mad": mad,
        "robust_sigma": 1.4826 * mad,
        "q05": float(np.quantile(clean, 0.05)),
        "q25": float(np.quantile(clean, 0.25)),
        "q75": float(np.quantile(clean, 0.75)),
        "q95": float(np.quantile(clean, 0.95)),
        "minimum": float(np.min(clean)),
        "maximum": float(np.max(clean)),
    }


def select_stable_baseline_windows(
    window_features: pd.DataFrame,
    *,
    edge_guard_s: float = 5.0,
) -> pd.DataFrame:
    """挑選不與角度移動重疊，且遠離資料端點的純穩定視窗。"""
    required = {
        "method",
        "mode",
        "center_time_s",
        "angle_state_label",
        "dominant_state_label",
        "moving_fraction",
    }
    missing = sorted(required - set(window_features.columns))

    if missing:
        raise ValueError(f"滑動視窗資料缺少欄位：{missing}")

    if edge_guard_s < 0:
        raise ValueError("edge_guard_s不可小於0。")

    maximum_time_s = float(window_features["center_time_s"].max())
    stable_mask = (
        window_features["angle_state_label"].isin(STABLE_STATES)
        & window_features["dominant_state_label"].isin(STABLE_STATES)
        & (
            window_features["angle_state_label"]
            == window_features["dominant_state_label"]
        )
        & np.isclose(window_features["moving_fraction"], 0.0)
        & (window_features["center_time_s"] >= edge_guard_s)
        & (
            window_features["center_time_s"]
            <= maximum_time_s - edge_guard_s
        )
    )
    return window_features.loc[stable_mask].copy()


def build_state_conditioned_baseline(
    window_features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_BASELINE_FEATURES,
    edge_guard_s: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依方法、模態與角度狀態建立median/MAD正常參考值。"""
    missing_features = sorted(set(feature_columns) - set(window_features.columns))

    if missing_features:
        raise ValueError(f"滑動視窗資料缺少特徵欄位：{missing_features}")

    stable_windows = select_stable_baseline_windows(
        window_features,
        edge_guard_s=edge_guard_s,
    )
    rows: list[dict] = []
    group_columns = [
        "method",
        "mode",
        "physical_role_candidate",
        "global_peak_frequency_hz",
        "angle_state_label",
    ]

    for group_key, group in stable_windows.groupby(group_columns, sort=False):
        method, mode, role, global_peak_hz, state = group_key

        for feature in feature_columns:
            rows.append(
                {
                    "method": method,
                    "mode": mode,
                    "physical_role_candidate": role,
                    "global_peak_frequency_hz": float(global_peak_hz),
                    "angle_state_label": state,
                    "feature": feature,
                    **_robust_statistics(group[feature]),
                }
            )

    baseline = pd.DataFrame(rows).sort_values(
        ["method", "global_peak_frequency_hz", "angle_state_label", "feature"]
    ).reset_index(drop=True)
    return baseline, stable_windows
