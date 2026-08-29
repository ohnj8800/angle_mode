from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.normal_baseline import build_state_conditioned_baseline


WINDOW_ROOT = PROJECT_ROOT / "results" / "window_analysis"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "normal_baseline"
EDGE_GUARD_S = 5.0
METHODS = ("VMD", "IOVMD", "AVMD", "SVMD", "STVMD")


def _create_state_comparison(baseline: pd.DataFrame) -> pd.DataFrame:
    rms = baseline.loc[baseline["feature"] == "rms_nm"]
    comparison = rms.pivot_table(
        index=[
            "method",
            "mode",
            "physical_role_candidate",
            "global_peak_frequency_hz",
        ],
        columns="angle_state_label",
        values="median",
    ).reset_index()
    comparison.columns.name = None

    for column in ("stable_10deg", "stable_43deg"):
        if column not in comparison.columns:
            comparison[column] = np.nan

    comparison = comparison.rename(
        columns={
            "stable_10deg": "stable_10deg_median_rms_nm",
            "stable_43deg": "stable_43deg_median_rms_nm",
        }
    )
    comparison["rms_ratio_43_to_10"] = (
        comparison["stable_43deg_median_rms_nm"]
        / comparison["stable_10deg_median_rms_nm"]
    )
    comparison["state_contrast_ratio"] = np.maximum(
        comparison["rms_ratio_43_to_10"],
        1.0 / comparison["rms_ratio_43_to_10"],
    )
    return comparison.sort_values(
        ["method", "global_peak_frequency_hz"]
    ).reset_index(drop=True)


def _plot_state_rms_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(METHODS),
        1,
        figsize=(13, 10),
        sharex=False,
        constrained_layout=True,
    )

    for axis, method in zip(axes, METHODS):
        subset = comparison.loc[
            (comparison["method"] == method)
            & (comparison["global_peak_frequency_hz"] >= 0.5)
        ]
        x = np.arange(len(subset))
        width = 0.38
        axis.bar(
            x - width / 2,
            subset["stable_10deg_median_rms_nm"],
            width,
            label="stable 10 deg",
            color="tab:blue",
        )
        axis.bar(
            x + width / 2,
            subset["stable_43deg_median_rms_nm"],
            width,
            label="stable 43 deg",
            color="tab:orange",
        )
        labels = [
            f"{row.mode}\n{row.global_peak_frequency_hz:.1f} Hz"
            for row in subset.itertuples(index=False)
        ]
        axis.set_xticks(x, labels=labels, fontsize=8)
        axis.set_ylabel("Median RMS (nm)")
        axis.set_xlabel("Mode and global peak frequency")
        axis.set_title(method)
        axis.grid(axis="y", alpha=0.2)

    axes[0].legend(loc="upper right")
    figure.suptitle("Normal modal RMS baseline separated by angle state")
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    input_path = WINDOW_ROOT / "all_method_mode_window_features.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "缺少滑動視窗特徵，請先執行05_extract_window_features.py。"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    window_features = pd.read_csv(input_path, encoding="utf-8-sig")
    baseline, stable_windows = build_state_conditioned_baseline(
        window_features,
        edge_guard_s=EDGE_GUARD_S,
    )
    comparison = _create_state_comparison(baseline)
    baseline.to_csv(
        OUTPUT_ROOT / "normal_baseline_by_state.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stable_windows.to_csv(
        OUTPUT_ROOT / "selected_stable_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        OUTPUT_ROOT / "angle_state_mode_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "baseline_source": "stable windows in angle-repeat dataset",
                "edge_guard_s": EDGE_GUARD_S,
                "moving_fraction_required": 0.0,
                "states": "stable_10deg, stable_43deg",
                "statistics": "median, MAD, robust_sigma, q05-q95",
                "independent_validation_baseline": False,
            }
        ]
    ).to_csv(
        OUTPUT_ROOT / "normal_baseline_settings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_state_rms_comparison(
        comparison,
        OUTPUT_ROOT / "01_angle_state_rms_baseline.png",
    )
    print("=" * 70)
    print("依角度狀態建立正常模態基準完成")
    print(
        stable_windows.groupby(["method", "angle_state_label"])[
            "window_index"
        ]
        .nunique()
        .to_string()
    )
    print(f"正常基準列數：{len(baseline)}")
    print(f"結果位置：{OUTPUT_ROOT}")
    print("注意：這是同一份資料內的穩定狀態參考，不是獨立驗證資料。")


if __name__ == "__main__":
    main()
