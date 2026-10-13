from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import load_fbg_csv, parse_logger_time


def test_parse_logger_time_supports_both_formats() -> None:
    values = parse_logger_time(
        pd.Series(["26:27.724", "16:26:09.2"])
    )
    assert np.allclose(values, [1587.724, 59169.2])


def test_load_fbg_csv_accepts_chinese_logger_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "無負載10度 (1).csv"
    pd.DataFrame(
        {
            "時間戳記": ["16:26:09.2", "16:26:09.2", "16:26:09.3"],
            "Ch1": [1549.1, 1549.2, 1549.3],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    data, information = load_fbg_csv(csv_path, sampling_rate=200.0)
    assert data["ch1_nm"].tolist() == [1549.1, 1549.2, 1549.3]
    assert information["source_time_column"] == "時間戳記"
    assert information["source_ch1_column"] == "Ch1"
    assert information["sample_count"] == 3


def test_load_fbg_csv_accepts_ts_wl_and_removes_exact_duplicates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "無負載10-43-10 (1).csv"
    pd.DataFrame(
        {
            "ts": ["17:21:53.6", "17:21:53.6", "17:21:53.7"],
            "wl": [1549.1, 1549.1, 1549.3],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    data, information = load_fbg_csv(csv_path, sampling_rate=200.0)

    assert data["ch1_nm"].tolist() == [1549.1, 1549.3]
    assert information["source_time_column"] == "ts"
    assert information["source_ch1_column"] == "wl"
    assert information["original_row_count"] == 3
    assert information["cleaned_row_count"] == 2
    assert information["exact_duplicate_row_count"] == 1
    assert information["removed_exact_duplicate_row_count"] == 1
