from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"time", "CH1"}


def parse_logger_time(time_column: pd.Series) -> np.ndarray:
    """
    將CSV中的「分:秒.毫秒」轉成秒數。

    例如：
    26:27.724 -> 26 * 60 + 27.724 秒

    這個時間只用於資料品質檢查。
    真正的頻譜分析會依固定取樣率建立均勻時間軸。
    """
    text = time_column.astype(str).str.strip()
    parts = text.str.split(":", n=1, expand=True)

    if parts.shape[1] != 2:
        raise ValueError(
            "time欄位格式錯誤，預期格式為「分:秒.毫秒」，"
            "例如26:27.724。"
        )

    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce")
    logger_seconds = minutes * 60.0 + seconds

    if logger_seconds.isna().any():
        bad_count = int(logger_seconds.isna().sum())
        raise ValueError(f"time欄位中有 {bad_count} 筆資料無法轉成秒數。")

    return logger_seconds.to_numpy(dtype=float)


def load_fbg_csv(
    csv_path: str | Path,
    sampling_rate: float = 200.0,
) -> tuple[pd.DataFrame, dict]:
    """
    讀取FBG CSV，檢查資料格式，並建立均勻時間軸。

    Parameters
    ----------
    csv_path:
        CSV檔案位置。
    sampling_rate:
        系統取樣率，預設為200 Hz。

    Returns
    -------
    data:
        整理後的資料，包含：
        sample_index、time_s、raw_time、ch1_nm

    information:
        資料品質與取樣資訊。
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到CSV檔案：{csv_path}")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate必須大於0。")

    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw.columns = raw.columns.astype(str).str.strip()

    missing_columns = REQUIRED_COLUMNS - set(raw.columns)

    if missing_columns:
        raise ValueError(
            f"CSV缺少必要欄位：{sorted(missing_columns)}；"
            f"目前欄位為：{raw.columns.tolist()}"
        )

    ch1 = pd.to_numeric(raw["CH1"], errors="coerce")
    logger_seconds = parse_logger_time(raw["time"])

    valid_mask = np.isfinite(ch1.to_numpy())

    invalid_ch1_count = int((~valid_mask).sum())

    ch1 = ch1.loc[valid_mask].reset_index(drop=True)
    raw_time = raw.loc[valid_mask, "time"].reset_index(drop=True)
    logger_seconds = logger_seconds[valid_mask]

    if len(ch1) < 2:
        raise ValueError("有效CH1資料少於2筆，無法進行訊號分析。")

    # 不以原始time直接建立頻譜時間軸，因為紀錄時間存在重複與抖動。
    # 使用已知取樣率200 Hz建立均勻時間軸。
    sample_index = np.arange(len(ch1), dtype=int)
    time_s = sample_index / sampling_rate

    logger_dt = np.diff(logger_seconds)
    positive_dt = logger_dt[logger_dt > 0]

    duplicate_or_backward_count = int(np.sum(logger_dt <= 0))

    if len(positive_dt) > 0:
        median_positive_dt = float(np.median(positive_dt))
    else:
        median_positive_dt = float("nan")

    data = pd.DataFrame(
        {
            "sample_index": sample_index,
            "time_s": time_s,
            "raw_time": raw_time,
            "ch1_nm": ch1.to_numpy(dtype=float),
        }
    )

    information = {
        "file_name": csv_path.name,
        "sample_count": len(data),
        "sampling_rate_hz": sampling_rate,
        "duration_s": float((len(data) - 1) / sampling_rate),
        "nyquist_frequency_hz": float(sampling_rate / 2.0),
        "ch1_mean_nm": float(data["ch1_nm"].mean()),
        "ch1_std_nm": float(data["ch1_nm"].std(ddof=1)),
        "ch1_min_nm": float(data["ch1_nm"].min()),
        "ch1_max_nm": float(data["ch1_nm"].max()),
        "invalid_ch1_count": invalid_ch1_count,
        "raw_time_nonpositive_steps": duplicate_or_backward_count,
        "raw_time_median_positive_dt_s": median_positive_dt,
    }

    return data, information