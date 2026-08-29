from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_fbg_csv


DATA_FILES = {
    "stable_10deg": (
        PROJECT_ROOT / "data" / "raw" / "angle_10deg_stable.csv"
    ),
    "angle_repeat": (
        PROJECT_ROOT / "data" / "raw" / "angle_10deg_43deg_repeat.csv"
    ),
}


def main() -> None:
    for dataset_name, csv_path in DATA_FILES.items():
        data, information = load_fbg_csv(
            csv_path=csv_path,
            sampling_rate=200.0,
        )

        print("=" * 70)
        print(f"資料名稱：{dataset_name}")

        for key, value in information.items():
            print(f"{key}: {value}")

        print("\n整理後前5筆資料：")
        print(data.head())


if __name__ == "__main__":
    main()