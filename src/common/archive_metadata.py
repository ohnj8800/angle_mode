from __future__ import annotations

from pathlib import Path
import re


ANGLE_PATTERN = re.compile(r"(?P<angle>\d+)\s*度")
REPEAT_PATTERN = re.compile(r"\((?P<repeat>\d+)\)\s*$")


def classify_archive_csv(path: str | Path, archive_root: str | Path) -> dict:
    path = Path(path)
    archive_root = Path(archive_root)
    relative_path = path.relative_to(archive_root)
    name = path.stem.strip()
    full_text = "/".join(relative_path.parts)

    if "偏心" in full_text:
        condition = "偏心"
    elif "無負載開關" in full_text:
        condition = "無負載開關"
    elif "有負載" in full_text:
        condition = "有負載"
    elif "無負載" in full_text:
        condition = "無負載"
    elif "260607" in full_text:
        condition = "260607"
    else:
        condition = "其他"

    angle_match = ANGLE_PATTERN.search(name)
    repeat_match = REPEAT_PATTERN.search(name)
    angle_deg = int(angle_match.group("angle")) if angle_match else None
    repeat = int(repeat_match.group("repeat")) if repeat_match else None

    if angle_deg is None:
        analysis_type = "transition_sequence"
    elif any(token in name for token in ("低中", "低高", "中高", "動態")):
        analysis_type = "fixed_angle_speed_sequence"
    else:
        analysis_type = "fixed_angle"

    sequence_label = REPEAT_PATTERN.sub("", name).strip()

    return {
        "file_path": str(path),
        "relative_path": relative_path.as_posix(),
        "file_name": path.name,
        "condition": condition,
        "analysis_type": analysis_type,
        "angle_deg": angle_deg,
        "repeat": repeat,
        "sequence_label": sequence_label,
        "source_folder": relative_path.parent.as_posix(),
    }


def discover_archive_csvs(archive_root: str | Path) -> list[dict]:
    archive_root = Path(archive_root)
    if not archive_root.exists():
        raise FileNotFoundError(f"找不到資料根目錄：{archive_root}")

    paths = sorted(
        path for path in archive_root.rglob("*.csv") if path.is_file()
    )
    if not paths:
        raise ValueError(f"資料根目錄內沒有CSV：{archive_root}")

    return [classify_archive_csv(path, archive_root) for path in paths]
