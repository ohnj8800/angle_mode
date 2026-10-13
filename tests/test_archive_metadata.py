from pathlib import Path

from src.common.archive_metadata import classify_archive_csv


def test_classify_fixed_angle_file() -> None:
    root = Path("260720")
    result = classify_archive_csv(
        root / "200hz" / "260602 30hz" / "有負載" / "有負載28度 (7).csv",
        root,
    )
    assert result["condition"] == "有負載"
    assert result["analysis_type"] == "fixed_angle"
    assert result["angle_deg"] == 28
    assert result["repeat"] == 7


def test_classify_transition_file_without_forcing_an_angle() -> None:
    root = Path("260720")
    result = classify_archive_csv(
        root
        / "200hz"
        / "260602 30hz"
        / "無負載10-43-10"
        / "無負載10-43-10 (1).csv",
        root,
    )
    assert result["condition"] == "無負載"
    assert result["analysis_type"] == "transition_sequence"
    assert result["angle_deg"] is None
    assert result["repeat"] == 1


def test_classify_fixed_angle_speed_sequence() -> None:
    root = Path("260720")
    result = classify_archive_csv(
        root / "200hz" / "260607" / "28度低中低中高低 (2).csv",
        root,
    )
    assert result["condition"] == "260607"
    assert result["analysis_type"] == "fixed_angle_speed_sequence"
    assert result["angle_deg"] == 28
