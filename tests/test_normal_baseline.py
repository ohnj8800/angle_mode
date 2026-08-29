import numpy as np
import pandas as pd

from src.common.normal_baseline import build_state_conditioned_baseline


def test_baseline_separates_angle_states() -> None:
    rows = []

    for index in range(20):
        state = "stable_10deg" if index < 10 else "stable_43deg"
        rms = 1.0 if state == "stable_10deg" else 2.0
        rows.append(
            {
                "method": "TEST",
                "mode": "IMF1",
                "physical_role_candidate": "background_dynamic_mode",
                "global_peak_frequency_hz": 10.0,
                "center_time_s": float(index + 10),
                "angle_state_label": state,
                "dominant_state_label": state,
                "moving_fraction": 0.0,
                "local_peak_frequency_hz": 10.0,
                "rms_nm": rms,
                "energy_nm2": 400.0 * rms**2,
                "envelope_mean_nm": np.sqrt(2.0) * rms,
            }
        )

    baseline, selected = build_state_conditioned_baseline(
        pd.DataFrame(rows), edge_guard_s=0.0
    )
    rms_rows = baseline.loc[baseline["feature"] == "rms_nm"]
    medians = rms_rows.set_index("angle_state_label")["median"]
    assert len(selected) == 20
    assert medians["stable_10deg"] == 1.0
    assert medians["stable_43deg"] == 2.0
