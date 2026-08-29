import numpy as np

from src.common.vmd import variational_mode_decomposition


def test_vmd_separates_known_frequencies() -> None:
    sampling_rate = 200.0
    duration_s = 10.0

    time_s = np.arange(
        0.0,
        duration_s,
        1.0 / sampling_rate,
    )

    random_generator = np.random.default_rng(42)

    signal = (
        0.8 * np.sin(2.0 * np.pi * 8.0 * time_s)
        + 0.4 * np.sin(2.0 * np.pi * 25.0 * time_s)
        + 0.2 * np.sin(2.0 * np.pi * 60.0 * time_s)
        + 0.01 * random_generator.standard_normal(
            len(time_s)
        )
    )

    result = variational_mode_decomposition(
        signal=signal,
        sampling_rate=sampling_rate,
        initial_center_frequencies_hz=[
            8.0,
            25.0,
            60.0,
        ],
        alpha=2000.0,
        tau=0.0,
        tolerance=1e-7,
        maximum_iterations=500,
    )

    calculated_frequencies = result[
        "center_frequencies_hz"
    ]

    expected_frequencies = np.array(
        [8.0, 25.0, 60.0]
    )

    assert result["converged"]

    assert np.allclose(
        calculated_frequencies,
        expected_frequencies,
        atol=0.5,
    )

    assert result["correlation"] > 0.99

    assert result["modes"].shape == (
        3,
        len(signal),
    )