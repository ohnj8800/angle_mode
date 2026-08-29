import numpy as np
import pandas as pd

from src.common.consensus_evidence import select_consensus_evidence, time_window_mask


def test_selects_only_cross_method_clusters_and_builds_time_mask():
    clusters = pd.DataFrame(
        {
            "consensus_cluster_id": ["C001", "C002"],
            "method_count": [3, 1],
            "cluster_start_time_s": [5.0, 20.0],
        }
    )
    membership = pd.DataFrame(
        {
            "consensus_cluster_id": ["C001", "C001", "C002"],
            "method": ["IOVMD", "AVMD", "SVMD"],
            "mode": ["IMF2", "IMF3", "IMF4"],
        }
    )
    selected, evidence = select_consensus_evidence(clusters, membership)
    mask = time_window_mask(np.arange(0.0, 11.0), 4.0, 6.0, padding_s=2.0)

    assert selected["consensus_cluster_id"].tolist() == ["C001"]
    assert len(evidence) == 2
    assert np.flatnonzero(mask).tolist() == [2, 3, 4, 5, 6, 7, 8]
