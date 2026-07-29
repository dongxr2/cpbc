"""Structural tests for complete-match resampling and cross-fit isolation."""
from __future__ import annotations

import numpy as np
import pandas as pd

import full_match_bootstrap as fb
import legal_score_path_cpbc as lp


def synthetic_base():
    rows = []
    for match in range(6):
        for point in range(4 + match):
            rows.append({
                "match_id": f"M{match}", "pt_idx": point + 1, "tour": "atp",
                "server_won": point % 2, "server_name": f"S{match}",
                "returner_name": f"R{match}", "era": "sim", "SetNo": 1,
                "GameNo": 1, "PointNumber": point + 1, "PointServer": 1,
                "PointWinner": 1 if point % 2 else 2, "set_late": 0,
            })
    return pd.DataFrame(rows)


def main():
    base = synthetic_base().sort_values(
        ["match_id", "pt_idx"]
    ).reset_index(drop=True)
    keys = base["match_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(base)]
    base["_source_match"] = np.repeat(np.arange(len(starts)), ends - starts)
    fb._BASE, fb._STARTS, fb._ENDS, fb._TOUR = base, starts, ends, "atp"
    sample1, diag1 = fb.resample_complete_matches(12345, folds=3)
    sample2, diag2 = fb.resample_complete_matches(12345, folds=3)
    pd.testing.assert_frame_equal(sample1, sample2)
    assert diag1 == diag2
    assert diag1["sampled_matches"] == 6
    assert sample1["match_id"].nunique() == 6
    for _, group in sample1.groupby("match_id"):
        assert group["_source_match"].nunique() == 1
        assert np.array_equal(group["pt_idx"], np.arange(1, len(group) + 1))
    assert sample1.groupby(
        "_source_match"
    )["_crossfit_fold"].nunique().max() == 1
    legal_mask = np.ones(len(sample1), dtype=bool)
    _, _, fold_id, _ = lp.crossfit_null_components(
        sample1, legal_mask, folds=3, prior_n=2.0
    )
    assert np.array_equal(
        fold_id, sample1["_crossfit_fold"].to_numpy(np.int8)
    )
    print({"status": "PASS", **diag1})


if __name__ == "__main__":
    main()
