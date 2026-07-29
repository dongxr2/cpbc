"""Targeted empirical extensions.

The script corrects pre-point score reconstruction in tiebreak games, estimates
nested score and set-score CPBC orbits, audits source-data retention, and places
the preferred score-restricted effects in the prespecified eight-test Holm family.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import tennis_momentum_advanced as tm
import mca_analysis as rev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "momentum_output"
OUT.mkdir(parents=True, exist_ok=True)


def source_coverage_audit(retained: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted(Path(tm.PBP_DIR).glob("*-points.csv")):
        name = path.name.lower()
        if "doubles" in name or "mixed" in name:
            continue
        head = pd.read_csv(path, nrows=0)
        winner = next((c for c in ("PointWinner", "Pointwinner", "point_winner") if c in head), None)
        server = next((c for c in ("PointServer", "Pointserver", "point_server") if c in head), None)
        if winner is None or server is None or "match_id" not in head:
            continue
        d = pd.read_csv(path, usecols=["match_id", winner, server], low_memory=False)
        d = d.rename(columns={winner: "winner", server: "server"})
        d["winner"] = pd.to_numeric(d["winner"], errors="coerce")
        d["server"] = pd.to_numeric(d["server"], errors="coerce")
        d["valid"] = d["winner"].isin([1, 2]) & d["server"].isin([1, 2])
        frames.append(d[["match_id", "valid"]])
    source = pd.concat(frames, ignore_index=True)
    match = source.groupby("match_id", sort=False).agg(
        source_rows=("valid", "size"), valid_rows=("valid", "sum")
    ).reset_index()
    code = match["match_id"].astype(str).str.extract(r"-([^-]+)$")[0].str.upper()
    match["tour"] = np.select(
        [code.str.startswith(("1", "MS")), code.str.startswith(("2", "WS"))],
        ["atp", "wta"], default="unknown")
    match["year"] = pd.to_numeric(match["match_id"].astype(str).str[:4], errors="coerce")
    retained_counts = retained.groupby(["tour", "year"]).agg(
        retained_matches=("match_id", "nunique"), retained_points=("match_id", "size")
    ).reset_index()
    annual = match[match["tour"].isin(tm.TOURS)].groupby(["tour", "year"]).agg(
        source_matches=("match_id", "nunique"),
        source_rows=("source_rows", "sum"),
        matches_with_50_valid_points=("valid_rows", lambda x: int((x >= 50).sum())),
        valid_rows=("valid_rows", "sum")
    ).reset_index().merge(retained_counts, on=["tour", "year"], how="left").fillna(0)
    for c in ("retained_matches", "retained_points"):
        annual[c] = annual[c].astype(int)
    annual["match_retention_rate"] = annual["retained_matches"] / annual["source_matches"]
    summary = annual.groupby("tour").agg(
        source_matches=("source_matches", "sum"),
        matches_with_50_valid_points=("matches_with_50_valid_points", "sum"),
        retained_matches=("retained_matches", "sum"),
        source_rows=("source_rows", "sum"),
        valid_rows=("valid_rows", "sum"),
        retained_points=("retained_points", "sum")
    ).reset_index()
    summary["match_retention_rate"] = summary["retained_matches"] / summary["source_matches"]
    summary["valid_point_retention_rate"] = summary["retained_points"] / summary["valid_rows"]
    return summary, annual


def tiebreak_state_audit(points: pd.DataFrame) -> pd.DataFrame:
    d = points.sort_values(["match_id", "pt_idx"]).reset_index(drop=True).copy()
    game = d["match_id"].astype(str) + "|" + d["SetNo"].astype(str) + "|" + d["GameNo"].astype(str)
    codes = pd.factorize(game)[0]
    multi_server = pd.Series(d["PointServer"].to_numpy()).groupby(codes).transform("nunique").to_numpy() > 1
    state = rev._point_score_states(d)
    rows = []
    for tour in tm.TOURS:
        keep = d["tour"].eq(tour).to_numpy()
        rows.append({
            "tour": tour,
            "points": int(keep.sum()),
            "tiebreak_points": int((keep & multi_server).sum()),
            "tiebreak_point_share": float((keep & multi_server).sum() / keep.sum()),
            "tiebreak_games": int(pd.Series(codes[keep & multi_server]).nunique()),
            "distinct_tiebreak_states": int(pd.Series(state[keep & multi_server]).nunique()),
        })
    return pd.DataFrame(rows)


def preferred_confirmatory_family(preferred: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(ROOT / "output" / "replication" / "results" / "revision" /
                      "confirmatory_eight_test_holm.csv")
    global_rows = old[old["estimand"] == "global"][["estimand", "hypothesis", "raw_p"]].copy()
    local = preferred[preferred["stratification"] == "match_server_score"].copy()
    local = pd.DataFrame({
        "estimand": "local_score_restricted",
        "hypothesis": local["tour"] + "_point_score_restricted",
        "raw_p": local["p_two_sided"],
    })
    family = pd.concat([global_rows, local], ignore_index=True)
    reject, adjusted, _, _ = multipletests(family["raw_p"].to_numpy(), method="holm")
    family["p_holm_eight_test_family"] = adjusted
    family["reject_holm_0_05"] = reject.astype(int)
    return family


def main(B: int = 999) -> None:
    points = tm.load_points()
    coverage, annual = source_coverage_audit(points)
    coverage.to_csv(OUT / "sample_coverage_summary.csv", index=False)
    annual.to_csv(OUT / "sample_coverage_by_year.csv", index=False)
    tiebreak_state_audit(points).to_csv(OUT / "tiebreak_state_audit.csv", index=False)

    rows, null_rows, bootstrap_rows = [], [], []
    modes = (
        {"set_restricted": False, "exclude_tiebreak": False},
        {"set_restricted": True, "exclude_tiebreak": False},
        {"set_restricted": False, "exclude_tiebreak": True},
    )
    for tour in tm.TOURS:
        pt = points[points["tour"] == tour].copy()
        for mode in modes:
            result, null = rev.calibrated_pm_point(
                pt, tour, B, score_restricted=True, **mode)
            rows.append(result)
            null_rows.append(pd.DataFrame({
                "tour": tour, "stratification": result["stratification"],
                "draw": np.arange(1, B + 1), "null_beta": null}))
            bootstrap_rows.append({
                "stratification": result["stratification"],
                **rev.bootstrap_calibrated_point(pt, tour, result["null_mean"], R=999)})
    preferred = pd.DataFrame(rows)
    preferred.to_csv(OUT / "nested_score_orbit_results.csv", index=False)
    pd.concat(null_rows, ignore_index=True).to_csv(OUT / "nested_score_orbit_nulls.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "nested_score_orbit_bootstrap.csv", index=False)
    preferred_confirmatory_family(preferred).to_csv(
        OUT / "preferred_eight_test_holm.csv", index=False)
    print(coverage.to_string(index=False))
    print(preferred.to_string(index=False))


if __name__ == "__main__":
    main()