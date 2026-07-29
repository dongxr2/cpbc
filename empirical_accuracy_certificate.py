"""Cross-fitted empirical accuracy certificate for the Grand Slam CPBC analysis.

The null baseline is estimated out of fold at the match level.  It contains no
lagged outcome: server, returner, pre-point score state, and era enter through
additive, shrunk empirical logits.  Conditional permutations then estimate the
score covariance kappa on exactly the orbit used by CPBC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import expit, logit

import mca_revision_analysis as rev
import tennis_momentum_advanced as tm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "momentum_output" / "revision"
SEED = 20260704


def _shrunk_probability(train: pd.DataFrame, key: str, global_p: float,
                        prior_n: float = 50.0) -> pd.Series:
    stats = train.groupby(key, observed=True)["server_won"].agg(["sum", "count"])
    return (stats["sum"] + prior_n * global_p) / (stats["count"] + prior_n)


def cross_fitted_null_probability(full: pd.DataFrame, folds: int = 5,
                                  prior_n: float = 50.0) -> np.ndarray:
    """Five-fold match-level predictions from a prespecified additive null."""
    work = full.copy()
    work["score_state"] = rev._point_score_states(work)
    work["era_null"] = work["era"].astype(str).fillna("unknown")
    match_hash = pd.util.hash_pandas_object(
        work["match_id"].astype(str), index=False
    ).to_numpy(np.uint64)
    fold_id = (match_hash % folds).astype(np.int8)
    p0 = np.empty(len(work), dtype=np.float64)
    keys = ("server_name", "returner_name", "score_state", "era_null")

    for fold in range(folds):
        train = work.loc[fold_id != fold]
        test = work.loc[fold_id == fold]
        global_p = float(train["server_won"].mean())
        base_logit = float(logit(np.clip(global_p, 1e-5, 1 - 1e-5)))
        linear = np.full(len(test), base_logit, dtype=np.float64)
        for key in keys:
            probs = _shrunk_probability(train, key, global_p, prior_n)
            mapped = test[key].map(probs).fillna(global_p).to_numpy(float)
            linear += logit(np.clip(mapped, 1e-5, 1 - 1e-5)) - base_logit
        p0[test.index.to_numpy()] = np.clip(expit(linear), 0.02, 0.98)
    return p0


@njit(cache=True)
def _orbit_coefficient_score(y0, p0, server_slot, packed_indices, starts,
                             analysis_indices, group_codes, control_within,
                             group_counts, seeds):
    coefficients = np.empty(len(seeds), dtype=np.float64)
    scores = np.empty(len(seeds), dtype=np.float64)
    ngroups = len(group_counts)
    n_analysis = len(analysis_indices)
    for b in range(len(seeds)):
        np.random.seed(seeds[b])
        y = y0.copy()
        for h in range(len(starts) - 1):
            left, right = starts[h], starts[h + 1]
            for j in range(right - 1, left, -1):
                k = left + np.random.randint(0, j - left + 1)
                ij, ik = packed_indices[j], packed_indices[k]
                y[ij], y[ik] = y[ik], y[ij]

        sum_y = np.zeros(ngroups, dtype=np.float64)
        sum_x = np.zeros(ngroups, dtype=np.float64)
        x_values = np.empty(n_analysis, dtype=np.float64)
        score = 0.0
        for q in range(n_analysis):
            idx = analysis_indices[q]
            previous_winner = (server_slot[idx - 1] if y[idx - 1] == 1
                               else 3 - server_slot[idx - 1])
            x = 1.0 if previous_winner == server_slot[idx] else 0.0
            x_values[q] = x
            g = group_codes[q]
            sum_x[g] += x
            sum_y[g] += y[idx]
            score += x * (y[idx] - p0[idx])

        xx = xz = zz = xy = zy = 0.0
        for q in range(n_analysis):
            g = group_codes[q]
            xw = x_values[q] - sum_x[g] / group_counts[g]
            yw = y[analysis_indices[q]] - sum_y[g] / group_counts[g]
            zw = control_within[q]
            xx += xw * xw
            xz += xw * zw
            zz += zw * zw
            xy += xw * yw
            zy += zw * yw
        determinant = xx * zz - xz * xz
        coefficients[b] = ((zz * xy - xz * zy) / determinant
                           if abs(determinant) > 1e-14 else xy / xx)
        scores[b] = score
    return coefficients, scores


def certificate(points: pd.DataFrame, tour: str, score_restricted: bool,
                B: int = 499) -> dict:
    full = points.sort_values(["match_id", "pt_idx"]).reset_index(drop=True).copy()
    p0 = cross_fitted_null_probability(full)
    match_codes = pd.factorize(full["match_id"])[0]
    first = np.r_[True, match_codes[1:] != match_codes[:-1]]
    y = full["server_won"].to_numpy(np.int8)
    server_slot = full["PointServer"].to_numpy(np.int8)
    stratum = full["match_id"].astype(str) + "|" + full["PointServer"].astype(str)
    if score_restricted:
        stratum += "|" + pd.Series(rev._point_score_states(full), index=full.index)
    packed, starts = rev._packed_groups(stratum.to_numpy())

    counts = full.loc[~first, "server_name"].value_counts()
    keep_players = set(counts[counts >= 200].index)
    analysis_mask = (~first) & full["server_name"].isin(keep_players).to_numpy()
    analysis_indices = np.flatnonzero(analysis_mask).astype(np.int64)
    player_match = (full.loc[analysis_mask, "match_id"].astype(str) + "|" +
                    full.loc[analysis_mask, "server_name"].astype(str)).to_numpy()
    group_codes = pd.factorize(player_match)[0].astype(np.int64)
    group_counts = np.bincount(group_codes).astype(np.float64)
    control = full.loc[analysis_mask, "set_late"].to_numpy(float)
    control_within = rev._demean(control, group_codes).astype(np.float64)
    rng = np.random.default_rng(SEED + (1000 if tour == "wta" else 0) +
                                (10000 if score_restricted else 0))
    seeds = rng.integers(1, 2**31 - 1, size=B, dtype=np.int64)
    coefs, scores = _orbit_coefficient_score(
        y, p0, server_slot, packed, starts, analysis_indices, group_codes,
        control_within, group_counts, seeds
    )
    kappa = float(np.mean((coefs - coefs.mean()) * (scores - scores.mean())))
    tau1 = float(np.mean(p0[analysis_mask] * (1.0 - p0[analysis_mask])))
    block_ratios = []
    for block in np.array_split(np.arange(B), 10):
        bt, bs = coefs[block], scores[block]
        block_kappa = float(np.mean((bt - bt.mean()) * (bs - bs.mean())))
        block_ratios.append(block_kappa / tau1)
    return {
        "tour": tour.upper(),
        "orbit": "match-server-score" if score_restricted else "match-server",
        "folds": 5,
        "prior_n": 50,
        "B": B,
        "n": int(analysis_mask.sum()),
        "mean_p0": float(p0[analysis_mask].mean()),
        "tau_prime_0": tau1,
        "kappa": kappa,
        "kappa_over_tau_prime": kappa / tau1,
        "block_ratio_sd": float(np.std(block_ratios, ddof=1)),
        "block_ratio_min": float(np.min(block_ratios)),
        "block_ratio_max": float(np.max(block_ratios)),
    }


def main(B: int = 499) -> None:
    points = tm.load_points()
    rows = []
    for tour in ("atp", "wta"):
        subset = points.loc[points["tour"] == tour].copy()
        for restricted in (False, True):
            result = certificate(subset, tour, restricted, B=B)
            rows.append(result)
            print(result, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "empirical_accuracy_certificate.csv", index=False)


if __name__ == "__main__":
    main()
