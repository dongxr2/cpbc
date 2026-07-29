"""Additional analyses requested during the MCA manuscript revision.

The script implements four additions:
1. Conditional-permutation calibration of player-by-match fixed effects.
2. Sparse high-dimensional server-returner logistic regressions.
3. Match, player, and two-way match-player clustered standard errors.
4. Holm adjustment for the prespecified family of primary tests.

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit
from scipy import sparse
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

import tennis_momentum_advanced as tm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "momentum_output"
OUT.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20260628


def _demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    idx = pd.factorize(groups)[0]
    counts = np.bincount(idx).astype(float)
    means = np.bincount(idx, weights=values) / counts
    return values - means[idx]


def _within_slope(y: np.ndarray, x: np.ndarray, control: np.ndarray,
                  groups: np.ndarray) -> float:
    yw = _demean(y.astype(float), groups)
    xw = _demean(x.astype(float), groups)
    zw = _demean(control.astype(float), groups)
    design = np.column_stack([xw, zw])
    gram = design.T @ design
    rhs = design.T @ yw
    beta = np.linalg.pinv(gram) @ rhs
    return float(beta[0])


def _permutation_p(null: np.ndarray, observed: float) -> tuple[float, float]:
    """Relabeling-invariant Monte Carlo randomization probabilities."""
    reference = np.r_[observed, np.asarray(null, dtype=float)]
    center = float(np.mean(reference))
    right = np.mean(reference >= observed)
    two = np.mean(np.abs(reference - center) >= abs(observed - center))
    return float(right), float(two)


def _add_reference_uncertainty(result: dict) -> dict:
    """Conditional-null reference interval and normal-approximation power metrics."""
    sd = float(result["null_sd"])
    effect = float(result["calibrated_effect"])
    result["reference_ci_low"] = effect - 1.96 * sd
    result["reference_ci_high"] = effect + 1.96 * sd
    result["mde_80_two_sided"] = (1.959964 + 0.841621) * sd
    return result


def _shuffle_within(values: np.ndarray, index_groups: list[np.ndarray],
                    rng: np.random.Generator) -> np.ndarray:
    shuffled = values.copy()
    for idx in index_groups:
        if len(idx) > 1:
            shuffled[idx] = rng.permutation(values[idx])
    return shuffled


def _index_groups(keys: np.ndarray) -> list[np.ndarray]:
    codes = pd.factorize(keys)[0]
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    cuts = np.flatnonzero(np.diff(sorted_codes)) + 1
    return [part for part in np.split(order, cuts) if len(part)]


def _packed_groups(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = _index_groups(keys)
    indices = np.concatenate(groups).astype(np.int64)
    starts = np.r_[0, np.cumsum([len(g) for g in groups])].astype(np.int64)
    return indices, starts


@njit(cache=True)
def _fast_null_coefficients(y0, server_slot, packed_indices, starts,
                            analysis_indices, group_codes, control_within,
                            group_counts, seeds):
    """Numba implementation of the conditional shuffles and within slopes."""
    output = np.empty(len(seeds), dtype=np.float64)
    ngroups = len(group_counts)
    n_analysis = len(analysis_indices)
    for b in range(len(seeds)):
        np.random.seed(seeds[b])
        y = y0.copy()
        for h in range(len(starts) - 1):
            left = starts[h]
            right = starts[h + 1]
            for j in range(right - 1, left, -1):
                k = left + np.random.randint(0, j - left + 1)
                ij = packed_indices[j]
                ik = packed_indices[k]
                tmp = y[ij]
                y[ij] = y[ik]
                y[ik] = tmp

        sum_y = np.zeros(ngroups, dtype=np.float64)
        sum_x = np.zeros(ngroups, dtype=np.float64)
        x_values = np.empty(n_analysis, dtype=np.float64)
        for q in range(n_analysis):
            idx = analysis_indices[q]
            previous_winner = server_slot[idx - 1] if y[idx - 1] == 1 else 3 - server_slot[idx - 1]
            x = 1.0 if previous_winner == server_slot[idx] else 0.0
            x_values[q] = x
            g = group_codes[q]
            sum_x[g] += x
            sum_y[g] += y[idx]

        xx = 0.0
        xz = 0.0
        zz = 0.0
        xy = 0.0
        zy = 0.0
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
        output[b] = (zz * xy - xz * zy) / determinant if abs(determinant) > 1e-14 else xy / xx
    return output


def _fast_conditional_null(y_original: np.ndarray, server_slot: np.ndarray,
                           stratum_key: np.ndarray, analysis_mask: np.ndarray,
                           player_match: np.ndarray, control: np.ndarray,
                           B: int, seed: int) -> np.ndarray:
    packed, starts = _packed_groups(stratum_key)
    analysis_indices = np.flatnonzero(analysis_mask).astype(np.int64)
    group_codes = pd.factorize(player_match)[0].astype(np.int64)
    group_counts = np.bincount(group_codes).astype(np.float64)
    control_within = _demean(control.astype(float), group_codes).astype(np.float64)
    seeds = np.random.default_rng(seed).integers(1, 2**31 - 1, size=B, dtype=np.int64)
    return _fast_null_coefficients(
        y_original.astype(np.int8), server_slot.astype(np.int8), packed, starts,
        analysis_indices, group_codes, control_within, group_counts, seeds
    )


def _point_score_states(points: pd.DataFrame) -> np.ndarray:
    """Pre-point tennis score from the current server's perspective."""
    game_key = (points["match_id"].astype(str) + "|" + points["SetNo"].astype(str) +
                "|" + points["GameNo"].astype(str))
    codes = pd.factorize(game_key)[0]
    winner = points["PointWinner"].to_numpy(np.int8)
    server = points["PointServer"].to_numpy(np.int8)
    p1_win = (winner == 1).astype(np.int8)
    p2_win = (winner == 2).astype(np.int8)
    p1_before = pd.Series(p1_win).groupby(codes).cumsum().to_numpy() - p1_win
    p2_before = pd.Series(p2_win).groupby(codes).cumsum().to_numpy() - p2_win
    server_before = np.where(server == 1, p1_before, p2_before)
    return_before = np.where(server == 1, p2_before, p1_before)
    server_counts = pd.Series(server).groupby(codes).transform("nunique").to_numpy()
    tiebreak = server_counts > 1
    states = []
    for s, r, tb in zip(server_before, return_before, tiebreak):
        if tb:
            states.append(f"TB-{int(s)}-{int(r)}")
        elif s >= 3 and r >= 3:
            states.append("D" if s == r else ("AI" if s > r else "AO"))
        else:
            states.append(f"{min(int(s), 3)}-{min(int(r), 3)}")
    return np.asarray(states)


def calibrated_pm_point(points: pd.DataFrame, tour: str, B: int,
                        score_restricted: bool = False,
                        set_restricted: bool = False,
                        exclude_tiebreak: bool = False,
                        transition_scope: str = "all") -> tuple[dict, np.ndarray]:
    full = points.sort_values(["match_id", "pt_idx"]).reset_index(drop=True).copy()
    match_codes = pd.factorize(full["match_id"])[0]
    first = np.r_[True, match_codes[1:] != match_codes[:-1]]
    y_original = full["server_won"].to_numpy(np.int8)
    server_slot = full["PointServer"].to_numpy(np.int8)
    stratum = full["match_id"].astype(str) + "|" + full["PointServer"].astype(str)
    if score_restricted:
        stratum = stratum + "|" + pd.Series(_point_score_states(full), index=full.index)
    if set_restricted:
        stratum = stratum + "|SET" + full["SetNo"].astype(int).astype(str)
    stratum_key = stratum.to_numpy()

    game_key = (full["match_id"].astype(str) + "|" + full["SetNo"].astype(str) +
                "|" + full["GameNo"].astype(str)).to_numpy()
    boundary = np.r_[False, game_key[1:] != game_key[:-1]] & (~first)
    game_codes = pd.factorize(game_key)[0]
    tiebreak = (pd.Series(full["PointServer"].to_numpy()).groupby(game_codes)
                .transform("nunique").to_numpy() > 1)
    scope_mask = ~first
    if exclude_tiebreak:
        scope_mask &= ~tiebreak
    if transition_scope == "within_game":
        scope_mask &= ~boundary
    elif transition_scope == "boundary":
        scope_mask &= boundary
    elif transition_scope != "all":
        raise ValueError(f"Unknown transition scope: {transition_scope}")

    counts = full.loc[~first, "server_name"].value_counts()
    keep_players = set(counts[counts >= 200].index)
    analysis_mask = scope_mask & full["server_name"].isin(keep_players).to_numpy()
    player_match = (full.loc[analysis_mask, "match_id"].astype(str) + "|" +
                    full.loc[analysis_mask, "server_name"].astype(str)).to_numpy()
    control = full.loc[analysis_mask, "set_late"].to_numpy(float)

    def estimate(y_full: np.ndarray) -> float:
        winner = np.where(y_full == 1, server_slot, 3 - server_slot)
        previous = np.roll(winner, 1)
        x_full = (previous == server_slot).astype(float)
        return _within_slope(
            y_full[analysis_mask].astype(float), x_full[analysis_mask],
            control, player_match
        )

    observed = estimate(y_original)
    scope_offset = {"all": 0, "within_game": 20000, "boundary": 30000}[transition_scope]
    seed = (RNG_SEED + (0 if tour == "atp" else 1000) +
            (10000 if score_restricted else 0) +
            (40000 if set_restricted else 0) +
            (80000 if exclude_tiebreak else 0) + scope_offset)
    null = _fast_conditional_null(
        y_original, server_slot, stratum_key, analysis_mask, player_match,
        control, B, seed
    )
    p_right, p_two = _permutation_p(null, observed)
    stratum_codes = pd.factorize(stratum_key)[0]
    stratum_sizes = np.bincount(stratum_codes)
    stratum_success = np.bincount(stratum_codes, weights=y_original)
    effective = ((stratum_sizes > 1) & (stratum_success > 0) &
                 (stratum_success < stratum_sizes))
    result = {
        "tour": tour, "level": "point" if transition_scope == "all" else
        f"point_{transition_scope}", "transition_scope": transition_scope, "stratification":
        ("match_server_score_no_tiebreak" if score_restricted and exclude_tiebreak else
         "match_server_set_score" if score_restricted and set_restricted else
         "match_server_score" if score_restricted else
         "match_server_set" if set_restricted else "match_server"), "B": B,
        "observed_pmfe": observed, "null_mean": null.mean(),
        "null_sd": null.std(ddof=1), "calibrated_effect": observed - null.mean(),
        "p_positive": p_right, "p_two_sided": p_two,
        "n": int(analysis_mask.sum()), "n_strata": len(np.unique(stratum_key)),
        "median_stratum_size": float(np.median(stratum_sizes)),
        "stratum_size_q25": float(np.quantile(stratum_sizes, 0.25)),
        "stratum_size_q75": float(np.quantile(stratum_sizes, 0.75)),
        "effective_strata": int(effective.sum()),
        "effective_observation_share": float(stratum_sizes[effective].sum() / len(full)),
    }
    return _add_reference_uncertainty(result), null


def calibrated_pm_game(games: pd.DataFrame, tour: str, B: int,
                       set_restricted: bool = False) -> tuple[dict, np.ndarray]:
    full = games.sort_values(["match_id", "g_idx"]).reset_index(drop=True).copy()
    match_codes = pd.factorize(full["match_id"])[0]
    first = np.r_[True, match_codes[1:] != match_codes[:-1]]
    y_original = full["held"].to_numpy(np.int8)
    server_slot = full["server"].to_numpy(np.int8)
    stratum = full["match_id"].astype(str) + "|" + full["server"].astype(str)
    if set_restricted and "SetNo" in full:
        stratum = stratum + "|" + full["SetNo"].astype(str)
    stratum_key = stratum.to_numpy()

    counts = full.loc[~first, "server_name"].value_counts()
    keep_players = set(counts[counts >= 80].index)
    analysis_mask = (~first) & full["server_name"].isin(keep_players).to_numpy()
    player_match = (full.loc[analysis_mask, "match_id"].astype(str) + "|" +
                    full.loc[analysis_mask, "server_name"].astype(str)).to_numpy()
    control = full.loc[analysis_mask, "set_late"].to_numpy(float)

    def estimate(y_full: np.ndarray) -> float:
        winner = np.where(y_full == 1, server_slot, 3 - server_slot)
        previous = np.roll(winner, 1)
        x_full = (previous == server_slot).astype(float)
        return _within_slope(
            y_full[analysis_mask].astype(float), x_full[analysis_mask],
            control, player_match
        )

    observed = estimate(y_original)
    seed = RNG_SEED + 2000 + (0 if tour == "atp" else 1000) + (10000 if set_restricted else 0)
    null = _fast_conditional_null(
        y_original, server_slot, stratum_key, analysis_mask, player_match,
        control, B, seed
    )
    p_right, p_two = _permutation_p(null, observed)
    result = {
        "tour": tour, "level": "game", "stratification":
        "match_server_set" if set_restricted else "match_server", "B": B,
        "observed_pmfe": observed, "null_mean": null.mean(),
        "null_sd": null.std(ddof=1), "calibrated_effect": observed - null.mean(),
        "p_positive": p_right, "p_two_sided": p_two,
        "n": int(analysis_mask.sum()), "n_strata": len(np.unique(stratum_key)),
    }
    return _add_reference_uncertainty(result), null


def bootstrap_calibrated_point(points: pd.DataFrame, tour: str, null_mean: float,
                               R: int = 999) -> dict:
    """Match-cluster bootstrap with the conditional-null mean held fixed."""
    full = points.sort_values(["match_id", "pt_idx"]).reset_index(drop=True).copy()
    match_codes_full = pd.factorize(full["match_id"])[0]
    first = np.r_[True, match_codes_full[1:] != match_codes_full[:-1]]
    y = full["server_won"].to_numpy(float)
    slot = full["PointServer"].to_numpy(np.int8)
    winner = np.where(y == 1, slot, 3 - slot)
    previous = np.roll(winner, 1)
    x = (previous == slot).astype(float)
    counts = full.loc[~first, "server_name"].value_counts()
    keep_players = set(counts[counts >= 200].index)
    keep = (~first) & full["server_name"].isin(keep_players).to_numpy()
    pm = (full.loc[keep, "match_id"].astype(str) + "|" +
          full.loc[keep, "server_name"].astype(str)).to_numpy()
    yw = _demean(y[keep], pm)
    xw = _demean(x[keep], pm)
    zw = _demean(full.loc[keep, "set_late"].to_numpy(float), pm)
    match = pd.factorize(full.loc[keep, "match_id"])[0]
    G = int(match.max()) + 1
    components = np.column_stack([
        np.bincount(match, weights=xw * xw, minlength=G),
        np.bincount(match, weights=xw * zw, minlength=G),
        np.bincount(match, weights=zw * zw, minlength=G),
        np.bincount(match, weights=xw * yw, minlength=G),
        np.bincount(match, weights=zw * yw, minlength=G),
    ])
    rng = np.random.default_rng(RNG_SEED + 50000 + (0 if tour == "atp" else 1000))
    estimates = np.empty(R)
    for r in range(R):
        weights = np.bincount(rng.integers(0, G, size=G), minlength=G)
        xx, xz, zz, xy, zy = weights @ components
        det = xx * zz - xz * xz
        estimates[r] = (zz * xy - xz * zy) / det - null_mean
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "tour": tour, "replications": R, "bootstrap_type": "match_fixed_null_mean",
        "bootstrap_mean": float(estimates.mean()), "bootstrap_se": float(estimates.std(ddof=1)),
        "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
        "null_mean_held_fixed": float(null_mean),
    }


def _cluster_rows(fit, term_index: int, match, player, label: dict) -> list[dict]:
    rows = []
    match_codes = pd.factorize(np.asarray(match))[0]
    player_codes = pd.factorize(np.asarray(player))[0]
    covariance_specs = {
        "match": cov_cluster(fit, match_codes),
        "player": cov_cluster(fit, player_codes),
        "match_player": cov_cluster_2groups(fit, match_codes, player_codes)[0],
    }
    beta = float(fit.params[term_index])
    for clustering, covariance in covariance_specs.items():
        variance = float(covariance[term_index, term_index])
        se = math.sqrt(max(variance, 0.0))
        p = 2 * norm.sf(abs(beta / se)) if se > 0 else np.nan
        rows.append({**label, "clustering": clustering, "beta": beta, "se": se, "p": p})
    return rows


def cluster_robustness_tour(points: pd.DataFrame, tour: str) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    d = tm.build_point_lags(points, 1).rename(columns={"L1": "prev_win"})
    counts = d["server_name"].value_counts()
    d = d[d["server_name"].isin(counts[counts >= 200].index)].copy()
    w = tm.demean_twoway(d, ["server_won", "prev_win", "set_late"],
                         "server_name", "returner_name")
    X = sm.add_constant(w[["tw_prev_win", "tw_set_late"]].to_numpy())
    fit = sm.OLS(w["tw_server_won"].to_numpy(), X).fit()
    rows.extend(_cluster_rows(fit, 1, w["match_id"], w["server_name"],
                              {"tour": tour, "level": "point", "test": "tour_effect"}))

    games = tm.build_games(points)
    g = games.sort_values(["match_id", "g_idx"]).copy()
    prev = g.groupby("match_id")["game_winner"].shift(1)
    g["prev_game_win"] = (prev == g["server"]).astype(float)
    g.loc[prev.isna(), "prev_game_win"] = np.nan
    g = g.dropna(subset=["prev_game_win"])
    counts = g["server_name"].value_counts()
    g = g[g["server_name"].isin(counts[counts >= 80].index)].copy()
    gw = tm.demean_twoway(g, ["held", "prev_game_win", "set_late"],
                          "server_name", "returner_name")
    Xg = sm.add_constant(gw[["tw_prev_game_win", "tw_set_late"]].to_numpy())
    fitg = sm.OLS(gw["tw_held"].to_numpy(), Xg).fit()
    rows.extend(_cluster_rows(fitg, 1, gw["match_id"], gw["server_name"],
                              {"tour": tour, "level": "game", "test": "tour_effect"}))
    games["tour"] = tour
    return rows, games


def cluster_robustness_interaction(points: pd.DataFrame, games: pd.DataFrame) -> list[dict]:
    rows = []
    frames = []
    for tour in tm.TOURS:
        d = tm.build_point_lags(points[points["tour"] == tour], 1).rename(
            columns={"L1": "prev_win"})
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    counts = d["server_name"].value_counts()
    d = d[d["server_name"].isin(counts[counts >= 200].index)].copy()
    d["female"] = (d["tour"] == "wta").astype(float)
    d["xf"] = d["prev_win"] * d["female"]
    w = tm.demean_twoway(d, ["server_won", "prev_win", "xf", "set_late"],
                         "server_name", "returner_name")
    X = sm.add_constant(w[["tw_prev_win", "tw_xf", "tw_set_late"]].to_numpy())
    fit = sm.OLS(w["tw_server_won"].to_numpy(), X).fit()
    rows.extend(_cluster_rows(fit, 2, w["match_id"], w["server_name"],
                              {"tour": "pooled", "level": "point", "test": "gender_interaction"}))

    frames = []
    for tour in tm.TOURS:
        g = games[games["tour"] == tour].sort_values(["match_id", "g_idx"]).copy()
        prev = g.groupby("match_id")["game_winner"].shift(1)
        g["prev_game_win"] = (prev == g["server"]).astype(float)
        g.loc[prev.isna(), "prev_game_win"] = np.nan
        frames.append(g.dropna(subset=["prev_game_win"]))
    g = pd.concat(frames, ignore_index=True)
    counts = g["server_name"].value_counts()
    g = g[g["server_name"].isin(counts[counts >= 80].index)].copy()
    g["female"] = (g["tour"] == "wta").astype(float)
    g["xf"] = g["prev_game_win"] * g["female"]
    gw = tm.demean_twoway(g, ["held", "prev_game_win", "xf", "set_late"],
                          "server_name", "returner_name")
    Xg = sm.add_constant(gw[["tw_prev_game_win", "tw_xf", "tw_set_late"]].to_numpy())
    fitg = sm.OLS(gw["tw_held"].to_numpy(), Xg).fit()
    rows.extend(_cluster_rows(fitg, 2, gw["match_id"], gw["server_name"],
                              {"tour": "pooled", "level": "game", "test": "gender_interaction"}))
    return rows


def _sparse_hdfe_logit(df: pd.DataFrame, outcome: str, lag: str,
                       max_iter: int = 100) -> dict:
    server_codes, server_levels = pd.factorize(df["server_name"])
    returner_codes, returner_levels = pd.factorize(df["returner_name"])
    n = len(df)
    reference_returner = len(returner_levels) - 1
    nonreference = returner_codes != reference_returner
    rows = np.r_[np.arange(n), np.arange(n)[nonreference]]
    cols = np.r_[server_codes, len(server_levels) + returner_codes[nonreference]]
    data = np.ones(len(rows), dtype=float)
    fixed = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n, len(server_levels) + len(returner_levels) - 1)
    )
    lag_values = df[lag].to_numpy(float)
    controls = sparse.csr_matrix(np.column_stack([
        lag_values, df["set_late"].to_numpy(float)
    ]))
    design = sparse.hstack([fixed, controls], format="csr")
    y = df[outcome].to_numpy(np.int8)
    model = LogisticRegression(
        penalty=None, fit_intercept=False, solver="newton-cg", max_iter=max_iter,
        tol=1e-6
    )
    model.fit(design, y)
    beta = float(model.coef_[0, -2])
    eta = model.decision_function(design)
    p1 = expit(eta + beta * (1.0 - lag_values))
    p0 = expit(eta - beta * lag_values)
    return {
        "log_odds": beta, "odds_ratio": math.exp(beta),
        "average_marginal_effect": float(np.mean(p1 - p0)),
        "n": n, "iterations": int(model.n_iter_[0]),
        "converged": int(model.n_iter_[0] < max_iter),
    }


def nonlinear_robustness(points: pd.DataFrame) -> list[dict]:
    rows = []
    for tour in tm.TOURS:
        pt = points[points["tour"] == tour]
        d = tm.build_point_lags(pt, 1).rename(columns={"L1": "prev_win"})
        counts = d["server_name"].value_counts()
        d = d[d["server_name"].isin(counts[counts >= 200].index)].copy()
        rows.append({"tour": tour, "level": "point", **_sparse_hdfe_logit(
            d, "server_won", "prev_win")})

        g = tm.build_games(pt).sort_values(["match_id", "g_idx"]).copy()
        prev = g.groupby("match_id")["game_winner"].shift(1)
        g["prev_game_win"] = (prev == g["server"]).astype(float)
        g.loc[prev.isna(), "prev_game_win"] = np.nan
        g = g.dropna(subset=["prev_game_win"])
        counts = g["server_name"].value_counts()
        g = g[g["server_name"].isin(counts[counts >= 80].index)].copy()
        rows.append({"tour": tour, "level": "game", **_sparse_hdfe_logit(
            g, "held", "prev_game_win")})
    return rows


def holm_primary(cluster_df: pd.DataFrame) -> pd.DataFrame:
    primary = cluster_df[
        (cluster_df["clustering"] == "match_player") &
        (((cluster_df["test"] == "tour_effect") & cluster_df["tour"].isin(tm.TOURS)) |
         (cluster_df["test"] == "gender_interaction"))
    ].copy()
    reject, adjusted, _, _ = multipletests(primary["p"].to_numpy(), method="holm")
    primary["p_holm"] = adjusted
    primary["reject_holm_0_05"] = reject.astype(int)
    return primary


def exact_sequence_diagnostics(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tour in tm.TOURS:
        d = points[points["tour"] == tour].sort_values(["match_id", "pt_idx"]).copy()
        d["seq_key"] = d["match_id"].astype(str) + "|" + d["focal"].astype(int).astype(str)
        total = short = degenerate = insufficient = valid = 0
        successor_counts = []
        for _, g in d.groupby("seq_key", sort=False):
            total += 1
            y = g["server_won"].to_numpy(np.int8)
            if len(y) < 10:
                short += 1
                continue
            k = int(y.sum())
            if k == 0 or k == len(y):
                degenerate += 1
                continue
            eligible = int((y[:-1] == 1).sum())
            if eligible < 2:
                insufficient += 1
                continue
            valid += 1
            successor_counts.append(eligible)
        rows.append({
            "tour": tour, "total_sequences": total, "valid_equal_weight_sequences": valid,
            "excluded_length_below_10": short, "excluded_all_zero_or_all_one": degenerate,
            "excluded_fewer_than_2_success_successors": insufficient,
            "median_eligible_successors": float(np.median(successor_counts)),
            "q25_eligible_successors": float(np.quantile(successor_counts, 0.25)),
            "q75_eligible_successors": float(np.quantile(successor_counts, 0.75)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--robustness-permutations", type=int, default=999)
    parser.add_argument("--skip-logit", action="store_true")
    args = parser.parse_args()

    print("Loading corrected point data")
    points = tm.load_points()
    calibration_rows = []
    robustness_rows = []
    transition_rows = []
    bootstrap_rows = []
    null_files = []
    cluster_rows = []
    all_games = []

    for tour in tm.TOURS:
        pt = points[points["tour"] == tour].copy()
        print(f"Calibrating {tour.upper()} point model")
        point_result, point_null = calibrated_pm_point(pt, tour, args.permutations)
        calibration_rows.append(point_result)
        bootstrap_rows.append(bootstrap_calibrated_point(
            pt, tour, point_result["null_mean"], R=999
        ))
        null_files.append(pd.DataFrame({"tour": tour, "level": "point", "null_beta": point_null}))
        print(f"Calibrating {tour.upper()} point model within score states")
        score_result, score_null = calibrated_pm_point(
            pt, tour, args.robustness_permutations, score_restricted=True
        )
        robustness_rows.append(score_result)
        null_files.append(pd.DataFrame({
            "tour": tour, "level": "point_score_restricted", "null_beta": score_null
        }))
        for scope in ("within_game", "boundary"):
            print(f"Calibrating {tour.upper()} point model for {scope} transitions")
            scope_result, scope_null = calibrated_pm_point(
                pt, tour, args.robustness_permutations, transition_scope=scope
            )
            transition_rows.append(scope_result)
            null_files.append(pd.DataFrame({
                "tour": tour, "level": f"point_{scope}", "null_beta": scope_null
            }))

        games = tm.build_games(pt)
        games["tour"] = tour
        all_games.append(games)
        print(f"Calibrating {tour.upper()} game model")
        game_result, game_null = calibrated_pm_game(games, tour, args.permutations)
        calibration_rows.append(game_result)
        null_files.append(pd.DataFrame({"tour": tour, "level": "game", "null_beta": game_null}))
        rows, _ = cluster_robustness_tour(pt, tour)
        cluster_rows.extend(rows)

    games_all = pd.concat(all_games, ignore_index=True)
    cluster_rows.extend(cluster_robustness_interaction(points, games_all))
    cluster_df = pd.DataFrame(cluster_rows)
    holm = holm_primary(cluster_df)
    exact_diagnostics = exact_sequence_diagnostics(points)

    pd.DataFrame(calibration_rows).to_csv(
        OUT / "conditional_permutation_calibration.csv", index=False
    )
    calibrated_primary = pd.DataFrame(calibration_rows)
    calibrated_primary = calibrated_primary[calibrated_primary["level"] == "point"].copy()
    reject_local, p_local, _, _ = multipletests(
        calibrated_primary["p_two_sided"].to_numpy(), method="holm"
    )
    calibrated_primary["p_holm_local_family"] = p_local
    calibrated_primary["reject_holm_0_05"] = reject_local.astype(int)
    calibrated_primary.to_csv(OUT / "calibrated_primary_holm.csv", index=False)
    global_primary = cluster_df[
        (cluster_df["clustering"] == "match_player") &
        (((cluster_df["test"] == "tour_effect") & cluster_df["tour"].isin(tm.TOURS)) |
         (cluster_df["test"] == "gender_interaction"))
    ].copy()
    confirmatory_all = pd.concat([
        global_primary.assign(
            estimand="global", raw_p=global_primary["p"],
            hypothesis=global_primary["tour"] + "_" + global_primary["level"] +
            "_" + global_primary["test"]
        )[["estimand", "hypothesis", "raw_p"]],
        calibrated_primary.assign(
            estimand="local_calibrated", raw_p=calibrated_primary["p_two_sided"],
            hypothesis=calibrated_primary["tour"] + "_point_calibrated"
        )[["estimand", "hypothesis", "raw_p"]],
    ], ignore_index=True)
    reject_all, adjusted_all, _, _ = multipletests(
        confirmatory_all["raw_p"].to_numpy(), method="holm"
    )
    confirmatory_all["p_holm_eight_test_family"] = adjusted_all
    confirmatory_all["reject_holm_0_05"] = reject_all.astype(int)
    confirmatory_all.to_csv(OUT / "confirmatory_eight_test_holm.csv", index=False)
    pd.DataFrame(robustness_rows).to_csv(
        OUT / "restricted_exchangeability_robustness.csv", index=False
    )
    pd.DataFrame(transition_rows).to_csv(
        OUT / "transition_scope_robustness.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        OUT / "calibrated_point_bootstrap.csv", index=False
    )
    pd.concat(null_files, ignore_index=True).to_csv(
        OUT / "conditional_permutation_nulls.csv", index=False
    )
    cluster_df.to_csv(OUT / "cluster_robustness.csv", index=False)
    holm.to_csv(OUT / "primary_tests_holm.csv", index=False)
    exact_diagnostics.to_csv(OUT / "exact_sequence_diagnostics.csv", index=False)

    if not args.skip_logit:
        logistic = pd.DataFrame(nonlinear_robustness(points))
        logistic.to_csv(OUT / "hdfe_logit_robustness.csv", index=False)
        print(logistic.to_string(index=False))

    print(pd.DataFrame(calibration_rows).to_string(index=False))
    print(pd.DataFrame(robustness_rows).to_string(index=False))
    print(pd.DataFrame(transition_rows).to_string(index=False))
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))
    print(cluster_df.to_string(index=False))
    print(holm.to_string(index=False))
    print(exact_diagnostics.to_string(index=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
