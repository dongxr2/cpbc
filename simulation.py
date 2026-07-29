"""Simulation study for the MCA manuscript.

The script evaluates two components of the proposed framework. First, it checks
whether the exact conditional finite-sequence correction centers the post-win
statistic under independent Bernoulli sampling more accurately than the raw and
first-order alternatives. Second, it evaluates raw, player fixed-effect,
server-returner fixed-effect, and player-match fixed-effect lag estimators under
a tennis-like binary data-generating process with alternating service blocks.

Outputs are written to momentum_output/simulation.
"""

from __future__ import annotations

import argparse
import math
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "momentum_output" / "simulation"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260627
COLORS = {
    "Raw gap": "#999999",
    "Player FE": "#E69F00",
    "Server-returner FE": "#0072B2",
    "Player-match FE": "#009E73",
}


@lru_cache(maxsize=None)
def exact_ms_bias(n: int, k: int) -> float:
    """Exact conditional bias of P(Y_t=1 | Y_{t-1}=1) minus k/n."""
    if k < 2 or n < 3 or k > n:
        return 0.0
    z = n - k
    total = math.comb(n, k)
    numerator = 0.0
    for runs in range(1, min(k, z + 1) + 1):
        common = math.comb(k - 1, runs - 1)
        ending_one = math.comb(z, runs - 1) * common
        ending_zero = math.comb(z, runs) * common
        numerator += ending_one * (k - runs) / (k - 1)
        numerator += ending_zero * (k - runs) / k
    return numerator / total - k / n


def first_order_bias(n: int, p: float) -> float:
    if n < 3 or not 0 < p < 1 or n * p <= 1:
        return 0.0
    return -p * (1.0 - p) / (n * p - 1.0)


def sequence_statistic(y: np.ndarray) -> tuple[float, float, float] | None:
    n = len(y)
    k = int(y.sum())
    if k < 2 or k == n:
        return None
    eligible = y[:-1] == 1
    if eligible.sum() < 2:
        return None
    p = k / n
    raw = y[1:][eligible].mean() - p
    return raw, raw - first_order_bias(n, p), raw - exact_ms_bias(n, k)


def finite_sequence_experiment(reps: int, sequences_per_rep: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in (12, 24, 48):
        for p in (0.50, 0.65):
            estimates = {"Raw": [], "First-order": [], "Exact conditional": []}
            rejected = {key: 0 for key in estimates}
            valid_counts = []
            for _ in range(reps):
                per_sequence = []
                for _ in range(sequences_per_rep):
                    y = rng.binomial(1, p, size=n).astype(float)
                    value = sequence_statistic(y)
                    if value is not None:
                        per_sequence.append(value)
                arr = np.asarray(per_sequence)
                valid_counts.append(len(arr))
                if len(arr) < 10:
                    continue
                for j, method in enumerate(estimates):
                    values = arr[:, j]
                    estimates[method].append(values.mean())
                    rejected[method] += int(ttest_1samp(values, 0.0).pvalue < 0.05)
            for method, values in estimates.items():
                values = np.asarray(values)
                rows.append({
                    "n": n,
                    "p": p,
                    "method": method,
                    "mean_estimate": values.mean(),
                    "absolute_bias": abs(values.mean()),
                    "rmse": np.sqrt(np.mean(values**2)),
                    "type1_error": rejected[method] / len(values),
                    "mean_valid_sequences": np.mean(valid_counts),
                    "replications": len(values),
                })
    return pd.DataFrame(rows)


def _alternating_demean(values: np.ndarray, g1: np.ndarray, g2: np.ndarray,
                        max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    out = values.astype(float).copy()
    i1, i2 = pd.factorize(g1)[0], pd.factorize(g2)[0]
    c1 = np.bincount(i1)
    c2 = np.bincount(i2)
    for _ in range(max_iter):
        m1 = np.bincount(i1, weights=out) / c1
        out -= m1[i1]
        m2 = np.bincount(i2, weights=out) / c2
        out -= m2[i2]
        check = np.bincount(i1, weights=out) / c1
        if np.max(np.abs(check)) < tol:
            break
    return out


def _oneway_demean(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    idx = pd.factorize(group)[0]
    counts = np.bincount(idx)
    means = np.bincount(idx, weights=values) / counts
    return values - means[idx]


def _cluster_slope(y: np.ndarray, x: np.ndarray,
                   cluster: np.ndarray) -> tuple[float, float, float]:
    denominator = float(x @ x)
    if denominator < 1e-12:
        return np.nan, np.nan, np.nan
    beta = float(x @ y / denominator)
    residual = y - beta * x
    cluster_index = pd.factorize(cluster)[0]
    scores = np.bincount(cluster_index, weights=x * residual)
    groups = len(scores)
    n = len(y)
    correction = groups / max(groups - 1, 1) * (n - 1) / max(n - 1, 1)
    variance = correction * float(scores @ scores) / denominator**2
    se = math.sqrt(max(variance, 0.0))
    pvalue = 2.0 * norm.sf(abs(beta / se)) if se > 0 else np.nan
    return beta, se, pvalue


def generate_tennis_like(rng: np.random.Generator, matches: int,
                         points_per_match: int, lag_logit: float,
                         form_sd: float, players: int = 80) -> tuple[pd.DataFrame, float]:
    serve = rng.normal(0.0, 0.38, players)
    return_strength = rng.normal(0.0, 0.30, players)
    records = []
    marginal_effects = []
    for match in range(matches):
        p1, p2 = rng.choice(players, size=2, replace=False)
        pair = np.array([p1, p2])
        form = rng.normal(0.0, form_sd, size=2)
        block_server = int(rng.integers(0, 2))
        remaining = int(rng.integers(4, 9))
        previous_winner = -1
        for point in range(points_per_match):
            if remaining == 0:
                block_server = 1 - block_server
                remaining = int(rng.integers(4, 9))
            server_slot = block_server
            returner_slot = 1 - block_server
            server = int(pair[server_slot])
            returner = int(pair[returner_slot])
            prev_win = np.nan if point == 0 else float(previous_winner == server)
            base = 0.58 + serve[server] - return_strength[returner] + form[server_slot]
            base_prob = float(expit(base))
            marginal_effects.append(float(expit(base + lag_logit) - base_prob))
            eta = base + (0.0 if point == 0 else lag_logit * prev_win)
            server_won = int(rng.random() < expit(eta))
            previous_winner = server if server_won else returner
            records.append((match, point, server, returner, server_won, prev_win,
                            base_prob))
            remaining -= 1
    df = pd.DataFrame(records, columns=[
        "match", "point", "server", "returner", "outcome", "prev_win",
        "base_prob"
    ]).dropna().reset_index(drop=True)
    return df, float(np.mean(marginal_effects))


def estimate_methods(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    y = df["outcome"].to_numpy(float)
    x = df["prev_win"].to_numpy(float)
    match = df["match"].to_numpy()
    server = df["server"].to_numpy()
    returner = df["returner"].to_numpy()

    x_raw = x - x.mean()
    y_raw = y - y.mean()
    result = {"Raw gap": _cluster_slope(y_raw, x_raw, match)}

    result["Player FE"] = _cluster_slope(
        _oneway_demean(y, server), _oneway_demean(x, server), match
    )
    result["Server-returner FE"] = _cluster_slope(
        _alternating_demean(y, server, returner),
        _alternating_demean(x, server, returner), match
    )
    player_match = match.astype(np.int64) * 10000 + server.astype(np.int64)
    result["Player-match FE"] = _cluster_slope(
        _oneway_demean(y, player_match),
        _oneway_demean(x, player_match), match
    )
    return result


def panel_experiment(reps: int) -> pd.DataFrame:
    settings = [
        ("Null, short", 0.00, 0.00, 80),
        ("Dependence, short", 0.12, 0.00, 80),
        ("Dependence and form, short", 0.12, 0.30, 80),
        ("Dependence and form, long", 0.12, 0.30, 220),
    ]
    rng = np.random.default_rng(SEED + 1)
    storage = []
    for scenario, lag, form_sd, length in settings:
        for replication in range(reps):
            df, target = generate_tennis_like(
                rng, matches=100, points_per_match=length,
                lag_logit=lag, form_sd=form_sd
            )
            for method, (estimate, se, pvalue) in estimate_methods(df).items():
                storage.append({
                    "scenario": scenario,
                    "replication": replication,
                    "method": method,
                    "target_ame": target,
                    "estimate": estimate,
                    "se": se,
                    "pvalue": pvalue,
                    "covered": int(estimate - 1.96 * se <= target <= estimate + 1.96 * se),
                    "rejected": int(pvalue < 0.05),
                    "n": len(df),
                })
    raw = pd.DataFrame(storage)
    summary = (raw.groupby(["scenario", "method"], sort=False)
               .apply(lambda g: pd.Series({
                   "target_ame": g["target_ame"].mean(),
                   "mean_estimate": g["estimate"].mean(),
                   "bias": (g["estimate"] - g["target_ame"]).mean(),
                   "absolute_bias": abs((g["estimate"] - g["target_ame"]).mean()),
                   "rmse": np.sqrt(np.mean((g["estimate"] - g["target_ame"])**2)),
                   "coverage": g["covered"].mean(),
                   "rejection_rate": g["rejected"].mean(),
                   "mean_n": g["n"].mean(),
                   "replications": len(g),
               }), include_groups=False).reset_index())
    raw.to_csv(OUT / "simulation_panel_replications.csv", index=False)
    return summary


def _pm_slope_from_ordered(df: pd.DataFrame, outcome: np.ndarray) -> float:
    d = df.sort_values(["match", "point"]).reset_index(drop=True)
    y = np.asarray(outcome, dtype=float)
    match = d["match"].to_numpy()
    server = d["server"].to_numpy()
    returner = d["returner"].to_numpy()
    first = np.r_[True, match[1:] != match[:-1]]
    winner = np.where(y == 1, server, returner)
    previous = np.roll(winner, 1)
    x = (previous == server).astype(float)
    keep = ~first
    player_match = match[keep].astype(np.int64) * 10000 + server[keep].astype(np.int64)
    yw = _oneway_demean(y[keep], player_match)
    xw = _oneway_demean(x[keep], player_match)
    denominator = float(xw @ xw)
    return float(xw @ yw / denominator)


def _pm_slope_influence(df: pd.DataFrame, outcome: np.ndarray,
                        subset: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Player-match within slope and match-cluster influence contributions."""
    d = df.sort_values(["match", "point"]).reset_index(drop=True)
    y = np.asarray(outcome, dtype=float)
    match = d["match"].to_numpy(np.int64)
    server = d["server"].to_numpy(np.int64)
    returner = d["returner"].to_numpy(np.int64)
    first = np.r_[True, match[1:] != match[:-1]]
    winner = np.where(y == 1, server, returner)
    previous = np.roll(winner, 1)
    x = (previous == server).astype(float)
    keep = ~first
    if subset is not None:
        keep &= np.asarray(subset, dtype=bool)
    pm = match[keep] * 10000 + server[keep]
    yw = _oneway_demean(y[keep], pm)
    xw = _oneway_demean(x[keep], pm)
    denominator = float(xw @ xw)
    beta = float(xw @ yw / denominator)
    residual = yw - beta * xw
    all_matches = int(match.max()) + 1
    scores = np.bincount(match[keep], weights=xw * residual,
                         minlength=all_matches) / denominator
    return beta, scores


def _influence_se(influence: np.ndarray) -> float:
    groups = len(influence)
    return math.sqrt(groups / max(groups - 1, 1) * float(influence @ influence))


def _dynamic_score_components(df: pd.DataFrame,
                              outcome: np.ndarray) -> tuple[float, float, float, float]:
    """Null score derivatives and AME derivatives for the tennis-like logit path."""
    d = df.sort_values(["match", "point"]).reset_index(drop=True)
    y = np.asarray(outcome, dtype=float)
    match = d["match"].to_numpy(np.int64)
    server = d["server"].to_numpy(np.int64)
    returner = d["returner"].to_numpy(np.int64)
    p0 = d["base_prob"].to_numpy(float)
    first = np.r_[True, match[1:] != match[:-1]]
    winner = np.where(y == 1, server, returner)
    previous = np.roll(winner, 1)
    x = (previous == server).astype(float)
    keep = ~first
    score1 = float(np.sum(x[keep] * (y[keep] - p0[keep])))
    score2 = float(-np.sum(x[keep] * p0[keep] * (1.0 - p0[keep])))
    tau1 = float(np.mean(p0[keep] * (1.0 - p0[keep])))
    tau2 = float(np.mean(p0[keep] * (1.0 - p0[keep]) * (1.0 - 2.0 * p0[keep])))
    return score1, score2, tau1, tau2


def _prepare_pm_design(df: pd.DataFrame) -> dict[str, np.ndarray | int]:
    d = df.sort_values(["match", "point"]).reset_index(drop=True)
    match = d["match"].to_numpy(np.int64)
    server = d["server"].to_numpy(np.int64)
    returner = d["returner"].to_numpy(np.int64)
    first = np.r_[True, match[1:] != match[:-1]]
    pm_raw = match * 10000 + server
    pm_codes = pd.factorize(pm_raw)[0]
    position = pd.Series(pm_raw).groupby(pm_raw).cumcount().to_numpy()
    size = pd.Series(pm_raw).groupby(pm_raw).transform("size").to_numpy()
    return {"match": match, "server": server, "returner": returner,
            "first": first, "pm_codes": pm_codes,
            "first_half": position < np.floor(size / 2),
            "groups": int(pm_codes.max()) + 1,
            "matches": int(match.max()) + 1}


def _fast_pm_slope_influence(prep: dict[str, np.ndarray | int],
                             outcome: np.ndarray,
                             subset: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    y = np.asarray(outcome, dtype=float)
    match = np.asarray(prep["match"])
    server = np.asarray(prep["server"])
    returner = np.asarray(prep["returner"])
    first = np.asarray(prep["first"])
    codes = np.asarray(prep["pm_codes"])
    winner = np.where(y == 1, server, returner)
    x = (np.roll(winner, 1) == server).astype(float)
    keep = ~first
    if subset is not None:
        keep &= np.asarray(subset, dtype=bool)
    cc = codes[keep]
    groups = int(prep["groups"])
    counts = np.bincount(cc, minlength=groups).astype(float)
    yv = y[keep]
    xv = x[keep]
    ymean = np.bincount(cc, weights=yv, minlength=groups) / np.maximum(counts, 1)
    xmean = np.bincount(cc, weights=xv, minlength=groups) / np.maximum(counts, 1)
    yw = yv - ymean[cc]
    xw = xv - xmean[cc]
    denominator = float(xw @ xw)
    beta = float(xw @ yw / denominator)
    residual = yw - beta * xw
    scores = np.bincount(match[keep], weights=xw * residual,
                         minlength=int(prep["matches"])) / denominator
    return beta, scores


def _fast_split_panel_jackknife(prep: dict[str, np.ndarray | int],
                                outcome: np.ndarray) -> tuple[float, float]:
    first_half = np.asarray(prep["first_half"])
    full, if_full = _fast_pm_slope_influence(prep, outcome)
    half1, if_half1 = _fast_pm_slope_influence(prep, outcome, first_half)
    half2, if_half2 = _fast_pm_slope_influence(prep, outcome, ~first_half)
    estimate = 2.0 * full - 0.5 * (half1 + half2)
    influence = 2.0 * if_full - 0.5 * (if_half1 + if_half2)
    return estimate, _influence_se(influence)


def split_panel_jackknife(df: pd.DataFrame, outcome: np.ndarray) -> tuple[float, float, float]:
    """Half-panel jackknife applied within every chronological player-match panel."""
    d = df.sort_values(["match", "point"]).reset_index(drop=True)
    pm = d["match"].to_numpy(np.int64) * 10000 + d["server"].to_numpy(np.int64)
    position = pd.Series(pm).groupby(pm).cumcount().to_numpy()
    size = pd.Series(pm).groupby(pm).transform("size").to_numpy()
    first_half = position < np.floor(size / 2)
    second_half = ~first_half
    full, if_full = _pm_slope_influence(d, outcome)
    half1, if_half1 = _pm_slope_influence(d, outcome, first_half)
    half2, if_half2 = _pm_slope_influence(d, outcome, second_half)
    estimate = 2.0 * full - 0.5 * (half1 + half2)
    influence = 2.0 * if_full - 0.5 * (if_half1 + if_half2)
    groups = len(influence)
    se = math.sqrt(groups / max(groups - 1, 1) * float(influence @ influence))
    pvalue = 2.0 * norm.sf(abs(estimate / se)) if se > 0 else np.nan
    return estimate, se, pvalue


def linear_panel_gmm(df: pd.DataFrame, system: bool = False,
                     max_lag: int = 4) -> dict[str, float]:
    """Collapsed one-step panel GMM for the CPBC linear estimand.

    The lag indicator is treated as predetermined.  Difference-equation moments
    use levels dated t-2 through t-max_lag.  The system version adds a level-
    equation moment instrumented by the first lagged difference.  Collapsing and
    the lag cap are prespecified because the unrestricted instrument count is
    larger than the number of player-match panels in the long design.
    """
    d = df.sort_values(["match", "server", "point"]).reset_index(drop=True)
    panel = (d["match"].to_numpy(np.int64) * 10000
             + d["server"].to_numpy(np.int64))
    y_all = d["outcome"].to_numpy(float)
    x_all = d["prev_win"].to_numpy(float)
    cuts = np.flatnonzero(np.diff(panel)) + 1
    groups = np.split(np.arange(len(d)), cuts)
    q_diff = max_lag - 1
    q = q_diff + int(system)
    ztz = np.zeros((q, q))
    ztx = np.zeros(q)
    zty = np.zeros(q)
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    max_t = 0
    for idx in groups:
        y = y_all[idx]
        x = x_all[idx]
        max_t = max(max_t, len(y))
        zr: list[np.ndarray] = []
        xr: list[float] = []
        yr: list[float] = []
        for t in range(2, len(y)):
            z = np.zeros(q)
            for lag in range(2, min(max_lag, t) + 1):
                z[lag - 2] = x[t - lag]
            zr.append(z)
            xr.append(x[t] - x[t - 1])
            yr.append(y[t] - y[t - 1])
        if system:
            for t in range(2, len(y)):
                z = np.zeros(q)
                z[-1] = x[t - 1] - x[t - 2]
                zr.append(z)
                xr.append(x[t])
                yr.append(y[t])
        if not zr:
            continue
        Z = np.vstack(zr)
        xv = np.asarray(xr)
        yv = np.asarray(yr)
        ztz += Z.T @ Z
        ztx += Z.T @ xv
        zty += Z.T @ yv
        blocks.append((Z, xv, yv))
    result = {
        "estimate": np.nan, "se": np.nan, "pvalue": np.nan,
        "hansen_p": np.nan, "instrument_count": float(q),
        "full_instrument_count": float((max_t - 2) * (max_t - 1) // 2
                                         + ((max_t - 2) if system else 0)),
        "failed": 1.0,
    }
    try:
        if len(blocks) < 3 or np.linalg.matrix_rank(ztz) < q:
            return result
        weight = np.linalg.inv(ztz)
        denominator = float(ztx @ weight @ ztx)
        if denominator <= 1e-12:
            return result
        beta = float(ztx @ weight @ zty / denominator)
        scores = np.vstack([Z.T @ (yv - beta * xv) for Z, xv, yv in blocks])
        meat = scores.T @ scores
        variance = float(ztx @ weight @ meat @ weight @ ztx) / denominator**2
        se = math.sqrt(max(variance, 0.0))
        pvalue = 2.0 * norm.sf(abs(beta / se)) if se > 0 else np.nan
        rank = int(np.linalg.matrix_rank(meat))
        if rank > 1:
            moment = zty - beta * ztx
            hansen = float(moment @ np.linalg.pinv(meat) @ moment)
            hansen_p = float(chi2.sf(hansen, rank - 1))
        else:
            hansen_p = np.nan
        result.update({"estimate": beta, "se": se, "pvalue": pvalue,
                       "hansen_p": hansen_p, "failed": 0.0})
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        pass
    return result


def _shuffle_sim_outcomes(df: pd.DataFrame, y: np.ndarray,
                          rng: np.random.Generator,
                          extra_key: np.ndarray | None = None) -> np.ndarray:
    shuffled = y.copy()
    keys = df["match"].to_numpy(np.int64) * 10000 + df["server"].to_numpy(np.int64)
    if extra_key is not None:
        keys = keys * 100 + np.asarray(extra_key, dtype=np.int64)
    codes = pd.factorize(keys)[0]
    order = np.argsort(codes, kind="stable")
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    for idx in np.split(order, cuts):
        if len(idx) > 1:
            shuffled[idx] = rng.permutation(y[idx])
    return shuffled


def _symmetric_permutation_p(observed: float, null: np.ndarray,
                             alternative: str = "two-sided") -> float:
    """Monte Carlo probability with a relabeling-invariant common center."""
    reference = np.r_[float(observed), np.asarray(null, dtype=float)]
    center = float(reference.mean())
    if alternative == "greater":
        return float(np.mean(reference >= observed))
    if alternative == "two-sided":
        return float(np.mean(
            np.abs(reference - center) >= abs(observed - center)))
    raise ValueError(f"Unknown alternative: {alternative}")


def calibration_experiment(reps: int = 100, B: int = 39) -> pd.DataFrame:
    """Head-to-head validation of PMFE, half-panel jackknife, and calibration."""
    settings = [
        ("Null and form, short", 0.00, 0.30, 80),
        ("Dependence and form, short", 0.12, 0.30, 80),
        ("Dependence and form, long", 0.12, 0.30, 220),
    ]
    rng = np.random.default_rng(SEED + 30)
    rows = []
    for scenario, lag, form_sd, length in settings:
        for replication in range(reps):
            df, target = generate_tennis_like(
                rng, matches=100, points_per_match=length,
                lag_logit=lag, form_sd=form_sd
            )
            df = df.sort_values(["match", "point"]).reset_index(drop=True)
            y = df["outcome"].to_numpy(float)
            observed = _pm_slope_from_ordered(df, y)
            _, observed_if = _pm_slope_influence(df, y)
            observed_se = math.sqrt(
                len(observed_if) / max(len(observed_if) - 1, 1) *
                float(observed_if @ observed_if)
            )
            observed_p = 2.0 * norm.sf(abs(observed / observed_se))
            spj, spj_se, spj_p = split_panel_jackknife(df, y)
            difference_gmm = linear_panel_gmm(df, system=False)
            system_gmm = linear_panel_gmm(df, system=True)
            null = np.empty(B)
            for b in range(B):
                null[b] = _pm_slope_from_ordered(
                    df, _shuffle_sim_outcomes(df, y, rng)
                )
            center = null.mean()
            calibrated = observed - center
            p_calibrated = _symmetric_permutation_p(observed, null)
            rows.extend([
                {"scenario": scenario, "replication": replication,
                 "method": "Player-match FE", "target_ame": target,
                 "estimate": observed, "pvalue": observed_p},
                {"scenario": scenario, "replication": replication,
                 "method": "Half-panel jackknife", "target_ame": target,
                 "estimate": spj, "pvalue": spj_p},
                {"scenario": scenario, "replication": replication,
                 "method": "Permutation-calibrated PMFE", "target_ame": target,
                 "estimate": calibrated, "pvalue": p_calibrated},
                {"scenario": scenario, "replication": replication,
                 "method": "Difference-GMM", "target_ame": target,
                 **difference_gmm},
                {"scenario": scenario, "replication": replication,
                 "method": "System-GMM", "target_ame": target,
                 **system_gmm},
            ])
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["scenario", "method"], sort=False)
               .apply(lambda g: pd.Series({
                   "target_ame": g["target_ame"].mean(),
                   "mean_estimate": g["estimate"].mean(),
                   "bias": (g["estimate"] - g["target_ame"]).mean(),
                   "rmse": np.sqrt(np.mean((g["estimate"] - g["target_ame"])**2)),
                   "rejection_rate": np.mean(g["pvalue"] <= 0.05),
                   "failure_rate": g.get("failed", pd.Series(0.0, index=g.index)).fillna(0).mean(),
                   "instrument_count": g.get("instrument_count", pd.Series(np.nan, index=g.index)).mean(),
                   "full_instrument_count": g.get("full_instrument_count", pd.Series(np.nan, index=g.index)).mean(),
                   "median_hansen_p": g.get("hansen_p", pd.Series(np.nan, index=g.index)).median(),
                   "hansen_p_gt_0_9": np.mean(g.get("hansen_p", pd.Series(np.nan, index=g.index)).dropna() > 0.9)
                       if g.get("hansen_p", pd.Series(np.nan, index=g.index)).notna().any() else np.nan,
                   "replications": len(g), "permutations": B,
               }), include_groups=False).reset_index())
    raw.to_csv(OUT / "simulation_calibration_replications.csv", index=False)
    summary.to_csv(OUT / "simulation_calibration_summary.csv", index=False)
    return summary


def generate_context_tennis_like(
        rng: np.random.Generator, matches: int, points_per_match: int,
        lag_logit: float, context_effect: float, form_sd: float = 0.30,
        players: int = 80) -> tuple[pd.DataFrame, float]:
    """Tennis-like panels with a persistent observed context state."""
    serve = rng.normal(0.0, 0.38, players)
    return_strength = rng.normal(0.0, 0.30, players)
    records: list[tuple] = []
    marginal_effects: list[float] = []
    for match in range(matches):
        p1, p2 = rng.choice(players, size=2, replace=False)
        pair = np.array([p1, p2])
        form = rng.normal(0.0, form_sd, size=2)
        block_server = int(rng.integers(0, 2))
        remaining = int(rng.integers(4, 9))
        context = int(rng.integers(0, 2))
        previous_winner = -1
        for point in range(points_per_match):
            if remaining == 0:
                block_server = 1 - block_server
                remaining = int(rng.integers(4, 9))
            if point > 0 and rng.random() > 0.85:
                context = 1 - context
            server_slot = block_server
            returner_slot = 1 - block_server
            server = int(pair[server_slot])
            returner = int(pair[returner_slot])
            prev_win = np.nan if point == 0 else float(previous_winner == server)
            context_term = context_effect * (2.0 * context - 1.0)
            base = (0.58 + serve[server] - return_strength[returner] +
                    form[server_slot] + context_term)
            base_prob = float(expit(base))
            if point > 0:
                marginal_effects.append(float(
                    expit(base + lag_logit) - base_prob))
            eta = base + (0.0 if point == 0 else lag_logit * prev_win)
            server_won = int(rng.random() < expit(eta))
            previous_winner = server if server_won else returner
            records.append((match, point, server, returner, server_won,
                            prev_win, base_prob, context))
            remaining -= 1
    df = pd.DataFrame(records, columns=[
        "match", "point", "server", "returner", "outcome", "prev_win",
        "base_prob", "context"
    ]).dropna().reset_index(drop=True)
    return df, float(np.mean(marginal_effects))


def exchangeability_refinement_experiment(reps: int = 300,
                                           B: int = 99) -> pd.DataFrame:
    """Measure how orbit refinement responds to observed context persistence."""
    settings = [
        ("Exchangeable null", 0.00, 0.00),
        ("Persistent-context null", 0.00, 0.55),
        ("Persistent context and dependence", 0.12, 0.55),
    ]
    rng = np.random.default_rng(SEED + 430)
    rows: list[dict] = []
    for scenario, lag, context_effect in settings:
        for replication in range(reps):
            df, target = generate_context_tennis_like(
                rng, matches=80, points_per_match=80, lag_logit=lag,
                context_effect=context_effect)
            df = df.sort_values(["match", "point"]).reset_index(drop=True)
            y = df["outcome"].to_numpy(float)
            observed = _pm_slope_from_ordered(df, y)
            for method, extra_key in (
                    ("Match-server orbit", None),
                    ("Context-restricted orbit", df["context"].to_numpy())):
                null = np.empty(B)
                for b in range(B):
                    shuffled = _shuffle_sim_outcomes(
                        df, y, rng, extra_key=extra_key)
                    null[b] = _pm_slope_from_ordered(df, shuffled)
                estimate = observed - float(null.mean())
                pvalue = _symmetric_permutation_p(observed, null)
                rows.append({
                    "scenario": scenario, "replication": replication,
                    "method": method, "target_ame": target,
                    "estimate": estimate, "pvalue": pvalue,
                    "context_effect": context_effect, "permutations": B,
                })
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["scenario", "method"], sort=False)
               .apply(lambda g: pd.Series({
                   "target_ame": g["target_ame"].mean(),
                   "mean_estimate": g["estimate"].mean(),
                   "bias": (g["estimate"] - g["target_ame"]).mean(),
                   "rmse": np.sqrt(np.mean(
                       (g["estimate"] - g["target_ame"]) ** 2)),
                   "rejection_rate": np.mean(g["pvalue"] <= 0.05),
                   "replications": len(g),
                   "permutations": int(g["permutations"].iloc[0]),
               }), include_groups=False).reset_index())
    raw.to_csv(OUT / "simulation_exchangeability_replications.csv", index=False)
    summary.to_csv(OUT / "simulation_exchangeability_summary.csv", index=False)
    return summary

def calibration_sweep_experiment(reps: int = 200, B: int = 49) -> pd.DataFrame:
    """Assess approximate bias additivity over dependence and panel length."""
    rng = np.random.default_rng(SEED + 130)
    rows = []
    for length in (40, 80, 220):
        for lag in (0.00, 0.05, 0.10, 0.20):
            for replication in range(reps):
                df, target = generate_tennis_like(
                    rng, matches=100, points_per_match=length,
                    lag_logit=lag, form_sd=0.30
                )
                df = df.sort_values(["match", "point"]).reset_index(drop=True)
                y = df["outcome"].to_numpy(float)
                observed = _pm_slope_from_ordered(df, y)
                null = np.empty(B)
                for b in range(B):
                    null[b] = _pm_slope_from_ordered(
                        df, _shuffle_sim_outcomes(df, y, rng)
                    )
                rows.append({
                    "length": length, "lag_logit": lag,
                    "replication": replication, "target_ame": target,
                    "pmfe": observed, "null_mean": null.mean(),
                    "calibrated": observed - null.mean()
                })
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["length", "lag_logit"], sort=True)
               .apply(lambda g: pd.Series({
                   "target_ame": g["target_ame"].mean(),
                   "pmfe_bias": (g["pmfe"] - g["target_ame"]).mean(),
                   "mean_null_bias": g["null_mean"].mean(),
                   "calibrated_bias": (g["calibrated"] - g["target_ame"]).mean(),
                   "calibrated_rmse": np.sqrt(np.mean(
                       (g["calibrated"] - g["target_ame"]) ** 2)),
                   "replications": len(g), "permutations": B,
               }), include_groups=False).reset_index())
    raw.to_csv(OUT / "simulation_calibration_sweep_replications.csv", index=False)
    summary.to_csv(OUT / "simulation_calibration_sweep_summary.csv", index=False)
    return summary


def derivative_calibration_experiment(reps: int = 100, B: int = 99) -> pd.DataFrame:
    """Score-covariance validation of derivative-calibrated CPBC (D-CPBC)."""
    rng = np.random.default_rng(SEED + 230)
    rows = []
    for length in (40, 80, 220):
        for lag in (0.00, 0.05, 0.10, 0.20):
            for replication in range(reps):
                df, target = generate_tennis_like(
                    rng, matches=100, points_per_match=length,
                    lag_logit=lag, form_sd=0.30
                )
                df = df.sort_values(["match", "point"]).reset_index(drop=True)
                y = df["outcome"].to_numpy(float)
                observed = _pm_slope_from_ordered(df, y)
                null_t = np.empty(B)
                null_s1 = np.empty(B)
                null_s2 = np.empty(B)
                tau1 = np.nan
                tau2 = np.nan
                for b in range(B):
                    shuffled = _shuffle_sim_outcomes(df, y, rng)
                    null_t[b] = _pm_slope_from_ordered(df, shuffled)
                    null_s1[b], null_s2[b], tau1, tau2 = (
                        _dynamic_score_components(df, shuffled)
                    )
                center_t = float(null_t.mean())
                center_s1 = float(null_s1.mean())
                kappa = float(np.mean((null_t - center_t) *
                                      (null_s1 - center_s1)))
                h2 = null_s2 + (null_s1 - center_s1) ** 2
                m2 = float(np.mean((null_t - center_t) * (h2 - h2.mean())))
                cpbc = observed - center_t
                if abs(kappa) > 1e-10 and tau1 > 0:
                    scale = tau1 / kappa
                    dcpbc = cpbc * scale
                    predicted_cpbc_bias = lag * kappa + 0.5 * lag**2 * m2 - target
                    predicted_dcpbc_bias = scale * (
                        lag * kappa + 0.5 * lag**2 * m2
                    ) - target
                else:
                    scale = np.nan
                    dcpbc = np.nan
                    predicted_cpbc_bias = np.nan
                    predicted_dcpbc_bias = np.nan
                rows.append({
                    "length": length, "lag_logit": lag,
                    "replication": replication, "target_ame": target,
                    "observed": observed, "cpbc": cpbc, "dcpbc": dcpbc,
                    "kappa": kappa, "tau1": tau1, "tau2": tau2,
                    "m2": m2, "scale": scale,
                    "first_order_envelope": abs(lag * (kappa - tau1)),
                    "predicted_cpbc_bias": predicted_cpbc_bias,
                    "predicted_dcpbc_bias": predicted_dcpbc_bias,
                })
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["length", "lag_logit"], sort=False)
               .apply(lambda g: pd.Series({
                   "target_ame": g["target_ame"].mean(),
                   "cpbc_bias": (g["cpbc"] - g["target_ame"]).mean(),
                   "cpbc_rmse": np.sqrt(np.mean((g["cpbc"] - g["target_ame"])**2)),
                   "dcpbc_bias": (g["dcpbc"] - g["target_ame"]).mean(),
                   "dcpbc_rmse": np.sqrt(np.mean((g["dcpbc"] - g["target_ame"])**2)),
                   "analytic_cpbc_bias": g["predicted_cpbc_bias"].mean(),
                   "analytic_dcpbc_bias": g["predicted_dcpbc_bias"].mean(),
                   "first_order_envelope": g["first_order_envelope"].mean(),
                   "mean_kappa": g["kappa"].mean(),
                   "mean_tau1": g["tau1"].mean(),
                   "mean_scale": g["scale"].mean(),
                   "scale_sd": g["scale"].std(),
                   "failed_rate": g["dcpbc"].isna().mean(),
                   "replications": len(g), "permutations": B,
               }), include_groups=False).reset_index())
    raw.to_csv(OUT / "simulation_derivative_cpbc_replications.csv", index=False)
    summary.to_csv(OUT / "simulation_derivative_cpbc_summary.csv", index=False)
    return summary


def enhanced_calibration_test_experiment(reps: int = 500,
                                         B: int = 99) -> pd.DataFrame:
    """Compare conditional tests with the analytic half-panel jackknife."""
    rng = np.random.default_rng(SEED + 330)
    rows = []
    settings = [
        ("Null, short", 0.00, 80),
        ("Dependence, short", 0.12, 80),
        ("Null, long", 0.00, 220),
        ("Dependence, long", 0.12, 220),
    ]
    for scenario, lag, length in settings:
        for replication in range(reps):
            df, target = generate_tennis_like(
                rng, matches=100, points_per_match=length,
                lag_logit=lag, form_sd=0.30
            )
            df = df.sort_values(["match", "point"]).reset_index(drop=True)
            y = df["outcome"].to_numpy(float)
            prep = _prepare_pm_design(df)
            observed, observed_if = _fast_pm_slope_influence(prep, y)
            observed_se = _influence_se(observed_if)
            hpj, hpj_se = _fast_split_panel_jackknife(prep, y)
            null = np.empty(B)
            null_se = np.empty(B)
            null_hpj = np.empty(B)
            null_hpj_se = np.empty(B)
            for b in range(B):
                shuffled = _shuffle_sim_outcomes(df, y, rng)
                null[b], influence = _fast_pm_slope_influence(prep, shuffled)
                null_se[b] = _influence_se(influence)
                null_hpj[b], null_hpj_se[b] = _fast_split_panel_jackknife(
                    prep, shuffled)
            center = float(null.mean())
            center_hpj = float(null_hpj.mean())
            calibrated = observed - center
            calibrated_hpj = hpj - center_hpj
            common_center = float(np.r_[observed, null].mean())
            raw_reference = np.r_[observed, null] - common_center
            p_raw_two = float(np.mean(
                np.abs(raw_reference) >= abs(raw_reference[0])))
            p_raw_pos = float(np.mean(raw_reference >= raw_reference[0]))
            z_obs = (observed - common_center) / observed_se
            z_null = (null - common_center) / null_se
            z_reference = np.r_[z_obs, z_null]
            p_stud_two = float(np.mean(
                np.abs(z_reference) >= abs(z_reference[0])))
            p_stud_pos = float(np.mean(z_reference >= z_reference[0]))
            common_center_hpj = float(np.r_[hpj, null_hpj].mean())
            z_hpj_obs = (hpj - common_center_hpj) / hpj_se
            z_hpj_null = (null_hpj - common_center_hpj) / null_hpj_se
            z_hpj_reference = np.r_[z_hpj_obs, z_hpj_null]
            p_hpj_two = float(np.mean(
                np.abs(z_hpj_reference) >= abs(z_hpj_reference[0])))
            p_hpj_pos = float(np.mean(
                z_hpj_reference >= z_hpj_reference[0]))
            p_analytic_two = 2.0 * norm.sf(abs(hpj / hpj_se)) if hpj_se > 0 else np.nan
            p_analytic_pos = norm.sf(hpj / hpj_se) if hpj_se > 0 else np.nan
            rows.extend([
                {"scenario": scenario, "length": length,
                 "replication": replication, "method": "Coefficient CPBC",
                 "target_ame": target, "estimate": calibrated,
                 "positive_p": p_raw_pos, "two_sided_p": p_raw_two},
                {"scenario": scenario, "length": length,
                 "replication": replication, "method": "Studentized CPBC",
                 "target_ame": target, "estimate": calibrated,
                 "positive_p": p_stud_pos, "two_sided_p": p_stud_two},
                {"scenario": scenario, "length": length,
                 "replication": replication, "method": "Permutation HPJ",
                 "target_ame": target, "estimate": calibrated_hpj,
                 "positive_p": p_hpj_pos, "two_sided_p": p_hpj_two},
                {"scenario": scenario, "length": length,
                 "replication": replication, "method": "Analytic HPJ",
                 "target_ame": target, "estimate": hpj,
                 "positive_p": p_analytic_pos,
                 "two_sided_p": p_analytic_two},
            ])
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["scenario", "length", "method"], sort=False)
               .apply(lambda g: pd.Series({
                   "target_ame": g["target_ame"].mean(),
                   "bias": (g["estimate"] - g["target_ame"]).mean(),
                   "rmse": np.sqrt(np.mean((g["estimate"] - g["target_ame"])**2)),
                   "positive_rejection_rate": np.mean(g["positive_p"] <= 0.05),
                   "two_sided_rejection_rate": np.mean(g["two_sided_p"] <= 0.05),
                   "replications": len(g),
                   "permutations": (0 if g.name[2] == "Analytic HPJ" else B),
               }), include_groups=False).reset_index())
    raw.to_csv(OUT / "simulation_enhanced_test_replications.csv", index=False)
    summary.to_csv(OUT / "simulation_enhanced_test_summary.csv", index=False)
    return summary


def augment_enhanced_test_with_analytic_hpj(reps: int = 500,
                                            B: int = 99) -> pd.DataFrame:
    """Add analytic HPJ to an existing enhanced-test run on the identical draws.

    The original experiment uses one fixed random stream for both DGP draws and
    conditional permutations.  Advancing that stream through the same shuffles
    reproduces every observed data set without needlessly refitting the three
    already stored permutation statistics.
    """
    raw_path = OUT / "simulation_enhanced_test_replications.csv"
    raw = pd.read_csv(raw_path)
    raw = raw[raw["method"] != "Analytic HPJ"].copy()
    expected = reps * 4 * 3
    if len(raw) != expected:
        raise ValueError(f"expected {expected} stored conditional rows, found {len(raw)}")
    rng = np.random.default_rng(SEED + 330)
    rows = []
    settings = [
        ("Null, short", 0.00, 80),
        ("Dependence, short", 0.12, 80),
        ("Null, long", 0.00, 220),
        ("Dependence, long", 0.12, 220),
    ]
    for scenario, lag, length in settings:
        for replication in range(reps):
            df, target = generate_tennis_like(
                rng, matches=100, points_per_match=length,
                lag_logit=lag, form_sd=0.30
            )
            df = df.sort_values(["match", "point"]).reset_index(drop=True)
            y = df["outcome"].to_numpy(float)
            hpj, hpj_se = _fast_split_panel_jackknife(
                _prepare_pm_design(df), y)
            p_two = 2.0 * norm.sf(abs(hpj / hpj_se)) if hpj_se > 0 else np.nan
            p_pos = norm.sf(hpj / hpj_se) if hpj_se > 0 else np.nan
            rows.append({
                "scenario": scenario, "length": length,
                "replication": replication, "method": "Analytic HPJ",
                "target_ame": target, "estimate": hpj,
                "positive_p": p_pos, "two_sided_p": p_two,
            })
            for _ in range(B):
                _shuffle_sim_outcomes(df, y, rng)
    raw = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)
    summary = (raw.groupby(["scenario", "length", "method"], sort=False)
               .apply(lambda g: pd.Series({
                   "target_ame": g["target_ame"].mean(),
                   "bias": (g["estimate"] - g["target_ame"]).mean(),
                   "rmse": np.sqrt(np.mean((g["estimate"] - g["target_ame"])**2)),
                   "positive_rejection_rate": np.mean(g["positive_p"] <= 0.05),
                   "two_sided_rejection_rate": np.mean(g["two_sided_p"] <= 0.05),
                   "replications": len(g),
                   "permutations": (0 if g.name[2] == "Analytic HPJ" else B),
               }), include_groups=False).reset_index())
    raw.to_csv(raw_path, index=False)
    summary.to_csv(OUT / "simulation_enhanced_test_summary.csv", index=False)
    return summary


def calibration_test_experiment(reps: int = 500, B: int = 99) -> pd.DataFrame:
    """Estimate randomization-test size and power in short and long panels."""
    rng = np.random.default_rng(SEED + 230)
    rows = []
    settings = [
        ("Null, short", 0.00, 80),
        ("Dependence, short", 0.12, 80),
        ("Null, long", 0.00, 220),
        ("Dependence, long", 0.12, 220),
    ]
    for scenario, lag, length in settings:
        for replication in range(reps):
            df, target = generate_tennis_like(
                rng, matches=100, points_per_match=length,
                lag_logit=lag, form_sd=0.30
            )
            df = df.sort_values(["match", "point"]).reset_index(drop=True)
            y = df["outcome"].to_numpy(float)
            observed = _pm_slope_from_ordered(df, y)
            null = np.empty(B)
            for b in range(B):
                null[b] = _pm_slope_from_ordered(
                    df, _shuffle_sim_outcomes(df, y, rng)
                )
            center = null.mean()
            p_positive = _symmetric_permutation_p(
                observed, null, alternative="greater")
            p_two = _symmetric_permutation_p(observed, null)
            rows.append({
                "scenario": scenario, "replication": replication,
                "length": length, "lag_logit": lag, "target_ame": target,
                "observed": observed, "calibrated": observed - center,
                "p_positive": p_positive, "p_two_sided": p_two,
                "reject_positive_0_05": int(p_positive <= 0.05),
                "reject_two_sided_0_05": int(p_two <= 0.05),
            })
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["scenario", "length", "lag_logit"], sort=False)
               .agg(target_ame=("target_ame", "mean"),
                    mean_calibrated=("calibrated", "mean"),
                    positive_rejection_rate=("reject_positive_0_05", "mean"),
                    two_sided_rejection_rate=("reject_two_sided_0_05", "mean"),
                    replications=("replication", "size")).reset_index())
    summary["permutations"] = B
    raw.to_csv(OUT / "simulation_calibration_test_replications.csv", index=False)
    summary.to_csv(OUT / "simulation_calibration_test_summary.csv", index=False)
    return summary


def runtime_experiment() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for matches in (25, 100, 400, 1000):
        df, _ = generate_tennis_like(
            rng, matches=matches, points_per_match=120,
            lag_logit=0.12, form_sd=0.30
        )
        estimate_methods(df)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            estimate_methods(df)
            timings.append(time.perf_counter() - start)
        rows.append({
            "matches": matches,
            "observations": len(df),
            "median_seconds_all_estimators": float(np.median(timings)),
        })
    return pd.DataFrame(rows)


def cpbc_runtime_experiment() -> pd.DataFrame:
    """Wall-clock scaling of the complete observed-plus-permutations CPBC loop."""
    rng = np.random.default_rng(SEED + 302)
    rows = []
    for matches in (25, 100, 400):
        df, _ = generate_tennis_like(
            rng, matches=matches, points_per_match=120,
            lag_logit=0.12, form_sd=0.30
        )
        df = df.sort_values(["match", "point"]).reset_index(drop=True)
        y = df["outcome"].to_numpy(float)
        for B in (99, 999):
            start = time.perf_counter()
            _pm_slope_from_ordered(df, y)
            for _ in range(B):
                shuffled = _shuffle_sim_outcomes(df, y, rng)
                _pm_slope_from_ordered(df, shuffled)
            elapsed = time.perf_counter() - start
            rows.append({"matches": matches, "observations": len(df),
                         "permutations": B, "seconds": elapsed,
                         "milliseconds_per_fit": 1000 * elapsed / (B + 1)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "simulation_cpbc_runtime.csv", index=False)
    return result


def make_derivative_figure(summary: pd.DataFrame) -> None:
    """Plot empirical D-CPBC performance against the score-based local expansion."""
    colors = {40: "#D55E00", 80: "#0072B2", 220: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for length, group in summary.groupby("length", sort=True):
        d = group.sort_values("target_ame")
        x = d["target_ame"].to_numpy()
        axes[0].plot(x, d["cpbc_bias"], marker="o", color=colors[length],
                     label=f"CPBC, T={length}")
        axes[0].plot(x, d["analytic_cpbc_bias"], linestyle="--",
                     color=colors[length], alpha=0.85,
                     label=f"Score expansion, T={length}")
        axes[0].plot(x, d["dcpbc_bias"], marker="s", linestyle=":",
                     color=colors[length], alpha=0.9,
                     label=f"D-CPBC, T={length}")
        envelope = d["first_order_envelope"].to_numpy()
        axes[0].fill_between(x, -envelope, envelope, color=colors[length],
                             alpha=0.06)
        axes[1].plot(x, d["cpbc_rmse"], marker="o", color=colors[length],
                     label=f"CPBC, T={length}")
        axes[1].plot(x, d["dcpbc_rmse"], marker="s", linestyle=":",
                     color=colors[length], label=f"D-CPBC, T={length}")
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_xlabel("True average marginal effect")
    axes[0].set_ylabel("Bias")
    axes[0].set_title("Empirical bias and score expansion")
    axes[1].set_xlabel("True average marginal effect")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Cost of derivative calibration")
    method_handles = [
        Line2D([0], [0], color="#333333", marker="o", label="CPBC"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Score expansion"),
        Line2D([0], [0], color="#333333", marker="s", linestyle=":", label="D-CPBC"),
    ]
    length_handles = [
        Line2D([0], [0], color=colors[length], linewidth=2, label=f"T={length}")
        for length in (40, 80, 220)
    ]
    first = axes[0].legend(handles=method_handles, frameon=False, fontsize=8,
                           loc="best")
    axes[0].add_artist(first)
    axes[0].legend(handles=length_handles, frameon=False, fontsize=8,
                   loc="lower right")
    axes[1].legend(handles=[method_handles[0], method_handles[2], *length_handles],
                   frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_derivative_cpbc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(finite: pd.DataFrame, panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    subset = finite[finite["p"] == 0.65]
    methods = ["Raw", "First-order", "Exact conditional"]
    colors = ["#999999", "#E69F00", "#0072B2"]
    width = 0.24
    x = np.arange(3)
    for j, method in enumerate(methods):
        d = subset[subset["method"] == method].sort_values("n")
        axes[0].bar(x + (j - 1) * width, d["absolute_bias"], width,
                    color=colors[j], label=method)
    axes[0].set_xticks(x, ["12", "24", "48"])
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Absolute bias")
    axes[0].legend(frameon=False)

    scenario_order = panel["scenario"].drop_duplicates().tolist()
    method_order = list(COLORS)
    width = 0.19
    x = np.arange(len(scenario_order))
    for j, method in enumerate(method_order):
        d = panel[panel["method"] == method].set_index("scenario").loc[scenario_order]
        axes[1].bar(x + (j - 1.5) * width, d["rmse"], width,
                    color=COLORS[method], label=method)
    axes[1].set_xticks(x, ["Null,\nshort", "Dependence,\nshort",
                           "Form,\nshort", "Form,\nlong"])
    axes[1].set_ylabel("RMSE")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_simulation_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=250)
    parser.add_argument("--finite-reps", type=int, default=500)
    parser.add_argument("--sequences", type=int, default=150)
    parser.add_argument("--calibration-reps", type=int, default=100)
    parser.add_argument("--calibration-perm", type=int, default=39)
    parser.add_argument("--followup-reps", type=int, default=0)
    parser.add_argument("--derivative-reps", type=int, default=0)
    parser.add_argument("--derivative-perm", type=int, default=199)
    parser.add_argument("--enhanced-test-reps", type=int, default=0)
    parser.add_argument("--enhanced-test-perm", type=int, default=99)
    parser.add_argument("--exchangeability-reps", type=int, default=0)
    parser.add_argument("--exchangeability-perm", type=int, default=99)
    parser.add_argument("--exchangeability-only", action="store_true")
    args = parser.parse_args()

    if args.exchangeability_only:
        result = exchangeability_refinement_experiment(
            max(args.exchangeability_reps, 1), args.exchangeability_perm)
        print(result.to_string(index=False))
        return

    finite = finite_sequence_experiment(args.finite_reps, args.sequences)
    panel = panel_experiment(args.reps)
    runtime = runtime_experiment()
    calibration = calibration_experiment(args.calibration_reps, args.calibration_perm)
    if args.followup_reps > 0:
        sweep = calibration_sweep_experiment(args.followup_reps, 49)
        tests = calibration_test_experiment(max(500, args.followup_reps), 99)
        print(sweep.to_string(index=False))
        print(tests.to_string(index=False))
    if args.derivative_reps > 0:
        derivative = derivative_calibration_experiment(
            args.derivative_reps, args.derivative_perm)
        make_derivative_figure(derivative)
        print(derivative.to_string(index=False))
    if args.enhanced_test_reps > 0:
        enhanced = enhanced_calibration_test_experiment(
            args.enhanced_test_reps, args.enhanced_test_perm)
        print(enhanced.to_string(index=False))
    finite.to_csv(OUT / "simulation_finite_sequence.csv", index=False)
    panel.to_csv(OUT / "simulation_panel_summary.csv", index=False)
    runtime.to_csv(OUT / "simulation_runtime.csv", index=False)
    make_figures(finite, panel)
    print(finite.to_string(index=False))
    print(panel.to_string(index=False))
    print(runtime.to_string(index=False))
    print(calibration.to_string(index=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
