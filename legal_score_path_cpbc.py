"""Legal-score-path conditional Monte Carlo calibration for tennis point panels.

The sampler conditions on each ordinary game's observed length and winner, which
preserves point count, game result, set/match schedule, server schedule, and the
match-server service-point total. Candidate outcome paths must obey the standard
tennis game stopping rule. Tiebreak paths and incomplete/irregular games are held
fixed; irregular ordinary-game points are excluded from the reported statistic.

Candidate paths are weighted by a five-fold match-level cross-fitted null model
with server, returner, era, and dynamically recomputed pre-point score effects.
No lagged outcome enters that null model.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import expit, logit, logsumexp
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mca_revision_analysis as rev
import tennis_momentum_advanced as tm

OUT = ROOT / "output" / "replication" / "v20_legal_path" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260715
STANDARD_STATES = tuple(
    [f"{a}-{b}" for a in range(4) for b in range(4) if not (a == 3 and b == 3)]
    + ["D", "AI", "AO"]
)
STATE_INDEX = {s: i for i, s in enumerate(STANDARD_STATES)}
N_STATES = len(STANDARD_STATES)


class PathClass(NamedTuple):
    kind: int          # 0: explicit short path; 1: compressed deuce path
    mask: int          # explicit path mask, or six-point prefix mask
    r: int             # number of split deuce pairs before the terminal pair
    m: int             # number of (win, loss) split pairs
    winner: int        # server-won indicator for the game winner
    length: int
    multiplicity: int
    losses: tuple[int, ...]
    wins: tuple[int, ...]


def standard_score_state(server_points: int, return_points: int) -> str:
    if server_points >= 3 and return_points >= 3:
        if server_points == return_points:
            return "D"
        return "AI" if server_points > return_points else "AO"
    return f"{min(server_points, 3)}-{min(return_points, 3)}"


def is_legal_standard_game(path) -> bool:
    values = tuple(int(x) for x in path)
    if len(values) < 4 or any(x not in (0, 1) for x in values):
        return False
    sw = rw = 0
    for i, value in enumerate(values):
        sw += value
        rw += 1 - value
        terminal = max(sw, rw) >= 4 and abs(sw - rw) >= 2
        if terminal != (i == len(values) - 1):
            return False
    return True


def path_signature(path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    losses = np.zeros(N_STATES, dtype=np.int16)
    wins = np.zeros(N_STATES, dtype=np.int16)
    sw = rw = 0
    for value in path:
        state = STATE_INDEX[standard_score_state(sw, rw)]
        if int(value):
            wins[state] += 1
            sw += 1
        else:
            losses[state] += 1
            rw += 1
    return tuple(int(x) for x in losses), tuple(int(x) for x in wins)


def _mask_from_path(path) -> int:
    mask = 0
    for i, value in enumerate(path):
        mask |= int(value) << i
    return mask


def _path_from_class(cls: PathClass, rng: np.random.Generator) -> np.ndarray:
    if cls.kind == 0:
        return np.asarray([(cls.mask >> i) & 1 for i in range(cls.length)], dtype=np.int8)
    path = [(cls.mask >> i) & 1 for i in range(6)]
    orientations = np.r_[np.ones(cls.m, dtype=np.int8),
                         np.zeros(cls.r - cls.m, dtype=np.int8)]
    rng.shuffle(orientations)
    for flag in orientations:
        path.extend((1, 0) if flag else (0, 1))
    path.extend((cls.winner, cls.winner))
    return np.asarray(path, dtype=np.int8)


@lru_cache(maxsize=None)
def candidate_classes(length: int, winner: int) -> tuple[PathClass, ...]:
    """Compressed enumeration of every legal path with fixed length and winner."""
    length, winner = int(length), int(winner)
    if winner not in (0, 1):
        raise ValueError("winner must be 0 or 1")
    classes: list[PathClass] = []
    if length < 8:
        for prefix in itertools.product((0, 1), repeat=max(length - 1, 0)):
            path = prefix + (winner,)
            if is_legal_standard_game(path):
                losses, wins = path_signature(path)
                classes.append(PathClass(
                    0, _mask_from_path(path), 0, 0, winner, length, 1, losses, wins
                ))
    elif length % 2 == 0:
        r = (length - 8) // 2
        for ones in itertools.combinations(range(6), 3):
            prefix = [0] * 6
            for j in ones:
                prefix[j] = 1
            prefix_mask = _mask_from_path(prefix)
            for m in range(r + 1):
                representative = list(prefix)
                representative.extend([1, 0] * m)
                representative.extend([0, 1] * (r - m))
                representative.extend([winner, winner])
                losses, wins = path_signature(representative)
                classes.append(PathClass(
                    1, prefix_mask, r, m, winner, length, math.comb(r, m),
                    losses, wins
                ))
    if not classes:
        raise ValueError(f"No legal standard-game path for length={length}, winner={winner}")
    expected = 1 if length == 4 else (
        4 if length == 5 else 10 if length == 6 else
        20 * (2 ** ((length - 8) // 2))
    )
    if sum(c.multiplicity for c in classes) != expected:
        raise AssertionError("Compressed path count does not match the exact count")
    return tuple(classes)


def exact_path_distribution(length: int, winner: int, p_state: np.ndarray):
    """Return normalized class probabilities under state-dependent point success."""
    classes = candidate_classes(length, winner)
    p = np.clip(np.asarray(p_state, dtype=float), 1e-8, 1 - 1e-8)
    logp, logq = np.log(p), np.log1p(-p)
    logw = np.asarray([
        math.log(c.multiplicity)
        + np.dot(np.asarray(c.wins, dtype=float), logp)
        + np.dot(np.asarray(c.losses, dtype=float), logq)
        for c in classes
    ])
    probs = np.exp(logw - logsumexp(logw))
    return classes, probs


def _game_audit(full: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Classify points: 0 irregular ordinary, 1 legal ordinary, 2 tiebreak."""
    game_key = (full["match_id"].astype(str) + "|" + full["SetNo"].astype(str)
                + "|" + full["GameNo"].astype(str)).to_numpy()
    cuts = np.r_[0, np.flatnonzero(game_key[1:] != game_key[:-1]) + 1, len(full)]
    status = np.zeros(len(full), dtype=np.int8)
    rows = []
    y = full["server_won"].to_numpy(np.int8)
    server = full["PointServer"].to_numpy(np.int8)
    for h in range(len(cuts) - 1):
        left, right = int(cuts[h]), int(cuts[h + 1])
        servers = np.unique(server[left:right])
        if len(servers) > 1:
            status[left:right] = 2
            rows.append((left, right - left, 2, -1, True))
            continue
        path = y[left:right]
        legal = is_legal_standard_game(path)
        if legal:
            status[left:right] = 1
            rows.append((left, right - left, 1, int(path[-1]), True))
        else:
            rows.append((left, right - left, 0, int(path[-1]) if len(path) else -1, False))
    games = pd.DataFrame(rows, columns=["start", "length", "status", "winner", "legal"])
    return status, games


def _shrunk_map(train: pd.DataFrame, key: str, global_p: float, prior_n: float):
    stats = train.groupby(key, observed=True)["server_won"].agg(["sum", "count"])
    return (stats["sum"] + prior_n * global_p) / (stats["count"] + prior_n)


def crossfit_null_components(full: pd.DataFrame, legal_mask: np.ndarray,
                             folds: int = 5, prior_n: float = 50.0):
    """Cross-fitted row baselines plus fold-specific dynamic score components."""
    work = full.copy()
    work["score_state_null"] = rev._point_score_states(work)
    work["era_null"] = work["era"].astype(str).fillna("unknown")
    if "_crossfit_fold" in work.columns:
        fold_id = work["_crossfit_fold"].to_numpy(np.int8)
        if np.any((fold_id < 0) | (fold_id >= folds)):
            raise ValueError("_crossfit_fold must be in [0, folds)")
        if "_source_match" in work.columns:
            fold_counts = work.groupby("_source_match", observed=True)["_crossfit_fold"].nunique()
            if int(fold_counts.max()) != 1:
                raise ValueError("Copies of a source match cross fold boundaries")
    else:
        hashes = pd.util.hash_pandas_object(work["match_id"].astype(str), index=False)
        fold_id = (hashes.to_numpy(np.uint64) % folds).astype(np.int8)
    base_no_score = np.empty(len(work), dtype=np.float64)
    score_delta = np.zeros((folds, N_STATES), dtype=np.float64)
    observed_p = np.full(len(work), np.nan, dtype=np.float64)
    keys = ("server_name", "returner_name", "era_null")
    for fold in range(folds):
        train = work.loc[legal_mask & (fold_id != fold)]
        test_idx = np.flatnonzero(fold_id == fold)
        test = work.iloc[test_idx]
        global_p = float(train["server_won"].mean())
        base_logit = float(logit(np.clip(global_p, 1e-6, 1 - 1e-6)))
        linear = np.full(len(test), base_logit, dtype=np.float64)
        for key in keys:
            mapping = _shrunk_map(train, key, global_p, prior_n)
            mapped = test[key].map(mapping).fillna(global_p).to_numpy(float)
            linear += logit(np.clip(mapped, 1e-6, 1 - 1e-6)) - base_logit
        base_no_score[test_idx] = linear
        score_map = _shrunk_map(train, "score_state_null", global_p, prior_n)
        for s, j in STATE_INDEX.items():
            ps = float(score_map.get(s, global_p))
            score_delta[fold, j] = float(logit(np.clip(ps, 1e-6, 1 - 1e-6))
                                           - base_logit)
        obs_state = test["score_state_null"].map(STATE_INDEX)
        obs_delta = np.asarray([
            score_delta[fold, int(v)] if pd.notna(v) else 0.0 for v in obs_state
        ])
        observed_p[test_idx] = np.clip(expit(linear + obs_delta), 0.02, 0.98)
    return base_no_score, score_delta, fold_id, observed_p


def _class_arrays(length: int, winner: int):
    classes = candidate_classes(length, winner)
    losses = np.asarray([c.losses for c in classes], dtype=np.float64)
    wins = np.asarray([c.wins for c in classes], dtype=np.float64)
    logmult = np.log(np.asarray([c.multiplicity for c in classes], dtype=float))
    meta = {
        "kind": np.asarray([c.kind for c in classes], dtype=np.int8),
        "mask": np.asarray([c.mask for c in classes], dtype=np.int64),
        "r": np.asarray([c.r for c in classes], dtype=np.int8),
        "m": np.asarray([c.m for c in classes], dtype=np.int8),
        "winner": np.asarray([c.winner for c in classes], dtype=np.int8),
        "length": np.asarray([c.length for c in classes], dtype=np.int16),
    }
    return classes, losses, wins, logmult, meta


def build_legal_sampler(full: pd.DataFrame, games: pd.DataFrame,
                        base_no_score: np.ndarray, score_delta: np.ndarray,
                        fold_id: np.ndarray):
    random_games = games.loc[games["status"] == 1].copy()
    random_games["fold"] = fold_id[random_games["start"].to_numpy(int)]
    random_games["base"] = base_no_score[random_games["start"].to_numpy(int)]
    starts_all, lengths_all, offsets = [], [], [0]
    cum_all, kind_all, mask_all, r_all, m_all, winner_all = [], [], [], [], [], []
    pstate_all = []
    type_rows = []
    for (length, winner), group in random_games.groupby(["length", "winner"], sort=True):
        length, winner = int(length), int(winner)
        _, losses, wins, logmult, meta = _class_arrays(length, winner)
        row_index = group.index.to_numpy()
        gstarts = group["start"].to_numpy(np.int64)
        gfolds = group["fold"].to_numpy(np.int64)
        gbases = group["base"].to_numpy(float)
        pstate = np.clip(expit(gbases[:, None] + score_delta[gfolds]), 0.02, 0.98)
        logw = (np.log(pstate) @ wins.T + np.log1p(-pstate) @ losses.T
                + logmult[None, :])
        logw -= logsumexp(logw, axis=1, keepdims=True)
        probs = np.exp(logw)
        cumulative = np.cumsum(probs, axis=1)
        cumulative[:, -1] = 1.0
        c = probs.shape[1]
        starts_all.extend(gstarts.tolist())
        lengths_all.extend([length] * len(group))
        pstate_all.append(pstate.astype(np.float32))
        cum_all.append(cumulative.ravel())
        kind_all.append(np.tile(meta["kind"], len(group)))
        mask_all.append(np.tile(meta["mask"], len(group)))
        r_all.append(np.tile(meta["r"], len(group)))
        m_all.append(np.tile(meta["m"], len(group)))
        winner_all.append(np.tile(meta["winner"], len(group)))
        for _ in range(len(group)):
            offsets.append(offsets[-1] + c)
        type_rows.append({
            "length": length, "winner": winner, "games": len(group),
            "classes": c, "exact_paths": int(sum(x.multiplicity for x in candidate_classes(length, winner)))
        })
    arrays = {
        "game_starts": np.asarray(starts_all, dtype=np.int64),
        "game_lengths": np.asarray(lengths_all, dtype=np.int16),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "cumprob": np.concatenate(cum_all).astype(np.float64),
        "kind": np.concatenate(kind_all).astype(np.int8),
        "mask": np.concatenate(mask_all).astype(np.int64),
        "r": np.concatenate(r_all).astype(np.int8),
        "m": np.concatenate(m_all).astype(np.int8),
        "winner": np.concatenate(winner_all).astype(np.int8),
        "pstate": np.concatenate(pstate_all, axis=0).astype(np.float32),
    }
    return arrays, pd.DataFrame(type_rows)


def _analysis_spec(full: pd.DataFrame, status: np.ndarray, include_tiebreak: bool,
                   min_player_points: int = 200):
    match_codes = pd.factorize(full["match_id"])[0]
    first = np.r_[True, match_codes[1:] != match_codes[:-1]]
    eligible = status == 1
    if include_tiebreak:
        eligible |= status == 2
    counts = full.loc[eligible & ~first, "server_name"].value_counts()
    keep_players = set(counts[counts >= min_player_points].index)
    mask = eligible & ~first & full["server_name"].isin(keep_players).to_numpy()
    indices = np.flatnonzero(mask).astype(np.int64)
    groups = (full.loc[mask, "match_id"].astype(str) + "|"
              + full.loc[mask, "server_name"].astype(str)).to_numpy()
    group_codes = pd.factorize(groups)[0].astype(np.int64)
    group_counts = np.bincount(group_codes).astype(np.float64)
    control = full.loc[mask, "set_late"].to_numpy(float)
    control_within = rev._demean(control, group_codes).astype(np.float64)
    return {
        "mask": mask, "indices": indices, "groups": group_codes,
        "counts": group_counts, "control": control_within,
        "n": int(mask.sum()), "players": len(keep_players)
    }


@njit(cache=True)
def _coefficient(y, server_slot, indices, groups, control_within, group_counts):
    ngroups = len(group_counts)
    n = len(indices)
    sum_y = np.zeros(ngroups, dtype=np.float64)
    sum_x = np.zeros(ngroups, dtype=np.float64)
    x_values = np.empty(n, dtype=np.float64)
    for q in range(n):
        idx = indices[q]
        prev_winner = server_slot[idx - 1] if y[idx - 1] == 1 else 3 - server_slot[idx - 1]
        x = 1.0 if prev_winner == server_slot[idx] else 0.0
        x_values[q] = x
        g = groups[q]
        sum_x[g] += x
        sum_y[g] += y[idx]
    xx = xz = zz = xy = zy = 0.0
    for q in range(n):
        g = groups[q]
        xw = x_values[q] - sum_x[g] / group_counts[g]
        yw = y[indices[q]] - sum_y[g] / group_counts[g]
        zw = control_within[q]
        xx += xw * xw
        xz += xw * zw
        zz += zw * zw
        xy += xw * yw
        zy += zw * yw
    det = xx * zz - xz * xz
    return (zz * xy - xz * zy) / det if abs(det) > 1e-14 else xy / xx


@njit(cache=True)
def _legal_null_kernel(y0, server_slot, game_starts, game_lengths, offsets,
                       cumprob, kind, masks, rs, ms, winners,
                       idx_primary, grp_primary, ctl_primary, cnt_primary,
                       idx_all, grp_all, ctl_all, cnt_all, seeds):
    out_primary = np.empty(len(seeds), dtype=np.float64)
    out_all = np.empty(len(seeds), dtype=np.float64)
    flags = np.zeros(32, dtype=np.int8)
    for b in range(len(seeds)):
        np.random.seed(seeds[b])
        y = y0.copy()
        for g in range(len(game_starts)):
            u = np.random.random()
            left, right = offsets[g], offsets[g + 1]
            c = left
            while c < right - 1 and u > cumprob[c]:
                c += 1
            start = game_starts[g]
            n = game_lengths[g]
            if kind[c] == 0:
                mask = masks[c]
                for t in range(n):
                    y[start + t] = (mask >> t) & 1
            else:
                mask = masks[c]
                for t in range(6):
                    y[start + t] = (mask >> t) & 1
                r, m = rs[c], ms[c]
                for j in range(r):
                    flags[j] = 1 if j < m else 0
                for j in range(r - 1, 0, -1):
                    k = np.random.randint(0, j + 1)
                    tmp = flags[j]
                    flags[j] = flags[k]
                    flags[k] = tmp
                pos = start + 6
                for j in range(r):
                    if flags[j] == 1:
                        y[pos], y[pos + 1] = 1, 0
                    else:
                        y[pos], y[pos + 1] = 0, 1
                    pos += 2
                y[pos] = winners[c]
                y[pos + 1] = winners[c]
        out_primary[b] = _coefficient(
            y, server_slot, idx_primary, grp_primary, ctl_primary, cnt_primary
        )
        out_all[b] = _coefficient(
            y, server_slot, idx_all, grp_all, ctl_all, cnt_all
        )
    return out_primary, out_all


def _observed_coefficient(y: np.ndarray, server_slot: np.ndarray, spec: dict) -> float:
    return float(_coefficient(
        y.astype(np.int8), server_slot.astype(np.int8), spec["indices"],
        spec["groups"], spec["control"], spec["counts"]
    ))


def _summarize(observed: float, null: np.ndarray, tour: str, mode: str,
               spec: dict, diagnostics: dict):
    p_right, p_two = rev._permutation_p(null, observed)
    effect = observed - float(null.mean())
    sd = float(null.std(ddof=1))
    return {
        "tour": tour.upper(), "mode": mode, "B": len(null),
        "observed_pmfe": observed, "null_mean": float(null.mean()),
        "null_sd": sd, "calibrated_effect": effect,
        "reference_low": effect - 1.96 * sd,
        "reference_high": effect + 1.96 * sd,
        "p_positive": p_right, "p_two_sided": p_two,
        "n": spec["n"], "players": spec["players"], **diagnostics
    }


def run_tour(points: pd.DataFrame, tour: str, B: int = 999, folds: int = 5,
             prior_n: float = 50.0, min_player_points: int = 200,
             seed: int | None = None):
    full = points.loc[points["tour"] == tour].sort_values(
        ["match_id", "pt_idx"]
    ).reset_index(drop=True).copy()
    status, games = _game_audit(full)
    legal_mask = status == 1
    base, score_delta, fold_id, p_obs = crossfit_null_components(
        full, legal_mask, folds=folds, prior_n=prior_n
    )
    sampler, type_diag = build_legal_sampler(full, games, base, score_delta, fold_id)
    spec_primary = _analysis_spec(
        full, status, include_tiebreak=False, min_player_points=min_player_points
    )
    spec_all = _analysis_spec(
        full, status, include_tiebreak=True, min_player_points=min_player_points
    )
    y = full["server_won"].to_numpy(np.int8)
    slot = full["PointServer"].to_numpy(np.int8)
    observed_primary = _observed_coefficient(y, slot, spec_primary)
    observed_all = _observed_coefficient(y, slot, spec_all)
    if seed is None:
        seed = SEED + (10000 if tour == "wta" else 0)
    seeds = np.random.default_rng(seed).integers(
        1, 2**31 - 1, size=B, dtype=np.int64
    )
    null_primary, null_all = _legal_null_kernel(
        y, slot, sampler["game_starts"], sampler["game_lengths"],
        sampler["offsets"], sampler["cumprob"], sampler["kind"], sampler["mask"],
        sampler["r"], sampler["m"], sampler["winner"],
        spec_primary["indices"], spec_primary["groups"], spec_primary["control"],
        spec_primary["counts"], spec_all["indices"], spec_all["groups"],
        spec_all["control"], spec_all["counts"], seeds
    )
    valid_games = int((games["status"] == 1).sum())
    tiebreak_games = int((games["status"] == 2).sum())
    invalid_games = int((games["status"] == 0).sum())
    diagnostics = {
        "randomized_legal_games": valid_games,
        "fixed_tiebreak_games": tiebreak_games,
        "excluded_irregular_games": invalid_games,
        "randomized_point_share": float((status == 1).mean()),
        "crossfit_folds": folds, "prior_n": prior_n,
        "null_model_brier": float(np.mean(
            (y[legal_mask] - p_obs[legal_mask]) ** 2
        )),
    }
    rows = [
        _summarize(observed_primary, null_primary, tour,
                   "ordinary_legal_games_tiebreak_excluded", spec_primary, diagnostics),
        _summarize(observed_all, null_all, tour,
                   "ordinary_legal_games_tiebreak_paths_fixed", spec_all, diagnostics),
    ]
    nulls = pd.DataFrame({
        "draw": np.arange(1, B + 1), "tour": tour.upper(),
        "ordinary_only": null_primary, "tiebreak_fixed": null_all
    })
    audit = games["status"].value_counts().rename_axis("status").reset_index(name="games")
    audit["tour"] = tour.upper()
    return pd.DataFrame(rows), nulls, type_diag.assign(tour=tour.upper()), audit


def recompute_holm(results: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(
        ROOT / "output" / "replication" / "v18_package" / "results"
        / "revision_v18" / "preferred_eight_test_holm.csv"
    )
    local = results.loc[
        results["mode"] == "ordinary_legal_games_tiebreak_excluded",
        ["tour", "calibrated_effect", "null_sd", "p_two_sided"]
    ].copy()
    local["hypothesis"] = local["tour"].str.lower() + "_point_legal_score_local"
    local = local.rename(columns={"p_two_sided": "raw_p"})
    # The revised family replaces both frozen-score local tests; it must contain
    # six unchanged global tests plus the two legal-path local tests (eight total).
    globals_old = old.loc[~old["hypothesis"].str.contains("score_restricted")].copy()
    combined = pd.concat([
        local[["hypothesis", "raw_p"]],
        globals_old[["hypothesis", "raw_p"]]
    ], ignore_index=True)
    reject, adjusted, _, _ = multipletests(combined["raw_p"], method="holm")
    combined["p_holm"] = adjusted
    combined["reject_holm_0_05"] = reject.astype(int)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--prior-n", type=float, default=50.0)
    parser.add_argument("--min-player-points", type=int, default=200)
    args = parser.parse_args()
    points = tm.load_points()
    result_parts, null_parts, type_parts, audit_parts = [], [], [], []
    for tour in ("atp", "wta"):
        result, nulls, types, audit = run_tour(
            points, tour, B=args.permutations, folds=args.folds,
            prior_n=args.prior_n, min_player_points=args.min_player_points
        )
        result_parts.append(result)
        null_parts.append(nulls)
        type_parts.append(types)
        audit_parts.append(audit)
        print(result.to_string(index=False), flush=True)
    results = pd.concat(result_parts, ignore_index=True)
    nulls = pd.concat(null_parts, ignore_index=True)
    types = pd.concat(type_parts, ignore_index=True)
    audits = pd.concat(audit_parts, ignore_index=True)
    holm = recompute_holm(results)
    results.to_csv(OUT / "legal_path_score_results.csv", index=False)
    nulls.to_csv(OUT / "legal_path_score_nulls.csv", index=False)
    types.to_csv(OUT / "legal_path_candidate_diagnostics.csv", index=False)
    audits.to_csv(OUT / "legal_path_game_audit.csv", index=False)
    holm.to_csv(OUT / "legal_path_eight_test_holm.csv", index=False)
    print(holm.to_string(index=False), flush=True)
    print(f"Outputs written to {OUT}", flush=True)


if __name__ == "__main__":
    main()