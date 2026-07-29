"""Null-size simulations for legal-path versus frozen-score calibration.

The data generator has score-dependent point probabilities but no dependence on
the previous point winner. It optionally adds an unobserved match-by-server form
shock to probe nuisance-model misspecification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import legal_score_path_cpbc as lp
import mca_revision_analysis as rev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "output" / "replication" / "legal_path" / "results"


def simulate_panel(seed: int, matches: int, games_per_match: int,
                   players: int, shock_sd: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = np.asarray([f"P{j:02d}" for j in range(players)])
    ability = rng.normal(0.0, 0.35, players)
    # Fixed dynamic-score effects; no lagged outcome enters the generator.
    score_delta = np.linspace(-0.28, 0.28, lp.N_STATES)
    rows = []
    for match in range(matches):
        p1, p2 = rng.choice(players, size=2, replace=False)
        shock = rng.normal(0.0, shock_sd, 2)
        point_index = 0
        for game in range(1, games_per_match + 1):
            slot = 1 if game % 2 else 2
            server = p1 if slot == 1 else p2
            returner = p2 if slot == 1 else p1
            a = b = 0
            game_point = 0
            while True:
                state = lp.standard_score_state(a, b)
                true_base = (0.35 + ability[server] - 0.45 * ability[returner]
                             + shock[slot - 1])
                eta = true_base + score_delta[lp.STATE_INDEX[state]]
                won = int(rng.random() < 1.0 / (1.0 + np.exp(-eta)))
                point_index += 1
                game_point += 1
                rows.append({
                    "match_id": f"M{match:04d}", "pt_idx": point_index,
                    "SetNo": 1 + (game - 1) // 12, "GameNo": game,
                    "PointNumber": game_point, "PointServer": slot,
                    "PointWinner": slot if won else 3 - slot,
                    "server_won": won, "server_name": names[server],
                    "returner_name": names[returner],
                    "set_late": int(game % 12 >= 9 or game % 12 == 0),
                    "era": "sim", "tour": "atp",
                    "_true_base": true_base,
                })
                a += won
                b += 1 - won
                if (a >= 4 or b >= 4) and abs(a - b) >= 2:
                    break
    return pd.DataFrame(rows)


def oracle_legal_path_pvalue(points: pd.DataFrame, B: int, seed: int) -> float:
    """Legal-path test using the data generator's true dynamic-score model."""
    full = points.sort_values(["match_id", "pt_idx"]).reset_index(drop=True)
    status, games = lp._game_audit(full)
    score_delta = np.linspace(-0.28, 0.28, lp.N_STATES)[None, :]
    fold_id = np.zeros(len(full), dtype=np.int8)
    sampler, _ = lp.build_legal_sampler(
        full, games, full["_true_base"].to_numpy(float), score_delta, fold_id
    )
    spec = lp._analysis_spec(full, status, include_tiebreak=False,
                             min_player_points=0)
    y = full["server_won"].to_numpy(np.int8)
    slot = full["PointServer"].to_numpy(np.int8)
    observed = lp._observed_coefficient(y, slot, spec)
    seeds = np.random.default_rng(seed).integers(
        1, 2**31 - 1, size=B, dtype=np.int64
    )
    null, _ = lp._legal_null_kernel(
        y, slot, sampler["game_starts"], sampler["game_lengths"],
        sampler["offsets"], sampler["cumprob"], sampler["kind"],
        sampler["mask"], sampler["r"], sampler["m"], sampler["winner"],
        spec["indices"], spec["groups"], spec["control"], spec["counts"],
        spec["indices"], spec["groups"], spec["control"], spec["counts"], seeds
    )
    return float(rev._permutation_p(null, observed)[1])


def frozen_score_pvalue(points: pd.DataFrame, B: int, seed: int) -> float:
    full = points.sort_values(["match_id", "pt_idx"]).reset_index(drop=True)
    status, _ = lp._game_audit(full)
    spec = lp._analysis_spec(full, status, include_tiebreak=False,
                             min_player_points=0)
    score = rev._point_score_states(full)
    stratum = (full["match_id"].astype(str) + "|"
               + full["PointServer"].astype(str) + "|" + pd.Series(score))
    groups = (full.loc[spec["mask"], "match_id"].astype(str) + "|"
              + full.loc[spec["mask"], "server_name"].astype(str)).to_numpy()
    y = full["server_won"].to_numpy(np.int8)
    slot = full["PointServer"].to_numpy(np.int8)
    observed = lp._observed_coefficient(y, slot, spec)
    null = rev._fast_conditional_null(
        y, slot, stratum.to_numpy(), spec["mask"], groups,
        full.loc[spec["mask"], "set_late"].to_numpy(float), B, seed
    )
    return float(rev._permutation_p(null, observed)[1])


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--matches", type=int, default=80)
    parser.add_argument("--games-per-match", type=int, default=14)
    parser.add_argument("--players", type=int, default=24)
    parser.add_argument("--shock-sd", type=float, nargs="+", default=[0.0, 0.45])
    args = parser.parse_args()
    records = []
    for scenario, shock_sd in enumerate(args.shock_sd):
        for replication in range(args.replications):
            seed = 202607150 + scenario * 100000 + replication
            panel = simulate_panel(
                seed, args.matches, args.games_per_match, args.players, shock_sd
            )
            result, _, _, _ = lp.run_tour(
                panel, "atp", B=args.permutations, folds=5,
                prior_n=50.0, min_player_points=0
            )
            legal_p = float(result.loc[
                result["mode"] == "ordinary_legal_games_tiebreak_excluded",
                "p_two_sided"
            ].iloc[0])
            oracle_p = oracle_legal_path_pvalue(
                panel, args.permutations, seed + 800000
            )
            frozen_p = frozen_score_pvalue(panel, args.permutations, seed + 900000)
            records.extend([
                {"scenario": f"shock_sd_{shock_sd:g}", "replication": replication,
                 "method": "legal_path_crossfit", "p_two_sided": legal_p,
                 "points": len(panel)},
                {"scenario": f"shock_sd_{shock_sd:g}", "replication": replication,
                 "method": "legal_path_oracle", "p_two_sided": oracle_p,
                 "points": len(panel)},
                {"scenario": f"shock_sd_{shock_sd:g}", "replication": replication,
                 "method": "frozen_score_label", "p_two_sided": frozen_p,
                 "points": len(panel)},
            ])
            if (replication + 1) % 10 == 0:
                print(f"scenario={shock_sd:g} replication={replication + 1}", flush=True)
    draws = pd.DataFrame(records)
    summary_rows = []
    for (scenario, method), group in draws.groupby(["scenario", "method"]):
        n = len(group)
        k = int((group["p_two_sided"] <= 0.05).sum())
        low, high = wilson(k, n)
        summary_rows.append({
            "scenario": scenario, "method": method,
            "replications": n, "B": args.permutations,
            "rejections_0_05": k, "empirical_size": k / n,
            "wilson_low": low, "wilson_high": high,
            "mean_p": float(group["p_two_sided"].mean()),
            "median_points": float(group["points"].median()),
        })
    summary = pd.DataFrame(summary_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUT / "legal_path_null_simulation_draws.csv", index=False)
    summary.to_csv(OUT / "legal_path_null_simulation_summary.csv", index=False)
    config = vars(args)
    (OUT / "legal_path_null_simulation_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
