"""Full match-level bootstrap for LP-CMC.

Every outer replicate resamples complete matches within tour, assigns duplicate
matches unique cluster identifiers, reapplies the player-frequency rule, assigns
all copies of one source match to the same freshly randomized cross-fitting fold,
refits the no-lag path model, and recomputes the legal-path calibration mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

import legal_score_path_cpbc as lp
import tennis_momentum_advanced as tm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "output" / "replication" / "legal_path" / "results"
BASE_SEED = 2026071501
_BASE = None
_STARTS = None
_ENDS = None
_TOUR = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def balanced_source_folds(n_sources: int, folds: int,
                          rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(n_sources)
    values = np.empty(n_sources, dtype=np.int8)
    values[order] = np.arange(n_sources, dtype=np.int64) % folds
    return values


def initialize_worker(tour: str):
    global _BASE, _STARTS, _ENDS, _TOUR
    points = tm.load_points()
    base = points.loc[points["tour"] == tour].sort_values(
        ["match_id", "pt_idx"]
    ).reset_index(drop=True).copy()
    del points
    keys = base["match_id"].astype(str).to_numpy()
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(base)]
    base["_source_match"] = np.repeat(
        np.arange(len(starts), dtype=np.int32), ends - starts
    )
    _BASE = base
    _STARTS = starts.astype(np.int64)
    _ENDS = ends.astype(np.int64)
    _TOUR = tour


def resample_complete_matches(seed: int, folds: int):
    if _BASE is None:
        raise RuntimeError("worker not initialized")
    rng = np.random.default_rng(seed)
    n_matches = len(_STARTS)
    selected = rng.integers(0, n_matches, size=n_matches, dtype=np.int64)
    lengths = _ENDS[selected] - _STARTS[selected]
    offsets = np.r_[0, np.cumsum(lengths)]
    take = np.empty(int(offsets[-1]), dtype=np.int64)
    for j, source in enumerate(selected):
        take[offsets[j]:offsets[j + 1]] = np.arange(
            _STARTS[source], _ENDS[source], dtype=np.int64
        )
    sample = _BASE.iloc[take].copy()
    sample["match_id"] = np.repeat(np.arange(n_matches, dtype=np.int64), lengths)
    fold_map = balanced_source_folds(n_matches, folds, rng)
    sample["_crossfit_fold"] = fold_map[
        sample["_source_match"].to_numpy(np.int64)
    ]
    unique = int(np.unique(selected).size)
    return sample, {
        "sampled_matches": int(n_matches),
        "sampled_points": int(len(sample)),
        "unique_source_matches": unique,
        "duplicate_draws": int(n_matches - unique),
    }


def worker_task(task):
    replication, outer_seed, inner_B, folds, prior_n, min_player_points = task
    started = time.perf_counter()
    sample, boot_diag = resample_complete_matches(outer_seed, folds)
    result, _, _, _ = lp.run_tour(
        sample, _TOUR, B=inner_B, folds=folds, prior_n=prior_n,
        min_player_points=min_player_points, seed=outer_seed + 500_000_003
    )
    elapsed = float(time.perf_counter() - started)
    rows = []
    for row in result.to_dict("records"):
        rows.append({
            "tour": str(row["tour"]),
            "replication": int(replication),
            "outer_seed": int(outer_seed),
            "mode": str(row["mode"]),
            "inner_B": int(inner_B),
            "observed_pmfe": float(row["observed_pmfe"]),
            "null_mean": float(row["null_mean"]),
            "null_sd": float(row["null_sd"]),
            "calibrated_effect": float(row["calibrated_effect"]),
            "inner_mc_se": float(row["null_sd"] / np.sqrt(inner_B)),
            "analysis_n": int(row["n"]),
            "players": int(row["players"]),
            "null_model_brier": float(row["null_model_brier"]),
            **boot_diag,
            "elapsed_seconds": elapsed,
            "status": "ok",
        })
    return rows


def append_rows(path: Path, rows):
    pd.DataFrame(rows).to_csv(
        path, mode="a", index=False, header=not path.exists()
    )


def point_estimates() -> pd.DataFrame:
    path = RESULTS_ROOT / "legal_path_score_results.csv"
    return pd.read_csv(path)[["tour", "mode", "calibrated_effect"]].rename(
        columns={"calibrated_effect": "point_estimate"}
    )


def summarize(draws: pd.DataFrame, requested_R: int) -> pd.DataFrame:
    points = point_estimates()
    rows = []
    for (tour, mode), group in draws.groupby(["tour", "mode"], sort=True):
        theta = float(points.loc[
            (points["tour"] == tour) & (points["mode"] == mode),
            "point_estimate"
        ].iloc[0])
        values = group["calibrated_effect"].to_numpy(float)
        q025, q975 = np.quantile(values, [0.025, 0.975])
        raw_var = float(np.var(values, ddof=1))
        inner_var = float(np.mean(group["inner_mc_se"].to_numpy(float) ** 2))
        corrected_var = max(raw_var - inner_var, 0.0)
        corrected_se = float(np.sqrt(corrected_var))
        rows.append({
            "tour": tour,
            "mode": mode,
            "requested_R": requested_R,
            "successful_R": len(group),
            "inner_B": int(group["inner_B"].iloc[0]),
            "point_estimate_4999": theta,
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_bias": float(np.mean(values) - theta),
            "bootstrap_se_raw": float(np.sqrt(raw_var)),
            "inner_mc_se_rms": float(np.sqrt(inner_var)),
            "inner_mc_variance_fraction": (
                float(inner_var / raw_var) if raw_var > 0 else np.nan
            ),
            "bootstrap_se_mc_corrected": corrected_se,
            "percentile_low": float(q025),
            "percentile_high": float(q975),
            "basic_low": float(2 * theta - q975),
            "basic_high": float(2 * theta - q025),
            "normal_mc_corrected_low": float(theta - 1.96 * corrected_se),
            "normal_mc_corrected_high": float(theta + 1.96 * corrected_se),
            "mean_analysis_n": float(group["analysis_n"].mean()),
            "mean_players": float(group["players"].mean()),
            "mean_null_model_brier": float(group["null_model_brier"].mean()),
            "median_elapsed_seconds": float(group["elapsed_seconds"].median()),
        })
    return pd.DataFrame(rows)


def run_tour_bootstrap(tour: str, args, checkpoint: Path):
    completed = set()
    if checkpoint.exists() and args.resume:
        old = pd.read_csv(checkpoint)
        counts = old.groupby("replication")["mode"].nunique()
        completed = set(counts[counts == 2].index.astype(int))
    tasks = []
    tour_offset = 10_000_000 if tour == "wta" else 0
    for replication in range(1, args.replications + 1):
        if replication not in completed:
            seed = args.seed + tour_offset + replication * 1009
            tasks.append((replication, seed, args.inner_draws, args.folds,
                          args.prior_n, args.min_player_points))
    if not tasks:
        print(f"{tour.upper()}: already complete", flush=True)
        return
    print(f"{tour.upper()}: starting {len(tasks)} replications", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker, initargs=(tour,)
    ) as pool:
        futures = {pool.submit(worker_task, task): task[0] for task in tasks}
        finished = len(completed)
        for future in as_completed(futures):
            rows = future.result()
            append_rows(checkpoint, rows)
            finished += 1
            if finished % args.progress_every == 0 or finished == args.replications:
                print(
                    f"{tour.upper()}: {finished}/{args.replications}; "
                    f"worker_seconds={rows[0]['elapsed_seconds']:.1f}",
                    flush=True
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=999)
    parser.add_argument("--inner-draws", type=int, default=199)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--prior-n", type=float, default=50.0)
    parser.add_argument("--min-player-points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--tours", nargs="+", choices=("atp", "wta"),
                        default=["atp", "wta"])
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.replications < 2 or args.inner_draws < 2:
        raise ValueError("replications and inner-draws must be at least 2")
    run_name = (
        f"R{args.replications}_B{args.inner_draws}_F{args.folds}_"
        f"P{args.prior_n:g}_M{args.min_player_points}_S{args.seed}"
    )
    out = RESULTS_ROOT / "full_match_bootstrap" / run_name
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    for tour in args.tours:
        run_tour_bootstrap(tour, args, out / f"{tour}_bootstrap_draws.csv")
    draws = pd.concat(
        [pd.read_csv(out / f"{tour}_bootstrap_draws.csv") for tour in args.tours],
        ignore_index=True
    )
    summary = summarize(draws, args.replications)
    if np.any(summary["successful_R"] != args.replications):
        raise RuntimeError("not every tour/mode has the requested successful R")
    summary.to_csv(out / "full_match_bootstrap_summary.csv", index=False)
    config = {
        **vars(args),
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "bootstrap_unit": "complete_match_within_tour",
        "duplicate_cluster_handling": "unique bootstrap match_id",
        "crossfit_duplicate_handling": (
            "all copies of one source match share one freshly assigned fold"
        ),
        "eligibility_rule": "reapplied within every outer sample",
        "nuisance_refit": True,
        "legal_path_mean_recomputed": True,
        "inner_mc_variance_correction": (
            "raw variance minus mean(null_sd^2/inner_B), truncated at zero"
        ),
    }
    (out / "full_match_bootstrap_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    manifest = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "sha256_manifest.json":
            manifest[path.name] = {
                "sha256": sha256(path), "bytes": path.stat().st_size
            }
    (out / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Outputs written to {out}", flush=True)


if __name__ == "__main__":
    main()
