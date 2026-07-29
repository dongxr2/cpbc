"""Deterministic and Monte Carlo checks for the legal-score-path sampler."""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

import legal_score_path_cpbc as lp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "output" / "replication" / "legal_path" / "results"


def brute_paths(length: int, winner: int):
    return [
        np.asarray(bits, dtype=np.int8)
        for bits in itertools.product((0, 1), repeat=length)
        if bits[-1] == winner and lp.is_legal_standard_game(bits)
    ]


def path_probability(path: np.ndarray, p_state: np.ndarray) -> float:
    a = b = 0
    out = 1.0
    for y in path:
        j = lp.STATE_INDEX[lp.standard_score_state(a, b)]
        out *= p_state[j] if y else 1.0 - p_state[j]
        a += int(y)
        b += int(1 - y)
    return float(out)


def main():
    rng = np.random.default_rng(20260715)
    count_checks = []
    probability_checks = []
    for length in (4, 5, 6, 8, 10):
        for winner in (0, 1):
            brute = brute_paths(length, winner)
            classes = lp.candidate_classes(length, winner)
            exact_count = int(sum(c.multiplicity for c in classes))
            assert exact_count == len(brute)
            generated = []
            for cls in classes:
                reps = cls.multiplicity if length <= 10 else 1
                for _ in range(reps):
                    path = lp._path_from_class(cls, rng)
                    assert len(path) == length
                    assert int(path[-1]) == winner
                    assert lp.is_legal_standard_game(path)
                    generated.append(tuple(path.tolist()))
            assert set(generated) <= {tuple(x.tolist()) for x in brute}
            count_checks.append({
                "length": length, "winner": winner,
                "classes": len(classes), "exact_paths": exact_count,
            })

            logits = np.linspace(-0.55, 0.55, lp.N_STATES)
            p_state = 1.0 / (1.0 + np.exp(-logits))
            classes2, probs = lp.exact_path_distribution(length, winner, p_state)
            class_total = 0.0
            for cls, prob in zip(classes2, probs):
                representative = lp._path_from_class(cls, np.random.default_rng(7))
                class_total += cls.multiplicity * path_probability(representative, p_state)
            brute_total = sum(path_probability(x, p_state) for x in brute)
            assert np.isclose(class_total, brute_total, rtol=1e-12, atol=1e-14)
            assert np.isclose(probs.sum(), 1.0)
            probability_checks.append({
                "length": length, "winner": winner,
                "unnormalized_mass": brute_total,
                "normalized_sum": float(probs.sum()),
            })

    # Long-game compression: 12 points has 80 paths represented by 60 classes.
    long_classes = lp.candidate_classes(12, 1)
    assert len(long_classes) == 60
    assert sum(c.multiplicity for c in long_classes) == 80

    # Weighted sampling check on a nontrivial 10-point game.
    p_state = 1.0 / (1.0 + np.exp(-np.linspace(-0.8, 0.8, lp.N_STATES)))
    classes, target = lp.exact_path_distribution(10, 1, p_state)
    draws = rng.choice(len(classes), size=100000, p=target)
    freq = np.bincount(draws, minlength=len(classes)) / len(draws)
    max_abs_error = float(np.max(np.abs(freq - target)))
    assert max_abs_error < 0.01

    report = {
        "status": "PASS",
        "count_checks": count_checks,
        "probability_checks": probability_checks,
        "long_game_classes": len(long_classes),
        "long_game_exact_paths": int(sum(c.multiplicity for c in long_classes)),
        "weighted_sampling_draws": len(draws),
        "weighted_sampling_max_abs_error": max_abs_error,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legal_path_test_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
