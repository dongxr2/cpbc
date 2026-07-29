"""Fit latent-state validation models on the corrected tour classification."""

import pandas as pd

import tennis_momentum_advanced as tm


def main():
    tm.HMM_RESTARTS = 2
    tm.HMM_MAX_ITER = 30
    points = tm.load_points()
    player_frames = []
    summaries = []
    for tour in tm.TOURS:
        tour_points = points[points["tour"] == tour].copy()
        point_sequences = tm.point_sequences_by_player(tour_points)
        point_sequences = dict(sorted(
            point_sequences.items(), key=lambda item: sum(map(len, item[1])), reverse=True
        )[:100])
        point_players, point_summary = tm.hmm_by_player(
            point_sequences, tm.HMM_MIN_POINTS, tour, "point"
        )
        if len(point_players):
            point_players["tour"] = tour
            point_players["level"] = "point"
            player_frames.append(point_players)
        if point_summary:
            summaries.append(point_summary)

        games = tm.build_games(tour_points)
        if games is not None:
            game_sequences = tm.game_sequences_by_player(games)
            game_sequences = dict(sorted(
                game_sequences.items(), key=lambda item: sum(map(len, item[1])), reverse=True
            )[:80])
            game_players, game_summary = tm.hmm_by_player(
                game_sequences, tm.HMM_MIN_GAMES, tour, "game"
            )
            if len(game_players):
                game_players["tour"] = tour
                game_players["level"] = "game"
                player_frames.append(game_players)
            if game_summary:
                summaries.append(game_summary)

    pd.DataFrame(summaries).to_csv(
        tm.out("corrected_hmm_summary.csv"), index=False, encoding="utf-8-sig"
    )
    if player_frames:
        pd.concat(player_frames, ignore_index=True).to_csv(
            tm.out("corrected_hmm_players.csv"), index=False, encoding="utf-8-sig"
        )
    tm.fig_hmm(summaries)


if __name__ == "__main__":
    main()
