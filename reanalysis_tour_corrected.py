"""Fast, auditable reanalysis after correcting ATP/WTA draw classification.

This runner intentionally omits the slow exploratory per-player HMM.  It produces
all confirmatory regression, exact-bias, distributed-lag, permutation, placebo,
and player-by-match fixed-effect results used by the revised manuscript.
"""

import pandas as pd

import tennis_momentum_advanced as tm


def main():
    pts = tm.load_points()
    audit = (pts.groupby("tour")
             .agg(matches=("match_id", "nunique"), valid_points=("match_id", "size"),
                  players=("server_name", "nunique"),
                  first_year=("year", "min"), last_year=("year", "max"))
             .reset_index())
    audit.to_csv(tm.out("corrected_sample_audit.csv"), index=False)
    print(audit.to_string(index=False), flush=True)

    ms_rows, robustness_rows, decay_frames = [], [], []
    perm_rows, placebo_rows, games_frames = [], [], []
    for tour in tm.TOURS:
        pt = pts[pts["tour"] == tour].copy()
        ms_rows.append(tm.run_ms_comparison(pt, tour))
        robustness_rows.extend(tm.fe_robustness_point(pt, tour))
        dlp, _ = tm.distributed_lag_point(pt, tour)
        decay_frames.append(dlp)

        games = tm.build_games(pt)
        games["tour"] = tour
        games_frames.append(games)
        robustness_rows.extend(tm.fe_robustness_game(games, tour))
        dlg, _ = tm.distributed_lag_game(games, tour)
        decay_frames.append(dlg)

        pres, _ = tm.within_sequence_permutation(pt, tour, B=tm.N_PERM)
        perm_rows.append(pres)
        placebo_rows.append(tm.cross_match_placebo(pt, tour))

    all_games = pd.concat(games_frames, ignore_index=True)
    fit_pt = tm.pooled_interaction_point(pts)
    fit_g = tm.pooled_interaction_game(all_games)
    inter_rows = [
        {"level": "point", "base_men": fit_pt.params["tw_prev_win"],
         "female_interaction": fit_pt.params["tw_xf"], "se": fit_pt.bse["tw_xf"],
         "p": fit_pt.pvalues["tw_xf"], "n": int(fit_pt.nobs)},
        {"level": "game", "base_men": fit_g.params["tw_prev_game_win"],
         "female_interaction": fit_g.params["tw_xf"], "se": fit_g.bse["tw_xf"],
         "p": fit_g.pvalues["tw_xf"], "n": int(fit_g.nobs)},
    ]
    pm_inter = [tm.pooled_player_match_interaction_point(pts),
                tm.pooled_player_match_interaction_game(all_games)]
    lab_res, _ = tm.label_permutation_gender(pts, B=tm.N_PERM)

    pd.DataFrame(ms_rows).to_csv(tm.out("corrected_ms_exact.csv"), index=False)
    pd.DataFrame(robustness_rows).to_csv(tm.out("corrected_fe_robustness.csv"), index=False)
    decay = pd.concat(decay_frames, ignore_index=True)
    decay.to_csv(tm.out("corrected_distributed_lag.csv"), index=False)
    pd.DataFrame(perm_rows).to_csv(tm.out("corrected_sequence_permutation.csv"), index=False)
    pd.DataFrame(placebo_rows).to_csv(tm.out("corrected_global_shuffle_placebo.csv"), index=False)
    pd.DataFrame(inter_rows).to_csv(tm.out("corrected_twoway_interaction.csv"), index=False)
    pd.DataFrame(pm_inter).to_csv(tm.out("corrected_player_match_interaction.csv"), index=False)
    pd.DataFrame([lab_res]).to_csv(tm.out("corrected_label_permutation.csv"), index=False)
    tm.fig_decay(decay)
    print("Corrected confirmatory reanalysis complete.", flush=True)


if __name__ == "__main__":
    main()

