"""
build_week.py -- rebuild the per-week Break the Plane model inputs from nflverse.

Produces, for a given week W (using only games played in weeks < W):
  lambda_2026wk{W}.csv            team trailing TDs, rush share, implied totals
  slate_2026wk{W}_players_raw.parquet   per-player last-4-team-games volume + priors
  qb_influence_2026wk{W}.csv      starting QB + (carried-over) fitted QB influence
  qb_depth_rank_2026.csv          top-3 QBs per team from the latest depth chart

Reverse-engineered from the existing Week 2-8 files (the original weekly scripts
are not in the repo). Rules verified by backtest against Week 8:
  * L4_* = sum over the TEAM's last 4 played games / 4 (a missed game counts as 0)
  * final_* = L4 when the player has history on his current team, else the prior
    (mean_*) carried from the previous week's slate
  * trailing_*_tds = mean of the team's last 4 games' passing / rushing TDs
"""
import argparse, json, os, re, sys, unicodedata
import numpy as np, pandas as pd

SEASON = 2026


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace(".", "").replace("'", "").strip().lower()
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def fill_lines_from_espn(sched_all, W, log=print):
    """nflverse's schedule snapshot has no spread/total until lines are loaded by hand. Fill any blank
    Week W line from ESPN's public scoreboard (DraftKings). Live OpticOdds lines still override at runtime."""
    import urllib.request
    fix = {"LAR": "LA", "WSH": "WAS"}
    m = (sched_all.season == SEASON) & (sched_all.week == W)
    need = m & (sched_all.spread_line.isna() | sched_all.total_line.isna())
    if not need.any():
        return sched_all
    url = (f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={W}&dates={SEASON}")
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"}), timeout=20))
    n = 0
    for ev in d.get("events", []):
        c = ev["competitions"][0]; o = (c.get("odds") or [None])[0]
        if not o or o.get("overUnder") is None or o.get("spread") is None:
            continue
        ab = {t["homeAway"]: fix.get(t["team"]["abbreviation"], t["team"]["abbreviation"]) for t in c["competitors"]}
        idx = sched_all.index[m & (sched_all.home_team == ab["home"]) & (sched_all.away_team == ab["away"])]
        if len(idx) == 1:
            # ESPN spread is from the HOME side (negative = home favored); nflverse spread_line is the opposite sign
            sched_all.loc[idx, "spread_line"] = -float(o["spread"])
            sched_all.loc[idx, "total_line"] = float(o["overUnder"])
            n += 1
    log(f"filled {n}/{int(need.sum())} blank Week {W} game lines from ESPN (DraftKings)")
    return sched_all


def build(W, data_dir, nfl_dir, out_dir, prev_week, depth_asof=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    const = json.load(open(f"{data_dir}/model_constants_2026.json"))
    sched_all = pd.read_csv(f"{data_dir}/nflverse/schedules_2026.csv")
    sched_all = fill_lines_from_espn(sched_all, W, log)
    sched_all.to_csv(f"{out_dir}/schedules_2026.csv", index=False)
    sched = sched_all[(sched_all.season == SEASON) & (sched_all.week == W)].copy()
    if sched.empty:
        sys.exit(f"No Week {W} games in schedules_2026.csv")
    wk = pd.read_csv(f"{nfl_dir}/wk.csv", low_memory=False)
    tm = pd.read_csv(f"{nfl_dir}/teamwk.csv", low_memory=False)
    wk, tm = wk[wk.week < W], tm[tm.week < W]
    teams_playing = sorted(set(sched.home_team) | set(sched.away_team))
    last4 = tm.groupby("team").week.apply(lambda s: sorted(set(s))[-4:]).to_dict()

    # ---------------- lambda ----------------
    rows = []
    for _, g in sched.iterrows():
        for team, is_home in ((g.home_team, True), (g.away_team, False)):
            tot, spr = g.total_line, g.spread_line
            imp = tot / 2 + (spr / 2 if is_home else -spr / 2)
            t4 = tm[(tm.team == team) & (tm.week.isin(last4.get(team, [])))]
            rush = t4.rushing_tds.mean() if len(t4) else np.nan
            pas = t4.passing_tds.mean() if len(t4) else np.nan
            off = rush + pas
            rs = rush / off if off and off > 0 else const["league_rush_share"]
            itd = imp / const["pts_to_td_ratio"]
            lam = max(0.1, const["intercept"] + const["w_implied"] * itd + const["w_trailing"] * off)
            rows.append(dict(game_id=g.game_id, team=team, season=SEASON, week=W, implied_total=imp,
                             implied_tds=itd, trailing_off_tds=off, trailing_rush_tds=rush,
                             trailing_pass_tds=pas, lambda_team=lam, rush_share=rs,
                             lambda_rush=lam * rs, lambda_rec=lam * (1 - rs)))
    lam = pd.DataFrame(rows)
    lam.to_csv(f"{out_dir}/lambda_2026wk{W}.csv", index=False)

    # ---------------- slate ----------------
    prev = pd.read_parquet(f"{data_dir}/slate_2026wk{prev_week}_players_raw.parquet")
    ros = pd.read_csv(f"{nfl_dir}/roster.csv", low_memory=False)
    if "week" in ros.columns:
        ros = ros.sort_values("week").drop_duplicates("gsis_id", keep="last")
    ros = ros[ros.position.isin(["QB", "RB", "WR", "TE"]) & ros.gsis_id.notna() & (ros.status.isin(["ACT", "RES", "INA"]))].copy()
    # keep anyone from the previous slate who is still active somewhere
    ros = ros[ros.team.isin(teams_playing) | True]
    pl = wk[wk.position.isin(["QB", "RB", "WR", "TE", "FB"])]  # nflverse tags fullbacks "FB"
    tt = wk.groupby(["team", "week"])[["carries", "targets"]].sum().reset_index()
    pri_cols = ["mean_targets", "mean_carries", "mean_target_share", "mean_rush_share"]
    prev_pri = prev.drop_duplicates("gsis_id").set_index("gsis_id")[pri_cols + ["prior_rank", "pos_rank"]]
    fallback = prev[prev.confidence == "rookie"].groupby("position")[pri_cols].median()

    out = []
    for _, r in ros.iterrows():
        T, pid = r.team, r.gsis_id
        weeks4 = last4.get(T, [])
        g_all = pl[(pl.player_id == pid) & (pl.team == T)]
        g_win = g_all[g_all.week.isin(weeks4)]
        # verified vs Week 8: "has history" == any carry/target in the team's last 4 games
        has_hist = (g_win.carries.sum() + g_win.targets.sum()) > 0
        row = {c: r.get(c) for c in ["season", "team", "position", "depth_chart_position", "jersey_number", "status",
               "full_name", "first_name", "last_name", "birth_date", "height", "weight", "college", "gsis_id", "espn_id",
               "sportradar_id", "yahoo_id", "rotowire_id", "pff_id", "pfr_id", "fantasy_data_id", "sleeper_id",
               "years_exp", "headshot_url", "ngs_position", "game_type"]}
        row["week"] = W
        # espn_id must be a plain-integer STRING: refresh_live matches it against ESPN's ids as strings
        row["espn_id"] = str(int(float(row["espn_id"]))) if pd.notna(row.get("espn_id")) else None
        if pid in prev_pri.index:
            pri = prev_pri.loc[pid]
        elif r.position in fallback.index:
            pri = fallback.loc[r.position]
        else:
            pri = pd.Series({c: 0.0 for c in pri_cols})
        for c in pri_cols:
            row[c] = float(pri[c]) if pd.notna(pri.get(c)) else 0.0
        if has_hist:
            g = g_win
            T4 = tt[(tt.team == T) & (tt.week.isin(weeks4))]
            car, tgt = g.carries.sum() / 4, g.targets.sum() / 4
            rsh = g.carries.sum() / max(T4.carries.sum(), 1)
            gw = g.set_index("week").target_share.reindex(weeks4).fillna(0)
            tsh = gw.mean() if len(weeks4) else 0.0
            row.update(L4_targets=tgt, L4_target_share=tsh, L4_carries=car, L4_rush_share=rsh, L4_games=4.0,
                       confidence="full_history", final_L4_targets=tgt, final_L4_target_share=tsh,
                       final_L4_carries=car, final_rush_share=rsh)
        else:
            row.update(L4_targets=0.0, L4_target_share=0.0, L4_carries=0.0, L4_rush_share=0.0, L4_games=0.0,
                       confidence="rookie" if (len(g_all) == 0 and r.get("years_exp") in (0, "0", 0.0)) else "new_team",
                       final_L4_targets=row["mean_targets"], final_L4_target_share=row["mean_target_share"],
                       final_L4_carries=row["mean_carries"], final_rush_share=row["mean_rush_share"])
        row["pos_abb"] = r.position
        out.append(row)
    slate = pd.DataFrame(out)
    slate.to_parquet(f"{out_dir}/slate_2026wk{W}_players_raw.parquet", index=False)

    # ---------------- QB depth + influence ----------------
    dep = pd.read_csv(f"{nfl_dir}/depth.csv", low_memory=False)
    dep = dep[dep.pos_abb == "QB"]
    if depth_asof:
        dep = dep[dep.dt <= depth_asof]
    dep = dep[dep.dt == dep.dt.max()].sort_values(["team", "pos_rank"])
    qd = dep.drop_duplicates(["team", "gsis_id"]).groupby("team").head(3)[["team", "gsis_id", "player_name", "pos_rank"]]
    qd.to_csv(f"{out_dir}/qb_depth_rank_2026.csv", index=False)
    prev_q = pd.read_csv(f"{data_dir}/qb_influence_2026wk{prev_week}.csv")
    qmap = {norm(n): v for n, v in zip(prev_q.starting_qb_name, prev_q.qb_influence_on_rec_td)}
    starters = qd[qd.pos_rank == 1].set_index("team").player_name.to_dict()
    qrows, missing = [], []
    for team in sorted(set(prev_q.team) | set(starters)):
        name = starters.get(team)
        if name is None:
            old = prev_q[prev_q.team == team]
            name = old.starting_qb_name.iloc[0] if len(old) else ""
        val = qmap.get(norm(name))
        if val is None:
            missing.append((team, name)); val = 0.0
        qrows.append(dict(team=team, starting_qb_name=name, qb_influence_on_rec_td=val))
    pd.DataFrame(qrows).to_csv(f"{out_dir}/qb_influence_2026wk{W}.csv", index=False)
    log(f"week {W}: lambda {len(lam)} teams | slate {len(slate)} players "
        f"({(slate.confidence=='full_history').sum()} with history) | QB influence carried over; "
        f"new/unknown starters (set to 0): {missing}")
    return lam, slate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--prev-week", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nfl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-asof")
    a = ap.parse_args()
    build(a.week, a.data, a.nfl, a.out, a.prev_week, a.depth_asof)
