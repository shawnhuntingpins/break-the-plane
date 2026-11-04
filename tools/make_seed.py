"""
make_seed.py -- roll data.json forward to a new week.

refresh_live.py seeds from the existing data.json and only rewrites probabilities,
totals and market odds. It never updates week, opponent, home/away, kickoff, defense
fields or TD streaks, and it never drops bye-week players. This script does that
rollover once, at the start of each week, BEFORE the hourly refresh takes over.

Rules verified by self-test against the Week 8 data.json:
  def_tds_allowed = opponent's mean offensive TDs allowed over its last 4 games
  def_rank        = rank of def_tds_allowed, descending, ties share the lowest rank
  def_tier        = by rank in buckets of 8: Worst / Tough / Fair / Elite
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

TIERS = ["Worst", "Tough", "Fair", "Elite"]


def build_seed(W, prev_json, data_dir, nfl_dir, slate_parquet, repo_dir, out_json):
    sys.path.insert(0, repo_dir)
    import refresh_live as rl  # reuse the exact kickoff formatter

    d = json.load(open(prev_json))
    sched = pd.read_csv(f"{data_dir}/nflverse/schedules_2026.csv")
    sched = sched[(sched.season == 2026) & (sched.week == W)]
    teams = set(sched.home_team) | set(sched.away_team)
    opp, home, roof, div = {}, {}, {}, {}
    for _, g in sched.iterrows():
        opp[g.home_team], opp[g.away_team] = g.away_team, g.home_team
        home[g.home_team], home[g.away_team] = True, False
        roof[g.home_team] = roof[g.away_team] = g.roof if pd.notna(g.roof) else "TBD"
        div[g.home_team] = div[g.away_team] = bool(g.div_game)
    kick, kick_ts = rl.build_kickoff_maps(sched)

    tm = pd.read_csv(f"{nfl_dir}/teamwk.csv", low_memory=False)
    tm = tm[tm.week < W]
    tm["tds"] = tm.passing_tds + tm.rushing_tds
    last4 = tm.groupby("team").week.apply(lambda s: sorted(set(s))[-4:]).to_dict()
    allowed = tm.groupby(["opponent_team", "week"]).tds.sum().reset_index()
    dta = {t: allowed[(allowed.opponent_team == t) & (allowed.week.isin(last4[t]))].tds.mean() for t in last4}
    s = pd.Series(dta)
    rank = s.rank(ascending=False, method="min")

    wk = pd.read_csv(f"{nfl_dir}/wk.csv", low_memory=False)
    wk = wk[wk.week < W]
    wk["any_td"] = (wk.rushing_tds + wk.receiving_tds) > 0
    slate = pd.read_parquet(slate_parquet).drop_duplicates("gsis_id").set_index("gsis_id")

    out, dropped_bye, dropped_gone = [], [], []
    for p in d["players"]:
        pid = p["id"]
        if pid not in slate.index:
            dropped_gone.append(p["name"]); continue
        team = slate.loc[pid, "team"]
        if team not in teams:
            dropped_bye.append(p["name"]); continue
        p = dict(p)
        p["team"] = team
        o = opp[team]
        p.update(opp=o, isHome=home[team], roof=roof[team], divGame=div[team],
                 kickoff=kick.get(team), kickoffTs=kick_ts.get(team),
                 def_tds_allowed=round(float(s[o]), 2), def_rank=int(rank[o]),
                 def_tier=TIERS[min((int(rank[o]) - 1) // 8, 3)])
        # TD streak fields: team's last 4 games window
        tw = last4.get(team, [])
        g = wk[(wk.player_id == pid) & (wk.team == team)].set_index("week").any_td
        flags = [bool(g.get(w, False)) for w in tw]
        p["tds_in_last4"] = int(sum(flags))
        gs = 0
        for f in reversed(flags):
            if f:
                break
            gs += 1
        p["games_since_td"] = min(gs, 3)  # original caps at 3 (verified vs Week 8)
        # fields that are only valid for the week they were built; cleared so nothing stale shows
        p.update(actual_any_td=None, gameStatus="scheduled", liveOut=False, rz_share=None, scoreBreakdown=None,
                 roleBump=None, has_market_odds=False, market_price=None, market_book=None,
                 market_consensus=None, market_n_books=None, plus_ev_edge=None, books=[])
        out.append(p)
    # Teams that were on bye last week have no cards to roll forward. Add light cards for their skill players
    # (top 10 RB/WR/TE by last-4 volume + starting QB); refresh_live fills in every probability and line.
    have = {p["team"] for p in out}
    qd_path = os.path.join(os.path.dirname(slate_parquet), "qb_depth_rank_2026.csv")
    starters = set(pd.read_csv(qd_path).query("pos_rank == 1").gsis_id) if os.path.exists(qd_path) else set()
    added = []
    for team in sorted(teams - have):
        sl = slate[(slate.team == team) & slate.position.isin(["RB", "WR", "TE"])].copy()
        sl["vol"] = sl.final_L4_targets + sl.final_L4_carries
        pick = list(sl.sort_values("vol", ascending=False).head(10).index) + \
               [i for i in slate.index[(slate.team == team) & (slate.position == "QB")] if i in starters]
        for pid in pick:
            r = slate.loc[pid]
            slug = "-".join(str(r.full_name).lower().replace(".", "").replace("'", "").split())
            rid = int(float(r.rotowire_id)) if pd.notna(r.rotowire_id) else None
            o = opp[team]
            out.append(dict(id=pid, name=r.full_name, pos=r.position, team=team, opp=o, isHome=home[team], total=None,
                rank=None, p_final=0.0, p_topdown=0.0, p_bottomup=0.0, lambda_team=None,
                def_rank=int(rank[o]), def_tier=TIERS[min((int(rank[o]) - 1) // 8, 3)], def_tds_allowed=round(float(s[o]), 2),
                games_since_td=None, tds_in_last4=None, rz_share=None, target_share=None, rush_share=None, p_option_c=None,
                align=None, alignSource=None, alignBucket=None, alignMatchup=None,
                rotowireUrl=(f"https://www.rotowire.com/football/player/{slug}-{rid}" if rid else None),
                has_track_record=False, confidence=r.confidence, roof=roof[team], divGame=div[team], roleBump=None,
                headshot=r.headshot_url if pd.notna(r.headshot_url) else None,
                kickoff=kick.get(team), kickoffTs=kick_ts.get(team), actual_any_td=None, gameStatus="scheduled",
                has_market_odds=False, market_price=None, market_book=None, market_consensus=None, market_n_books=None,
                plus_ev_edge=None, books=[], seasonProd=None, scoreBreakdown=None,
                newsHeadline=None, newsDate=None, newsExcerpt=None, addedByLiveOdds=False, addedByRollover=True))
            tw = last4.get(team, [])
            g = wk[(wk.player_id == pid) & (wk.team == team)].set_index("week").any_td
            flags = [bool(g.get(w, False)) for w in tw]
            out[-1]["tds_in_last4"] = int(sum(flags))
            gs = 0
            for f in reversed(flags):
                if f: break
                gs += 1
            out[-1]["games_since_td"] = min(gs, 3)
            added.append(r.full_name)
    print(f"added {len(added)} light cards for bye-returning teams {sorted(teams - have)}")
    d = dict(d, week=W, players=out, updated_at=pd.Timestamp.utcnow().isoformat() + "Z")
    json.dump(d, open(out_json, "w"))
    return d, dropped_bye, dropped_gone


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--prev-json", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nfl", required=True)
    ap.add_argument("--slate", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, bye, gone = build_seed(a.week, a.prev_json, a.data, a.nfl, a.slate, a.repo, a.out)
    print(f"seed week {d['week']}: {len(d['players'])} players | dropped (bye team): {len(bye)} | dropped (not in slate): {len(gone)}")
