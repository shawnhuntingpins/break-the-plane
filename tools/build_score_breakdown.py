"""
build_score_breakdown.py -- fills p["scoreBreakdown"] for every player in data.json.

Four 0-1 components and a weighted composite (weights 35/30/20/15), reconstructed from the Week 8 values:
  opportunity  = wA*min(1, rz_share/capA) + (1-wA)*min(1, usage/capB)       (usage = target share for WR/TE, rush share for RB/QB)
                   WR .50 / .80 / .20      TE .30 / .25 / .34      RB .35 / .75 / .60      QB .10 / .05 / .25
  tdProduction = 0.6*min(1, tds_in_last4/4) + 0.4*min(1, season TDs per game/0.9)
  matchup      = (33 - def_rank)/32            (best matchup = worst defense = 1.0)
  gameContext  = min(1, lambda_team/3.6)
  composite    = 0.35*opportunity + 0.30*tdProduction + 0.20*matchup + 0.15*gameContext
Fit to the 104 players who carried a breakdown in Week 8: mean absolute error 0.0002 (tdProduction), 0.0005-0.003 (opportunity,
WR/RB/TE), 0.0009 (QB), matchup and gameContext exact to rounding.

Usage: python3 tools/build_score_breakdown.py --repo .      (safe to re-run; only rewrites scoreBreakdown)
"""
import argparse, json, os, urllib.request
import pandas as pd

PREV_SEASON = 2025
STATS_URL = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{PREV_SEASON}.csv"
MIN_GAMES = 4


def season_prod(repo, log=print):
    """Last season's TDs per game / yards per game for every RB/WR/TE/QB, cached in data/season_prod_{PREV_SEASON}.json.
    games = weeks with a carry or target; totalTds = rushing + receiving TDs (QBs: rushing TDs only).
    Matches the original seasonProd on TDs for 100% of players and on games for 97%."""
    cache = f"{repo}/data/season_prod_{PREV_SEASON}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    req = urllib.request.Request(STATS_URL, headers={"User-Agent": "break-the-plane-weekly/1.0"})
    path = f"/tmp/stats_{PREV_SEASON}.csv"
    open(path, "wb").write(urllib.request.urlopen(req, timeout=180).read())
    w = pd.read_csv(path, low_memory=False)
    w = w[(w.season_type == "REG") & w.position.isin(["QB", "RB", "WR", "TE"])]
    w = w[(w.carries.fillna(0) + w.targets.fillna(0)) > 0]
    out = {}
    for pid, g in w.groupby("player_id"):
        n = g.week.nunique()
        if n < MIN_GAMES:
            continue
        qb = g.position.iloc[0] == "QB"
        tds = g.rushing_tds.fillna(0).sum() + (0 if qb else g.receiving_tds.fillna(0).sum())
        ys = g.rushing_yards.fillna(0).sum() + (0 if qb else g.receiving_yards.fillna(0).sum())
        out[pid] = dict(games=int(n), totalTds=int(tds), tdsPerGame=round(float(tds) / n, 2),
                        ydsPerGame=round(float(ys) / n, 1), ydsLabel="rush yds/G" if qb else "scrim yds/G")
    json.dump(out, open(cache, "w"))
    log(f"Built {PREV_SEASON} season production for {len(out)} players.")
    return out

WEIGHTS = {"opportunity": 0.35, "tdProduction": 0.30, "matchup": 0.20, "gameContext": 0.15}
OPP = {"WR": ("target_share", 0.50, 0.80, 0.20), "TE": ("target_share", 0.30, 0.25, 0.34),
       "RB": ("rush_share", 0.35, 0.75, 0.60), "QB": ("rush_share", 0.10, 0.05, 0.25)}


def _c(x):
    return max(0.0, min(1.0, x))


def breakdown(p):
    pos = p.get("pos")
    sp = p.get("seasonProd") or {}
    need = [p.get("rz_share"), p.get(OPP.get(pos, ("x",))[0]), p.get("tds_in_last4"), sp.get("tdsPerGame"),
            p.get("def_rank"), p.get("lambda_team")]
    if pos not in OPP or any(v is None for v in need):
        return None
    usage, w, cap_rz, cap_u = OPP[pos]
    comp = {
        "opportunity": w * _c(p["rz_share"] / cap_rz) + (1 - w) * _c(p[usage] / cap_u),
        "tdProduction": 0.6 * _c(p["tds_in_last4"] / 4) + 0.4 * _c(sp["tdsPerGame"] / 0.9),
        "matchup": _c((33 - p["def_rank"]) / 32),
        "gameContext": _c(p["lambda_team"] / 3.6),
    }
    out = {k: round(v, 3) for k, v in comp.items()}
    out["composite"] = round(sum(WEIGHTS[k] * comp[k] for k in WEIGHTS), 3)
    out["weights"] = dict(WEIGHTS)
    return out


def patch(repo, log=print):
    path = f"{repo}/data.json"
    d = json.load(open(path))
    sp = season_prod(repo, log)
    n = 0
    for p in d["players"]:
        if not p.get("seasonProd") and p["id"] in sp:
            p["seasonProd"] = sp[p["id"]]
        b = breakdown(p)
        if b:
            p["scoreBreakdown"] = b
            n += 1
        else:
            p.pop("scoreBreakdown", None)
    json.dump(d, open(path, "w"))
    log(f"Score breakdown written for {n} of {len(d['players'])} players.")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    patch(ap.parse_args().repo)
