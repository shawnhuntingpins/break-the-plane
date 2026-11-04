"""
build_injuries.py -- sidelined list + next-man-up role bumps for Break the Plane.

1. Pulls RotoWire's NFLInjuries feed (env INJURY_FEED_URL, or NEWS_FEED_URL with feed=NFLNews swapped for
   feed=NFLInjuries) and keeps RB/WR/TE who are Out / IR / PUP / NFI / Reserve. Questionable and Doubtful are NOT
   zeroed (the hourly ESPN live pass handles game-day Outs).
2. Writes data/injury_sidelined_matched_2026wk{W}.csv  (team,gsis_id,name,pos_abb,injury,status).
3. Writes data/injury_role_bumps_2026wk{W}.csv: for each team/position, the next healthy player in the depth
   order picks up  BUMP_FRAC x (sum of the sidelined players' expected share)  of target share (WR/TE) or rush share (RB).
   BUMP_FRAC=0.125 was fitted to the original Week 8 file (median 0.125, correlation 0.92 on 24 WR/TE cases).

Usage: python3 tools/build_injuries.py --repo . [--week N] [--xml feed.xml]
"""
import argparse, json, os, re, sys, urllib.request
import numpy as np, pandas as pd

BUMP_FRAC = 0.125
OUT_STATUS = ("OUT", "IR", "IR-R", "PUP", "NFI", "RESERVE", "SUSPEND")
SKILL = {"RB", "WR", "TE"}  # QBs excluded, as in the original Week 8 file
TEAM_FIX = {"ARZ": "ARI", "LA": "LA", "LAR": "LA", "JAX": "JAC", "WSH": "WAS"}


def fetch(url=None):
    url = url or os.environ.get("INJURY_FEED_URL") or os.environ.get("NEWS_FEED_URL", "").replace("NFLNews", "NFLInjuries")
    if not url:
        raise SystemExit("Set INJURY_FEED_URL (or NEWS_FEED_URL).")
    req = urllib.request.Request(url, headers={"User-Agent": "break-the-plane-weekly/1.0"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


def parse(xml):
    rows = []
    for m in re.finditer(r'<Player Id="(\d+)">(.*?)</Player>', xml, re.S):
        b = m.group(2)
        g = lambda tag: (re.search(rf"<{tag}>(.*?)</{tag}>", b, re.S) or [None, ""])[1].strip()
        inj = re.search(r'<Injury Status="([^"]*)" Type="([^"]*)"', b)
        team = re.search(r'<Team Id="\d+" Code="([^"]*)"', b)
        rows.append(dict(rw_id=m.group(1), name=f"{g('FirstName')} {g('LastName')}".strip(), pos=g("Position"),
                         team=team.group(1) if team else "", status=inj.group(1) if inj else "",
                         injury=inj.group(2) if inj else ""))
    return pd.DataFrame(rows)


def sidelined(feed, players, slate=None):
    """feed rows -> gsis ids via the slate's rotowire_id column (all depth players), falling back to data.json rotowireUrl"""
    idmap = {}
    for p in players:
        m = re.search(r"(\d+)/?$", p.get("rotowireUrl") or "")
        if m:
            idmap[m.group(1)] = p["id"]
    if slate is not None:
        for r, g in zip(slate.rotowire_id, slate.gsis_id):
            if pd.notna(r):
                idmap[str(int(float(r)))] = g
    f = feed[feed.pos.isin(SKILL) & feed.status.str.upper().str.startswith(OUT_STATUS) & (feed.team != "")].copy()
    f["team"] = f.team.replace(TEAM_FIX)
    f["gsis_id"] = f.rw_id.map(idmap)
    f["status_out"] = np.where(f.status.str.upper() == "OUT", "Out", "Reserve/IR")
    return f


def bumps(out, slate):
    """out: sidelined rows with gsis_id; slate: slate_players_raw with pos_rank / shares"""
    s = slate.copy()
    s["is_out"] = s.gsis_id.isin(set(out.gsis_id.dropna()))
    rows = []
    for (team, pos), grp in s[s.is_out & s.position.isin(["WR", "TE", "RB"])].groupby(["team", "position"]):
        share_col, prior_col, l4_col, bump_col = (("L4_rush_share", "mean_rush_share", "L4_rush_share", "rush_share_bump")
                                                  if pos == "RB" else
                                                  ("L4_target_share", "mean_target_share", "L4_target_share", "target_share_bump"))
        exp = np.fmax(grp[prior_col].fillna(0), grp[l4_col].fillna(0)).sum()
        pool = s[(s.team == team) & (s.position == pos) & ~s.is_out].sort_values("pos_rank")
        floor = min(grp.pos_rank.min(), 5)
        cand = pool[pool.pos_rank >= floor]
        if cand.empty:
            continue
        ben = cand.iloc[0]
        top = grp.sort_values("pos_rank").iloc[0]
        rows.append(dict(team=team, position=pos, beneficiary_gsis_id=ben.gsis_id, beneficiary_name=ben.full_name,
                         beneficiary_rank=ben.pos_rank, out_name=top.full_name, out_rank=top.pos_rank,
                         n_contributors=len(grp), **{bump_col: round(BUMP_FRAC * exp, 6)}))
    df = pd.DataFrame(rows)
    for c in ("target_share_bump", "rush_share_bump"):
        if c not in df:
            df[c] = 0.0
    return df.fillna({"target_share_bump": 0.0, "rush_share_bump": 0.0})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--week", type=int)
    ap.add_argument("--xml")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import current_week as cw
    W = a.week or cw.refresh_week(repo)
    xml = open(a.xml).read() if a.xml else fetch()
    feed = parse(xml)
    d = json.load(open(f"{repo}/data.json"))
    slate = pd.read_parquet(f"{repo}/data/slate_2026wk{W}_players_raw.parquet")
    sl = sidelined(feed, d["players"], slate)
    sl = sl[sl.gsis_id.notna()]
    sl_out = pd.DataFrame(dict(team=sl.team, gsis_id=sl.gsis_id, name=sl.name, pos_abb=sl.pos, injury=sl.injury, status=sl.status_out))
    sl_out.to_csv(f"{repo}/data/injury_sidelined_matched_2026wk{W}.csv", index=False)
    b = bumps(sl, slate)
    b.to_csv(f"{repo}/data/injury_role_bumps_2026wk{W}.csv", index=False)
    print(f"Week {W}: {len(sl_out)} sidelined skill players, {len(b)} role bumps.")
