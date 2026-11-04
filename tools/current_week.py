"""
current_week.py -- which week should the hourly refresh build?

Prints one integer. Rule:
  schedule week = the first week whose last game (kickoff + 6h buffer) has not finished yet
  refresh week  = the highest week <= schedule week whose weekly inputs already exist in data/

The second half is what prevents the failure we hit on Week 9: the hourly job never jumps to
a new week until the weekly rollover has actually produced that week's files, so the board
keeps showing last week's results (instead of crashing) until the rollover runs.
"""
import argparse, glob, os, re, sys
import pandas as pd

SEASON = 2026
FINISHED_BUFFER = pd.Timedelta(hours=6)


def week_ends(repo):
    s = pd.read_csv(f"{repo}/data/nflverse/schedules_2026.csv")
    s = s[s.season == SEASON]
    if "game_type" in s.columns:
        s = s[s.game_type == "REG"]
    ko = pd.to_datetime(s.gameday.astype(str) + " " + s.gametime.fillna("13:00").astype(str), errors="coerce")
    ko = ko.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return ko.groupby(s.week).max() + FINISHED_BUFFER


def schedule_week(repo, now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    ends = week_ends(repo).dropna()
    pending = ends[ends > now]
    return int(pending.index.min()) if len(pending) else int(ends.index.max())


def built_weeks(repo):
    """Weeks that have every required weekly input."""
    out = []
    for f in glob.glob(f"{repo}/data/lambda_2026wk*.csv"):
        w = int(re.search(r"wk(\d+)\.csv$", f).group(1))
        if all(os.path.exists(f"{repo}/data/{n.format(w=w)}") for n in
               ["slate_2026wk{w}_players_raw.parquet", "qb_influence_2026wk{w}.csv"]):
            out.append(w)
    return sorted(out)


def refresh_week(repo, now=None):
    sw = schedule_week(repo, now)
    ok = [w for w in built_weeks(repo) if w <= sw]
    if not ok:
        sys.exit(f"No built weeks at or before schedule week {sw}; run the weekly rollover first.")
    return max(ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--now", help="ISO UTC timestamp, for testing")
    a = ap.parse_args()
    now = pd.Timestamp(a.now, tz="UTC") if a.now else None
    print(refresh_week(a.repo, now))
