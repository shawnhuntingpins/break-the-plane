"""
weekly_rollover.py -- build next week's Break the Plane inputs and roll data.json forward.

Run by .github/workflows/weekly-rollover.yml (Tue/Wed/Thu), or by hand:
    python3 tools/weekly_rollover.py --repo . [--week N] [--force]

Steps (all idempotent -- if the target week is already built it exits without changing anything):
  1. Work out the target week from the schedule (first week not yet finished), or take --week.
  2. Download nflverse player stats, team stats, rosters and depth charts.
  3. Check nflverse has the PREVIOUS week's finished games. If not, exit cleanly and let the next
     scheduled run retry (nothing is written).
  4. build_week.build()  -> lambda, slate, QB influence, QB depth, schedule with lines filled from ESPN
  5. make_seed.build_seed() -> data.json rolled forward (opponents, kickoffs, defense fields, byes)
  6. Copy the outputs into data/ and data.json, then run refresh_live.py once so the committed
     board already has probabilities, injuries and odds (not the zeroed seed).
The workflow then commits whatever changed.

Not rebuilt here (these were manual steps in the original pipeline): injury role bumps / next-man-up
reassignment, red-zone share, and the scoring breakdown. See README notes in the workflow summary.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import current_week as cw  # noqa: E402
import build_week  # noqa: E402
import make_seed  # noqa: E402

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {
    "wk.csv": f"{BASE}/stats_player/stats_player_week_2026.csv",
    "teamwk.csv": f"{BASE}/stats_team/stats_team_week_2026.csv",
    "roster.csv": f"{BASE}/rosters/roster_2026.csv",
    "depth.csv": f"{BASE}/depth_charts/depth_charts_2026.csv",
}


def say(msg):
    print(f"[weekly_rollover] {msg}", flush=True)
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a") as f:
            f.write(msg + "\n\n")


def download(url, dest, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "break-the-plane-weekly/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--week", type=int)
    ap.add_argument("--force", action="store_true", help="rebuild even if the week already exists")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    data = f"{repo}/data"

    sched = pd.read_csv(f"{data}/nflverse/schedules_2026.csv")
    sched = sched[sched.season == cw.SEASON]
    T = a.week or cw.schedule_week(repo)
    if T not in set(sched.week):
        say(f"Week {T} is not in the schedule; nothing to do.")
        return
    if T in cw.built_weeks(repo) and not a.force:
        say(f"Week {T} is already built; nothing to do.")
        return
    prev_candidates = [w for w in cw.built_weeks(repo) if w < T]
    if not prev_candidates:
        sys.exit(f"No earlier built week to carry priors from (needed for Week {T}). Seed the first week manually.")
    prev = max(prev_candidates)

    tmp = tempfile.mkdtemp(prefix="btp_")
    nfl, out = f"{tmp}/nfl", f"{tmp}/out"
    os.makedirs(nfl)
    for name, url in FILES.items():
        download(url, f"{nfl}/{name}")
    say(f"Downloaded nflverse data for the Week {T} build (priors carried from Week {prev}).")

    # nflverse must already contain every game from the week just played
    teams_prev = set(sched[sched.week == T - 1].home_team) | set(sched[sched.week == T - 1].away_team)
    have = set(pd.read_csv(f"{nfl}/teamwk.csv", usecols=["team", "week"]).query("week == @T - 1").team)
    if teams_prev - have:
        say(f"**Waiting on nflverse:** Week {T-1} stats not complete yet (missing {sorted(teams_prev - have)}). "
            f"No changes made; the next scheduled run will retry.")
        return

    lam, slate = build_week.build(T, data, nfl, out, prev, log=say)
    d, dropped_bye, _ = make_seed.build_seed(T, f"{repo}/data.json", data, nfl, f"{out}/slate_2026wk{T}_players_raw.parquet",
                                             repo, f"{out}/data.json")

    for f in [f"lambda_2026wk{T}.csv", f"slate_2026wk{T}_players_raw.parquet", f"qb_influence_2026wk{T}.csv",
              "qb_depth_rank_2026.csv"]:
        shutil.copy(f"{out}/{f}", f"{data}/{f}")
    shutil.copy(f"{out}/schedules_2026.csv", f"{data}/nflverse/schedules_2026.csv")
    shutil.copy(f"{out}/data.json", f"{repo}/data.json")

    env = dict(os.environ, BTP_ND=f"{data}/nflverse", BTP_OUT=data, BTP_SITE=repo)
    subprocess.run([sys.executable, f"{repo}/refresh_live.py", "--week", str(T)], cwd=repo, env=env, check=True)

    n_teams = len({p["team"] for p in json.load(open(f"{repo}/data.json"))["players"]})
    say(f"**Week {T} built.** {len(d['players'])} players on the seed, {n_teams} teams, "
        f"{len(dropped_bye)} bye-week cards dropped. Odds key present: {bool(os.environ.get('ODDSJAM_KEY'))}.")
    say("Not reassigned this week (manual steps in the original pipeline): injury role bumps / next-man-up, "
        "red-zone share, scoring breakdown. ESPN Out/IR players are still zeroed by the hourly job.")


if __name__ == "__main__":
    main()
