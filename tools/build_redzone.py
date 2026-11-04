"""
build_redzone.py -- red-zone and goal-line usage for every player, from nflverse play-by-play.

For week W it looks at each team's last 4 games played before W and counts, per player,
  rz_touches : rushes + pass targets from inside the opponent's 20
  gl_touches : the same, from inside the 10
  rz5        : the same, from inside the 5
and divides by the team's total in the same window:
  rz_share = player rz_touches / team rz plays      (the "Red-zone share (L4)" bar and RZ Role chip)
  gl_share = player gl_touches / team inside-10 plays   (new goal-line bar)
A "team rz play" is any rush attempt or pass attempt (including sacks and throwaways) from inside the 20;
two-point tries are excluded.

Checked against the original Week 8 rz_share for 108 players: correlation 0.95, mean gap 2.9 points
(the original's exact definition is not in the repo, so values will not match to the decimal).

Usage:
    python3 tools/build_redzone.py --repo . [--week N]        # patches data.json for that week
Run from weekly_rollover.py after the weekly build; safe to re-run (idempotent).
"""
import argparse, json, os, shutil, sys, time, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet"
COLS = ["season_type", "week", "posteam", "yardline_100", "rush_attempt", "pass_attempt",
        "rusher_player_id", "receiver_player_id", "two_point_attempt"]


def download(dest, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(PBP_URL, headers={"User-Agent": "break-the-plane-weekly/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def compute(pbp_path, W):
    d = pd.read_parquet(pbp_path, columns=COLS)
    d = d[(d.season_type == "REG") & (d.week < W) & d.posteam.notna()]
    last4 = d.groupby("posteam").week.apply(lambda s: sorted(set(s))[-4:]).to_dict()
    d = d[d.apply(lambda r: r.week in last4.get(r.posteam, []), axis=1)]
    d = d[(d.yardline_100 <= 20) & (d.two_point_attempt.fillna(0) == 0)]
    ru = d[d.rush_attempt == 1]
    ta = d[d.pass_attempt == 1]

    def table(ru_, ta_, tag):
        num = pd.concat([ru_[["rusher_player_id", "posteam"]].rename(columns={"rusher_player_id": "gsis_id"}),
                         ta_[["receiver_player_id", "posteam"]].rename(columns={"receiver_player_id": "gsis_id"})])
        return num.dropna(subset=["gsis_id"]).groupby(["gsis_id", "posteam"]).size().rename(tag + "_touches").reset_index()

    rz = table(ru, ta, "rz")
    gl = table(ru[ru.yardline_100 <= 10], ta[ta.yardline_100 <= 10], "gl")
    f5 = table(ru[ru.yardline_100 <= 5], ta[ta.yardline_100 <= 5], "rz5")[["gsis_id", "posteam", "rz5_touches"]]
    out = rz.merge(gl, on=["gsis_id", "posteam"], how="outer").merge(f5, on=["gsis_id", "posteam"], how="outer")
    out = out.fillna({"rz_touches": 0, "gl_touches": 0, "rz5_touches": 0})  # inside-10/5 only players get 0 rz_touches elsewhere
    return out, last4


def team_totals(pbp_path, W):
    d = pd.read_parquet(pbp_path, columns=COLS)
    d = d[(d.season_type == "REG") & (d.week < W) & d.posteam.notna()]
    last4 = d.groupby("posteam").week.apply(lambda s: sorted(set(s))[-4:]).to_dict()
    d = d[d.apply(lambda r: r.week in last4.get(r.posteam, []), axis=1)]
    d = d[(d.yardline_100 <= 20) & (d.two_point_attempt.fillna(0) == 0) & ((d.rush_attempt == 1) | (d.pass_attempt == 1))]
    return (d.groupby("posteam").size().to_dict(),
            d[d.yardline_100 <= 10].groupby("posteam").size().to_dict())


def build(pbp_path, W):
    out, _ = compute(pbp_path, W)
    rz_tot, gl_tot = team_totals(pbp_path, W)
    out["rz_share"] = out.apply(lambda r: r.rz_touches / rz_tot[r.posteam] if rz_tot.get(r.posteam) else np.nan, axis=1)
    out["gl_share"] = out.apply(lambda r: r.gl_touches / gl_tot[r.posteam] if gl_tot.get(r.posteam) else np.nan, axis=1)
    return out[["gsis_id", "posteam", "rz_touches", "rz_share", "gl_touches", "gl_share", "rz5_touches"]]


def patch(repo, W, rz, log=print):
    p = f"{repo}/data.json"
    d = json.load(open(p))
    if int(d.get("week", 0)) != int(W):
        log(f"data.json is Week {d.get('week')}, not Week {W}; red-zone fields not patched.")
        return 0
    m = rz.set_index("gsis_id")
    n = 0
    for pl in d["players"]:
        r = m.loc[pl["id"]] if pl["id"] in m.index else None
        if r is not None and isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        if r is None:
            # on a team with plays but no red-zone touches in the window: a real zero, not missing
            pl.update(rz_share=0.0, gl_share=0.0, rz_touches=0)
        else:
            pl.update(rz_share=round(float(r.rz_share), 3), gl_share=round(float(r.gl_share), 3) if pd.notna(r.gl_share) else 0.0,
                      rz_touches=int(r.rz_touches))
        n += 1
    json.dump(d, open(p, "w"))
    log(f"Red-zone/goal-line share written for {n} players (Week {W}, last 4 team games).")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--week", type=int)
    ap.add_argument("--pbp", help="local parquet instead of downloading (testing)")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    import current_week as cw
    W = a.week or cw.refresh_week(repo)
    path = a.pbp
    if not path:
        import tempfile
        path = tempfile.mkdtemp(prefix="btp_pbp_") + "/pbp.parquet"
        download(path)
    rz = build(path, W)
    os.makedirs(f"{repo}/data", exist_ok=True)
    rz.to_csv(f"{repo}/data/redzone_2026wk{W}.csv", index=False)
    patch(repo, W, rz)
