#!/usr/bin/env python3
"""
ca_crs_harness.py  --  CA-CRS+ evaluation harness
==================================================
Turns per-frame CROWD COUNTS + per-frame MOTION (S, C) into the CA-CRS+ risk
timeline and runs the two headline reviewer tests.

Subcommands
-----------
  score       (1) single-zone per-frame risk timeline:
                  D,S,C -> score, level, dominant cause, gate action, CRR
              (3) conflict-window extraction from the per-frame C series
  ripple      (2) inter-zone RIPPLE-EFFECT counterfactual (Reviewer 2, ask #1):
                  BASELINE (REDIRECT always) vs CA-CRS+ (REDIRECT -> HOLD when
                  the downstream zone score > ripple_thr)
  motion-csv      dump a per-frame motion CSV from a frame folder (wraps the
                  verified frame_motion.process_frames_motion; needs cv2)

Design notes
------------
* Scoring is the REAL engine imported from ca_crs_core.py -- nothing numerical
  is re-implemented, so harness output == paper engine output.
* Counter-agnostic: feed DM-Count (recommended, calibrated) or APGCC counts.
  The count enters risk ONLY through the density term D, exactly as in the core.
* NO scene number is hard-coded. Density normalisation (area OR capacity),
  per-camera calibration (a,b) and the zone layout are all explicit inputs, so
  nothing is fabricated. `python ca_crs_harness.py` (no args) runs a self-test.

Inputs you must supply to run on REAL data (see chat 'what I need from you')
----------------------------------------------------------------------------
  * counts CSV   : columns frame,pred_count  (you already have s2l1_dmcount_qnrf.csv)
  * motion CSV   : columns frame_index,S,C   (generate with the motion-csv subcommand)
  * density map  : EITHER --capacity-max N   OR   --area-m2 A [--rho-crush 7]
  * calibration  : --cal-a --cal-b           (DM-Count per-camera affine; e.g. 0.571 / 1.080)
  * ripple       : two zones -> --up-* and --dn-* counts+motion CSVs
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

from ca_crs_core import (
    CACRSConfig,
    score,
    classify,
    attribute,
    gate_command,
    project_post_intervention,
)


# ---------------------------------------------------------------------------
# CSV loading with tolerant column detection (adapt to whatever header exists)
# ---------------------------------------------------------------------------
def _pick(cols, candidates, override=None, what=""):
    if override:
        if override not in cols:
            raise KeyError(f"column '{override}' not in {cols}")
        return override
    low = {c.lower(): c for c in cols}
    for cand in candidates:            # exact (case-insensitive) match first
        if cand in low:
            return low[cand]
    for cand in candidates:            # then loose contains-match
        for c in cols:
            if cand in c.lower():
                return c
    raise KeyError(f"could not find a {what} column in {cols}; pass an explicit override")


def load_counts(path, frame_col=None, count_col=None):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []
    fc = _pick(cols, ["frame", "frame_index", "frame_id", "img", "image", "idx"], frame_col, "frame")
    cc = _pick(cols, ["pred_count", "count", "dm_count", "apgcc_count", "prediction", "pred", "gt_count", "n"], count_col, "count")
    out = {}
    for row in rows:
        out[int(float(row[fc]))] = float(row[cc])
    return out, fc, cc


def load_motion(path, frame_col=None, s_col=None, c_col=None):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []
    fc = _pick(cols, ["frame_index", "frame", "frame_id", "idx"], frame_col, "frame")
    sc = _pick(cols, ["s", "speed"], s_col, "S")
    cc = _pick(cols, ["c", "conflict"], c_col, "C")
    out = {}
    for row in rows:
        out[int(float(row[fc]))] = (float(row[sc]), float(row[cc]))
    return out, fc, sc, cc


# ---------------------------------------------------------------------------
# count -> density  and  per-camera calibration
# ---------------------------------------------------------------------------
def calibrate(count, a=1.0, b=0.0):
    """Per-camera linear calibration: count_cal = max(0, a*count + b)."""
    return max(0.0, a * float(count) + b)


def count_to_density(count, area_m2=None, rho_crush=7.0, capacity_max=None):
    """
    Normalise a person COUNT to D in [0,1]. Provide EXACTLY one of:
      capacity_max : D = count / capacity_max               (persons at D=1)
      area_m2      : D = (count / area_m2) / rho_crush       (rho_crush persons/m^2)
    rho_crush defaults to 7.0 (matches ca_crs_core crush band, D>=6/7).
    """
    if (capacity_max is None) == (area_m2 is None):
        raise ValueError("Provide EXACTLY one of capacity_max or area_m2.")
    if capacity_max is not None:
        d = float(count) / float(capacity_max)
    else:
        d = (float(count) / float(area_m2)) / float(rho_crush)
    return float(np.clip(d, 0.0, 1.0))


def crs_reduction_rate(pre, post):
    return 0.0 if pre <= 1e-12 else (pre - post) / pre


# ---------------------------------------------------------------------------
# (1) single-zone per-frame scoring
# ---------------------------------------------------------------------------
def score_series(frames, counts, S_series, C_series, cfg, *,
                 area_m2=None, rho_crush=7.0, capacity_max=None,
                 cal_a=1.0, cal_b=0.0, downstream_scores=None):
    rows = []
    for i, fr in enumerate(frames):
        cnt = calibrate(counts[i], cal_a, cal_b)
        D = count_to_density(cnt, area_m2, rho_crush, capacity_max)
        S = float(np.clip(S_series[i], 0.0, 1.0))
        C = float(np.clip(C_series[i], 0.0, 1.0))
        s = score(D, S, C, cfg)
        level = classify(s, cfg).value
        cause, ratios = attribute(D, S, C, cfg)
        dn = None if downstream_scores is None else downstream_scores[i]
        g = gate_command(cause, dn, cfg)
        action = g["action"]
        post = project_post_intervention(D, S, C, action, cfg)
        rows.append({
            "frame": fr, "count_cal": round(cnt, 3), "D": round(D, 4),
            "S": round(S, 4), "C": round(C, 4), "score": round(s, 4),
            "level": level, "cause": cause.value,
            "ratio_density": round(ratios["DENSITY"], 3),
            "ratio_speed": round(ratios["SPEED"], 3),
            "ratio_conflict": round(ratios["CONFLICT"], 3),
            "gate": action, "ripple_downgraded": g["ripple_downgraded"],
            "post_score": round(post, 4),
            "risk_reduction": round(crs_reduction_rate(s, post), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# (2) inter-zone ripple-effect counterfactual  (Reviewer 2, ask #1)
# ---------------------------------------------------------------------------
def ripple_counterfactual(frames, up, dn, cfg):
    """
    up, dn : dict frame -> (D,S,C) for the upstream and downstream zones.
    Compares BASELINE (REDIRECT always executed, ignoring the downstream state)
    vs CA-CRS+ (REDIRECT downgraded to HOLD when downstream score > ripple_thr).
    """
    n_redirect = n_downgrade = n_unsafe_baseline = 0
    rows = []
    for fr in frames:
        uD, uS, uC = up[fr]
        dD, dS, dC = dn[fr]
        u_s = score(uD, uS, uC, cfg)
        u_cause, _ = attribute(uD, uS, uC, cfg)
        d_s = score(dD, dS, dC, cfg)
        base = gate_command(u_cause, None, cfg)["action"]     # ignores downstream
        aware = gate_command(u_cause, d_s, cfg)               # ripple-aware
        if base == "REDIRECT":
            n_redirect += 1
            if d_s > cfg.ripple_thr:
                n_unsafe_baseline += 1
            if aware["ripple_downgraded"]:
                n_downgrade += 1
        rows.append({
            "frame": fr, "up_score": round(u_s, 4), "up_cause": u_cause.value,
            "baseline_gate": base, "downstream_score": round(d_s, 4),
            "cacrs_gate": aware["action"], "ripple_downgraded": aware["ripple_downgraded"],
        })
    return {"n_frames": len(frames), "n_redirect": n_redirect,
            "n_downgraded_to_hold": n_downgrade,
            "n_unsafe_redirects_prevented": n_unsafe_baseline, "rows": rows}


# ---------------------------------------------------------------------------
# (3) conflict-window extraction from the per-frame C series
# ---------------------------------------------------------------------------
def extract_conflict_windows(frames, C_series, thr=0.5, min_len=3):
    C = np.asarray(C_series, dtype=float)
    frames = list(frames)
    windows, i, n = [], 0, len(C)
    while i < n:
        if C[i] >= thr:
            j = i
            while j < n and C[j] >= thr:
                j += 1
            if (j - i) >= min_len:
                seg = C[i:j]
                windows.append({"start_frame": frames[i], "end_frame": frames[j - 1],
                                "n_frames": int(j - i), "C_mean": float(seg.mean()),
                                "C_peak": float(seg.max())})
            i = j
        else:
            i += 1
    return windows


# ---------------------------------------------------------------------------
# helpers: join + write + motion-csv builder
# ---------------------------------------------------------------------------
def join_counts_motion(counts, motion, motion_frame_offset=0):
    m = {k + motion_frame_offset: v for k, v in motion.items()}
    frames = sorted(set(counts) & set(m))
    cnts = [counts[fr] for fr in frames]
    Ss = [m[fr][0] for fr in frames]
    Cs = [m[fr][1] for fr in frames]
    return frames, cnts, Ss, Cs


def write_rows_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_motion_csv(frame_dir, out, fps=7.0, stride=1):
    from frame_motion import process_frames_motion  # lazy: needs motion_features + cv2
    res = process_frames_motion(frame_dir, fps=fps, frame_stride=stride)
    cols = ["frame_index", "t_seconds", "S", "C", "n_moving", "mean_speed_fps"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow({k: r.get(k) for k in cols})
    return len(res)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_density_args(p):
    p.add_argument("--capacity-max", type=float, default=None,
                   help="persons at which D=1 for this zone (use EITHER this OR --area-m2)")
    p.add_argument("--area-m2", type=float, default=None,
                   help="zone floor area in m^2 -> D=(count/area)/rho-crush")
    p.add_argument("--rho-crush", type=float, default=7.0,
                   help="crush density persons/m^2 (default 7.0)")
    p.add_argument("--cal-a", type=float, default=1.0, help="calibration slope a")
    p.add_argument("--cal-b", type=float, default=0.0, help="calibration intercept b")
    p.add_argument("--motion-frame-offset", type=int, default=0,
                   help="added to motion frame keys before joining with counts")


def _zone(counts_path, motion_path, off, a, b, cf=None, cc=None, mf=None, sc=None, ccc=None):
    counts, _, _ = load_counts(counts_path, cf, cc)
    motion, _, _, _ = load_motion(motion_path, mf, sc, ccc)
    frames, cnts, Ss, Cs = join_counts_motion(counts, motion, off)
    return frames, cnts, Ss, Cs


def main(argv):
    ap = argparse.ArgumentParser(description="CA-CRS+ evaluation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="single-zone per-frame risk timeline")
    ps.add_argument("--counts", required=True)
    ps.add_argument("--motion", required=True)
    ps.add_argument("--counts-frame-col")
    ps.add_argument("--count-col")
    ps.add_argument("--motion-frame-col")
    ps.add_argument("--s-col")
    ps.add_argument("--c-col")
    ps.add_argument("--conflict-thr", type=float, default=0.5)
    ps.add_argument("--conflict-min-len", type=int, default=3)
    ps.add_argument("--out", default="ca_crs_timeline.csv")
    _add_density_args(ps)

    pr = sub.add_parser("ripple", help="two-zone ripple-effect counterfactual")
    pr.add_argument("--up-counts", required=True)
    pr.add_argument("--up-motion", required=True)
    pr.add_argument("--dn-counts", required=True)
    pr.add_argument("--dn-motion", required=True)
    pr.add_argument("--out", default="ca_crs_ripple.csv")
    _add_density_args(pr)

    pm = sub.add_parser("motion-csv", help="dump per-frame motion CSV from a frame folder")
    pm.add_argument("--frame-dir", required=True)
    pm.add_argument("--fps", type=float, default=7.0)
    pm.add_argument("--stride", type=int, default=1)
    pm.add_argument("--out", default="motion.csv")

    args = ap.parse_args(argv)
    cfg = CACRSConfig()

    if args.cmd == "motion-csv":
        n = build_motion_csv(args.frame_dir, args.out, fps=args.fps, stride=args.stride)
        print(f"wrote {args.out} ({n} motion steps)")
        return

    if args.cmd == "score":
        frames, cnts, Ss, Cs = _zone(args.counts, args.motion, args.motion_frame_offset,
                                     args.cal_a, args.cal_b, args.counts_frame_col,
                                     args.count_col, args.motion_frame_col, args.s_col, args.c_col)
        if not frames:
            print("ERROR: no overlapping frames between counts and motion (try --motion-frame-offset)")
            sys.exit(2)
        rows = score_series(frames, cnts, Ss, Cs, cfg,
                            area_m2=args.area_m2, rho_crush=args.rho_crush,
                            capacity_max=args.capacity_max, cal_a=args.cal_a, cal_b=args.cal_b)
        write_rows_csv(args.out, rows)
        n = len(rows)
        lv = Counter(r["level"] for r in rows)
        ca = Counter(r["cause"] for r in rows)
        print(f"frames scored: {n}  ->  {args.out}")
        print("  level mix :", dict(lv))
        print("  cause mix :", dict(ca))
        print(f"  crowd-risk rate (share DANGER, paper CRR): {100.0 * lv['DANGER'] / n:.1f}%")
        print(f"  mean score: {np.mean([r['score'] for r in rows]):.3f}"
              f"   mean risk-reduction if gate applied: {np.mean([r['risk_reduction'] for r in rows]):.3f}")
        wins = extract_conflict_windows(frames, Cs, args.conflict_thr, args.conflict_min_len)
        print(f"  conflict windows (C>={args.conflict_thr}, >= {args.conflict_min_len} frames): {len(wins)}")
        for w in wins[:12]:
            print("     ", w)
        return

    if args.cmd == "ripple":
        upf, upc, upS, upC = _zone(args.up_counts, args.up_motion, args.motion_frame_offset, args.cal_a, args.cal_b)
        dnf, dnc, dnS, dnC = _zone(args.dn_counts, args.dn_motion, args.motion_frame_offset, args.cal_a, args.cal_b)

        def dvec(frames, cnts, Ss, Cs):
            d = {}
            for i, fr in enumerate(frames):
                cnt = calibrate(cnts[i], args.cal_a, args.cal_b)
                D = count_to_density(cnt, args.area_m2, args.rho_crush, args.capacity_max)
                d[fr] = (D, float(np.clip(Ss[i], 0, 1)), float(np.clip(Cs[i], 0, 1)))
            return d

        up = dvec(upf, upc, upS, upC)
        dn = dvec(dnf, dnc, dnS, dnC)
        frames = sorted(set(up) & set(dn))
        if not frames:
            print("ERROR: no overlapping frames between the two zones")
            sys.exit(2)
        res = ripple_counterfactual(frames, up, dn, cfg)
        write_rows_csv(args.out, res["rows"])
        print(f"frames: {res['n_frames']}  ->  {args.out}")
        print(f"  upstream REDIRECT frames                 : {res['n_redirect']}")
        print(f"  ...of which downstream already unsafe    : {res['n_unsafe_redirects_prevented']}")
        print(f"  -> CA-CRS+ downgraded REDIRECT to HOLD   : {res['n_downgraded_to_hold']}")
        print("  (these downgrades are the secondary-crush events the ripple mechanism prevents)")
        return


# ---------------------------------------------------------------------------
# self-test (no args) -- synthetic data + real DM-Count plumbing check
# ---------------------------------------------------------------------------
def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def selftest():
    cfg = CACRSConfig()
    print("== CA-CRS+ harness self-test ==")

    # 1) count_to_density
    assert _approx(count_to_density(50, capacity_max=100), 0.5)
    assert _approx(count_to_density(1000, capacity_max=100), 1.0)          # clip
    assert _approx(count_to_density(35, area_m2=10, rho_crush=7), 0.5)
    try:
        count_to_density(5)
        raise AssertionError("expected ValueError when neither area nor capacity given")
    except ValueError:
        pass
    print("  [ok] count_to_density (capacity / area / clip / validation)")

    # 2) calibrate
    assert _approx(calibrate(8.354, 0.571, 1.080), 0.571 * 8.354 + 1.080)
    assert calibrate(-99, 1.0, 0.0) == 0.0
    print("  [ok] calibrate")

    # 3) single-zone scoring: dense frame -> DANGER/DENSITY, quiet frame -> SAFE
    rows = score_series([1, 2], [95, 5], [0.05, 0.05], [0.0, 0.0], cfg, capacity_max=100)
    assert rows[0]["level"] == "DANGER", rows[0]
    assert rows[0]["cause"] == "DENSITY", rows[0]
    assert rows[1]["level"] == "SAFE", rows[1]
    assert 0.0 <= rows[0]["risk_reduction"] <= 1.0
    print("  [ok] score_series:", {k: rows[0][k] for k in ("D", "score", "level", "cause", "gate", "risk_reduction")})

    # 4) conflict-window extraction
    Cseq = [0.1, 0.6, 0.7, 0.8, 0.2, 0.55, 0.9, 0.6, 0.1]
    frames = list(range(1, len(Cseq) + 1))
    wins = extract_conflict_windows(frames, Cseq, thr=0.5, min_len=3)
    assert len(wins) == 2, wins
    assert wins[0]["start_frame"] == 2 and wins[0]["end_frame"] == 4, wins[0]
    assert wins[1]["start_frame"] == 6 and wins[1]["end_frame"] == 8, wins[1]
    print("  [ok] extract_conflict_windows:", wins)

    # 5) ripple counterfactual: upstream conflict, downstream first-half dangerous
    fr = [1, 2, 3, 4, 5, 6]
    up = {f: (0.30, 0.20, 0.90) for f in fr}                 # conflict-dominant -> REDIRECT
    dn = {f: ((0.90, 0.10, 0.00) if f <= 3 else (0.10, 0.10, 0.10)) for f in fr}
    assert gate_command(attribute(0.30, 0.20, 0.90, cfg)[0], None, cfg)["action"] == "REDIRECT"
    assert score(0.90, 0.10, 0.00, cfg) > cfg.ripple_thr
    assert score(0.10, 0.10, 0.10, cfg) <= cfg.ripple_thr
    res = ripple_counterfactual(fr, up, dn, cfg)
    assert res["n_redirect"] == 6, res
    assert res["n_unsafe_redirects_prevented"] == 3, res
    assert res["n_downgraded_to_hold"] == 3, res
    print("  [ok] ripple_counterfactual:", {k: res[k] for k in ("n_redirect", "n_unsafe_redirects_prevented", "n_downgraded_to_hold")})

    # 6) REAL-DATA plumbing (no fabricated motion): load real DM-Count counts, calibrate, report D
    real = "s2l1_dmcount_qnrf.csv"
    if os.path.exists(real):
        counts, fc, cc = load_counts(real)
        raw = np.array(list(counts.values()))
        cal = np.array([calibrate(v, 0.571, 1.080) for v in raw])
        print(f"  [real] {real}: cols=({fc},{cc}) n={len(raw)} raw_mean={raw.mean():.2f} "
              f"calibrated_mean={cal.mean():.2f}  (S2.L1 GT mean = 5.85)")
        for cap in (30, 47, 70):
            D = np.clip(cal / cap, 0, 1)
            print(f"         capacity_max={cap:3d} -> D mean={D.mean():.3f} max={D.max():.3f} "
                  f"crush-band frames(D>=0.857)={int((D >= cfg.conflict_unreliable_D).sum())}")
        print("  [note] the capacity_max values above are ILLUSTRATIVE placeholders -- real value needed from you.")
    else:
        print("  [real] skipped (s2l1_dmcount_qnrf.csv not in this dir)")

    print("== all self-tests passed ==")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main(sys.argv[1:])
