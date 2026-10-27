#!/usr/bin/env python3
"""
zone_runner.py -- Option A per-zone data generator for the CA-CRS+ ripple test.

Splits ONE camera view (e.g. PETS09-S2L1 View_001) into an UPSTREAM and a
DOWNSTREAM zone, then emits, per zone, the exact CSVs the CA-CRS+ harness
`ripple` subcommand consumes:

    <prefix>_up_counts.csv   frame,pred_count
    <prefix>_up_motion.csv   frame_index,t_seconds,S,C,n_moving,mean_speed_fps
    <prefix>_dn_counts.csv   ...
    <prefix>_dn_motion.csv   ...

plus <prefix>_zones_overlay.png so you can VISUALLY verify the geometry (which
half the crowd enters = upstream) BEFORE trusting the result. I cannot see the
footage, so the split is a configurable default you confirm from the overlay
(swap with --flip, or supply exact polygons with --zones zones.json).

Per-zone count  = sum of DM-Count's density map inside the zone polygon. Density
                  maps integrate to the head count, so a masked sum IS the count
                  in that region -- the density-map-integration-over-a-zone that
                  replaces point-in-polygon in the paper.
Per-zone motion = motion_features() restricted to the zone mask (masked Farneback
                  flow) -> normalized speed S and circular-variance conflict C.

Then run the ripple counterfactual:
  python ca_crs_harness.py ripple \\
      --up-counts s2l1_up_counts.csv --up-motion s2l1_up_motion.csv \\
      --dn-counts s2l1_dn_counts.csv --dn-motion s2l1_dn_motion.csv \\
      --capacity-max <N> --cal-a 0.571 --cal-b 1.080 --motion-frame-offset 1

Run with NO args for a self-test (synthetic; no cv2 frames, no DM-Count weights).
Colab run needs: cv2, torch, the DM-Count repo (models.py) + weights, and
ca_crs_core.py / motion_features.py / dmcount_detector.py / ca_crs_harness.py
in the same folder.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np

WORK_WIDTH_DEFAULT = 640


# --------------------------- geometry (self-tested) ---------------------------
def polygon_mask(h, w, pts):
    import cv2
    m = np.zeros((h, w), np.uint8)
    poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def default_zones(h, w, split="vertical", frac=0.5):
    frac = float(frac)
    if split == "vertical":
        x = int(round(w * frac))
        return {"upstream": [[0, 0], [x, 0], [x, h], [0, h]],
                "downstream": [[x, 0], [w, 0], [w, h], [x, h]]}
    if split == "horizontal":
        y = int(round(h * frac))
        return {"upstream": [[0, 0], [w, 0], [w, y], [0, y]],
                "downstream": [[0, y], [w, y], [w, h], [0, h]]}
    raise ValueError("split must be 'vertical' or 'horizontal' (or pass --zones json)")


def scale_pts(pts, sx, sy):
    return [[int(round(x * sx)), int(round(y * sy))] for x, y in pts]


def zone_count_from_density(dmap, mask_full):
    """Sum DM-Count density inside a zone. mask_full is at ORIGINAL image res;
    it is resized (nearest) to the density-map resolution before summing, so the
    result is the head count integrated over that zone."""
    import cv2
    dh, dw = dmap.shape[:2]
    if mask_full.shape != (dh, dw):
        m = cv2.resize(mask_full.astype(np.uint8), (dw, dh),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        m = mask_full
    return float(np.asarray(dmap, dtype=np.float64)[m].sum())


def zone_motion(flow, mask_work, dt, work_width, v_max_fps, motion_thresh_frac):
    """motion_features restricted to a zone mask (resized to the flow resolution)."""
    from motion_features import motion_features
    fh, fw = flow.shape[:2]
    if mask_work.shape != (fh, fw):
        import cv2
        mask_work = cv2.resize(mask_work.astype(np.uint8), (fw, fh),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    return motion_features(flow, dt_seconds=dt, frame_width=work_width, mask=mask_work,
                           v_max_fps=v_max_fps, motion_thresh_frac=motion_thresh_frac)


def save_overlay(frame_bgr, zones, path):
    import cv2
    ov = frame_bgr.copy()
    colors = {"upstream": (0, 180, 0), "downstream": (0, 0, 220)}
    for name, pts in zones.items():
        poly = np.asarray(pts, np.int32).reshape(-1, 1, 2)
        col = colors.get(name, (200, 160, 0))
        tmp = ov.copy()
        cv2.fillPoly(tmp, [poly], col)
        ov = cv2.addWeighted(tmp, 0.30, ov, 0.70, 0)
        cv2.polylines(ov, [poly], True, col, 2)
        cx = int(np.mean([p[0] for p in pts])); cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(ov, name, (max(5, cx - 45), cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(path, ov)


def write_motion_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "t_seconds", "S", "C", "n_moving", "mean_speed_fps"])
        for r in rows:
            w.writerow([r["frame_index"], round(r["t_seconds"], 6), round(r["S"], 6),
                        round(r["C"], 6), r["n_moving"], round(r["mean_speed_fps"], 6)])


# --------------------------- folder runner (Colab) ---------------------------
def run_folder(frame_dir, weights, out_prefix="s2l1", zones_json=None, split="vertical",
               frac=0.5, flip=False, fps=7.0, stride=1, work_width=WORK_WIDTH_DEFAULT,
               v_max_fps=0.05, motion_thresh_frac=0.002, device=None,
               models_module="models", limit=None):
    import cv2
    from dmcount_detector import (DMCountCounter, list_frames,
                                  frame_index_from_name, write_counts_csv)
    from motion_features import compute_flow

    paths = list_frames(frame_dir)
    if not paths:
        raise SystemExit("no frames in " + frame_dir)
    if limit:
        paths = paths[:limit]
    first = cv2.imread(paths[0])
    if first is None:
        raise SystemExit("cannot read " + paths[0])
    H, W = first.shape[:2]

    zones = json.load(open(zones_json)) if zones_json else default_zones(H, W, split, frac)
    if flip:
        zones = {"upstream": zones["downstream"], "downstream": zones["upstream"]}
    save_overlay(first, zones, out_prefix + "_zones_overlay.png")
    print("wrote", out_prefix + "_zones_overlay.png",
          "-- OPEN IT and confirm 'upstream' is where the crowd ENTERS (use --flip to swap).")

    masks_full = {n: polygon_mask(H, W, p) for n, p in zones.items()}
    work_h = max(1, int(round(H * work_width / float(W))))
    sx, sy = work_width / float(W), work_h / float(H)
    masks_work = {n: polygon_mask(work_h, work_width, scale_pts(p, sx, sy)) for n, p in zones.items()}

    counter = DMCountCounter(weights, device=device, models_module=models_module)
    dt = stride / float(fps)
    counts = {n: [] for n in zones}
    motion = {n: [] for n in zones}
    prev = None
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        fr = frame_index_from_name(p) or (i + 1)
        dmap = counter.predict_density(img)
        for n in zones:
            counts[n].append((fr, zone_count_from_density(dmap, masks_full[n])))
        small = cv2.resize(img, (work_width, work_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = compute_flow(prev, gray)
            for n in zones:
                feat = zone_motion(flow, masks_work[n], dt, work_width,
                                   v_max_fps, motion_thresh_frac)
                feat["frame_index"] = i          # flow arriving at image i (1-based enumerate)
                feat["t_seconds"] = (i - 1) * dt
                motion[n].append(dict(feat))
        prev = gray
        if (i + 1) % 50 == 0:
            print("  ", i + 1, "/", len(paths), "frames", flush=True)

    for n, tag in (("upstream", "up"), ("downstream", "dn")):
        write_counts_csv(f"{out_prefix}_{tag}_counts.csv", counts[n])
        write_motion_csv(f"{out_prefix}_{tag}_motion.csv", motion[n])
        print("wrote", f"{out_prefix}_{tag}_counts.csv", "+", f"{out_prefix}_{tag}_motion.csv",
              f"({len(counts[n])} counts, {len(motion[n])} motion)")
    print("NEXT: python ca_crs_harness.py ripple --up-counts %s_up_counts.csv "
          "--up-motion %s_up_motion.csv --dn-counts %s_dn_counts.csv "
          "--dn-motion %s_dn_motion.csv --capacity-max <N> --cal-a 0.571 --cal-b 1.080 "
          "--motion-frame-offset 1" % (out_prefix, out_prefix, out_prefix, out_prefix))


def main(argv):
    ap = argparse.ArgumentParser(description="Option A per-zone ripple data generator")
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--weights", required=True, help="DM-Count weights, e.g. pretrained/qnrf.pth")
    ap.add_argument("--out-prefix", default="s2l1")
    ap.add_argument("--zones", help="zones.json: {'upstream':[[x,y]...],'downstream':[[x,y]...]} in original px")
    ap.add_argument("--split", default="vertical", choices=["vertical", "horizontal"])
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--flip", action="store_true", help="swap which side is upstream/downstream")
    ap.add_argument("--fps", type=float, default=7.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--work-width", type=int, default=WORK_WIDTH_DEFAULT)
    ap.add_argument("--v-max-fps", type=float, default=0.05)
    ap.add_argument("--motion-thresh-frac", type=float, default=0.002)
    ap.add_argument("--device", default=None)
    ap.add_argument("--models-module", default="models")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    run_folder(a.frame_dir, a.weights, a.out_prefix, a.zones, a.split, a.frac, a.flip,
               a.fps, a.stride, a.work_width, a.v_max_fps, a.motion_thresh_frac,
               a.device, a.models_module, a.limit)


# --------------------------- self-test ---------------------------
def selftest():
    print("== zone_runner self-test ==")
    m = polygon_mask(100, 200, [[0, 0], [100, 0], [100, 100], [0, 100]])
    assert m.shape == (100, 200), m.shape
    assert m[10:90, 10:90].all(), "interior not filled"
    assert not m[:, 105:].any(), "far side leaked"
    assert m[:, :100].mean() > 0.95, ("left coverage", float(m[:, :100].mean()))
    print("  [ok] polygon_mask (left cover %.3f, right cover %.3f)" % (float(m[:, :100].mean()), float(m[:, 100:].mean())))

    dmap = np.ones((60, 80), np.float32) * 0.02              # total 96
    left = polygon_mask(120, 160, [[0, 0], [80, 0], [80, 120], [0, 120]])
    c = zone_count_from_density(dmap, left)
    assert abs(c - 48.0) < 2.0, c
    dmap2 = np.zeros((60, 80), np.float32); dmap2[20:40, 60:75] = 0.5
    rightm = polygon_mask(120, 160, [[80, 0], [160, 0], [160, 120], [80, 120]])
    cr = zone_count_from_density(dmap2, rightm)
    cl = zone_count_from_density(dmap2, left)
    assert cr > 10 * max(cl, 1e-6), (cl, cr)
    print("  [ok] zone_count_from_density (split=%.1f/%.1f ; localized L=%.2f R=%.2f)" % (c, 96 - c, cl, cr))

    H, W = 120, 640
    fcoh = np.zeros((H, W, 2), np.float32); fcoh[..., 0] = 5.0
    full = np.ones((H, W), bool)
    a = zone_motion(fcoh, full, 0.5, W, 0.05, 0.002)
    dxf = np.full((H, W), 5.0, np.float32); dxf[:, W // 2:] = -5.0
    fopp = np.zeros((H, W, 2), np.float32); fopp[..., 0] = dxf
    b = zone_motion(fopp, full, 0.5, W, 0.05, 0.002)
    lonly = np.zeros((H, W), bool); lonly[:, :W // 2] = True
    d = zone_motion(fopp, lonly, 0.5, W, 0.05, 0.002)
    assert a["C"] < 1e-3 and b["C"] > 0.99 and d["C"] < 1e-3, (a, b, d)
    print("  [ok] zone_motion (coherent C=%.3f, opposing C=%.3f, masked-left C=%.3f)" % (a["C"], b["C"], d["C"]))

    import cv2
    frame = np.zeros((120, 200, 3), np.uint8)
    save_overlay(frame, default_zones(120, 200, "vertical", 0.5), "/tmp/_zov.png")
    assert os.path.exists("/tmp/_zov.png") and cv2.imread("/tmp/_zov.png") is not None
    print("  [ok] save_overlay + default_zones")

    from dmcount_detector import write_counts_csv
    from ca_crs_harness import load_counts, load_motion
    write_counts_csv("/tmp/_c.csv", [(1, 3.2), (2, 4.1)])
    write_motion_csv("/tmp/_m.csv", [
        {"frame_index": 1, "t_seconds": 0.0, "S": 0.5, "C": 0.4, "n_moving": 10, "mean_speed_fps": 0.03},
        {"frame_index": 2, "t_seconds": 0.14, "S": 0.6, "C": 0.5, "n_moving": 12, "mean_speed_fps": 0.04}])
    cnts, _, _ = load_counts("/tmp/_c.csv")
    mot, _, _, _ = load_motion("/tmp/_m.csv")
    assert abs(cnts[1] - 3.2) < 1e-9 and abs(cnts[2] - 4.1) < 1e-9, cnts
    assert abs(mot[1][0] - 0.5) < 1e-9 and abs(mot[1][1] - 0.4) < 1e-9, mot
    print("  [ok] CSV round-trip -> harness load_counts/load_motion")
    print("== all zone_runner self-tests passed ==")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main(sys.argv[1:])
