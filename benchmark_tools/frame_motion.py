"""
frame_motion.py
===============
Run CA-CRS+ motion features over a DIRECTORY OF ORDERED FRAMES.
PETS2009 ships as JPEG image sequences (768x576, ~7 fps, per camera view), NOT
as video files -- so use this instead of process_video_motion() from
motion_features.py.

Purpose: empirically confirm which PETS2009 sub-sequence fits which pathway:
    S1.L1 (13-57, walking in two directions)  -> expect HIGH conflict (C)
    S1.L3 (14-17, crowd begins to run)        -> expect HIGH speed (S), low-ish C

All the flow math is the VERIFIED motion_features() from motion_features.py --
nothing numerical is re-implemented here; this only reads ordered image files.

Point it at a SINGLE camera-view folder, e.g.:
    .../Crowd_PETS09/S1/L1/Time_13-57/View_001/
    python frame_motion.py /path/to/View_001 7
"""
from __future__ import annotations
import os
import re
import sys
import numpy as np
from motion_features import motion_features, compute_flow  # verified core

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _natural_key(name: str):
    # numeric-aware sort so frame_2 < frame_10 (NOT lexicographic 10 < 2)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frames(frame_dir: str) -> list[str]:
    files = [f for f in os.listdir(frame_dir) if f.lower().endswith(_IMG_EXTS)]
    files.sort(key=_natural_key)
    return [os.path.join(frame_dir, f) for f in files]


def process_frames_motion(frame_dir: str, fps: float = 7.0, work_width: int = 640,
                          v_max_fps: float = 0.05, motion_thresh_frac: float = 0.002,
                          frame_stride: int = 1) -> list[dict]:
    """
    frame_dir    : folder of ordered image frames (one camera view)
    fps          : capture rate (PETS2009 ~7). Affects ONLY absolute speed;
                   conflict is fps-independent.
    frame_stride : process every Nth frame (dt scales with it)
    Returns per-step dicts: {frame_index, t_seconds, S, C, n_moving, mean_speed_fps}
    cv2 is imported lazily inside compute_flow, so importing this file needs no cv2.
    """
    import cv2
    frames = list_frames(frame_dir)
    if len(frames) < 2:
        raise FileNotFoundError(f"Need >=2 image frames in {frame_dir}; found {len(frames)}")

    dt = frame_stride / float(fps)
    results, prev_gray, kept = [], None, 0
    for i in range(0, len(frames), frame_stride):
        img = cv2.imread(frames[i])
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = work_width / float(w)
        small = cv2.resize(img, (work_width, max(1, int(round(h * scale)))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = compute_flow(prev_gray, gray)
            feat = motion_features(flow, dt_seconds=dt, frame_width=work_width,
                                   v_max_fps=v_max_fps, motion_thresh_frac=motion_thresh_frac)
            feat.update({"frame_index": i, "t_seconds": kept * dt})
            results.append(feat)
            kept += 1
        prev_gray = gray
    return results


def summarize(results: list[dict], label: str = "") -> None:
    if not results:
        print(f"[{label}] no steps produced"); return
    S = np.array([r["S"] for r in results])
    C = np.array([r["C"] for r in results])
    print(f"[{label}] steps={len(results)}")
    print(f"    S(min/mean/max) = {S.min():.3f} / {S.mean():.3f} / {S.max():.3f}")
    print(f"    C(min/mean/max) = {C.min():.3f} / {C.mean():.3f} / {C.max():.3f}")
    # HONEST verdict guidance:
    #   C is v_max-INDEPENDENT -> trust it for identifying conflict sequences.
    #   S is v_max-DEPENDENT (placeholder) -> compare S ACROSS sequences relatively,
    #   do NOT read an absolute S threshold as meaningful until v_max_fps is calibrated.
    if C.mean() > 0.5:
        print("    -> CONFLICT-like (high directional variance; opposing/crossing flow).")
    elif C.mean() < 0.35:
        print("    -> COHERENT flow (low conflict). If S is also relatively high vs other")
        print("       sequences, this is your SPEED candidate -- but calibrate v_max_fps first.")
    else:
        print("    -> MIXED. Inspect frames; may be low-motion or partially crossing.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frame_motion.py <frame_dir> [fps=7]")
        sys.exit(1)
    d = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
    res = process_frames_motion(d, fps=fps)
    summarize(res, label=os.path.basename(d.rstrip("/\\")) or d)
