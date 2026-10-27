"""
motion_features.py
==================
Dense optical-flow motion features for CA-CRS+: normalized speed (S) and
directional conflict (C). COUNTER-INDEPENDENT -- works whether the front end
is a density-map counter (DM-Count) or a point-based one (APGCC/P2PNet),
because it operates on raw pixels, not detections.

Uses Farneback dense optical flow. Cob-Parro et al. [5] (your cited SOTA) build
their stampede feature vector from dense optical flow too, so this paradigm is
grounded in the literature, not invented.

TWO deliberate improvements over your published version, each flagged FIX:

  FIX-A (conflict). C = circular variance of flow DIRECTIONS over moving pixels
        (magnitude-weighted), via ca_crs_core.conflict_index. This REPLACES the
        published "fraction opposing the dominant direction" (which collapses at
        a 50/50 opposing split -- the exact PETS2009 case) and the hackathon's
        "IQR of head x-positions" (which measures spatial spread, not flow
        conflict). This is the metric that lets Scenario C actually fire.

  FIX-B (speed). Your paper fixed v_max = 15 px/frame. That is RESOLUTION- and
        FPS-DEPENDENT: the same physical speed yields a different S on a 1080p vs
        480p clip, and at different sampling rates -- so your cross-scenario
        speed numbers were not comparable. Here speed is expressed as
        (fraction of frame width) per SECOND, then normalized by v_max_fps. This
        is resolution- and fps-independent and MUST be described as a change in
        the paper.

Nothing here has been run on your actual videos; the self-test below uses
synthetic flow fields so you (and I) can verify the math with no cv2 and no data.
"""
from __future__ import annotations
import numpy as np
from ca_crs_core import conflict_index   # FIX-A lives there; single source of truth


def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Farneback dense optical flow between two grayscale frames.
    Returns HxWx2 (dx, dy) in pixels/frame at the input resolution.
    cv2 is imported lazily so this file loads (and the self-test runs) without it.
    """
    import cv2
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def motion_features(flow: np.ndarray, dt_seconds: float, frame_width: int,
                    mask: np.ndarray | None = None,
                    v_max_fps: float = 0.05,
                    motion_thresh_frac: float = 0.002) -> dict:
    """
    Convert a dense flow field into normalized speed S and conflict C.

    flow               : HxWx2 px/frame (from compute_flow)
    dt_seconds         : real time between the two frames = stride / source_fps
    frame_width        : width (px) of the frame the flow was computed on
    mask               : optional HxW bool to restrict to a zone (default: all)
    v_max_fps          : speed (fraction-of-width per second) that maps to S=1.0.
                         This is the ONE speed knob; calibrate it once and report it.
    motion_thresh_frac : min displacement (fraction of width per frame) for a
                         pixel to count as 'moving' -- filters static-background
                         optical-flow noise so C isn't diluted by still pixels.

    Returns dict: S (0..1), C (0..1), n_moving, mean_speed_fps (diagnostic).
    """
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag_px = np.sqrt(dx * dx + dy * dy)                 # px/frame
    mag_frac = mag_px / float(frame_width)              # FIX-B: resolution-independent

    region = mask if mask is not None else np.ones(mag_px.shape, dtype=bool)
    moving = region & (mag_frac > motion_thresh_frac)
    n_moving = int(moving.sum())
    if n_moving == 0:
        return {"S": 0.0, "C": 0.0, "n_moving": 0, "mean_speed_fps": 0.0}

    # Speed: fraction-of-width per SECOND, averaged over moving pixels, normalized.
    mean_speed_fps = float(mag_frac[moving].mean() / max(dt_seconds, 1e-6))
    S = float(np.clip(mean_speed_fps / v_max_fps, 0.0, 1.0))

    # Conflict: circular variance of moving-pixel flow directions, magnitude-weighted.
    angles = np.arctan2(dy[moving], dx[moving])
    C = conflict_index(angles, mag_px[moving])          # FIX-A

    return {"S": S, "C": C, "n_moving": n_moving, "mean_speed_fps": mean_speed_fps}


def process_video_motion(video_path: str, target_fps: float = 2.0,
                         work_width: int = 640, v_max_fps: float = 0.05,
                         motion_thresh_frac: float = 0.002) -> list[dict]:
    """
    Iterate a video at ~target_fps, compute per-step motion features.
    Downsampling to work_width is safe: mag_frac (px / frame_width) is invariant
    to the working resolution, so FIX-B still holds. cv2 imported lazily.

    Returns a list of dicts, one per processed step:
        {frame_index, t_seconds, S, C, n_moving, mean_speed_fps}
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / target_fps)))
    dt = stride / src_fps

    results, prev_gray, idx, kept = [], None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            scale = work_width / float(w)
            small = cv2.resize(frame, (work_width, max(1, int(round(h * scale)))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = compute_flow(prev_gray, gray)
                feat = motion_features(flow, dt_seconds=dt, frame_width=work_width,
                                       v_max_fps=v_max_fps,
                                       motion_thresh_frac=motion_thresh_frac)
                feat.update({"frame_index": idx, "t_seconds": kept * dt})
                results.append(feat)
                kept += 1
            prev_gray = gray
        idx += 1
    cap.release()
    return results


if __name__ == "__main__":
    # Synthetic flow fields -> verify FIX-A (conflict) and FIX-B (speed) with no cv2.
    H, W = 200, 640
    dt, vmax = 0.5, 0.05          # 2 fps sampling, v_max = 0.05 width/s

    def make_flow(dx_field, dy_field):
        f = np.zeros((H, W, 2), np.float32)
        f[..., 0] = dx_field
        f[..., 1] = dy_field
        return f

    # (1) Coherent rightward flow: everyone moving the same way -> low conflict.
    aligned = motion_features(make_flow(5.0, 0.0), dt, W, v_max_fps=vmax)

    # (2) Balanced opposing flow: left half +5px, right half -5px -> max conflict,
    #     same speed as (1). The published metric would MISS this; FIX-A catches it.
    dxf = np.full((H, W), 5.0, np.float32)
    dxf[:, W // 2:] = -5.0
    opposing = motion_features(make_flow(dxf, np.zeros((H, W), np.float32)), dt, W, v_max_fps=vmax)

    # (3) Resolution independence: same physical shift on a 2x-wide frame -> same S.
    hi = motion_features(make_flow(10.0, 0.0), dt, 2 * W, v_max_fps=vmax)  # 10px on 1280 == 5px on 640

    print("=" * 64)
    print("motion_features self-test (synthetic; verifies FIX-A and FIX-B)")
    print("=" * 64)
    print(f"(1) aligned  : S={aligned['S']:.3f}  C={aligned['C']:.3f}  (want C~0)")
    print(f"(2) opposing : S={opposing['S']:.3f}  C={opposing['C']:.3f}  (want C~1, same S as 1)")
    print(f"(3) 2x-res   : S={hi['S']:.3f}         (want == aligned S, proving FIX-B)")
    ok = abs(aligned['C']) < 1e-6 and opposing['C'] > 0.99 and abs(hi['S'] - aligned['S']) < 1e-6
    print(f"\nPASS={ok}")
