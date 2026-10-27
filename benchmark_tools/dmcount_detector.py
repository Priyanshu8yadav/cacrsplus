#!/usr/bin/env python3
"""
dmcount_detector.py
DM-Count adapter for the CA-CRS+ counter bake-off.

Wraps the official DM-Count (Wang et al., NeurIPS 2020) VGG-19 crowd counter so
it can be scored against MOT15 PETS09-S2L1 ground-truth counts side-by-side with
the APGCC detector. The predicted count for an image is the sum of DM-Count's
predicted density map.

Shared 'counter' contract used by the bake-off (the APGCC adapter should expose
the same method so the two are interchangeable):

    counter = DMCountCounter(weights='pretrained/qnrf.pth', device='cuda')
    n = counter.predict_count(image)   # image: path | np.ndarray(BGR HxWx3) | PIL.Image
                                       # returns float (density-map sum)

Folder runner -> CSV compatible with count_eval.py (columns: frame,pred_count):

    python3 dmcount_detector.py \
        --weights pretrained/qnrf.pth \
        --frames '<MOT15>/train/PETS09-S2L1/img1' \
        --out s2l1_dmcount_qnrf.csv

The DM-Count model definition (vgg19) comes from the official repo's models.py:
    https://github.com/cvlab-stonybrook/DM-Count
Run this next to that repo (so 'import models' works) or pass
--models-module <dotted.path.to.models>.

INTERFACE NOTE: APGCC is a point/localization model; its per-image count is the
number of predicted points above threshold. Give its adapter a
predict_count(image) -> float method (e.g. return float(len(points))) and it
plugs into the same runner and into count_eval.py unchanged.
"""

import argparse
import csv
import os
import re

import numpy as np

# ImageNet normalization (DM-Count preprocessing)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------- pure helpers (unit-tested) ---------------------------
def natural_key(name):
    """Natural sort key so 000002.jpg < 000010.jpg regardless of zero padding."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", name)]


def list_frames(folder, exts=IMG_EXTS):
    names = [n for n in os.listdir(folder) if n.lower().endswith(exts)]
    names.sort(key=natural_key)
    return [os.path.join(folder, n) for n in names]


def frame_index_from_name(path):
    """MOT15 frames are named like 000001.jpg -> 1. Returns None if no digits."""
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = re.findall(r"[0-9]+", stem)
    return int(digits[-1]) if digits else None


def compute_padding(h, w, divisor):
    """Right/bottom padding needed to make (h, w) multiples of divisor."""
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    return pad_h, pad_w


def preprocess(image_rgb_uint8, divisor=16):
    """RGB uint8 HxWx3 -> float32 CHW, ImageNet-normalized, padded to a multiple
    of divisor. Returns (chw_float32, (orig_h, orig_w))."""
    assert image_rgb_uint8.ndim == 3 and image_rgb_uint8.shape[2] == 3, "expected HxWx3 RGB"
    h, w = image_rgb_uint8.shape[:2]
    pad_h, pad_w = compute_padding(h, w, divisor)
    if pad_h or pad_w:
        image_rgb_uint8 = np.pad(
            image_rgb_uint8, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
        )
    x = image_rgb_uint8.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    chw = np.transpose(x, (2, 0, 1)).copy()
    return chw, (h, w)


def density_to_count(dmap):
    """DM-Count density map integrates to the head count."""
    return float(np.asarray(dmap, dtype=np.float64).sum())


def write_counts_csv(path, rows):
    """rows: iterable of (frame_index, count). Writes header frame,pred_count."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "pred_count"])
        for frame, count in rows:
            w.writerow([frame, round(float(count), 4)])


# --------------------------- image loading ---------------------------
def _to_rgb_uint8(image):
    """Accept a path, a PIL.Image, or an np.ndarray (BGR HxWx3 as from cv2) and
    return an RGB uint8 HxWx3 array."""
    if isinstance(image, str):
        from PIL import Image
        return np.array(Image.open(image).convert("RGB"))
    try:
        from PIL import Image as _PILImage
        if isinstance(image, _PILImage.Image):
            return np.array(image.convert("RGB"))
    except Exception:
        pass
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # assume BGR (cv2) -> RGB
    return np.ascontiguousarray(arr.astype(np.uint8))


# --------------------------- DM-Count model wrapper ---------------------------
class DMCountCounter:
    def __init__(self, weights, device=None, divisor=16, models_module="models"):
        import importlib
        import torch

        self.torch = torch
        self.divisor = divisor
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        try:
            mod = importlib.import_module(models_module)
        except Exception as e:
            raise ImportError(
                "Could not import DM-Count model module '" + models_module + "'. "
                "Run next to the DM-Count repo (so 'import models' works) or pass "
                "--models-module. Original error: " + repr(e)
            )
        if not hasattr(mod, "vgg19"):
            raise ImportError(
                "'" + models_module + "' has no vgg19() factory (expected DM-Count models.py)."
            )
        self.model = mod.vgg19()
        state = torch.load(weights, map_location="cpu")
        state = self._unwrap_state(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print("[dmcount] warning: missing keys:", len(missing))
        if unexpected:
            print("[dmcount] warning: unexpected keys:", len(unexpected))
        self.model.to(self.device).eval()

    @staticmethod
    def _unwrap_state(state):
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state.items()
            }
        return state

    def predict_density(self, image):
        torch = self.torch
        rgb = _to_rgb_uint8(image)
        chw, _ = preprocess(rgb, self.divisor)
        inp = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.squeeze().detach().cpu().numpy().astype(np.float32)

    def predict_count(self, image):
        return density_to_count(self.predict_density(image))


# --------------------------- folder runner ---------------------------
def run_folder(counter, frames_dir, out_csv, limit=None, progress_every=50):
    paths = list_frames(frames_dir)
    if not paths:
        raise SystemExit("No image frames found in " + frames_dir)
    if limit:
        paths = paths[:limit]
    rows = []
    for i, p in enumerate(paths, start=1):
        idx = frame_index_from_name(p)
        if idx is None:
            idx = i
        rows.append((idx, counter.predict_count(p)))
        if progress_every and i % progress_every == 0:
            print("  " + str(i) + "/" + str(len(paths)) + " frames processed", flush=True)
    write_counts_csv(out_csv, rows)
    vals = [c for _, c in rows]
    print("wrote " + str(len(rows)) + " rows -> " + out_csv)
    print("pred count min/mean/max: %.2f / %.2f / %.2f"
          % (min(vals), sum(vals) / len(vals), max(vals)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="DM-Count adapter / folder runner")
    ap.add_argument("--weights", required=True, help="path to DM-Count weights (.pth)")
    ap.add_argument("--frames", required=True, help="folder of frames (e.g. PETS09-S2L1/img1)")
    ap.add_argument("--out", required=True, help="output CSV (frame,pred_count)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--divisor", type=int, default=16, help="pad frames to a multiple of this")
    ap.add_argument("--models-module", default="models", help="dotted module exposing vgg19()")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N frames")
    args = ap.parse_args(argv)

    counter = DMCountCounter(
        weights=args.weights, device=args.device,
        divisor=args.divisor, models_module=args.models_module,
    )
    run_folder(counter, args.frames, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
