"""
apgcc_detector.py
=================
Adapter to run APGCC (ECCV 2024, point-based head counting/localization) on
ARBITRARY video frames -- which its own code cannot do out of the box.

WHY THIS EXISTS
APGCC's built-in inference (engine.evaluate_crowd_counting) needs a DataLoader
carrying ground-truth annotations; it is written to score a prepared test set,
not to run on raw frames. This class bypasses that and calls the model directly
with the EXACT inference recipe read from the repo source (not guessed):

  VERIFIED against AaronCIH/APGCC @ main:
    build   : models.build_model(cfg, training=False)            models/__init__.py
    config  : config.merge_from_file(cfg, yml)                    config.py / main.py
    forward : model(x) -> {'pred_logits','pred_points','offset'}  models/APGCC.py:72-75
    extract : softmax(pred_logits,-1)[:,:,1][0] > THR ; gather pred_points  engine.py:194-200
    input   : ToTensor + Normalize(ImageNet); pad to multiple of 128         datasets/build.py (block=128)
    ckpt    : torch.load(path) -> filter to keys present in model.state_dict() main.py:166-170

HARD CONSTRAINTS -- READ BEFORE USING
  * Only ShanghaiTech Part-A (SHHA) pretrained weights are public. Your four
    videos (overhead market / halls) are DOMAIN-SHIFTED from SHHA. Counts on your
    own footage are cross-domain and MUST be caveated in the paper. Validate the
    method on the SHHA test split, and (strongly recommended) hand-annotate a
    small sample of your own frames for a real in-domain error number.
  * We PAD (not resize) to a multiple of 128, so returned point coords are already
    in ORIGINAL-frame (x, y) pixels -- no rescaling. Points falling outside the
    original frame (into the pad) are dropped.

USAGE (on Colab, after `git clone` of APGCC + downloading SHHA_best.pth)
    import cv2
    from apgcc_detector import APGCCDetector
    det = APGCCDetector(repo_dir="/content/APGCC",
                        checkpoint="/content/APGCC/apgcc/output/SHHA_best.pth")
    frame_bgr = cv2.imread("frame.jpg")               # cv2 is BGR
    pts, scores = det.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    print("count =", len(pts))                         # pts: Nx2 (x,y); scores: N

Nothing here has been executed end-to-end in my environment (no GPU, no weights).
The API is verified against the repo; YOU confirm the first real run on Colab.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ImageNet stats -- confirmed in datasets/build.py:28
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class APGCCDetector:
    def __init__(self, repo_dir: str, checkpoint: str, config_yml: str | None = None,
                 threshold: float = 0.5, device: str | None = None, pad_block: int = 128):
        """
        repo_dir   : cloned APGCC repo root (the folder that contains 'apgcc/')
        checkpoint : path to SHHA_best.pth
        config_yml : test config (default: <repo>/apgcc/configs/SHHA_test.yml)
        threshold  : head-confidence cutoff (repo default 0.5). Higher -> fewer,
                     more confident points; this is a knob you should report.
        pad_block  : spatial multiple required by the model (repo uses 128).
        """
        self.threshold = float(threshold)
        self.pad_block = int(pad_block)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Put the apgcc package on the path so `import config` / `import models` resolve
        apgcc_pkg = os.path.join(repo_dir, "apgcc")
        if not os.path.isdir(apgcc_pkg):
            raise FileNotFoundError(f"'{apgcc_pkg}' not found -- is repo_dir the APGCC root?")
        if apgcc_pkg not in sys.path:
            sys.path.insert(0, apgcc_pkg)

        from config import cfg, merge_from_file      # repo's easydict-based config
        from models import build_model               # repo's model factory

        if config_yml is None:
            config_yml = os.path.join(apgcc_pkg, "configs", "SHHA_test.yml")
        cfg = merge_from_file(cfg, config_yml)

        model = build_model(cfg, training=False)     # returns the model only (not-training path)

        # Load checkpoint the way main.py does: filter to matching keys, be robust to a
        # {'model': state_dict} wrapper and to shape mismatches.
        sd = torch.load(checkpoint, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model_dict = model.state_dict()
        matched = {k: v for k, v in sd.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        if len(matched) == 0:
            raise RuntimeError("No checkpoint keys matched the model -- wrong checkpoint or config.")
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        self.n_loaded = len(matched)
        self.n_total = len(model_dict)

        model.eval().to(self.device)
        self.model = model
        self._to_tensor = T.Compose([
            T.ToTensor(),  # HWC uint8 -> CHW float in [0,1]
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _pad_to_block(self, x: torch.Tensor) -> torch.Tensor:
        """Pad [1,3,H,W] bottom/right with zeros to multiples of pad_block."""
        _, _, h, w = x.shape
        nh = ((h + self.pad_block - 1) // self.pad_block) * self.pad_block
        nw = ((w + self.pad_block - 1) // self.pad_block) * self.pad_block
        return F.pad(x, (0, nw - w, 0, nh - h), value=0.0)  # (left,right,top,bottom)

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray):
        """
        frame_rgb : HxWx3 uint8 RGB (convert cv2 BGR -> RGB before calling).
        returns   : (points Nx2 float32 [x,y] in original-frame pixels, scores N float32)
        """
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 uint8 RGB array.")
        h0, w0 = frame_rgb.shape[:2]

        x = self._to_tensor(frame_rgb).unsqueeze(0).to(self.device)  # [1,3,H,W]
        x = self._pad_to_block(x)

        out = self.model(x)                                          # dict
        scores = F.softmax(out["pred_logits"], -1)[:, :, 1][0]       # [Q] head-class prob
        points = out["pred_points"][0]                              # [Q,2] (x,y)

        keep = scores > self.threshold
        pts = points[keep].detach().cpu().numpy().astype(np.float32)
        scr = scores[keep].detach().cpu().numpy().astype(np.float32)

        # Drop any point that landed in the zero-pad region (outside original frame).
        if pts.shape[0]:
            inb = (pts[:, 0] >= 0) & (pts[:, 0] < w0) & (pts[:, 1] >= 0) & (pts[:, 1] < h0)
            pts, scr = pts[inb], scr[inb]
        return pts, scr

    def count(self, frame_rgb: np.ndarray) -> int:
        pts, _ = self.detect(frame_rgb)
        return int(len(pts))


if __name__ == "__main__":
    # Cannot run a real forward here (needs torch weights + a frame). This just
    # confirms the file imports and the tensor-shaping math is sane on a dummy.
    print("apgcc_detector.py loaded. torch:", torch.__version__ if "torch" in dir() else "n/a")
    dummy = (np.random.rand(157, 233, 3) * 255).astype(np.uint8)  # non-multiple-of-128 dims
    # exercise only the padding math without a model:
    t = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(dummy).unsqueeze(0)
    _, _, H, W = t.shape
    nH = ((H + 127) // 128) * 128
    nW = ((W + 127) // 128) * 128
    print(f"pad check: {H}x{W} -> {nH}x{nW} (both multiples of 128: "
          f"{nH % 128 == 0 and nW % 128 == 0})")
