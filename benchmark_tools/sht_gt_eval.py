#!/usr/bin/env python3
"""
sht_gt_eval.py -- ShanghaiTech / UCF-QNRF ground-truth extractor + counting
evaluator for CA-CRS+ (answers Reviewer 1 "no SOTA benchmarking" and gives a
high-density, GT-backed counting result WITHOUT any stock video).

Commands:
  gt   --gt-dir DIR --out gt.csv     parse GT_*.mat  -> image_id,gt_count
  eval --gt gt.csv --pred pred.csv   MAE/RMSE/bias/Pearson r vs GT (+SOTA table)
  (no args)                          run offline self-test

.mat parsing uses scipy.io.loadmat and runs on Colab (scipy present there). The
index-navigation logic is validated offline here against mock structures that
mirror scipy's output, so the parser is correct before you run it on real data.
"""
import sys, os, csv, glob, math, argparse
import numpy as np

# Published reference numbers. VERIFY each against the primary PDF before it
# goes into the manuscript. DM-Count SHA/SHB MAE are confirmed (59.7 / 7.4).
SOTA_REF = {
    "DM-Count (NeurIPS'20)": {"SHA": 59.7, "SHB": 7.4, "QNRF": 85.6},
    "APGCC (ECCV'24)": {},   # fill from APGCC paper after verification
}


def _iter_elements(a):
    if isinstance(a, np.ndarray):
        if a.dtype.names:
            for idx in np.ndindex(a.shape):
                v = a[idx]
                for f in a.dtype.names:
                    yield v[f]
        else:
            for idx in np.ndindex(a.shape):
                yield a[idx]
    elif isinstance(a, np.void):
        for name in (a.dtype.names or ()):
            yield a[name]


def _as_nx2(a):
    if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 2 \
            and a.shape[0] >= 1 and a.dtype.names is None and a.dtype != object \
            and np.issubdtype(a.dtype, np.number):
        return a.astype(float)
    return None


def find_points(obj, _depth=0):
    direct = _as_nx2(obj)
    if direct is not None:
        return direct
    if _depth > 12:
        return None
    if isinstance(obj, (np.ndarray, np.void)):
        for sub in _iter_elements(obj):
            r = find_points(sub, _depth + 1)
            if r is not None:
                return r
    return None


def extract_points(matdict):
    for key in ("annPoints", "image_info", "points", "annptrs"):
        if key in matdict:
            r = find_points(matdict[key])
            if r is not None:
                return r
    for k, v in matdict.items():
        if isinstance(k, str) and k.startswith("__"):
            continue
        r = find_points(v)
        if r is not None:
            return r
    return np.zeros((0, 2), float)


def gt_count_from_mat(path):
    import scipy.io as sio   # available on Colab
    m = sio.loadmat(path)
    pts = extract_points(m)
    return len(pts), pts


def _image_id_from_gt_name(fn):
    base = os.path.basename(fn)
    if base.lower().endswith(".mat"):
        base = base[:-4]
    if base.startswith("GT_"):
        base = base[3:]
    return base


def build_gt_csv(gt_dir, out_csv):
    mats = sorted(glob.glob(os.path.join(gt_dir, "*.mat")))
    if not mats:
        raise SystemExit("no .mat files in %s" % gt_dir)
    rows = []
    for p in mats:
        n, _ = gt_count_from_mat(p)
        rows.append((_image_id_from_gt_name(p), n))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "gt_count"]); w.writerows(rows)
    cnts = [r[1] for r in rows]
    print("wrote %d GT counts -> %s  (mean=%.2f min=%d max=%d)"
          % (len(rows), out_csv, float(np.mean(cnts)), min(cnts), max(cnts)))
    return rows


def _read_two_col(path):
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    idc = next((cols[c] for c in ("image_id", "id", "frame", "filename", "name") if c in cols), df.columns[0])
    valc = next((cols[c] for c in ("gt_count", "pred_count", "count", "pred", "gt") if c in cols), df.columns[-1])
    out = df[[idc, valc]].copy(); out.columns = ["image_id", "value"]
    out["image_id"] = out["image_id"].astype(str).str.replace(r"\.(jpg|png|mat)$", "", regex=True)
    return out


def eval_counts(gt_csv, pred_csv):
    import pandas as pd
    gt = _read_two_col(gt_csv).rename(columns={"value": "gt"})
    pr = _read_two_col(pred_csv).rename(columns={"value": "pred"})
    m = gt.merge(pr, on="image_id", how="inner")
    if len(m) == 0:
        raise SystemExit("no overlapping image_id between GT and predictions")
    g = m["gt"].to_numpy(float); p = m["pred"].to_numpy(float)
    mae = float(np.mean(np.abs(p - g))); rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    bias = float(np.mean(p - g)); r = float(np.corrcoef(g, p)[0, 1]) if len(m) > 1 else float("nan")
    print("n=%d  GT mean=%.2f  pred mean=%.2f" % (len(m), g.mean(), p.mean()))
    print("MAE=%.3f  RMSE=%.3f  bias(pred-gt)=%+.3f  Pearson r=%.4f" % (mae, rmse, bias, r))
    print("\nPublished SOTA reference (verify vs primary PDF before manuscript):")
    for meth, d in SOTA_REF.items():
        if d:
            print("  %s: %s" % (meth, ", ".join("%s MAE=%.1f" % (k, v) for k, v in d.items())))
    return {"n": len(m), "mae": mae, "rmse": rmse, "bias": bias, "r": r}


def selftest():
    print("== sht_gt_eval self-test ==")
    loc = np.array([[1., 2.], [3., 4.], [5., 6.]])
    inner = np.empty((1, 1), object); inner[0, 0] = loc
    ii = np.empty((1, 1), object); ii[0, 0] = inner
    pts = extract_points({"image_info": ii, "__header__": b"x"})
    assert pts.shape == (3, 2) and abs(pts.sum() - 21.0) < 1e-9, pts
    print("  [ok] object-nested image_info -> %d points" % len(pts))
    st = np.zeros((1, 1), dtype=[("location", "O"), ("number", "O")])
    st[0, 0]["location"] = np.array([[10., 20.], [30., 40.]]); st[0, 0]["number"] = np.array([[2]])
    ii2 = np.empty((1, 1), object); ii2[0, 0] = st
    p2 = extract_points({"image_info": ii2})
    assert p2.shape == (2, 2), p2.shape
    print("  [ok] structured image_info (location/number) -> %d points" % len(p2))
    p3 = extract_points({"annPoints": np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])})
    assert p3.shape == (4, 2)
    print("  [ok] QNRF annPoints -> %d points" % len(p3))
    import tempfile
    d = tempfile.mkdtemp()
    gtp = os.path.join(d, "gt.csv"); prp = os.path.join(d, "pred.csv")
    open(gtp, "w").write("image_id,gt_count\nIMG_1,10\nIMG_2,20\nIMG_3,30\n")
    open(prp, "w").write("image_id,pred_count\nIMG_1,12\nIMG_2,18\nIMG_3,33\n")
    res = eval_counts(gtp, prp)
    assert abs(res["mae"] - 7 / 3) < 1e-6, res["mae"]
    assert abs(res["rmse"] - math.sqrt(17 / 3)) < 1e-6, res["rmse"]
    print("  [ok] MAE/RMSE math")
    print("== all sht_gt_eval self-tests passed ==")


def main(argv):
    ap = argparse.ArgumentParser(description="ShanghaiTech/QNRF GT + counting eval")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gt"); g.add_argument("--gt-dir", required=True); g.add_argument("--out", default="gt_counts.csv")
    e = sub.add_parser("eval"); e.add_argument("--gt", required=True); e.add_argument("--pred", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "gt":
        build_gt_csv(a.gt_dir, a.out)
    elif a.cmd == "eval":
        eval_counts(a.gt, a.pred)
    else:
        selftest()


if __name__ == "__main__":
    main(sys.argv[1:])
