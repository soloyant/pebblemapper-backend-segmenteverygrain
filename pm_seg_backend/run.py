"""Segment Every Grain backend: subprocess entry point (runs in conda env 'pm-seg').

Reads the PebbleMapper job spec (``--spec <json>``), runs segmenteverygrain
(U-Net prompts + SAM 2.1 masks) on each job image and writes, for each job,
an *instance file* next to the CSV the core expects::

    <out_csv>.instances.npz   labels  int32 HxW  (0 = background, k = grain k)
                              scores  float32 N  (mean U-Net grain probability
                                                  under grain k; NOT a detector
                                                  confidence)
                              shape   (H, W)

It deliberately does NOT write the canonical CSV: measurement happens in the
core (detectors.measure.measure_mask) so the numbers are defined identically
to Mask R-CNN. This file must never import the PebbleMapper core.

Spec params honoured: ``devicemode`` ("gpu"/"cpu"), ``weights_dir``,
``metric_cropsize`` + ``resolution`` (ortho patch size), ``seg`` (dict of
segmenteverygrain options: min_area, dbs_max_dist, min_grain_area, patch_size,
overlap, remove_edge_grains, use_sam, dilation).
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
import argparse
import json
import os
import sys
import time
import tempfile

import numpy as np


def log(msg):
    print(f"[seg] {msg}", flush=True)


def _load_rgb(path, mode):
    """HxWx3 uint8 in the stored pixel frame (no EXIF transpose, like the core).
    Ortho GeoTIFFs are read with rasterio (first three bands, scaled to 8 bit)."""
    if mode == "ortho":
        import rasterio
        with rasterio.open(path) as ds:
            n = min(3, ds.count)
            arr = ds.read(list(range(1, n + 1)))
            arr = np.moveaxis(arr, 0, -1)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            lo, hi = np.nanpercentile(a, 0.5), np.nanpercentile(a, 99.5)
            a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255
            arr = a.astype(np.uint8)
        return np.ascontiguousarray(arr)
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _pick_device(devicemode):
    import torch
    forced = os.environ.get("PM_SEG_DEVICE", "").strip().lower()
    if forced:
        return forced
    if str(devicemode or "gpu").lower() == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_models(weights_dir, device):
    import keras
    import torch
    import segmenteverygrain as seg
    from sam2.build_sam import build_sam2

    unet_path = os.path.join(weights_dir, "seg_model_smooth_labels.keras")
    sam_ckpt = os.environ.get("PM_SEG_SAM_CKPT") or os.path.join(
        weights_dir, "sam2.1_hiera_large.pt")
    sam_cfg = os.environ.get("PM_SEG_SAM_CFG") or "configs/sam2.1/sam2.1_hiera_l.yaml"
    for p in (unet_path, sam_ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(f"weights missing: {p}")
    t0 = time.time()
    unet = keras.saving.load_model(
        unet_path, custom_objects={"weighted_crossentropy": seg.weighted_crossentropy})
    log(f"U-Net loaded ({time.time() - t0:.1f}s, tensorflow on CPU/GPU as available)")
    t0 = time.time()
    sam = build_sam2(sam_cfg, sam_ckpt, device=device)
    log(f"SAM 2.1 loaded from {os.path.basename(sam_ckpt)} on {device} "
        f"({time.time() - t0:.1f}s)")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}, "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    return unet, sam


def _segment(image, unet, sam, seg_opts, patch_size, overlap_px):
    """Run segmenteverygrain on one RGB array. Returns (polygons, image_pred)."""
    import segmenteverygrain as seg
    from PIL import Image

    # predict_large_image takes a *filename* (keras load_img), so the array
    # goes through a temporary PNG: this is how GeoTIFF/ortho windows get in.
    fd, tmp = tempfile.mkstemp(prefix="pm_seg_", suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(image).save(tmp)
        polys, image_pred, coords = seg.predict_large_image(
            tmp, unet, sam,
            use_sam=bool(seg_opts.get("use_sam", True)),
            dilation=int(seg_opts.get("dilation", 3)),
            min_area=float(seg_opts.get("min_area", 100.0)),
            patch_size=int(patch_size),
            overlap=int(overlap_px),
            dbs_max_dist=float(seg_opts.get("dbs_max_dist", 20.0)),
            min_grain_area=int(seg_opts.get("min_grain_area", 50)),
            remove_edge_grains=bool(seg_opts.get("remove_edge_grains", False)),
            verbose=False,
        )
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return polys, np.asarray(image_pred)


def _rasterize(polys, image):
    """int32 label image from shapely polygons (segmenteverygrain's own rasterizer)."""
    import segmenteverygrain as seg
    if not polys:
        return np.zeros(image.shape[:2], dtype=np.int32)
    return seg.rasterize_grains(list(polys), image).astype(np.int32)


def _scores(labels, image_pred, n):
    """Mean U-Net grain probability (channel 1) under each label, in [0, 1]."""
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    prob = image_pred[:, :, 1].astype(np.float64)
    sums = np.bincount(labels.ravel(), weights=prob.ravel(), minlength=n + 1)
    cnts = np.bincount(labels.ravel(), minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    return np.clip(s[1:n + 1], 0.0, 1.0).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Segment Every Grain backend for PebbleMapper.")
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    mode = str(spec.get("mode", "quadrat")).lower()
    params = spec.get("params", {})
    seg_opts = params.get("seg", {}) or {}
    jobs = spec.get("jobs", [])
    weights_dir = params.get("weights_dir") or os.environ.get("PM_SEG_WEIGHTS") or ""
    device = _pick_device(params.get("devicemode"))
    log(f"segmenteverygrain backend: mode={mode}, {len(jobs)} job(s), device={device}")

    unet, sam = _load_models(weights_dir, device)

    for ji, job in enumerate(jobs):
        path = job["path"]
        out_npz = job.get("instances_path") or (job["out_csv"] + ".instances.npz")
        t0 = time.time()
        log(f"job {ji + 1}/{len(jobs)}: {os.path.basename(path)}")
        image = _load_rgb(path, mode)
        h, w = image.shape[:2]
        if mode == "ortho":
            res = float(params.get("resolution") or 0)
            crop_m = float(params.get("metric_cropsize") or 0)
            patch = int(round(crop_m / res)) if (res > 0 and crop_m > 0) else 2000
            patch = max(512, min(patch, 4096))
            frac = float(params.get("overlap") or 0.0)
            overlap_px = int(seg_opts.get("overlap", max(64, patch * (frac if frac > 0 else 0.3))))
        else:
            patch = int(seg_opts.get("patch_size", 2000))
            overlap_px = int(seg_opts.get("overlap", 600))
        log(f"  image {w}x{h} px, patch={patch}px overlap={overlap_px}px")
        polys, image_pred = _segment(image, unet, sam, seg_opts, patch, overlap_px)
        labels = _rasterize(polys, image)
        n = len(polys)
        scores = _scores(labels, image_pred, n)
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        np.savez_compressed(out_npz, labels=labels, scores=scores,
                            shape=np.array([h, w], dtype=np.int64))
        log(f"  {n} grains -> {os.path.basename(out_npz)} ({time.time() - t0:.1f}s)")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
