"""Fetch the two weight files into weights/ (about 925 MB).

    python scripts/download_weights.py

seg_model_smooth_labels.keras comes from the segmenteverygrain repository
(Apache-2.0); sam2.1_hiera_large.pt from Meta (Apache-2.0). Set PM_SEG_WEIGHTS
to put them elsewhere. Neither file is redistributed by this repository.
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
import os
import sys
import urllib.request
from pathlib import Path

FILES = {
    "seg_model_smooth_labels.keras":
        "https://raw.githubusercontent.com/zsylvester/segmenteverygrain/main/models/seg_model_smooth_labels.keras",
    "sam2.1_hiera_large.pt":
        "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt",
}
dest = Path(os.environ.get("PM_SEG_WEIGHTS") or Path(__file__).resolve().parents[1] / "weights")
dest.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    out = dest / name
    if out.exists():
        print(f"have {out}")
        continue
    print(f"downloading {name} ...", flush=True)
    urllib.request.urlretrieve(url, out)
    print(f"  -> {out} ({out.stat().st_size / 2**20:.0f} MB)")
sys.exit(0)
