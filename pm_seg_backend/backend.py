"""Segmenteverygrain (Sylvester et al., 2025) as a PebbleMapper detection backend.

Declared through ``user_detectors.json`` (``{"module": "pm_seg_backend.backend",
"factory": "make_backend", "path": "<this repository>"}``); no PebbleMapper
source is touched. The driver is PebbleMapper's own
:class:`detectors.instance_backend.InstanceSubprocessBackend`: this file only
says what the model is, where its weights are, and which options its
``run.py`` takes.

``run.py`` runs in the ``pm-seg`` conda env (U-Net prompts + Segment Anything
2.1 masks) and leaves a label image; every instance is then measured by
PebbleMapper's own measurement step, so the table means exactly what Mask
R-CNN's does. Ortho mode: the subprocess patches the image and merges grains
across seams itself; no .run.csv resume, no tile-level nodata filter, no ROI
in ortho mode.
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
from pathlib import Path

from detectors.base import BackendInfo
from detectors.instance_backend import InstanceSubprocessBackend
from detectors import subprocess_runner

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
ENV_NAME = os.environ.get("PM_SEG_ENV", "pm-seg")
_SCRIPT = PKG_DIR / "run.py"
WEIGHTS_DIR = Path(os.environ.get("PM_SEG_WEIGHTS") or (REPO_DIR / "weights"))
UNET_FILE = "seg_model_smooth_labels.keras"
SAM_FILE = "sam2.1_hiera_large.pt"

CITATION = ("Sylvester, Z., Stockli, D. F., Howes, N., Roberts, K., Malkowski, "
            "M. A., Poros, Z., Martindale, R. C., & Bai, W. (2025). "
            "Segmenteverygrain: A Python module for segmentation of grains in "
            "images. Journal of Open Source Software, 10(112), 7953. "
            "https://doi.org/10.21105/joss.07953")

INFO = BackendInfo(
    name="seg",
    display_name="Segmenteverygrain (Sylvester)",
    framework="tensorflow+pytorch",
    license=("Apache-2.0 (segmenteverygrain code, Copyright 2023 Zoltan Sylvester); "
             "Apache-2.0 (SAM 2.1 code and sam2.1_hiera_large.pt weights, Meta); "
             "U-Net weights seg_model_smooth_labels.keras distributed in the "
             "segmenteverygrain repository under the same Apache-2.0 licence. "
             "None of it is redistributed with this adapter (MIT)."),
    output_type="instance",
    env=ENV_NAME,
    in_process=False,
    weights=f"{WEIGHTS_DIR / UNET_FILE} + {WEIGHTS_DIR / SAM_FILE}",
    install_hint=("Create the 'pm-seg' conda env (see README.md), run "
                  "scripts/download_weights.py, and declare the backend in "
                  "user_detectors.json."),
    description=("U-Net prompts + Segment Anything 2.1 masks for grain outlines "
                 "(segmenteverygrain v0.5.0). Instances are measured by "
                 "PebbleMapper's shared measurement step; Score holds the mean "
                 "U-Net grain probability under the outline, not a detector "
                 "confidence. Cite: " + CITATION),
)


def make_backend():
    """Zero-argument factory named in user_detectors.json."""
    return SegBackend()


class SegBackend(InstanceSubprocessBackend):
    info = INFO
    env_name = ENV_NAME
    required_modules = ("torch", "segmenteverygrain")
    script = _SCRIPT
    model_version = "segmenteverygrain 0.5.0"
    log_prefix = "seg"

    def is_available(self) -> bool:
        return (_SCRIPT.exists()
                and (WEIGHTS_DIR / UNET_FILE).exists()
                and (WEIGHTS_DIR / SAM_FILE).exists()
                and subprocess_runner.conda_env_exists(ENV_NAME))

    def spec_params(self, mode, kwargs, resolution):
        log_fn = kwargs.get("log_fn") or (lambda s: None)
        if kwargs.get("min_confidence") is not None:
            log_fn("[seg] note: min_confidence is a Mask R-CNN threshold; "
                   "segmenteverygrain has no detector confidence, so it is "
                   "recorded in the manifest but not applied.")
        opts = dict(kwargs.get("seg_options")
                    or json.loads(os.environ.get("PM_SEG_OPTIONS", "{}") or "{}"))
        return {"weights_dir": str(WEIGHTS_DIR), "seg": opts}
