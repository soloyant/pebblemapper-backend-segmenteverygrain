#!/bin/bash
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
#
# Build the pm-seg environment with the exact versions the adapter was tested
# with (the pip route of the README, step 1), in one go. Needs conda on PATH,
# or CONDA_EXE pointing at conda.exe (Git Bash on Windows).
set -ex
CONDA="${CONDA_EXE:-conda}"
"$CONDA" create -y -n pm-seg python=3.11 pip
PIP="$CONDA run -n pm-seg --no-capture-output python -m pip"
$PIP install --upgrade pip
$PIP install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
$PIP install "tensorflow==2.19.1"
SAM2_BUILD_CUDA=0 $PIP install --no-build-isolation sam2==1.1.0
$PIP install segmenteverygrain==0.5.0
$PIP list
echo ENV_DONE
