#!/bin/bash
# Activate medor environment for train/inference (no softgym required)
export PYTORCH_JIT=0
export PYTHONPATH=${PWD}:${PWD}/garmentnets:$PYTHONPATH
export MUJOCO_gl=egl
export EGL_GPU=$CUDA_VISIBLE_DEVICES
