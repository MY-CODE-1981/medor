#!/bin/bash
# Step 2: Finetune regularization experiments (A, B, C)
# Uses fan_pipe_v6 model (change model_path to fan_pipe_v5 if v6 isn't ready)
set -e
source activate_env.sh
MODEL=data/release/fan_release/fan_pipe_v6

echo "=== Experiment A: Strong regularization ==="
WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path $MODEL \
  --cloth_type fan_cloth --exp_name fan_eval_v6_ftA \
  --log_dir data/release/fan_release --max_test_num 14 --tt_finetune \
  --laplacian_w 0.5 --edge_w 0.5 --obs_consist_w 20

echo ""
echo "=== Experiment B: Lower LR + fewer iterations ==="
WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path $MODEL \
  --cloth_type fan_cloth --exp_name fan_eval_v6_ftB \
  --log_dir data/release/fan_release --max_test_num 14 --tt_finetune \
  --opt_lr 0.0001 --opt_iter_total 30

echo ""
echo "=== Experiment C: Combined (A+B) ==="
WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path $MODEL \
  --cloth_type fan_cloth --exp_name fan_eval_v6_ftC \
  --log_dir data/release/fan_release --max_test_num 14 --tt_finetune \
  --laplacian_w 0.5 --edge_w 0.5 --opt_lr 0.0001 --opt_iter_total 50
