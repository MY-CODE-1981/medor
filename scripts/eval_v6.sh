#!/bin/bash
# Step 1: Evaluate fan_pipe_v6 (no finetune + with finetune)
set -e
source activate_env.sh

echo "=== fan_eval_v6 (no finetune) ==="
WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path data/release/fan_release/fan_pipe_v6 \
  --cloth_type fan_cloth --exp_name fan_eval_v6 \
  --log_dir data/release/fan_release --max_test_num 14

echo ""
echo "=== fan_eval_v6_ft (with finetune) ==="
WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path data/release/fan_release/fan_pipe_v6 \
  --cloth_type fan_cloth --exp_name fan_eval_v6_ft \
  --log_dir data/release/fan_release --max_test_num 14 --tt_finetune
