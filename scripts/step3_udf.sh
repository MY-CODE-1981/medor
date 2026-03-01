#!/bin/bash
# Step 3: UDF pipeline (data conversion → training → evaluation)
set -e
source activate_env.sh

# 1. Convert data with UDF (creates fan_cloth_v3_udf dataset)
echo "=== Converting data with UDF ==="
python scripts/convert_fan_dataset.py --udf --output_dir dataset/fan_cloth_v3_udf

# 2. Train fan_pipe_v7
# NOTE: Before running this, modify train_pipeline_default.yaml for UDF:
#   - tsdf_clip_value: 0.1
#   - volume_absolute_value: True
# And modify predict_default.yaml for UDF:
#   - iso_surface_level: 0.05
#   - gradient_direction: descent
#   - use_gradient_filter: False
echo ""
echo "=== Training fan_pipe_v7 ==="
WANDB_MODE=offline python garmentnets/train_pipeline.py \
  --exp_name fan_pipe_v7 --log_dir data/release/fan_release \
  --ds fan_cloth_v3_udf --cloth_type fan_cloth \
  --canon_checkpoint data/release/fan_release/fan_canon_v2 --input_type depth

# 3. Evaluate
echo ""
echo "=== Evaluating fan_pipe_v7 (no finetune) ==="
WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path data/release/fan_release/fan_pipe_v7 \
  --cloth_type fan_cloth --exp_name fan_eval_v7 \
  --log_dir data/release/fan_release --max_test_num 14

echo ""
echo "=== Evaluating fan_pipe_v7 (with finetune) ==="
WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path data/release/fan_release/fan_pipe_v7 \
  --cloth_type fan_cloth --exp_name fan_eval_v7_ft \
  --log_dir data/release/fan_release --max_test_num 14 --tt_finetune
