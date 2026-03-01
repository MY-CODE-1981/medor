#!/bin/bash
set -e

# Environment setup
cd "$(dirname "$0")"
source activate_env.sh

# Activate conda (medor env: Python 3.10 + PyTorch 2.1.2 + CUDA 12.1)
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate medor
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
export WANDB_MODE=offline
echo "Python: $(which python) ($(python --version 2>&1))"

echo "========================================="
echo " Step 1: Data conversion (v7_back)"
echo "========================================="
if [ -d "dataset/fan_cloth_v2/train/data" ] && [ "$(ls dataset/fan_cloth_v2/train/data/ | wc -l)" -gt 0 ]; then
    echo "  Dataset already exists ($(ls dataset/fan_cloth_v2/train/data/ | wc -l) train samples). Skipping."
else
    python scripts/convert_fan_dataset.py
fi

echo "========================================="
echo " Step 2: Canonicalization (500 epochs)"
echo "========================================="
if [ -d "data/release/fan_release/fan_canon_v2" ] && ls data/release/fan_release/fan_canon_v2/*.ckpt 1>/dev/null 2>&1; then
    echo "  Canon checkpoint already exists. Skipping."
else
    python garmentnets/train_pointnet2.py \
      --exp_name fan_canon_v2 --log_dir data/release/fan_release \
      --ds fan_cloth_v2 --cloth_type fan_cloth --input_type depth
fi

echo "========================================="
echo " Step 3: Pipeline (300 epochs)"
echo "========================================="
python garmentnets/train_pipeline.py \
  --exp_name fan_pipe_v2 --log_dir data/release/fan_release \
  --ds fan_cloth_v2 --cloth_type fan_cloth \
  --canon_checkpoint data/release/fan_release/fan_canon_v2 --input_type depth

echo "========================================="
echo " Step 4: Evaluation (with finetune)"
echo "========================================="
python garmentnets/eval_pipeline.py \
  --model_path data/release/fan_release/fan_pipe_v2 \
  --cloth_type fan_cloth --exp_name fan_eval_v2 \
  --log_dir data/release/fan_release --max_test_num 10 --tt_finetune

echo "========================================="
echo " All done!"
echo "========================================="
