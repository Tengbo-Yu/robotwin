# ./finetune.sh pi0_base_aloha_robotwin_lora 8_dual_tasks 5

train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
export LEROBOT_HOME=/nvmessd/ssd_share/tengbo/cache
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite