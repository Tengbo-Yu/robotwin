# ./finetune.sh pi0_base_aloha_robotwin_lora 8_dual_tasks 5 true true
# 参数说明：
# $1: train_config_name - 训练配置名称
# $2: model_name - 模型名称
# $3: gpu_use - 使用的GPU编号
# $4: dual_arm_separate_denoise - 是否启用双臂分别降噪 (true/false)
# $5: enable_contrastive_learning_cl1 - 是否启用对比学习 (true/false)

train_config_name=$1
model_name=$2
gpu_use=$3
dual_arm_separate_denoise=${4:-true}  # 默认为true
enable_contrastive_learning_cl1=${5:-true}  # 默认为true

export CUDA_VISIBLE_DEVICES=$gpu_use
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model Settings:"
echo "- dual_arm_separate_denoise: $dual_arm_separate_denoise"
echo "- enable_contrastive_learning_cl1: $enable_contrastive_learning_cl1"

# 对于布尔值参数，tyro使用--flag或--no-flag的形式而不是--flag=true/false
# 根据变量值构建命令行参数
dual_arm_flag=""
if [ "$dual_arm_separate_denoise" = "true" ]; then
  dual_arm_flag="--model.dual-arm-separate-denoise"
else
  dual_arm_flag="--model.no-dual-arm-separate-denoise"
fi

cl1_flag=""
if [ "$enable_contrastive_learning_cl1" = "true" ]; then
  cl1_flag="--model.enable-contrastive-learning-cl1"
else
  cl1_flag="--model.no-enable-contrastive-learning-cl1"
fi

export LEROBOT_HOME=/nvmessd/ssd_share/tengbo/cache
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite $dual_arm_flag $cl1_flag