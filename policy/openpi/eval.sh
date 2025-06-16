DEBUG=False

task_name=${1}
head_camera_type=${2}
train_config_name=${3} 
model_name=${4}
checkpoint_num=${5}
seed=${6}
gpu_id=${7}
dual_arm_separate_denoise=${8}
enable_contrastive_learning_cl1=${9}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

source .venv/bin/activate
cd ../..
python ./script/eval_policy_pi.py $task_name $head_camera_type $train_config_name $model_name $checkpoint_num $seed $dual_arm_separate_denoise $enable_contrastive_learning_cl1