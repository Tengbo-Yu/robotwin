#!/bin/bash

# 批量评估脚本
# 使用方法: ./batch_eval.sh [gpu_id]
# 示例: ./batch_eval.sh 0

# 设置默认GPU ID
gpu_id=${1:-0}

# 公共参数配置
head_camera_type="D435"
train_config_name="pi0_base_aloha_robotwin_lora"
checkpoint_num="10000"
seed="0"

# 定义要评估的任务列表
# 格式: "task_name model_name"
tasks=(
    "block_handover dual_tasks_0529"
    "diverse_bottles_pick dual_tasks_0529"
    "dual_bottles_pick_easy dual_tasks_0529"
    "dual_bottles_pick_hard dual_tasks_0529"
    "dual_shoes_place dual_tasks_0529"
    # 在这里添加更多任务
    # "your_task_name your_model_name"
)

echo "开始批量评估，使用GPU: $gpu_id"
echo "总共 ${#tasks[@]} 个任务需要评估"
echo "================================"

# 记录开始时间
start_time=$(date)
echo "开始时间: $start_time"

# 逐个评估任务
for i in "${!tasks[@]}"; do
    # 解析任务名和模型名
    task_info=(${tasks[$i]})
    task_name=${task_info[0]}
    model_name=${task_info[1]}
    
    echo ""
    echo "[$((i+1))/${#tasks[@]}] 正在评估任务: $task_name (模型: $model_name)"
    echo "参数: $task_name $head_camera_type $train_config_name $model_name $checkpoint_num $seed $gpu_id"
    echo "----------------------------------------"
    
    # 调用原始评估脚本
    ./eval.sh $task_name $head_camera_type $train_config_name $model_name $checkpoint_num $seed $gpu_id
    
    # 检查评估是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 任务 $task_name 评估完成"
    else
        echo "✗ 任务 $task_name 评估失败"
        # 可以选择继续或停止
        # exit 1  # 取消注释这行会在失败时停止整个批量评估
    fi
    
    echo "----------------------------------------"
done

# 记录结束时间
end_time=$(date)
echo ""
echo "================================"
echo "批量评估完成!"
echo "开始时间: $start_time"
echo "结束时间: $end_time"
echo "总共评估了 ${#tasks[@]} 个任务" 