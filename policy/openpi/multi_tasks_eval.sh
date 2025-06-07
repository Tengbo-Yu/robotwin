#!/bin/bash

# 批量评估脚本
# 使用方法: ./batch_eval.sh [gpu_id] [delay_hours]
# 示例: ./batch_eval.sh 0      # 立即在GPU 0上运行
# 示例: ./batch_eval.sh 0 2    # 在2小时后在GPU 0上运行

# 设置默认GPU ID
gpu_id=${1:-0}
# 设置延迟时间（小时）
delay_hours=${2:-0}

# 如果指定了延迟时间，则等待
if [ $delay_hours -gt 0 ]; then
    echo "将在 $delay_hours 小时后开始评估..."
    delay_seconds=$((delay_hours * 3600))
    sleep $delay_seconds
fi

# 公共参数配置
head_camera_type="D435"
train_config_name="pi0_base_aloha_robotwin_lora"
checkpoint_num="25000"
seed="0"
model_name="pi0_7+7_0605"  # 单独设置model_name

# 创建结果目录
results_dir="eval_result/${model_name}_${checkpoint_num}"
mkdir -p $results_dir
summary_file="$results_dir/summary.txt"

# 定义要评估的任务列表
# 格式: "task_name"
tasks=(
    "block_handover"
    "diverse_bottles_pick"
    "dual_bottles_pick_easy"
    "dual_bottles_pick_hard"
    "dual_shoes_place"
    # 在这里添加更多任务
    # "your_task_name"
)

echo "开始批量评估，使用GPU: $gpu_id"
echo "总共 ${#tasks[@]} 个任务需要评估"
echo "评估结果将保存到目录: $results_dir"
echo "================================"

# 记录开始时间
start_time=$(date)
echo "开始时间: $start_time"
echo "开始时间: $start_time" > $summary_file
echo "评估任务总数: ${#tasks[@]}" >> $summary_file
echo "================================" >> $summary_file

# 逐个评估任务
for i in "${!tasks[@]}"; do
    # 解析任务名和模型名
    task_name=${tasks[$i]}
    
    # 为当前任务创建结果文件
    task_result_file="$results_dir/${task_name}.txt"
    
    echo ""
    echo "[$((i+1))/${#tasks[@]}] 正在评估任务: $task_name"
    echo "参数: $task_name $head_camera_type $train_config_name $model_name $checkpoint_num $seed $gpu_id"
    echo "----------------------------------------"
    
    # 记录任务开始信息到结果文件
    echo "[$((i+1))/${#tasks[@]}] 任务: $task_name" > $task_result_file
    echo "参数: $task_name $head_camera_type $train_config_name $model_name $checkpoint_num $seed $gpu_id" >> $task_result_file
    echo "----------------------------------------" >> $task_result_file
    
    # 调用原始评估脚本
    ./eval.sh $task_name $head_camera_type $train_config_name $model_name $checkpoint_num $seed $gpu_id
    
    # 检查评估是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 任务 $task_name 评估完成"
        echo "✓ 任务评估完成" >> $task_result_file
        
        # 找到评估结果文件并复制内容
        result_source_file="../../result_pi0/${train_config_name}_${task_name}_${head_camera_type}/ckpt_${checkpoint_num}_seed_${seed}.txt"
        if [ -f "$result_source_file" ]; then
            echo "" >> $task_result_file
            echo "评估详细结果:" >> $task_result_file
            cat "$result_source_file" >> $task_result_file
            
            # 提取成功率并添加到摘要文件
            success_rate=$(grep -A 1 "TopK 1 Success Rate:" "$result_source_file" | tail -n 1)
            echo "[$((i+1))/${#tasks[@]}] 任务: $task_name - 成功率: $success_rate" >> $summary_file
        else
            echo "! 警告: 未找到结果文件 $result_source_file" >> $task_result_file
            echo "[$((i+1))/${#tasks[@]}] 任务: $task_name - 未找到结果文件" >> $summary_file
        fi
    else
        echo "✗ 任务 $task_name 评估失败"
        echo "✗ 任务评估失败" >> $task_result_file
        echo "[$((i+1))/${#tasks[@]}] 任务: $task_name - 评估失败" >> $summary_file
        # 可以选择继续或停止
        # exit 1  # 取消注释这行会在失败时停止整个批量评估
    fi
    
    echo "----------------------------------------" >> $task_result_file
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
echo "评估结果保存在目录: $results_dir"

# 更新摘要文件
echo "" >> $summary_file
echo "================================" >> $summary_file
echo "批量评估完成!" >> $summary_file
echo "开始时间: $start_time" >> $summary_file
echo "结束时间: $end_time" >> $summary_file
echo "总共评估了 ${#tasks[@]} 个任务" >> $summary_file 