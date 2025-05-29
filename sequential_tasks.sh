#!/bin/bash

# 检查必需的参数
if [ "$#" -lt 1 ]; then
    echo "使用方法: $0 <gpu_id>"
    exit 1
fi

gpu_id=$1
tasks_file="task_name.txt"

# 检查任务文件是否存在
if [ ! -f "$tasks_file" ]; then
    echo "错误: 任务文件 $tasks_file 不存在"
    exit 1
fi

# 读取任务列表
mapfile -t task_names < "$tasks_file"

# 检查是否有任务
if [ ${#task_names[@]} -eq 0 ]; then
    echo "警告: 任务文件为空，没有任务需要执行"
    exit 0
fi

echo "共发现 ${#task_names[@]} 个任务需要执行"

# 逐个处理任务
for task_name in "${task_names[@]}"; do
    # 跳过空行
    if [ -z "$task_name" ]; then
        continue
    fi
    
    echo "===================="
    echo "开始执行任务: $task_name"
    echo "===================="
    
    # 运行任务
    ./run_task.sh "$task_name" "$gpu_id"
    
    # 检查任务是否成功
    if [ $? -eq 0 ]; then
        echo "任务 $task_name 成功完成"
    else
        echo "警告: 任务 $task_name 可能未正常完成，继续下一个任务"
    fi
    
    echo ""
done

echo "所有任务已处理完毕" 