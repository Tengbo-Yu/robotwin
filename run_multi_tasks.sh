#!/bin/bash

# 检查参数
if [ $# -lt 2 ]; then
  echo "用法: $0 <gpu_id> <task1> <task2> ... <taskN>"
  echo "示例: $0 0 blocks_stack_easy blocks_stack_hard bottle_adjust"
  exit 1
fi

# 获取GPU ID
gpu_id=$1
shift

# 输出配置信息
echo "========================================="
echo "将在GPU $gpu_id 上依次执行以下任务:"
for task in "$@"; do
  echo "- $task"
done
echo "========================================="

# 依次执行所有任务
for task in "$@"; do
  echo "开始执行任务: $task"
  ./run_task.sh "$task" "$gpu_id"
  echo "任务 $task 执行完成"
  echo "-----------------------------------------"
done

echo "所有任务执行完毕!"
