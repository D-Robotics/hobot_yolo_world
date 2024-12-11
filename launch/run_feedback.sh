#!/bin/bash

# 检查输入参数
if [ "$#" -ne 4 ]; then
  echo "用法: $0 <model_file_name> <score_threshold> <回灌图片文件夹> <回灌结果XML文件夹>"
  exit 1
fi

model_file_name="$1"
score_threshold="$2"
input_folder="$3"
output_folder="$4"

target_count=$(find "$input_folder" -type f -name "*.jpg" | wc -l)

# 启动 ROS 2 launch 文件
echo "污渍检测回灌模型：$model_file_name 阈值: $score_threshold"
echo "回灌图片数量：$target_count"
ros2 launch hobot_yolo_world yolo_world_feedback_align.launch.py yolo_world_model_file_name:=$model_file_name yolo_world_score_threshold:=$score_threshold publish_image_source:=$input_folder yolo_world_image_width:=1280 yolo_world_image_height:=960 yolo_world_dump_ai_result:=1 yolo_world_dump_ai_path:=$output_folder &
ros_pid=$!  # 获取 ROS 2 进程的 PID\

# 定义清理函数
cleanup() {
  echo "终止 ROS 2 进程 (PID: $ros_pid)..."
  kill -9 "$ros_pid" 2>/dev/null
  echo "ROS 2 进程已终止。回灌完成。"
}

# 捕捉退出信号并清理
trap cleanup EXIT

# 监控目标文件夹中的 XML 文件数量
echo "回灌结果XML文件夹: $output_folder"
while true; do
  current_count=$(find "$output_folder" -type f -name "*.xml" | wc -l)
  
  if [ "$current_count" -ge "$target_count" ]; then
    echo "目标文件夹回灌结束 XML 文件数量为 $target_count，终止 ROS 2 进程。"
    exit 0
  fi
  
  # 每隔 1 秒检查一次
  sleep 1
done