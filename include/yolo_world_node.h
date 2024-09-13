// Copyright (c) 2024，Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

#ifdef SHARED_MEM_ENABLED
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#endif

#include "ai_msgs/msg/perception_targets.hpp"
#include "ai_msgs/msg/perf.hpp"
#include "dnn_node/dnn_node_data.h"
#include "dnn_node/util/output_parser/perception_common.h"

#include "include/post_process/yolo_world_output_parser.h"

#ifndef YOLO_WORLD_NODE_H_
#define YOLO_WORLD_NODE_H_

using rclcpp::NodeOptions;

using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::output_parser::DnnParserResult;
using ai_msgs::msg::PerceptionTargets;

// dnn node输出数据类型
struct YoloWorldOutput : public DnnNodeOutput {
  // resize参数，用于算法检测结果的映射
  float ratio = 1.0;  //缩放比例系数，无需缩放为1

  std::vector<std::string> class_names; 

  // 图片数据用于渲染
  std::shared_ptr<hobot::dnn_node::DNNTensor> tensor_image;

  ai_msgs::msg::Perf perf_preprocess;
  int resized_w = 0; // 经过resize后图像的w
  int resized_h = 0; // 经过resize后图像的w
};

class YoloWorldNode : public DnnNode {
 public:
  YoloWorldNode(const std::string &node_name,
                 const NodeOptions &options = NodeOptions());
  ~YoloWorldNode() override;

 protected:
  // 集成DnnNode的接口，实现参数配置和后处理
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  // 解析配置文件，包好模型文件路径、解析方法等信息
  int LoadVocabulary();

  // 本地回灌进行算法推理
  int FeedFromLocal();

  int GetTextIndex(
        std::vector<std::string>& texts,
        std::vector<int>& indexs,
        std::vector<std::string>& target_texts);

  static std::shared_ptr<DNNTensor> GetEmbeddingsTensor(
      std::vector<int>& indexs,
      const std::vector<std::vector<float>>& embeddings,
      hbDNNTensorProperties& tensor_properties);

  // 订阅图片消息的topic和订阅者
  // 共享内存模式
#ifdef SHARED_MEM_ENABLED
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::ConstSharedPtr
      sharedmem_img_subscription_ = nullptr;
  std::string sharedmem_img_topic_name_ = "/hbmem_img";
  void SharedMemImgProcess(
      const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg);
#endif

  // 非共享内存模式
  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr
      ros_img_subscription_ = nullptr;
  // 目前只支持订阅原图，可以使用压缩图"/image_raw/compressed" topic
  // 和sensor_msgs::msg::CompressedImage格式扩展订阅压缩图
  std::string ros_img_sub_topic_name_ = "/image";
  void RosImgProcess(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // string msg 控制话题信息
  rclcpp::Subscription<std_msgs::msg::String>::ConstSharedPtr
      ros_string_subscription_ = nullptr;
  std::string ros_string_sub_topic_name_ = "/target_words";
  void RosStringProcess(const std_msgs::msg::String::ConstSharedPtr msg);

  std::shared_ptr<YoloOutputParser> parser = nullptr;

  // 用于解析的配置文件，以及解析后的数据
  std::string vocabulary_file_ = "config/offline_vocabulary_embeddings.json";
  std::string model_file_name_ = "config/yolo_world.bin";
  std::string model_name_ = "";

  float score_threshold_ = 0.05;
  float iou_threshold_ = 0.45;
  int nms_top_k_ = 50;

  // 存储字段名称和对应的值
  std::vector<std::string> indice_;
  std::vector<std::vector<float>> embeddings_;

  // 加载模型后，查询出模型输入分辨率
  int model_input_width_ = -1;
  int model_input_height_ = -1;

  // 用于预测的图片来源，0：本地图片；1：订阅到的image msg
  int feed_type_ = 0;

  // 是否在本地渲染并保存渲染后的图片
  int dump_render_img_ = 0;

  // 使用shared mem通信方式订阅图片
  int is_shared_mem_sub_ = 0;

  // 算法推理的任务数
  int task_num_ = 4;

  // 类别数量
  int num_class_ = 11;

  // 默认检测文本
  std::string texts = "red bottle,trash bin";
  std::vector<std::string> texts_ = {};
  // 在线程中执行推理，避免阻塞订阅IO通道，导致AI msg消息丢失
  // std::mutex mtx_texts_;
  // std::condition_variable cv_texts_;
  std::mutex mtx_text_;
  std::condition_variable cv_text_;

  // 用于回灌的本地图片信息
  std::string image_file_ = "config/yolo_world_test.jpg";

  // 发布AI消息的topic和发布者
  std::string ai_msg_pub_topic_name_ = "/hobot_yolo_world";
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr msg_publisher_ =
      nullptr;
};

#endif  // YOLO_WORLD_NODE_H_
