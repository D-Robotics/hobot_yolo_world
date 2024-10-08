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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include "rclcpp/rclcpp.hpp"
#include <cv_bridge/cv_bridge.h>
#include <unistd.h>

#include "dnn_node/dnn_node.h"
#include "include/image_utils.h"
#include "include/post_process/yolo_world_output_parser.h"

#include "include/yolo_world_node.h"

// 3x3矩阵乘以3x1向量的函数
std::vector<double> matrixMultiply(const std::vector<double>& H, const std::vector<double>& x1) {
    std::vector<double> x2 = {0, 0, 0};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // x2[i] += H[i][j] * x1[j];
            x2[i] += H[i * 3 + j] * x1[j];
        }
    }

    // 归一化，x2 = x2 / x2[2]
    if (x2[2] != 0) {
        for (int i = 0; i < 3; ++i) {
            x2[i] /= x2[2];
        }
    }

    return x2;
}

// 时间格式转换
builtin_interfaces::msg::Time ConvertToRosTime(
    const struct timespec &time_spec) {
  builtin_interfaces::msg::Time stamp;
  stamp.set__sec(time_spec.tv_sec);
  stamp.set__nanosec(time_spec.tv_nsec);
  return stamp;
}

// 根据起始时间计算耗时
int CalTimeMsDuration(const builtin_interfaces::msg::Time &start,
                      const builtin_interfaces::msg::Time &end) {
  return (end.sec - start.sec) * 1000 + end.nanosec / 1000 / 1000 -
         start.nanosec / 1000 / 1000;
}

int split(const std::string& str, 
          char delimiter,
          std::vector<std::string>& tokens) {
    std::string token;
    std::stringstream ss(str);
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }    
    return 0;
}

YoloWorldNode::YoloWorldNode(const std::string &node_name,
                               const NodeOptions &options)
    : DnnNode(node_name, options) {
  // 更新配置
  this->declare_parameter<int>("feed_type", feed_type_);
  this->declare_parameter<std::string>("image", image_file_);
  this->declare_parameter<int>("dump_render_img", dump_render_img_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<float>("score_threshold", score_threshold_);
  this->declare_parameter<float>("iou_threshold", iou_threshold_);
  this->declare_parameter<int>("nms_top_k", nms_top_k_);
  this->declare_parameter<double>("y_offset", y_offset_);
  this->declare_parameter<std::string>("texts",
                                       texts);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name_);
  this->declare_parameter<std::string>("ros_img_sub_topic_name",
                                       ros_img_sub_topic_name_);
  this->declare_parameter<std::string>("ros_string_sub_topic_name",
                                       ros_string_sub_topic_name_);

  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<std::string>("image", image_file_);
  this->get_parameter<int>("dump_render_img", dump_render_img_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<float>("score_threshold", score_threshold_);
  this->get_parameter<float>("iou_threshold", iou_threshold_);
  this->get_parameter<int>("nms_top_k", nms_top_k_);
  this->get_parameter<double>("y_offset", y_offset_);
  this->get_parameter<std::string>("texts", texts);
  this->get_parameter<std::string>("ai_msg_pub_topic_name", ai_msg_pub_topic_name_);
  this->get_parameter<std::string>("ros_img_sub_topic_name", ros_img_sub_topic_name_);
  this->get_parameter<std::string>("ros_string_sub_topic_name", ros_string_sub_topic_name_);

  {
    std::stringstream ss;
    ss << "Parameter:"
       << "\n feed_type(0:local, 1:sub): " << feed_type_
       << "\n image: " << image_file_
       << "\n dump_render_img: " << dump_render_img_
       << "\n is_shared_mem_sub: " << is_shared_mem_sub_
       << "\n score_threshold: " << score_threshold_
       << "\n iou_threshold: " << iou_threshold_
       << "\n nms_top_k: " << nms_top_k_
       << "\n y_offset: " << y_offset_
       << "\n texts: " << texts
       << "\n ai_msg_pub_topic_name: " << ai_msg_pub_topic_name_
       << "\n ros_img_sub_topic_name: " << ros_img_sub_topic_name_
       << "\n ros_string_sub_topic_name: " << ros_string_sub_topic_name_;
    RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"), "%s", ss.str().c_str());
  }
  LoadVocabulary();
  LoadHomography();
  {
    std::stringstream ss;
    ss << "Parameter:"
       << "\n model_file_name: " << model_file_name_
       << "\n model_name: " << model_name_;
    RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"), "%s", ss.str().c_str());
  }

  // 使用基类接口初始化，加载模型
  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Init failed!");
    rclcpp::shutdown();
    return;
  }

  // 未指定模型名，从加载的模型中查询出模型名
  if (model_name_.empty()) {
    if (!GetModel()) {
      RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Get model fail.");
    } else {
      model_name_ = GetModel()->GetName();
      RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"), "Get model name: %s from load model.", model_name_.c_str());
    }
  }

  // 加载模型后查询模型输入分辨率
  if (GetModelInputSize(0, model_input_width_, model_input_height_) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Get model input size fail!");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"),
                "The model input width is %d and height is %d",
                model_input_width_,
                model_input_height_);
  }

  texts_.clear();
  split(texts, ',', texts_);

  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 1);
  num_class_ = tensor_properties.alignedShape.dimensionSize[1];

  // 创建AI消息的发布者
  RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"),
              "Create ai msg publisher with topic_name: %s",
              ai_msg_pub_topic_name_.c_str());
  msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
      ai_msg_pub_topic_name_, 10);

  if (0 == feed_type_) {
    // 本地图片回灌
    RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"),
                "Dnn node feed with local image: %s",
                image_file_.c_str());
    FeedFromLocal();
  } 
  else if (1 == feed_type_) {
    // 创建图片消息的订阅者
    RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"),
                "Dnn node feed with subscription");
    
    RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"),
      "Create string subscription with topic_name: %s",
      ros_string_sub_topic_name_.c_str());
    ros_string_subscription_ =
          this->create_subscription<std_msgs::msg::String>(
              ros_string_sub_topic_name_,
              10,
              std::bind(
                  &YoloWorldNode::RosStringProcess, this, std::placeholders::_1));
    if (is_shared_mem_sub_) {
#ifdef SHARED_MEM_ENABLED
      RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"),
                  "Create img hbmem_subscription with topic_name: %s",
                  sharedmem_img_topic_name_.c_str());
      sharedmem_img_subscription_ =
          this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              sharedmem_img_topic_name_,
              rclcpp::SensorDataQoS(),
              std::bind(&YoloWorldNode::SharedMemImgProcess,
                        this,
                        std::placeholders::_1));
#else
      RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Unsupport shared mem");
#endif
    } else {
      RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"),
                  "Create img subscription with topic_name: %s",
                  ros_img_sub_topic_name_.c_str());
      ros_img_subscription_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              ros_img_sub_topic_name_,
              10,
              std::bind(
                  &YoloWorldNode::RosImgProcess, this, std::placeholders::_1));
    }
  } else {
    RCLCPP_ERROR(
        rclcpp::get_logger("hobot_yolo_world"), "Invalid feed_type:%d", feed_type_);
    rclcpp::shutdown();
    return;
  }
}

YoloWorldNode::~YoloWorldNode() {
  std::unique_lock<std::mutex> lg(mtx_text_);
  cv_text_.notify_all();
  lg.unlock();
}

int YoloWorldNode::LoadVocabulary() {
  
  // Parsing config
  std::ifstream ifs(vocabulary_file_.c_str());
  if (!ifs) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                "Read vacabulary file [%s] fail! File is not exit!",
                vocabulary_file_.data());
    return -1;
  }
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document document;
  document.ParseStream(isw);
  if (document.HasParseError()) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                "Parsing vocabulary file %s failed! Please check file format.",
                vocabulary_file_.data());
    return -1;
  }

  // 遍历所有字段
  for (rapidjson::Value::ConstMemberIterator itr = document.MemberBegin(); itr != document.MemberEnd(); ++itr) {
      std::string name = itr->name.GetString();
      indice_.push_back(name);
      std::vector<float> value;
      // 处理不同类型的值
      if (itr->value.IsArray()) {
        if (itr->value.Size() != 512) {
          RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
            "Load Vocabulary Failed! Index [%s] embedding size %d != 512.",
            name.c_str(), itr->value.Size());
          return -1;
        }

        for (rapidjson::SizeType i = 0; i < itr->value.Size(); ++i) {
            value.push_back(itr->value[i].GetFloat());
        }

      } else {
        RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
            "Load Vocabulary Failed! Index [%s] not embedding",
            name.c_str());
        return -1;
      }
      embeddings_.push_back(value);
  }
  return 0;
}

int YoloWorldNode::LoadHomography() {
  
  std::string homography_file = "config/homography.json";
  // Parsing config
  std::ifstream ifs(homography_file.c_str());
  if (!ifs) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                "Read h file [%s] fail! File is not exit!",
                homography_file.data());
    return -1;
  }
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document document;
  document.ParseStream(isw);
  if (document.HasParseError()) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                "Parsing h file %s failed! Please check file format.",
                homography_file.data());
    return -1;
  }

  for (rapidjson::Value::ConstMemberIterator itr = document.MemberBegin(); itr != document.MemberEnd(); ++itr) {
    std::string name = itr->name.GetString();
    
    // 处理不同类型的值
    if (itr->value.IsArray()) {
      if (itr->value.Size() != 9) {
        RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
          "Load h file Failed! size %d != 9.",
          name.c_str(), itr->value.Size());
        return -1;
      }

      for (rapidjson::SizeType i = 0; i < itr->value.Size(); ++i) {
          homography_.push_back(itr->value[i].GetDouble());
      }

    }
  }

  return 0;
}


int YoloWorldNode::SetNodePara() {
  RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"), "Set node para.");
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  dnn_node_para_ptr_->model_task_type =
      hobot::dnn_node::ModelTaskType::ModelInferType;
  dnn_node_para_ptr_->task_num = task_num_;

  RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"),
              "model_file_name_: %s, task_num: %d",
              model_file_name_.data(),
              dnn_node_para_ptr_->task_num);

  return 0;
}

int YoloWorldNode::GetTextIndex(
      std::vector<std::string>& texts,
      std::vector<int>& indexs,
      std::vector<std::string>& target_texts) {
  indexs.clear();
  target_texts.clear();
  int index = -1;
  std::string target_text = "";
  for (auto text: texts) {
    for (int i = 0; i < indice_.size(); i++) {
      if (text == indice_[i]) {
        index = i;
        target_text = text;
        indexs.push_back(index);
        target_texts.push_back(target_text);
        break;
      }
    }
  }

  if (indexs.size() == 0) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
              "Vocabullary has no target texts.");
    return -1;
  }
  int num = num_class_ - indexs.size();
  for (int i = 0; i < num; i++) {
    indexs.push_back(index);
    target_texts.push_back(target_text);
  }
  return 0;
}

std::shared_ptr<DNNTensor> YoloWorldNode::GetEmbeddingsTensor(
    std::vector<int>& indexs,
    const std::vector<std::vector<float>>& embeddings,
    hbDNNTensorProperties &tensor_properties) {
  auto *mem = new hbSysMem;
  hbSysAllocCachedMem(mem, tensor_properties.alignedByteSize);
  //内存初始化
  memset(mem->virAddr, 0, tensor_properties.alignedByteSize);
  auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem->virAddr);

  int num_class = tensor_properties.alignedShape.dimensionSize[1];
  for (int i = 0; i < num_class; ++i) {
    auto *raw = hb_mem_addr + i * 2048;
    auto *src = reinterpret_cast<const uint8_t *>(embeddings[indexs[i]].data());
    memcpy(raw, src, 2048);
  }

  hbSysFlushMem(mem, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new DNNTensor;

  input_tensor->properties = tensor_properties;
  input_tensor->sysMem[0].virAddr = reinterpret_cast<void *>(mem->virAddr);
  input_tensor->sysMem[0].phyAddr = mem->phyAddr;
  input_tensor->sysMem[0].memSize = tensor_properties.alignedByteSize;
  return std::shared_ptr<DNNTensor>(
      input_tensor, [mem](DNNTensor *input_tensor) {
        // Release memory after deletion
        hbSysFreeMem(mem);
        delete mem;
        delete input_tensor;
      });
}

int YoloWorldNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput> &node_output) {
  if (!rclcpp::ok()) {
    return -1;
  }

  // 1. 记录后处理开始时间
  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  auto parser_output = std::dynamic_pointer_cast<YoloWorldOutput>(node_output);
  if (parser_output) {
    std::stringstream ss;
    ss << "Output from frame_id: " << parser_output->msg_header->frame_id
       << ", stamp: " << parser_output->msg_header->stamp.sec << "."
       << parser_output->msg_header->stamp.nanosec;
    RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"), "%s", ss.str().c_str());
  }

  // 校验算法输出是否有效
  if (node_output->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                 "Invalid node_output->output_tensors");
    return -1;
  }

  // 2. 模型后处理解析
  auto parser = std::make_shared<YoloOutputParser>();
  parser->SetScoreThreshold(score_threshold_);
  parser->SetIouThreshold(iou_threshold_);
  parser->SetTopkThreshold(nms_top_k_);

  auto det_result = std::make_shared<DnnParserResult>();
  parser->Parse(det_result, parser_output->output_tensors, parser_output->class_names);

  // 3. 创建用于发布的AI消息
  if (!msg_publisher_) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Invalid msg_publisher_");
    return -1;
  }
  ai_msgs::msg::PerceptionTargets::UniquePtr pub_data(
      new ai_msgs::msg::PerceptionTargets());
  // 3.1 发布检测AI消息
  RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"),
              "out box size: %d",
              det_result->perception.det.size());
  for (auto &rect : det_result->perception.det) {
    if (rect.bbox.xmin < 0) rect.bbox.xmin = 0;
    if (rect.bbox.ymin < 0) rect.bbox.ymin = 0;
    if (rect.bbox.xmax >= model_input_width_) {
      rect.bbox.xmax = model_input_width_ - 1;
    }
    if (rect.bbox.ymax >= model_input_height_) {
      rect.bbox.ymax = model_input_height_ - 1;
    }

    std::stringstream ss;
    ss << "det rect: " << rect.bbox.xmin << " " << rect.bbox.ymin << " "
       << rect.bbox.xmax << " " << rect.bbox.ymax
       << ", det type: " << rect.class_name << ", score:" << rect.score;
    RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"), "%s", ss.str().c_str());

    ai_msgs::msg::Roi roi;
    roi.set__type(rect.class_name);
    roi.rect.set__x_offset(rect.bbox.xmin);
    roi.rect.set__y_offset(rect.bbox.ymin);
    roi.rect.set__width(rect.bbox.xmax - rect.bbox.xmin);
    roi.rect.set__height(rect.bbox.ymax - rect.bbox.ymin);
    roi.set__confidence(rect.score);

    ai_msgs::msg::Target target;
    target.set__type(rect.class_name);
    target.rois.emplace_back(roi);
    pub_data->targets.emplace_back(std::move(target));
  }

  pub_data->header.set__stamp(parser_output->msg_header->stamp);
  pub_data->header.set__frame_id(parser_output->msg_header->frame_id);

  // 如果开启了渲染，本地渲染并存储图片
  if (dump_render_img_ && parser_output->tensor_image) {
    ImageUtils::Render(parser_output->tensor_image, pub_data);
  }

  if (parser_output->ratio != 1.0) {
    // 前处理有对图片进行resize，需要将坐标映射到对应的订阅图片分辨率
    for (auto &target : pub_data->targets) {
      for (auto &roi : target.rois) {
        roi.rect.x_offset *= parser_output->ratio;
        roi.rect.y_offset *= parser_output->ratio;
        roi.rect.width *= parser_output->ratio;
        roi.rect.height *= parser_output->ratio;
      }
    }
  }

  for (auto &target : pub_data->targets) {
    for (auto &roi : target.rois) {
      std::vector<double> x1 = {
          static_cast<double>(roi.rect.x_offset), 
          static_cast<double>(roi.rect.y_offset + roi.rect.height / 2), 1.0};
      std::vector<double> x2 = matrixMultiply(homography_, x1);
      x2[0] = x2[0] - 960;
      x2[1] = y_offset_ - x2[1];

      auto attribute = ai_msgs::msg::Attribute();
      attribute.set__type("x_cm");
      // millimeter to centimeter
      attribute.set__value(x2[0] / 10.0);
      target.attributes.emplace_back(attribute);

      attribute.set__type("y_cm");
      // millimeter to centimeter
      attribute.set__value(x2[1] / 10.0);
      target.attributes.emplace_back(attribute);
    }
  }

  // 填充perf性能统计信息
  // 前处理统计
  ai_msgs::msg::Perf perf_preprocess = std::move(parser_output->perf_preprocess);
  perf_preprocess.set__time_ms_duration(CalTimeMsDuration(
      perf_preprocess.stamp_start, perf_preprocess.stamp_end));

  // dnn node有输出统计信息
  if (node_output->rt_stat) {
    struct timespec time_now = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_now);

    // 推理统计
    ai_msgs::msg::Perf perf;
    perf.set__type(model_name_ + "_predict_infer");
    perf.stamp_start =
        ConvertToRosTime(node_output->rt_stat->infer_timespec_start);
    perf.stamp_end = ConvertToRosTime(node_output->rt_stat->infer_timespec_end);
    perf.set__time_ms_duration(node_output->rt_stat->infer_time_ms);
    pub_data->perfs.push_back(perf);

    // 后处理统计
    ai_msgs::msg::Perf perf_postprocess;
    perf_postprocess.set__type(model_name_ + "_postprocess");
    perf_postprocess.stamp_start = ConvertToRosTime(time_start);
    clock_gettime(CLOCK_REALTIME, &time_now);
    perf_postprocess.stamp_end = ConvertToRosTime(time_now);
    perf_postprocess.set__time_ms_duration(CalTimeMsDuration(
        perf_postprocess.stamp_start, perf_postprocess.stamp_end));
    pub_data->perfs.emplace_back(perf_postprocess);

    // 推理输出帧率统计
    pub_data->set__fps(round(node_output->rt_stat->output_fps));

    // 如果当前帧有更新统计信息，输出统计信息
    if (node_output->rt_stat->fps_updated) {
      RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"),
                  "Sub img fps: %.2f, Smart fps: %.2f, pre process time ms: %d, "
                  "infer time ms: %d, "
                  "post process time ms: %d",
                  node_output->rt_stat->input_fps,
                  node_output->rt_stat->output_fps,
                  static_cast<int>(perf_preprocess.time_ms_duration),
                  node_output->rt_stat->infer_time_ms,
                  static_cast<int>(perf_postprocess.time_ms_duration));
    }
  }

  // 发布AI消息
  msg_publisher_->publish(std::move(pub_data));
  return 0;
}

int YoloWorldNode::FeedFromLocal() {
  if (access(image_file_.c_str(), R_OK) == -1) {
    RCLCPP_ERROR(
        rclcpp::get_logger("hobot_yolo_world"), "Image: %s not exist!", image_file_.c_str());
    return -1;
  }

  auto dnn_output = std::make_shared<YoloWorldOutput>();
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_start.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_now.tv_nsec;

  // 1. 获取图片数据DNNTensor
  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 0);
  std::shared_ptr<DNNTensor> tensor_image = nullptr;
  tensor_image = hobot::dnn_node::ImageProc::GetBGRTensorFromBGR(image_file_,
      model_input_height_, model_input_width_, tensor_properties, dnn_output->ratio,
      hobot::dnn_node::ImageType::RGB, true, false, false);

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("ClipImageNode"),
                 "Get tensor fail with image: %s",
                 image_file_.c_str());
    return -1;
  }

  // 2. 使用embeddings数据创建DNNTensor
  model->GetInputTensorProperties(tensor_properties, 1);
  std::shared_ptr<DNNTensor> tensor_embeddings = nullptr;
  std::vector<int> indexs;
  std::vector<std::string> class_names;
  if (GetTextIndex(texts_, indexs, class_names) != 0) {
    return -1;
  } 
  tensor_embeddings = GetEmbeddingsTensor(indexs, embeddings_, tensor_properties);

  // 3. 存储上面两个DNNTensor
  // inputs将会作为模型的输入通过InferTask接口传入
  std::vector<std::shared_ptr<DNNTensor>> inputs;
  inputs.push_back(tensor_image);
  inputs.push_back(tensor_embeddings);
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->class_names = std::move(class_names);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id("feedback");
  if (dump_render_img_) {
    dnn_output->tensor_image = tensor_image;
  }

  // 4. 开始预测
  if (Run(inputs, dnn_output, true) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Run predict failed!");
    return -1;
  }

  return 0;
}

void YoloWorldNode::RosImgProcess(
    const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
  if (!img_msg) {
    RCLCPP_DEBUG(rclcpp::get_logger("hobot_yolo_world"), "Get img failed");
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  std::stringstream ss;
  ss << "Recved img encoding: " << img_msg->encoding
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step
     << ", frame_id: " << img_msg->header.frame_id
     << ", stamp: " << img_msg->header.stamp.sec << "_"
     << img_msg->header.stamp.nanosec
     << ", data size: " << img_msg->data.size();
  RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"), "%s", ss.str().c_str());

  auto dnn_output = std::make_shared<YoloWorldOutput>();
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_start.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_now.tv_nsec;

  // 1. 将图片处理成模型输入数据类型DNNTensor
  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 0);
  std::shared_ptr<DNNTensor> tensor_image = nullptr;
  if ("rgb8" == img_msg->encoding) {
    auto cv_img =
        cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(img_msg), "bgr8");
    tensor_image = hobot::dnn_node::ImageProc::GetBGRTensorFromBGRImg(
      cv_img->image,
      model_input_height_,
      model_input_width_,
      tensor_properties,
      dnn_output->ratio,
      hobot::dnn_node::ImageType::BGR
    );
  } else if ("bgr8" == img_msg->encoding) {
    auto cv_img =
        cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(img_msg), "bgr8");
    tensor_image = hobot::dnn_node::ImageProc::GetBGRTensorFromBGRImg(
      cv_img->image,
      model_input_height_,
      model_input_width_,
      tensor_properties,
      dnn_output->ratio,
      hobot::dnn_node::ImageType::RGB);
  } else if ("nv12" == img_msg->encoding) {  // nv12格式使用hobotcv resize
    cv::Mat bgr_mat;
    hobot::dnn_node::ImageProc::Nv12ToBGR(reinterpret_cast<const char *>(img_msg->data.data()), img_msg->height, img_msg->width, bgr_mat);
    tensor_image = hobot::dnn_node::ImageProc::GetBGRTensorFromBGRImg(
      bgr_mat,
      model_input_height_,
      model_input_width_,
      tensor_properties,
      dnn_output->ratio,
      hobot::dnn_node::ImageType::RGB);
  }

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Get Tensor fail");
    return;
  }

  // 2. 使用embeddings数据创建DNNTensor
  model->GetInputTensorProperties(tensor_properties, 1);
  std::shared_ptr<DNNTensor> tensor_embeddings = nullptr;
  std::vector<int> indexs;
  std::vector<std::string> class_names;

  std::unique_lock<std::mutex> lg(mtx_text_);
  cv_text_.wait(lg, [this]() { return !texts_.empty() || !rclcpp::ok(); });
  if (GetTextIndex(texts_, indexs, class_names) != 0) {
    return;
  } 
  tensor_embeddings = GetEmbeddingsTensor(indexs, embeddings_, tensor_properties);

  // 3. 存储上面两个DNNTensor
  // inputs将会作为模型的输入通过InferTask接口传入
  auto inputs = std::vector<std::shared_ptr<DNNTensor>>{tensor_image, tensor_embeddings};
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(img_msg->header.frame_id);
  dnn_output->msg_header->set__stamp(img_msg->header.stamp);
  dnn_output->class_names = std::move(class_names);
  
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  if (dump_render_img_) {
    dnn_output->tensor_image = tensor_image;
  }

  // 4. 开始预测
  int ret = Run(inputs, dnn_output, false);
  if (ret != 0 && ret != HB_DNN_TASK_NUM_EXCEED_LIMIT) {
    RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"), "Run predict failed!");
    return;
  }
}

#ifdef SHARED_MEM_ENABLED
void YoloWorldNode::SharedMemImgProcess(
    const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr img_msg) {
  if (!img_msg) {
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  std::stringstream ss;
  ss << "Recved img encoding: "
     << std::string(reinterpret_cast<const char *>(img_msg->encoding.data()))
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step << ", index: " << img_msg->index
     << ", stamp: " << img_msg->time_stamp.sec << "_"
     << img_msg->time_stamp.nanosec << ", data size: " << img_msg->data_size;
  RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"), "%s", ss.str().c_str());

  auto dnn_output = std::make_shared<YoloWorldOutput>();
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_start.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_now.tv_nsec;

  // 1. 将图片处理成模型输入数据类型DNNTensor
  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 0);
  std::shared_ptr<DNNTensor> tensor_image = nullptr;
  if ("nv12" ==
      std::string(reinterpret_cast<const char *>(img_msg->encoding.data()))) {
    cv::Mat bgr_mat;
    hobot::dnn_node::ImageProc::Nv12ToBGR(reinterpret_cast<const char *>(img_msg->data.data()), img_msg->height, img_msg->width, bgr_mat);
    tensor_image = hobot::dnn_node::ImageProc::GetBGRTensorFromBGRImg(
          bgr_mat,
          model_input_height_,
          model_input_width_,
          tensor_properties,
          dnn_output->ratio,
          hobot::dnn_node::ImageType::RGB);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                 "Unsupported img encoding: %s, only nv12 img encoding is "
                 "supported for shared mem.",
                 img_msg->encoding.data());
    return;
  }

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Get Tensor fail");
    return;
  }

  {
    auto stamp_start = ConvertToRosTime(time_now);
    struct timespec time_end = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_end);
    auto stamp_end = ConvertToRosTime(time_end);
    RCLCPP_DEBUG(rclcpp::get_logger("hobot_yolo_world"),
            "image preforcess time: %d", 
            CalTimeMsDuration(stamp_start, stamp_end));
  }

  // 2. 使用embeddings数据创建DNNTensor
  model->GetInputTensorProperties(tensor_properties, 1);
  std::shared_ptr<DNNTensor> tensor_embeddings = nullptr;
  std::vector<int> indexs;
  std::vector<std::string> class_names;
  if (GetTextIndex(texts_, indexs, class_names) != 0) {
    return;
  } 
  tensor_embeddings = GetEmbeddingsTensor(indexs, embeddings_, tensor_properties);

  // 3. 初始化输出
  auto inputs = std::vector<std::shared_ptr<DNNTensor>>{tensor_image, tensor_embeddings};
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->msg_header->set__stamp(img_msg->time_stamp);
  dnn_output->class_names = std::move(class_names);
  
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  if (dump_render_img_) {
    dnn_output->tensor_image = tensor_image;
  }

  // 4. 开始预测
  int ret = Run(inputs, dnn_output, false);
  if (ret != 0 && ret != HB_DNN_TASK_NUM_EXCEED_LIMIT) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Run predict failed!");
    return;
  }
}
#endif

void YoloWorldNode::RosStringProcess(
    const std_msgs::msg::String::ConstSharedPtr msg) {
  if (!msg) {
    RCLCPP_DEBUG(rclcpp::get_logger("hobot_yolo_world"), "Get string failed");
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  std::stringstream ss;
  ss << "Recved string data: " << msg->data;
  RCLCPP_INFO(rclcpp::get_logger("hobot_yolo_world"), "%s", ss.str().c_str());

  std::unique_lock<std::mutex> lg(mtx_text_);
  texts_.clear();
  split(msg->data, ',', texts_);
  cv_text_.notify_one();
  lg.unlock();
}
