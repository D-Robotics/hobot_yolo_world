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

#include "hobot_cv/hobotcv_imgproc.h"
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

// 使用hobotcv resize nv12格式图片，固定图片宽高比
int ResizeNV12Img(const char *in_img_data,
                  const int &in_img_height,
                  const int &in_img_width,
                  int &resized_img_height,
                  int &resized_img_width,
                  const int &scaled_img_height,
                  const int &scaled_img_width,
                  cv::Mat &out_img,
                  float &ratio) {
  cv::Mat src(
      in_img_height * 3 / 2, in_img_width, CV_8UC1, (void *)(in_img_data));
  float ratio_w =
      static_cast<float>(in_img_width) / static_cast<float>(scaled_img_width);
  float ratio_h =
      static_cast<float>(in_img_height) / static_cast<float>(scaled_img_height);
  float dst_ratio = std::max(ratio_w, ratio_h);
  int resized_width, resized_height;
  if (dst_ratio == ratio_w) {
    resized_width = scaled_img_width;
    resized_height = static_cast<float>(in_img_height) / dst_ratio;
  } else if (dst_ratio == ratio_h) {
    resized_width = static_cast<float>(in_img_width) / dst_ratio;
    resized_height = scaled_img_height;
  }
  // hobot_cv要求输出宽度为16的倍数
  int remain = resized_width % 16;
  if (remain != 0) {
    //向下取16倍数，重新计算缩放系数
    resized_width -= remain;
    dst_ratio = static_cast<float>(in_img_width) / resized_width;
    resized_height = static_cast<float>(in_img_height) / dst_ratio;
  }
  //高度向下取偶数
  resized_height =
      resized_height % 2 == 0 ? resized_height : resized_height - 1;
  ratio = dst_ratio;

  resized_img_height = resized_height;
  resized_img_width = resized_width;

  return hobot_cv::hobotcv_resize(
      src, in_img_height, in_img_width, out_img, resized_height, resized_width);
}

YoloWorldNode::YoloWorldNode(const std::string &node_name,
                               const NodeOptions &options)
    : DnnNode(node_name, options) {
  // 更新配置
  this->declare_parameter<std::string>("model_file_name", model_file_name_);
  this->declare_parameter<std::string>("vocabulary_file_name", vocabulary_file_name_);
  this->declare_parameter<int>("feed_type", feed_type_);
  this->declare_parameter<std::string>("image", image_file_);
  this->declare_parameter<int>("dump_render_img", dump_render_img_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<float>("score_threshold", score_threshold_);
  this->declare_parameter<float>("iou_threshold", iou_threshold_);
  this->declare_parameter<int>("nms_top_k", nms_top_k_);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name_);
  this->declare_parameter<std::string>("ros_img_sub_topic_name",
                                       ros_img_sub_topic_name_);

  this->get_parameter<std::string>("model_file_name", model_file_name_);
  this->get_parameter<std::string>("vocabulary_file_name", vocabulary_file_name_);
  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<std::string>("image", image_file_);
  this->get_parameter<int>("dump_render_img", dump_render_img_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<float>("score_threshold", score_threshold_);
  this->get_parameter<float>("iou_threshold", iou_threshold_);
  this->get_parameter<int>("nms_top_k", nms_top_k_);
  this->get_parameter<std::string>("ai_msg_pub_topic_name", ai_msg_pub_topic_name_);
  this->get_parameter<std::string>("ros_img_sub_topic_name", ros_img_sub_topic_name_);

  {
    std::stringstream ss;
    ss << "Parameter:"
       << "\n model_file_name: " << model_file_name_
       << "\n vocabulary_file_name: " << vocabulary_file_name_
       << "\n feed_type(0:local, 1:sub): " << feed_type_
       << "\n image: " << image_file_
       << "\n dump_render_img: " << dump_render_img_
       << "\n is_shared_mem_sub: " << is_shared_mem_sub_
       << "\n score_threshold: " << score_threshold_
       << "\n iou_threshold: " << iou_threshold_
       << "\n nms_top_k: " << nms_top_k_
       << "\n ai_msg_pub_topic_name: " << ai_msg_pub_topic_name_
       << "\n ros_img_sub_topic_name: " << ros_img_sub_topic_name_;
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

  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetOutputTensorProperties(tensor_properties, 0);
  num_class_ = tensor_properties.alignedShape.dimensionSize[3];

  model->GetInputTensorProperties(tensor_properties, 0);
  is_nv12_ = tensor_properties.tensorType == HB_DNN_IMG_TYPE_NV12 ? true : false;

  if (LoadVocabulary() != 0) {
    return;
  }

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

YoloWorldNode::~YoloWorldNode() {}

int YoloWorldNode::LoadVocabulary() {
  
  // Parsing config
  std::ifstream ifs(vocabulary_file_name_.c_str());
  if (!ifs) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                "Read vocabulary file [%s] fail! File is not exit!",
                vocabulary_file_name_.data());
    return -1;
  }
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document document;
  document.ParseStream(isw);
  if (document.HasParseError()) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                "Parsing vocabulary file %s failed! Please check file format.",
                vocabulary_file_name_.data());
    return -1;
  }

  // 检查是否为数组
  if (!document.IsArray()) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"),
                "Vocabulary file format error! Expected an array.");
    return -1;
  }

  // 遍历数组并提取字符串
  
  int count = 0;
  for (const auto& item : document.GetArray()) {
  
    if (item.IsArray()) {
      for (const auto& sub_item : item.GetArray()) {
        if (sub_item.IsString()) {
          class_names_.emplace_back(sub_item.GetString());
          break;
        }
      }
    }
    if (item.IsString()) {
      class_names_.emplace_back(item.GetString());
    }
    count++;
  }

  if (class_names_.size() != num_class_) {
    RCLCPP_WARN(rclcpp::get_logger("hobot_yolo_world"),
              "Vocabulary num[%d] != Num Class[%d] !", class_names_.size(), num_class_);
    int num = class_names_.size();
    for (int i = 0; i < num_class_ - num; i++) {
      std::string name = "None"; 
      class_names_.emplace_back(name);
    }
    return 0;
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
  parser->Parse(det_result, parser_output->output_tensors, class_names_);

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
    ImageUtils::Render(parser_output->tensor_image, pub_data, parser_output->resized_h, parser_output->resized_w);
  }
  if (dump_render_img_ && parser_output->pyramid) {
    ImageUtils::Render(parser_output->pyramid, pub_data, parser_output->resized_h, parser_output->resized_w);
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

  if (is_nv12_) {

    // 1. 将图片处理成模型输入数据类型DNNInput
    // 使用图片生成pym，NV12PyramidInput为DNNInput的子类
    std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;
    // bgr img，支持将图片resize到模型输入size
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromBGR(
        image_file_,
        dnn_output->img_h,
        dnn_output->img_w,
        dnn_output->resized_h, 
        dnn_output->resized_w, 
        model_input_height_, 
        model_input_width_);
    if (!pyramid) {
      RCLCPP_ERROR(this->get_logger(),
                  "Get Nv12 pym fail with image: %s",
                  image_file_.c_str());
      return -1;
    }

    // 2. 存储上面两个DNNTensor
    // inputs将会作为模型的输入通过InferTask接口传入
    auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};
    clock_gettime(CLOCK_REALTIME, &time_now);
    dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
    dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
    dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
    dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
    dnn_output->msg_header->set__frame_id("feedback");
    if (dump_render_img_) {
      dnn_output->pyramid = pyramid;
    }

    // 3. 开始预测
    if (Run(inputs, dnn_output, nullptr) != 0) {
      RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Run predict failed!");
      return -1;
    }
    return 0;
  } 
    
  // 1. 获取图片数据DNNTensor
  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 0);
  std::shared_ptr<DNNTensor> tensor_image = nullptr;
  tensor_image = hobot::dnn_node::ImageProc::GetBGRTensorFromBGR(image_file_,
      model_input_height_, model_input_width_, tensor_properties, dnn_output->ratio,
      hobot::dnn_node::ImageType::RGB, true, false, true);

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("ClipImageNode"),
                "Get tensor fail with image: %s",
                image_file_.c_str());
    return -1;
  }

  cv::Mat bgr_mat = cv::imread(image_file_, cv::IMREAD_COLOR);
  int original_img_width = bgr_mat.cols;
  int original_img_height = bgr_mat.rows;
  dnn_output->resized_w = static_cast<int>(static_cast<float>(original_img_width) / dnn_output->ratio);
  dnn_output->resized_h = static_cast<int>(static_cast<float>(original_img_height) / dnn_output->ratio);

  // 2. 存储上面两个DNNTensor
  // inputs将会作为模型的输入通过InferTask接口传入
  std::vector<std::shared_ptr<DNNTensor>> inputs;
  inputs.push_back(tensor_image);
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id("feedback");
  if (dump_render_img_) {
    dnn_output->tensor_image = tensor_image;
  }

  // 3. 开始预测
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
    dnn_output->resized_h = static_cast<int>(static_cast<float>(cv_img->image.rows) / dnn_output->ratio);
    dnn_output->resized_w = static_cast<int>(static_cast<float>(cv_img->image.cols) / dnn_output->ratio);
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
    dnn_output->resized_h = static_cast<int>(static_cast<float>(cv_img->image.rows) / dnn_output->ratio);
    dnn_output->resized_w = static_cast<int>(static_cast<float>(cv_img->image.cols) / dnn_output->ratio);
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
    dnn_output->resized_h = static_cast<int>(static_cast<float>(bgr_mat.rows) / dnn_output->ratio);
    dnn_output->resized_w = static_cast<int>(static_cast<float>(bgr_mat.cols) / dnn_output->ratio);
  }

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Get Tensor fail");
    return;
  }

  // 2. 存储DNNTensor
  // inputs将会作为模型的输入通过InferTask接口传入
  auto inputs = std::vector<std::shared_ptr<DNNTensor>>{tensor_image};
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(img_msg->header.frame_id);
  dnn_output->msg_header->set__stamp(img_msg->header.stamp);
  
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  if (dump_render_img_) {
    dnn_output->tensor_image = tensor_image;
  }

  // 3. 开始预测
  int ret = Run(inputs, dnn_output, false);
  if (ret != 0 && ret != HB_DNN_TASK_NUM_EXCEED_LIMIT ) {
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

  if (is_nv12_) {
    // 1. 将图片处理成模型输入数据类型DNNInput
    // 使用图片生成pym，NV12PyramidInput为DNNInput的子类
    std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;
    if ("nv12" ==
        std::string(reinterpret_cast<const char *>(img_msg->encoding.data()))) {
      if (img_msg->height != static_cast<uint32_t>(model_input_height_) ||
          img_msg->width != static_cast<uint32_t>(model_input_width_)) {
        // 需要做resize处理
        cv::Mat out_img;
        if (ResizeNV12Img(reinterpret_cast<const char *>(img_msg->data.data()),
                          img_msg->height,
                          img_msg->width,
                          dnn_output->resized_h,
                          dnn_output->resized_w,
                          model_input_height_,
                          model_input_width_,
                          out_img,
                          dnn_output->ratio) < 0) {
          RCLCPP_ERROR(rclcpp::get_logger("dnn_node_example"),
                      "Resize nv12 img fail!");
          return;
        }

        uint32_t out_img_width = out_img.cols;
        uint32_t out_img_height = out_img.rows * 2 / 3;
        pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char *>(out_img.data),
            out_img_height,
            out_img_width,
            model_input_height_,
            model_input_width_);
      } else {
        
        int image_height = img_msg->height;
        int image_width = img_msg->width;
        dnn_output->resized_h = std::min(image_height, model_input_height_);
        dnn_output->resized_w = std::min(image_width, model_input_width_);

        //不需要进行resize
        pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char *>(img_msg->data.data()),
            img_msg->height,
            img_msg->width,
            model_input_height_,
            model_input_width_);
      }
    } else {
      RCLCPP_ERROR(this->get_logger(),
                  "Unsupported img encoding: %s, only nv12 img encoding is "
                  "supported for shared mem.",
                  img_msg->encoding.data());
      return;
    }

    // 初始化输出
    dnn_output->img_w = img_msg->width;
    dnn_output->img_h = img_msg->height;

    // 2. 初始化输出
    auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};
    dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
    dnn_output->msg_header->set__frame_id(std::to_string(img_msg->index));
    dnn_output->msg_header->set__stamp(img_msg->time_stamp);
  
    clock_gettime(CLOCK_REALTIME, &time_now);
    dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
    dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
    dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
    if (dump_render_img_) {
      dnn_output->pyramid = pyramid;
    }

    // 3. 开始预测
    int ret = Run(inputs, dnn_output, nullptr, false);
    if (ret != 0 && ret != HB_DNN_TASK_NUM_EXCEED_LIMIT) {
      RCLCPP_ERROR(this->get_logger(), "Run predict failed!");
      return;
    }
    return;
  } 

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
          hobot::dnn_node::ImageType::RGB,
          true,
          false,
          true);
    dnn_output->resized_h = static_cast<int>(static_cast<float>(bgr_mat.rows) / dnn_output->ratio);
    dnn_output->resized_w = static_cast<int>(static_cast<float>(bgr_mat.cols) / dnn_output->ratio);
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

  // 2. 初始化输出
  auto inputs = std::vector<std::shared_ptr<DNNTensor>>{tensor_image};
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->msg_header->set__stamp(img_msg->time_stamp);
  
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  if (dump_render_img_) {
    dnn_output->tensor_image = tensor_image;
  }

  // 3. 开始预测
  int ret = Run(inputs, dnn_output, false);
  if (ret != 0 && ret != HB_DNN_TASK_NUM_EXCEED_LIMIT ) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_yolo_world"), "Run predict failed!");
    return;
  }
}
#endif
