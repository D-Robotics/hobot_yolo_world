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

#ifndef YOLO_WORLD_OUTPUT_PARSER_H_
#define YOLO_WORLD_OUTPUT_PARSER_H_

#include <string>

#include <rclcpp/rclcpp.hpp>
#include "dnn_node/dnn_node_data.h"
#include "dnn_node/util/output_parser/perception_common.h"
#include "dnn_node/util/output_parser/detection/nms.h"


using hobot::dnn_node::output_parser::Bbox;
using hobot::dnn_node::output_parser::Detection;
using hobot::dnn_node::output_parser::DnnParserResult;
using hobot::dnn_node::output_parser::Perception;
using hobot::dnn_node::DNNTensor;

class YoloOutputParser {
 public:
  YoloOutputParser() {}
  ~YoloOutputParser() {}

  int32_t Parse(
      std::shared_ptr<DnnParserResult> &output,
      std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
      std::vector<std::string>& class_names);
 
  int32_t PostProcess(
      std::vector<std::shared_ptr<DNNTensor>> &tensors,
      std::vector<std::string>& class_names,
      Perception &perception);
  
  int32_t SetScoreThreshold(float score_threshold) {score_threshold_ = score_threshold;}
  int32_t SetIouThreshold(float iou_threshold) {iou_threshold_ = iou_threshold;}
  int32_t SetTopkThreshold(int nms_top_k) {nms_top_k_ = nms_top_k;}

 private:

  float score_threshold_ = 0.05;
  float iou_threshold_ = 0.45;
  int nms_top_k_ = 50;

};

#endif  // YOLO_WORLD_OUTPUT_PARSER_H_
