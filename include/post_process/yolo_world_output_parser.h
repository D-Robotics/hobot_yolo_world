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

#include <cmath>
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
 
  int32_t PostProcessWithoutDecode(
      std::vector<std::shared_ptr<DNNTensor>> &tensors,
      std::vector<std::string>& class_names,
      Perception &perception);

  int32_t PostProcess(
      std::vector<std::shared_ptr<DNNTensor>> &tensors,
      std::vector<std::string>& class_names,
      Perception &perception);
  
  int32_t DecodeLayerNCHW(const int16_t* output_data,
                        std::vector<std::string> &class_names,
                        std::vector<Detection> &dets,
                        const int num_class,
                        const float* scale_data,
                        const int vaild_h,
                        const int vaild_w,
                        const int aligned_w);

  int32_t SetScoreThreshold(float score_threshold) {score_threshold_ = score_threshold; return 0;}
  int32_t SetIouThreshold(float iou_threshold) {iou_threshold_ = iou_threshold; return 0;}
  int32_t SetTopkThreshold(int nms_top_k) {nms_top_k_ = nms_top_k; return 0;}
  int32_t SetFilterX(int filterx) {filterx_ = filterx; return 0;}
  int32_t SetFilterY(int filtery) {filtery_ = filtery; return 0;}

 private:

  int32_t Filter(std::vector<Detection> &input,
                    std::vector<Detection> &result);
  int32_t CheckObject(std::vector<Detection> &input,
                      int i,
                      std::vector<float> areas,
                      std::vector<Detection> &dets_defore,
                      std::vector<float> areas_before);

  float score_threshold_ = 0.22;
  float iou_threshold_ = 0.5;
  int nms_top_k_ = 100;

  int input_shape = 640;
  int filterx_ = 0;
  int filtery_ = 0;

  std::vector<Detection> dets1_;
  std::vector<Detection> dets2_;
  std::vector<Detection> dets3_;
  std::vector<Detection> dets4_;
};

#endif  // YOLO_WORLD_OUTPUT_PARSER_H_
