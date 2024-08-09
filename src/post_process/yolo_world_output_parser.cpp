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

#include "include/post_process/yolo_world_output_parser.h"

int32_t YoloOutputParser::Parse(
    std::shared_ptr<DnnParserResult> &output,
    std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
    std::vector<std::string>& class_names) {
    
  if (!output) {
    output = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(output_tensors, class_names, output->perception);
  if (ret != 0) {
    return ret;
  }

  return 0;
}

int32_t YoloOutputParser::PostProcess(
    std::vector<std::shared_ptr<DNNTensor>> &tensors,
    std::vector<std::string>& class_names,
    Perception &perception) {
  hbSysFlushMem(&(tensors[0]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  hbSysFlushMem(&(tensors[1]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *scores_data = reinterpret_cast<float *>(tensors[0]->sysMem[0].virAddr);
  auto *boxes_data = reinterpret_cast<float *>(tensors[1]->sysMem[0].virAddr);

  perception.type = Perception::DET;
  std::vector<Detection> dets;
  
  int num_pred = 0;
  int num_class = 0;
  if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    num_pred = tensors[0]->properties.alignedShape.dimensionSize[1];
    num_class = tensors[0]->properties.alignedShape.dimensionSize[2];
  } else if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    num_pred = tensors[0]->properties.alignedShape.dimensionSize[3];
    num_class = tensors[0]->properties.alignedShape.dimensionSize[1];
  }

  for (int i = 0; i < num_pred; i++) {
    float *score_data = scores_data + i * num_class;
    float *box_data = boxes_data + i * 4;
    float max_score = std::numeric_limits<float>::lowest(); // 初始最大值为最小可能值
    int max_index = -1;
    for (int k = 0; k < num_class; ++k) {
      if (score_data[k] > max_score) {
          max_score = score_data[k];
          max_index = k;
      }
    }
    if (max_score > score_threshold_) {
      float xmin = box_data[0];
      float ymin = box_data[1];
      float xmax = box_data[2];
      float ymax = box_data[3];
      Bbox bbox(xmin, ymin, xmax, ymax);
      dets.push_back(
          Detection(static_cast<int>(max_index),
                    max_score,
                    bbox,
                    class_names[max_index].c_str()));
    }
  }
  nms(dets, iou_threshold_, nms_top_k_, perception.det, false);
  return 0;
}