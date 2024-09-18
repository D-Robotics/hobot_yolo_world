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

  perception.type = Perception::DET;
  std::vector<Detection> dets;

  int num_class = 0;
  if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    num_class = tensors[0]->properties.alignedShape.dimensionSize[1];
  } else if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    num_class = tensors[0]->properties.alignedShape.dimensionSize[3];
  }

  for (auto &tensor: tensors) {
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    auto *output = reinterpret_cast<int16_t *>(tensor->sysMem[0].virAddr);
    DecodeLayerNCHW(output, class_names, dets, num_class - 4, 
                    tensor->properties.scale.scaleData, 
                    tensor->properties.validShape.dimensionSize[2], 
                    tensor->properties.validShape.dimensionSize[3], 
                    tensor->properties.alignedShape.dimensionSize[3]);
  }

  nms(dets, iou_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

float Sigmoid(float x) {
    return static_cast<float>(1 / (1 + exp(-static_cast<double>(x))));
}

float InverseSigmoid(float y) {
    if (y <= 0 || y >= 1) {
        throw std::invalid_argument("y must be in the open interval (0, 1)");
    }
    return static_cast<float>(-std::log(1 / static_cast<double>(y) - 1));
}

int32_t YoloOutputParser::DecodeLayerNCHW(const int16_t* output_data,
                                          std::vector<std::string> &class_names,
                                          std::vector<Detection> &dets,
                                          const int num_class,
                                          const float* scale_data,
                                          const int vaild_h,
                                          const int vaild_w,
                                          const int aligned_w) {
  int stride = vaild_h * aligned_w;
  float stride_x = input_shape / vaild_w;
  float stride_y = input_shape / vaild_h;
  float score_threshold_x = InverseSigmoid(score_threshold_) / scale_data[0];

  for (int h = 0; h < vaild_h; h++) {
    for (int w = 0; w < vaild_w; w++) {

      int16_t *score_data = const_cast<int16_t*>(output_data) + h * aligned_w + w;
      int16_t *box_data = const_cast<int16_t*>(output_data) + num_class * stride + h * aligned_w + w;
      
      float max_score = std::numeric_limits<float>::lowest(); // 初始最大值为最小可能值
      int max_index = -1;
      
      // for (int k = 0; k < num_class - 1; k++) {
      //   float score = Sigmoid(static_cast<float>(score_data[k * stride] * scale_data[0]));
      //   if (score > max_score) {
      //       max_score = score;
      //       max_index = k;
      //   }
      // }
      for (int k = 0; k < num_class - 1; k++) {
        float data = score_data[k * stride];
        if (data > max_score) {
            max_score = data;
            max_index = k;
        }
      }

      if (max_score > score_threshold_x) {
      // if (max_score > score_threshold_) {
        max_score = Sigmoid(max_score * scale_data[0]);

        float x = (0.5 + static_cast<float>(w)) * stride_x;
        float y = (0.5 + static_cast<float>(h)) * stride_y;

        float xmin = x - static_cast<float>(box_data[0]) * scale_data[0] * stride_x;
        float ymin = y - static_cast<float>(box_data[stride]) * scale_data[0] * stride_y;
        float xmax = x + static_cast<float>(box_data[2 * stride]) * scale_data[0] * stride_x;
        float ymax = y + static_cast<float>(box_data[3 * stride]) * scale_data[0] * stride_y;
        Bbox bbox(xmin, ymin, xmax, ymax);
        dets.push_back(
            Detection(static_cast<int>(max_index),
                      max_score,
                      bbox,
                      class_names[max_index].c_str()));
      }
    }
  }

  return 0;
}