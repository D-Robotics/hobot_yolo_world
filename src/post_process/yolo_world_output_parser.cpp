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

  int ret = -1;
  if (output_tensors.size() == 3) {
    ret = PostProcess(output_tensors, class_names, output->perception);
  } else if (output_tensors.size() == 2) {
    ret = PostProcessWithoutDecode(output_tensors, class_names, output->perception);
  } 
  
  if (ret != 0) {
    return ret;
  }

  return 0;
}

int YoloOutputParser::CheckObject(std::vector<Detection> &input,
                                int i,
                                std::vector<float> areas,
                                std::vector<Detection> &dets_defore,
                                std::vector<float> areas_before){
  for (size_t j = 0; j < dets_defore.size(); j++) {
    // intersection area
    float xx1 = std::max(input[i].bbox.xmin, dets_defore[j].bbox.xmin);
    float yy1 = std::max(input[i].bbox.ymin, dets_defore[j].bbox.ymin);
    float xx2 = std::min(input[i].bbox.xmax, dets_defore[j].bbox.xmax);
    float yy2 = std::min(input[i].bbox.ymax, dets_defore[j].bbox.ymax);
    if (xx2 > xx1 && yy2 > yy1) {
      float area_intersection = (xx2 - xx1) * (yy2 - yy1);
      float iou_ratio =
          area_intersection / (areas_before[j] + areas[i] - area_intersection);
      if (iou_ratio > iou_threshold_ && (input[i].class_name == dets_defore[j].class_name)) {
        return 1;
      }
    }
  }
  return 0;
}

int32_t YoloOutputParser::Filter(std::vector<Detection> &input,
         std::vector<Detection> &result) {

  std::vector<float> areas;
  areas.reserve(input.size());
  for (auto& det: input) {
    float width = det.bbox.xmax - det.bbox.xmin;
    float height = det.bbox.ymax - det.bbox.ymin;
    areas.push_back(width * height);
  }

  std::vector<float> areas_before1;
  areas_before1.reserve(dets1_.size());
  for (auto& det: dets1_) {
    float width = det.bbox.xmax - det.bbox.xmin;
    float height = det.bbox.ymax - det.bbox.ymin;
    areas_before1.push_back(width * height);
  }
  std::vector<float> areas_before2;
  areas_before2.reserve(dets2_.size());
  for (auto& det: dets2_) {
    float width = det.bbox.xmax - det.bbox.xmin;
    float height = det.bbox.ymax - det.bbox.ymin;
    areas_before2.push_back(width * height);
  }
  std::vector<float> areas_before3;
  areas_before3.reserve(dets3_.size());
  for (auto& det: dets3_) {
    float width = det.bbox.xmax - det.bbox.xmin;
    float height = det.bbox.ymax - det.bbox.ymin;
    areas_before3.push_back(width * height);
  }
  std::vector<float> areas_before4;
  areas_before4.reserve(dets4_.size());
  for (auto& det: dets4_) {
    float width = det.bbox.xmax - det.bbox.xmin;
    float height = det.bbox.ymax - det.bbox.ymin;
    areas_before4.push_back(width * height);
  }

  for (size_t i = 0; i < input.size(); i++) {
    int count = 0;
    count += CheckObject(input, i, areas, dets1_, areas_before1);
    count += CheckObject(input, i, areas, dets2_, areas_before2);
    count += CheckObject(input, i, areas, dets3_, areas_before3);
    count += CheckObject(input, i, areas, dets4_, areas_before4);

    if (count > (filterx_ - 1)) {
      result.push_back(input[i]);
    }
  }
  return 0;
}

int32_t YoloOutputParser::PostProcessWithoutDecode(
    std::vector<std::shared_ptr<DNNTensor>> &tensors,
    std::vector<std::string>& class_names,
    Perception &perception) {
  hbSysFlushMem(&(tensors[0]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  hbSysFlushMem(&(tensors[1]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *scores_data = reinterpret_cast<int16_t *>(tensors[0]->sysMem[0].virAddr);
  auto *boxes_data = reinterpret_cast<int16_t *>(tensors[1]->sysMem[0].virAddr);

  perception.type = Perception::DET;
  std::vector<Detection> dets;
  
  int num_pred = 0;
  int num_class = 0;
  int num_class_ailgned = 0;
  if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    num_pred = tensors[0]->properties.alignedShape.dimensionSize[1];
    num_class_ailgned = tensors[0]->properties.alignedShape.dimensionSize[2];
    num_class = tensors[0]->properties.validShape.dimensionSize[2];
  } else if (tensors[0]->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    num_pred = tensors[0]->properties.alignedShape.dimensionSize[3];
    num_class_ailgned = tensors[0]->properties.alignedShape.dimensionSize[1];
    num_class = tensors[0]->properties.validShape.dimensionSize[1];
  } else {
    num_pred = tensors[0]->properties.alignedShape.dimensionSize[2];
    num_class_ailgned = tensors[0]->properties.alignedShape.dimensionSize[3];
    num_class = tensors[0]->properties.validShape.dimensionSize[3];
  }

  for (int i = 0; i < num_pred; i++) {
    int16_t *score_data = scores_data + i * num_class_ailgned;
    int16_t *box_data = boxes_data + i * 8;
    float max_score = std::numeric_limits<float>::lowest(); // 初始最大值为最小可能值
    int max_index = -1;
    for (int k = 0; k < num_class; ++k) {
      float score = static_cast<float>(score_data[k]) * tensors[0]->properties.scale.scaleData[0];
      if (score > max_score) {
          max_score = score;
          max_index = k;
      }
    }
    if (max_score > score_threshold_) {
      float xmin = static_cast<float>(box_data[0]) * tensors[1]->properties.scale.scaleData[0];
      float ymin = static_cast<float>(box_data[1]) * tensors[1]->properties.scale.scaleData[0];
      float xmax = static_cast<float>(box_data[2]) * tensors[1]->properties.scale.scaleData[0];
      float ymax = static_cast<float>(box_data[3]) * tensors[1]->properties.scale.scaleData[0];
      Bbox bbox(xmin, ymin, xmax, ymax);
      dets.push_back(
          Detection(static_cast<int>(max_index),
                    max_score,
                    bbox,
                    class_names[max_index].c_str()));
    }
  }
  
  switch (filtery_) {
    default: nms(dets, iou_threshold_, nms_top_k_, perception.det, true); return 0;
    case 5: swap(dets3_, dets4_);
    case 4: swap(dets2_, dets3_);
    case 3: swap(dets1_, dets2_);
  }

  std::vector<Detection> tmpdets;
  nms(dets, iou_threshold_, nms_top_k_, tmpdets, true);

  dets1_.clear();
  for (auto &det: tmpdets) {
    dets1_.push_back(det);
  }
  Filter(tmpdets, perception.det);
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

  nms(dets, iou_threshold_, nms_top_k_, perception.det, true);
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
      for (int k = 0; k < num_class - 1; k++) {
        float data = score_data[k * stride];
        if (data > max_score) {
            max_score = data;
            max_index = k;
        }
      }

      if (max_score > score_threshold_x) {
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