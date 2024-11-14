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

// 计算直线的斜率和截距（通过两个点）
void lineEquation(const Point& p1, const Point& p2, float& k, float& b) {

    if ((p2.x - p1.x) == 0) {
      k = 0;
      b = p1.x;
      return;
    }
    k = (p2.y - p1.y) / (p2.x - p1.x);
    b = p1.y - k * p1.x;
}

// 判断检测框的点是否在梯形边界内
bool isPointInTrapezoid(const Point& point, float y_min, float y_max, double k1, double b1, double k2, double b2) {
    // 检查点的 y 值是否在上下边范围内
    if (point.y < y_min || point.y > y_max) {
        return false;
    }

    if (k1 == 0 && point.x < b1) {
      return false;
    }
    if (k2 == 0 && point.x > b2) {
      return false;
    }
    if (k1 != 0 && k2 != 0) {
      // 检查点是否在左边和右边的斜线内
      float y_left = k1 * point.x + b1;
      float y_right = k2 * point.x + b2;
      return (point.y >= y_left && point.y >= y_right);
    }
    return true;
}

// 判断检测框的四个顶点是否都在梯形内
bool isBoxInTrapezoid(const std::vector<Point>& trapezoid, const Detection& det) {
    
    // float y_min = det.bbox.ymin;
    // float y_max = det.bbox.ymax;
    float y_min = std::min(trapezoid[0].y, trapezoid[1].y);
    float y_max = std::max(trapezoid[2].y, trapezoid[3].y);

    // 计算梯形左边和右边的直线方程
    float k1, b1, k2, b2;

    lineEquation(trapezoid[0], trapezoid[3], k1, b1);  // 左边斜线
    lineEquation(trapezoid[1], trapezoid[2], k2, b2);  // 右边斜线

    Point point;
    // point.x = det.bbox.xmin;
    // point.y = det.bbox.ymin;
    // if (!isPointInTrapezoid(point, y_min, y_max, k1, b1, k2, b2)) {
    //   return false;
    // }
    // point.x = det.bbox.xmax;
    // point.y = det.bbox.ymin;
    // if (!isPointInTrapezoid(point, y_min, y_max, k1, b1, k2, b2)) {
    //   return false;
    // }
    point.x = (det.bbox.xmax + det.bbox.xmin) / 2;
    point.y = det.bbox.ymin;
    if (!isPointInTrapezoid(point, y_min, y_max, k1, b1, k2, b2)) {
      return false;
    }

    point.x = det.bbox.xmin;
    point.y = det.bbox.ymax;
    if (!isPointInTrapezoid(point, y_min, y_max, k1, b1, k2, b2)) {
      return false;
    }
    point.x = det.bbox.xmax;
    point.y = det.bbox.ymax;
    if (!isPointInTrapezoid(point, y_min, y_max, k1, b1, k2, b2)) {
      return false;
    }
    return true;
}

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
      Detection det = Detection(static_cast<int>(max_index),
                      max_score,
                      bbox,
                      class_names[max_index].c_str());
      if (roi_ && !isBoxInTrapezoid(points_, det)) {
        break;
      }
      if (class_mode_ == 1 && class_names[max_index] != "skein") {
        dets.push_back(det);
      } else if (class_mode_ == 0) {
        dets.push_back(det);
      }
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
    perception.det.push_back(
          Detection(det.id,
                    det.score,
                    det.bbox,
                    class_names[det.id + num_class_].c_str()));
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
  float stride_x = input_shape_w_ / vaild_w;
  float stride_y = input_shape_h_ / vaild_h;
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


// 将Detection写入标准VOC格式XML
int YoloOutputParser::WriteVOCXML(const std::string &filename, const std::string &imagePath, int imageWidth, int imageHeight, int depth, const std::vector<Detection> &detections) {
  XMLDocument doc;

  // 根节点annotation
  XMLElement *annotation = doc.NewElement("annotation");
  doc.InsertFirstChild(annotation);

  // folder
  XMLElement *folder = doc.NewElement("folder");
  folder->SetText("VOCImages");
  annotation->InsertEndChild(folder);

  // filename
  XMLElement *filenameElem = doc.NewElement("filename");
  filenameElem->SetText(imagePath.c_str());
  annotation->InsertEndChild(filenameElem);

  // size节点
  XMLElement *sizeElem = doc.NewElement("size");
  XMLElement *widthElem = doc.NewElement("width");
  widthElem->SetText(imageWidth);
  XMLElement *heightElem = doc.NewElement("height");
  heightElem->SetText(imageHeight);
  XMLElement *depthElem = doc.NewElement("depth");
  depthElem->SetText(depth);
  sizeElem->InsertEndChild(widthElem);
  sizeElem->InsertEndChild(heightElem);
  sizeElem->InsertEndChild(depthElem);
  annotation->InsertEndChild(sizeElem);

  // 添加object节点
  for (const auto &det : detections) {
      XMLElement *objectElem = doc.NewElement("object");

      XMLElement *nameElem = doc.NewElement("name");
      nameElem->SetText(det.class_name);
      objectElem->InsertEndChild(nameElem);

      XMLElement *poseElem = doc.NewElement("pose");
      poseElem->SetText("Unspecified");
      objectElem->InsertEndChild(poseElem);

      XMLElement *truncatedElem = doc.NewElement("truncated");
      truncatedElem->SetText(0);
      objectElem->InsertEndChild(truncatedElem);

      XMLElement *difficultElem = doc.NewElement("difficult");
      difficultElem->SetText(0);
      objectElem->InsertEndChild(difficultElem);

      // bndbox节点
      XMLElement *bndboxElem = doc.NewElement("bndbox");

      XMLElement *xminElem = doc.NewElement("xmin");
      xminElem->SetText(det.bbox.xmin);
      XMLElement *yminElem = doc.NewElement("ymin");
      yminElem->SetText(det.bbox.ymin);
      XMLElement *xmaxElem = doc.NewElement("xmax");
      xmaxElem->SetText(det.bbox.xmax);
      XMLElement *ymaxElem = doc.NewElement("ymax");
      ymaxElem->SetText(det.bbox.ymax);

      // 添加置信度score节点
      XMLElement *scoreElem = doc.NewElement("score");
      scoreElem->SetText(det.score);  // 假设confidence是float类型
      objectElem->InsertEndChild(scoreElem);

      bndboxElem->InsertEndChild(xminElem);
      bndboxElem->InsertEndChild(yminElem);
      bndboxElem->InsertEndChild(xmaxElem);
      bndboxElem->InsertEndChild(ymaxElem);
      objectElem->InsertEndChild(bndboxElem);

      annotation->InsertEndChild(objectElem);
  }

  // 保存文件
  XMLError eResult = doc.SaveFile(filename.c_str());
  if (eResult != XML_SUCCESS) {
      return -1;
  }
  return 0;
}