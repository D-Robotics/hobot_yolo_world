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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dnn/hb_sys.h"

#include "include/image_utils.h"

int ImageUtils::Render(
    const std::shared_ptr<hobot::dnn_node::NV12PyramidInput> &pyramid,
    const ai_msgs::msg::PerceptionTargets::UniquePtr &ai_msg,
    const int img_h,
    const int img_w) {
  if (!pyramid || !ai_msg) return -1;

  char *y_img = reinterpret_cast<char *>(pyramid->y_vir_addr);
  char *uv_img = reinterpret_cast<char *>(pyramid->uv_vir_addr);
  auto height = pyramid->height;
  auto width = pyramid->y_stride;
  auto img_y_size = height * width;
  auto img_uv_size = img_y_size / 2;
  char *buf = new char[img_y_size + img_uv_size];
  memcpy(buf, y_img, img_y_size);
  memcpy(buf + img_y_size, uv_img, img_uv_size);
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, buf);
  cv::Mat tmp;
  cv::cvtColor(nv12, tmp, CV_YUV2BGR_NV12);
  cv::Mat mat = tmp(cv::Rect(0, 0, img_w, img_h));
  delete[] buf;

  RCLCPP_INFO(rclcpp::get_logger("ImageUtils"),
              "target size: %d",
              ai_msg->targets.size());
  for (size_t idx = 0; idx < ai_msg->targets.size(); idx++) {
    const auto &target = ai_msg->targets.at(idx);
    RCLCPP_INFO(rclcpp::get_logger("ImageUtils"),
                "target type: %s, rois.size: %d",
                target.type.c_str(),
                target.rois.size());
    auto &color = colors[idx % colors.size()];
    for (const auto &roi : target.rois) {
      RCLCPP_INFO(
          rclcpp::get_logger("ImageUtils"),
          "roi.type: %s, x_offset: %d y_offset: %d width: %d height: %d",
          roi.type.c_str(),
          roi.rect.x_offset,
          roi.rect.y_offset,
          roi.rect.width,
          roi.rect.height);
      cv::rectangle(mat,
                    cv::Point(roi.rect.x_offset, roi.rect.y_offset),
                    cv::Point(roi.rect.x_offset + roi.rect.width,
                              roi.rect.y_offset + roi.rect.height),
                    color,
                    3);
      std::string roi_type = target.type;
      if (!roi.type.empty()) {
        roi_type = roi.type + ": " + std::to_string(roi.confidence);
      }
      if (!roi_type.empty()) {
        cv::putText(mat,
                    roi_type,
                    cv::Point2f(roi.rect.x_offset, roi.rect.y_offset - 10),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1.5);
      }
    }

    for (const auto &lmk : target.points) {
      for (const auto &pt : lmk.point) {
        cv::circle(mat, cv::Point(pt.x, pt.y), 3, color, 3);
      }
    }

    if (!target.captures.empty()) {
      float alpha_f = 0.5;
      for (auto capture: target.captures) {
        int parsing_width = capture.img.width;
        int parsing_height = capture.img.height;
        cv::Mat parsing_img(parsing_height, parsing_width, CV_8UC3);
        uint8_t *parsing_img_ptr = parsing_img.ptr<uint8_t>();

        for (int h = 0; h < parsing_height; ++h) {
          for (int w = 0; w < parsing_width; ++w) {
            auto id = static_cast<size_t>(capture.features[h * parsing_width + w]);
            id = id >= 80? (id % 80) + 1: id;
            *parsing_img_ptr++ = bgr_putpalette[id * 3];
            *parsing_img_ptr++ = bgr_putpalette[id * 3 + 1];
            *parsing_img_ptr++ = bgr_putpalette[id * 3 + 2];
          }
        }

        cv::resize(parsing_img, parsing_img, mat.size(), 0, 0);
        // alpha blending
        cv::Mat dst;
        addWeighted(mat, alpha_f, parsing_img, 1 - alpha_f, 0.0, dst);
        mat = std::move(dst);
      }
    }
  }

  std::string saving_path = ai_msg->header.frame_id + "_" +
                            std::to_string(ai_msg->header.stamp.sec) + "_" +
                            std::to_string(ai_msg->header.stamp.nanosec) + "_render" + 
                            ".jpg";
  RCLCPP_WARN(rclcpp::get_logger("ImageUtils"),
              "Draw result to file: %s",
              saving_path.c_str());
  cv::imwrite(saving_path, mat);
  return 0;
}

int ImageUtils::Render(
    const std::shared_ptr<hobot::dnn_node::DNNTensor> &tensor,
    const ai_msgs::msg::PerceptionTargets::UniquePtr &ai_msg,
    const int img_h,
    const int img_w) {
  if (!tensor || !ai_msg) return -1;

  int src_elem_size = 1;
  auto properties = tensor->properties;
  switch (properties.tensorType)
  {
    case HB_DNN_TENSOR_TYPE_U8: src_elem_size = 1; break;
    case HB_DNN_TENSOR_TYPE_F32: src_elem_size = 4; break;
  }

  char *img = reinterpret_cast<char *>(tensor->sysMem[0].virAddr);
  cv::Mat mat;
  if (properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    auto channel = properties.alignedShape.dimensionSize[1];
    auto height = properties.alignedShape.dimensionSize[2];
    auto width = properties.alignedShape.dimensionSize[3];
    
    auto img_size = height * width * channel * src_elem_size;
    char *buf = new char[img_size];

    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          auto *raw = buf + h * width * channel * src_elem_size + w * channel * src_elem_size + c * src_elem_size;
          auto *src = img + c * height * width * src_elem_size + h * width * src_elem_size + w * src_elem_size;
          memcpy(raw, src, src_elem_size);
        }
      }
    }
    cv::Mat bgr(height, width, CV_32FC3, buf);
    bgr = bgr * 255;
    mat = bgr.clone();
    delete[] buf;

  } else if (properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    auto height = properties.alignedShape.dimensionSize[1];
    auto width = properties.alignedShape.dimensionSize[2];
    auto channel = properties.alignedShape.dimensionSize[3];

    auto img_size = height * width * channel * src_elem_size;
    char *buf = new char[img_size];
    for (int h = 0; h < height; ++h) {
      auto *raw = buf + h * width * channel * src_elem_size;
      auto *src = img + h * width * channel * src_elem_size;
      memcpy(raw, src, width * channel * src_elem_size);
    }
    cv::Mat bgr(height, width, CV_32FC3, buf);
    mat = bgr.clone();
    delete[] buf;
  }
  
  cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

  RCLCPP_INFO(rclcpp::get_logger("ImageUtils"),
              "target size: %d",
              ai_msg->targets.size());
  bool hasRois = false;
  for (size_t idx = 0; idx < ai_msg->targets.size(); idx++) {
    const auto &target = ai_msg->targets.at(idx);
    if (target.rois.empty()) {
      continue;
    }
    hasRois = true;
    RCLCPP_INFO(rclcpp::get_logger("ImageUtils"),
                "target type: %s, rois.size: %d",
                target.type.c_str(),
                target.rois.size());
    auto &color = colors[idx % colors.size()];
    for (const auto &roi : target.rois) {
      RCLCPP_INFO(
          rclcpp::get_logger("ImageUtils"),
          "roi.type: %s, x_offset: %d y_offset: %d width: %d height: %d",
          roi.type.c_str(),
          roi.rect.x_offset,
          roi.rect.y_offset,
          roi.rect.width,
          roi.rect.height);
      cv::rectangle(mat,
                    cv::Point(roi.rect.x_offset, roi.rect.y_offset),
                    cv::Point(roi.rect.x_offset + roi.rect.width,
                              roi.rect.y_offset + roi.rect.height),
                    color,
                    3);
      std::string roi_type = target.type;
      if (!roi.type.empty()) {
        roi_type = roi.type;
      }
      if (!roi_type.empty()) {
        cv::putText(mat,
                    roi_type,
                    cv::Point2f(roi.rect.x_offset, roi.rect.y_offset - 10),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1.5);
      }
    }

    for (const auto &lmk : target.points) {
      for (const auto &pt : lmk.point) {
        cv::circle(mat, cv::Point(pt.x, pt.y), 3, color, 3);
      }
    }
  }

  if (!hasRois) {
    RCLCPP_WARN(rclcpp::get_logger("ImageUtils"),
                "Frame has no roi, skip the rendering");
    return 0;
  }
  mat = mat(cv::Rect(0, 0, img_w, img_h));
  std::string saving_path = "render_" + ai_msg->header.frame_id + "_" +
                            std::to_string(ai_msg->header.stamp.sec) + "_" +
                            std::to_string(ai_msg->header.stamp.nanosec) +
                            ".jpeg";
  RCLCPP_WARN(rclcpp::get_logger("ImageUtils"),
              "Draw result to file: %s",
              saving_path.c_str());
  cv::imwrite(saving_path, mat);
  return 0;
}
