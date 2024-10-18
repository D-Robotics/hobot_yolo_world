# Copyright (c) 2024，Horizon Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory
from ament_index_python.packages import get_package_prefix

def generate_launch_description():

    # args that can be set from the command line or a default will be used
    msg_pub_topic_name_launch_arg = DeclareLaunchArgument(
        "yolo_world_msg_pub_topic_name", default_value=TextSubstitution(text="hobot_yolo_world")
    )
    model_file_name_launch_arg = DeclareLaunchArgument(
        "yolo_world_model_file_name", default_value=TextSubstitution(text="config/DOSOD_L_4_without_nms_int16_nv12_conv_int8_v7_1016.bin")
    )
    vocabulary_file_name_launch_arg = DeclareLaunchArgument(
        "yolo_world_vocabulary_file_name", default_value=TextSubstitution(text="config/offline_vocabulary_embeddings.json")
    )
    dump_render_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_render_img", default_value=TextSubstitution(text="0")
    )
    port_interaction_arg = DeclareLaunchArgument(
        "ws_port_interaction", default_value=TextSubstitution(text="8081")
    )
    score_threshold_launch_arg = DeclareLaunchArgument(
        "yolo_world_score_threshold", default_value=TextSubstitution(text="0.22")
    )

    # jpeg->nv12
    nv12_codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py')),
        launch_arguments={
            'codec_in_mode': 'ros',
            'codec_out_mode': 'ros',
            'codec_sub_topic': '/image',
            'codec_pub_topic': '/image_raw'
        }.items()
    )

    # web展示pkg
    web_smart_topic_arg = DeclareLaunchArgument(
        'smart_topic',
        default_value='/hobot_yolo_world',
        description='websocket smart topic')
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image',
            'websocket_smart_topic': LaunchConfiguration('smart_topic')
        }.items()
    )

    # 算法pkg
    yolo_world_node = Node(
        package='hobot_yolo_world',
        executable='hobot_yolo_world',
        output='screen',
        parameters=[
            {"feed_type": 1},
            {"is_shared_mem_sub": 0},
            {"dump_render_img": LaunchConfiguration(
                "yolo_world_dump_render_img")},
            {"msg_pub_topic_name": LaunchConfiguration(
                "yolo_world_msg_pub_topic_name")},
            {"score_threshold": LaunchConfiguration(
                "yolo_world_score_threshold")},
            {"iou_threshold": 0.5},
            {"nms_top_k": 50},
            {"is_homography": 1},
            {"y_offset": 800.0},
            {"ros_img_sub_topic_name": '/image_raw'},
            {"ai_msg_pub_topic_name": '/hobot_yolo_world'},
            {"model_file_name": LaunchConfiguration(
                "yolo_world_model_file_name")},
            {"vocabulary_file_name": LaunchConfiguration(
                "yolo_world_vocabulary_file_name")},
            {"port_interaction": LaunchConfiguration("ws_port_interaction")}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    return LaunchDescription([
        web_smart_topic_arg,
        msg_pub_topic_name_launch_arg,
        dump_render_launch_arg,
        port_interaction_arg,
        model_file_name_launch_arg,
        vocabulary_file_name_launch_arg,
        score_threshold_launch_arg,
        # 图片编解码&发布pkg
        nv12_codec_node,
        # 启动yoloworld pkg
        yolo_world_node,
        # 启动web展示pkg
        web_node
    ])
