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
    image_width_launch_arg = DeclareLaunchArgument(
        "yolo_world_image_width", default_value=TextSubstitution(text="1920")
    )
    image_height_launch_arg = DeclareLaunchArgument(
        "yolo_world_image_height", default_value=TextSubstitution(text="1080")
    )
    msg_pub_topic_name_launch_arg = DeclareLaunchArgument(
        "yolo_world_msg_pub_topic_name", default_value=TextSubstitution(text="hobot_yolo_world")
    )
    dump_render_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_render_img", default_value=TextSubstitution(text="0")
    )
    model_file_name_launch_arg = DeclareLaunchArgument(
        "yolo_world_model_file_name", default_value=TextSubstitution(text="config/DOSOD_L_4_without_nms_int16_nv12_conv_int8_v7_1022.bin")
    )
    vocabulary_file_name_launch_arg = DeclareLaunchArgument(
        "yolo_world_vocabulary_file_name", default_value=TextSubstitution(text="config/offline_vocabulary_embeddings.json")
    )
    score_threshold_launch_arg = DeclareLaunchArgument(
        "yolo_world_score_threshold", default_value=TextSubstitution(text="0.6")
    )
    trigger_mode_launch_arg = DeclareLaunchArgument(
        "yolo_world_trigger_mode", default_value=TextSubstitution(text="0")
    )
    filterx_launch_arg = DeclareLaunchArgument(
        "yolo_world_filterx", default_value=TextSubstitution(text="0")
    )
    filtery_launch_arg = DeclareLaunchArgument(
        "yolo_world_filtery", default_value=TextSubstitution(text="0")
    )

    # web展示pkg
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_type': 'mjpeg',
            'websocket_smart_topic': LaunchConfiguration("yolo_world_msg_pub_topic_name")
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
            {"ros_img_sub_topic_name": "/image_combine_rectify"}
            {"model_file_name": LaunchConfiguration(
                "yolo_world_model_file_name")},
            {"vocabulary_file_name": LaunchConfiguration(
                "yolo_world_vocabulary_file_name")},
            {"trigger_mode": LaunchConfiguration(
                "yolo_world_trigger_mode")},
            {"filterx": LaunchConfiguration(
                "yolo_world_filterx")},
            {"filtery": LaunchConfiguration(
                "yolo_world_filtery")},
            {"score_threshold": LaunchConfiguration(
                "yolo_world_score_threshold")}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    shared_mem_node = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('hobot_shm'),
                        'launch/hobot_shm.launch.py'))
            )

    return LaunchDescription([
        image_width_launch_arg,
        image_height_launch_arg,
        msg_pub_topic_name_launch_arg,
        dump_render_launch_arg,
        model_file_name_launch_arg,
        vocabulary_file_name_launch_arg,
        score_threshold_launch_arg,
        trigger_mode_launch_arg,
        filterx_launch_arg,
        filtery_launch_arg,
        # 启动yoloworld pkg
        yolo_world_node,
        # 启动web展示pkg
        web_node
    ])
