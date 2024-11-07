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
    dump_ai_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_ai_result", default_value=TextSubstitution(text="0")
    )
    dump_render_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_render_img", default_value=TextSubstitution(text="0")
    )
    dump_raw_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_raw_img", default_value=TextSubstitution(text="0")
    )
    dump_raw_path_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_raw_path", default_value=TextSubstitution(text=".")
    )
    dump_ai_path_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_ai_path", default_value=TextSubstitution(text=".")
    )
    dump_render_path_launch_arg = DeclareLaunchArgument(
        "yolo_world_dump_render_path", default_value=TextSubstitution(text=".")
    )
    model_file_name_launch_arg = DeclareLaunchArgument(
        "yolo_world_model_file_name", default_value=TextSubstitution(text="config/DOSOD_L_without_nms_nv12_v3.bin")
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
    class_mode_launch_arg = DeclareLaunchArgument(
        "yolo_world_class_mode", default_value=TextSubstitution(text="0")
    )
    # 本地图片发布
    feedback_picture_arg = DeclareLaunchArgument(
        'publish_image_source',
        default_value='./config/00131.jpg',
        description='feedback picture')

    feedback_loop_arg = DeclareLaunchArgument(
        'publish_is_loop',
        default_value='True',
        description='feedback loop turn')

    fb_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_image_publisher'),
                'launch/hobot_image_publisher.launch.py')),
        launch_arguments={
            'publish_image_source': LaunchConfiguration('publish_image_source'),
            'publish_image_format': 'jpg',
            'publish_is_shared_mem': 'False',
            'publish_message_topic_name': '/image',
            'publish_is_compressed_img_pub': 'True',
            'publish_fps': '10',
            'publish_is_loop': LaunchConfiguration('publish_is_loop'),
            'publish_output_image_w': LaunchConfiguration('yolo_world_image_width'),
            'publish_output_image_h': LaunchConfiguration('yolo_world_image_height')
        }.items()
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
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image',
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
            {"ros_img_sub_topic_name": '/image_raw'},
            {"dump_ai_result": LaunchConfiguration(
                "yolo_world_dump_ai_result")},
            {"dump_raw_img": LaunchConfiguration(
                "yolo_world_dump_raw_img")},
            {"dump_render_img": LaunchConfiguration(
                "yolo_world_dump_render_img")},
            {"dump_ai_path": LaunchConfiguration(
                "yolo_world_dump_ai_path")},
            {"dump_render_path": LaunchConfiguration(
                "yolo_world_dump_render_path")},
            {"dump_raw_path": LaunchConfiguration(
                "yolo_world_dump_raw_path")},
            {"msg_pub_topic_name": LaunchConfiguration(
                "yolo_world_msg_pub_topic_name")},
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
            {"class_mode": LaunchConfiguration(
                "yolo_world_class_mode")},
            {"score_threshold": LaunchConfiguration(
                "yolo_world_score_threshold")}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    return LaunchDescription([
        image_width_launch_arg,
        image_height_launch_arg,
        msg_pub_topic_name_launch_arg,
        dump_ai_launch_arg,
        dump_render_launch_arg,
        dump_raw_launch_arg,
        dump_ai_path_launch_arg,
        dump_raw_path_launch_arg,
        dump_render_path_launch_arg,
        model_file_name_launch_arg,
        vocabulary_file_name_launch_arg,
        score_threshold_launch_arg,
        trigger_mode_launch_arg,
        filterx_launch_arg,
        filtery_launch_arg,
        class_mode_launch_arg,
        feedback_loop_arg,
        # 图片发布pkg
        fb_node,
        # 图片编解码&发布pkg
        nv12_codec_node,
        # 启动yoloworld pkg
        yolo_world_node,
        # 启动web展示pkg
        web_node
    ])
