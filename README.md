English| [简体中文](./README_cn.md)

Getting Started with Yolo World Example
=======


# Feature Introduction


The Yolo World package is an example of quantized deployment based on [Yolo-world](https://github.com/AILab-CVC/YOLO-World). The image data comes from local image feedback and subscribed image messages. Additionally, Yolo World supports changing detection categories based on input text, which is the biggest difference between yolo-world and conventional yolo. The text can be configured through parameters or controlled in real-time via string message topics during runtime. Text features are sourced from a local feature library, and the corresponding features are queried based on the input text and fed into model inference. Ultimately, intelligent results are published in the post-processing of Yolo World and can be viewed through a web interface.

# Development Environment

- Programming Language: C/C++
- Development Platform: X5
- System Version: Ubuntu 22.04
- Compilation Toolchain: Linaro GCC 11.4.0

# Compilation

- X5 Version: Supports compilation on the X5 Ubuntu system and cross-compilation using Docker on a PC.

- X86 Version: Supports compilation on the X86 Ubuntu system.

It also supports controlling the dependencies and functionality of the compiled pkg through compilation options.

## Dependency Libraries

- OpenCV: 3.4.5

ROS Packages:

- dnn node
- cv_bridge
- sensor_msgs
- hbm_img_msgs
- ai_msgs

hbm_img_msgs is a custom image message format used for image transmission in shared memory scenarios. The hbm_img_msgs pkg is defined in hobot_msgs; therefore, if shared memory is used for image transmission, this pkg is required.

## Compilation Options

1. SHARED_MEM

- Shared memory transmission switch, enabled by default (ON), can be turned off during compilation using the -DSHARED_MEM=OFF command.
- When enabled, compilation and execution depend on the hbm_img_msgs pkg and require the use of tros for compilation.
- When disabled, compilation and execution do not depend on the hbm_img_msgs pkg, supporting compilation using native ROS and tros.
- For shared memory communication, only subscription to nv12 format images is currently supported.## Compile on X3/Rdkultra Ubuntu System

1. Compilation Environment Verification

- The X5 Ubuntu system is installed on the board.
- The current compilation terminal has set up the TogetherROS environment variable: `source PATH/setup.bash`. Where PATH is the installation path of TogetherROS.
- The ROS2 compilation tool colcon is installed. If the installed ROS does not include the compilation tool colcon, it needs to be installed manually. Installation command for colcon: `pip install -U colcon-common-extensions`.
- The dnn node package has been compiled.

2. Compilation

- Compilation command: `colcon build --packages-select hobot_yolo_world`

## Docker Cross-Compilation for X5 Version

1. Compilation Environment Verification

- Compilation within docker, and TogetherROS has been installed in the docker environment. For instructions on docker installation, cross-compilation, TogetherROS compilation, and deployment, please refer to the README.md in the robot development platform's robot_dev_config repo.
- The dnn node package has been compiled.
- The hbm_img_msgs package has been compiled (see Dependency section for compilation methods).

2. Compilation

- Compilation command:

  ```shell
  # RDK X5
  bash robot_dev_config/build.sh -p X5 -s hobot_yolo_world
  ```

- Shared memory communication method is enabled by default in the compilation options.

## Compile X86 Version on X86 Ubuntu System

1. Compilation Environment Verification

X86 Ubuntu version: ubuntu22.04

2. Compilation

- Compilation command:

  ```shell
  colcon build --packages-select hobot_yolo_world \
     --merge-install \
     --cmake-force-configure \
     --cmake-args \
     --no-warn-unused-cli \
     -DPLATFORM_X86=ON \
     -DTHIRD_PARTY=`pwd`/../sysroot_docker
  ```

## Notes


# Instructions

## Dependencies

- mipi_cam package: Publishes image messages
- usb_cam package: Publishes image messages
- websocket package: Renders images and AI perception messages

## Parameters

| Parameter Name      | Explanation                            | Mandatory            | Default Value       | Remarks                                                                 |
| ------------------- | -------------------------------------- | -------------------- | ------------------- | ----------------------------------------------------------------------- |
| feed_type           | Image source, 0: local; 1: subscribe   | No                   | 0                   |                                                                         |
| image               | Local image path                       | No                   | config/test.jpg     |                                                                         |
| is_shared_mem_sub   | Subscribe to images using shared memory communication method | No  | 0                   |                                                                         |                                                                   |
| score_threshold | boxes confidence threshold | No | 0.05 | |
| iou_threshold | nms iou threshold | No | 0.45 | |
| nms_top_k | Detect the first k boxes | No | 50 | |
| texts | detect types | No | "dog,cat" | Separate each category with a comma in the middle |
| dump_render_img     | Whether to render, 0: no; 1: yes       | No                   | 0                   |                                                                         |
| ai_msg_pub_topic_name | Topic name for publishing intelligent results for web display | No                   | /hobot_yolo_world | |
| ros_img_sub_topic_name | Topic name for subscribing image msg | No                   | /image | |
| ros_string_sub_topic_name | Topic name for subscribing string msg to change detect types| No                   | /target_words | |

## Instructions

- Topic control: hobot_yolo_world supports controlling detection categories through string msg topic messages, which is the main difference between yolo-world and regular yolo. The example of using the string msg topic is as follows. Among them, /targetw_words is the topic name. The data in the 'data' field is a string string, which is separated by commas when setting multiple detection categories.

```shell
ros2 topic pub /target_words std_msgs/msg/String "{data: 'dog,person'}"
```

## Running

## Running on X5 Ubuntu System

Running method 1, use the executable file to start:
```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/local_setup.bash
# The config includes models used by the example and local images for filling
# Copy based on the actual installation path (the installation path in the docker is install/lib/hobot_yolo_world/config/, the copy command is cp -r install/lib/hobot_yolo_world/config/ .).
cp -r install/hobot_yolo_world/lib/hobot_yolo_world/config/ .

# Run mode 1:Use local JPG format images for backflow prediction, with input categories of dog and cat:

ros2 run hobot_yolo_world hobot_yolo_world --ros-args -p feed_type:=0 -p image:=config/dog.jpg -p image_type:=0 -p texts:="dog,cat"

# Run mode 2:Use the subscribed image msg (topic name: /image) for prediction, set the controlled topic name (topic name: /target_words) to and set the log level to warn. At the same time, send a string topic (topic name: /target_words) in another window to change the detection category:

ros2 run hobot_yolo_world hobot_yolo_world --ros-args -p feed_type:=1 --ros-args --log-level warn -p ros_string_sub_topic_name:="/target_words"

ros2 topic pub /target_words std_msgs/msg/String "{data: 'dog,person'}"

# Run mode 3: Use shared memory communication method (topic name: /hbmem_img) to perform inference in asynchronous mode and set the log level to warn, with input categories of dog and cat:

ros2 run hobot_yolo_world hobot_yolo_world --ros-args -p feed_type:=1 -p is_shared_mem_sub:=1 -p texts:="dog,cat" --ros-args --log-level warn
```

To run in mode 2 using a launch file:

```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/setup.bash
# Copy the configuration based on the actual installation path
cp -r install/lib/hobot_yolo_world/config/ .

# Configure MIPI camera
export CAM_TYPE=mipi

# Start the launch file, publish nv12 format images using shared memory with F37 sensor
# By default, it runs the fcos algorithm, switch algorithms using the config_file parameter in the launch command, e.g., to use unet algorithm: config_file:="config/mobilenet_unet_workconfig.json"
ros2 launch hobot_yolo_world yolo_world.launch.py
```

## Run on X5 Yocto system:

```shell
export ROS_LOG_DIR=/userdata/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./install/lib/

# Copy the configuration used by the example and the local image used for inference
cp -r install/lib/hobot_yolo_world/config/ .

# Run mode 1:Use local JPG format images for backflow prediction, with input categories of dog and cat:
./install/lib/hobot_yolo_world/hobot_yolo_world --ros-args -p feed_type:=0 -p image:=config/dog.jpg -p image_type:=0 -p texts:="dog,cat"

# Run mode 2:Use the subscribed image msg (topic name: /image) for prediction, set the controlled topic name (topic name: /target_words) to and set the log level to warn. At the same time, send a string topic (topic name: /target_words) in another window to change the detection category:
./install/lib/hobot_yolo_world/hobot_yolo_world --ros-args -p feed_type:=1 --ros-args --log-level warn -p ros_string_sub_topic_name:="/target_words"

ros2 topic pub /target_words std_msgs/msg/String "{data: 'dog,person'}"

# Run mode 3: Use shared memory communication method (topic name: /hbmem_img) to perform inference in asynchronous mode and set the log level to warn, with input categories of dog and cat:
./install/lib/hobot_yolo_world/hobot_yolo_world --ros-args -p feed_type:=1 -p is_shared_mem_sub:=1 -p texts:="dog,cat" --ros-args --log-level warn
```

## Run on X86 Ubuntu system:
```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/setup.bash
# Copy the model used in the config as an example, adjust based on the actual installation path
cp -r ./install/lib/hobot_yolo_world/config/ .

export CAM_TYPE=fb

# Launch the file.
ros2 launch hobot_yolo_world yolo_world.launch.py
```

# Results Analysis

## X5 Results Display

log:

Command executed: `ros2 run hobot_yolo_world hobot_yolo_world --ros-args -p feed_type:=0 -p image:=config/dog.jpg -p image_type:=0 -p texts:="dog,cat"`

```shell
[WARN] [0000105571.958746757] [hobot_yolo_world]: This is hobot yolo world!
[WARN] [0000105572.027101757] [hobot_yolo_world]: Parameter:
 feed_type(0:local, 1:sub): 0
 image: config/dog.jpg
 is_shared_mem_sub: 0
 score_threshold: 0.05
 iou_threshold: 0.45
 nms_top_k: 50
 texts: dog,cat
 ai_msg_pub_topic_name: /hobot_yolo_world
 ros_img_sub_topic_name: /image
 ros_string_sub_topic_name: /target_words
[WARN] [0000105572.200221257] [hobot_yolo_world]: Parameter:
 model_file_name: config/yolo_world.bin
 model_name:
[INFO] [0000105572.200349090] [dnn]: Node init.
[INFO] [0000105572.200386090] [hobot_yolo_world]: Set node para.
[WARN] [0000105572.200428507] [hobot_yolo_world]: model_file_name_: config/yolo_world.bin, task_num: 4
[INFO] [0000105572.200481298] [dnn]: Model init.
[BPU_PLAT]BPU Platform Version(1.3.6)!
[HBRT] set log level as 0. version = 3.15.49.0
[DNN] Runtime version = 1.23.8_(3.15.49 HBRT)
[A][DNN][packed_model.cpp:247][Model](1970-01-02,05:19:32.890.945) [HorizonRT] The model builder version = 1.23.6
[INFO] [0000105573.170551799] [dnn]: The model input 0 width is 640 and height is 640
[INFO] [0000105573.170660257] [dnn]: The model input 1 width is 1 and height is 512
[INFO] [0000105573.170822466] [dnn]:
Model Info:
name: yolo_world.
[input]
 - (0) Layout: NCHW, Shape: [1, 3, 640, 640], Type: HB_DNN_TENSOR_TYPE_F32.
 - (1) Layout: NCHW, Shape: [1, 32, 512, 1], Type: HB_DNN_TENSOR_TYPE_F32.
[output]
 - (0) Layout: NCHW, Shape: [1, 8400, 32, 1], Type: HB_DNN_TENSOR_TYPE_F32.
 - (1) Layout: NCHW, Shape: [1, 8400, 4, 1], Type: HB_DNN_TENSOR_TYPE_F32.

[INFO] [0000105573.170896757] [dnn]: Task init.
[INFO] [0000105573.173133966] [dnn]: Set task_num [4]
[WARN] [0000105573.173188007] [hobot_yolo_world]: Get model name: yolo_world from load model.
[INFO] [0000105573.173228924] [hobot_yolo_world]: The model input width is 640 and height is 640
[WARN] [0000105573.173286549] [hobot_yolo_world]: Create ai msg publisher with topic_name: /hobot_yolo_world
[INFO] [0000105573.198971966] [hobot_yolo_world]: Dnn node feed with local image: config/dog.jpg
[INFO] [0000105573.495347382] [hobot_yolo_world]: Output from frame_id: feedback, stamp: 0.0
[INFO] [0000105573.499860466] [hobot_yolo_world]: out box size: 1
[INFO] [0000105573.500038966] [hobot_yolo_world]: det rect: 0.495301 125.662 315.754 639.413, det type: dog, score:0.40681
```

## Render img:
![image](img/render.jpeg)

Note: Preprocessing Image involves scaling and padding.