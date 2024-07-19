[English](./README.md) | 简体中文

工具：用以生成文本编码词库。
=======

# 功能介绍
本工具可以让用户自定义yolo world推理节点检测所需的类别。具体用法：
1. 用户将需要检测的类别, 存入当前目录下 class.list 文件
2. 用户下载所需的编码模型文件 text_encoder.onnx
3. 运行脚本, 生成文本编码库 offline_vocabulary_embeddings.json

# 开发环境

- 编程语言: python
- 开发平台: X5/X86

# 使用说明

整个工程目录结构如下

```tool
├── __init__.py
├── class.list
├── main.py
├── text_encoder.onnx
└── clip
    ├── __init__.py
    ├── bpe_simple_vocab_16e6.txt.gz
    ├── clip.py
    └── simple_tokenizer.py
```

# 功能使用

```shell
# 下载模型并解压
wget http://sunrise.horizon.cc/models/yoloworld_encode_text/text_encoder.tar.gz
sudo tar -xf text_encoder.tar.gz -C tool

# 运行脚本
python3 main.py
```

log:
```shell
Current Process: 1/80
Current Process: 11/80
Current Process: 21/80
Current Process: 31/80
Current Process: 41/80
Current Process: 51/80
Current Process: 61/80
Current Process: 71/80
finish!
```