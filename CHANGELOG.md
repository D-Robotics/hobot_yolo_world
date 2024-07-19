# Changelog for package hobot_yolo_world

tros_0.2.0 (2024-07-19)
------------------
1. 新增python工具脚本, 用以生成词汇编码特征 offline_vocabulary_embeddings.json。
2. 新增可视化工具, 支持保存检测结果到本地。
3. 修复字典长度不够时, 生成text特征出错的问题。

tros_0.1.0 (2024-07-15)
------------------
1. 新增 yolo-world 推理功能。支持Ros话题数据、零拷贝话题数据输入。
2. 新增接收 std_msgs string 话题消息可以控制算法检测类别。