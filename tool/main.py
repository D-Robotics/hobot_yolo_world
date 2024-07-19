import json
import numpy as np
import onnxruntime as rt
import time

import clip

class YOLOEncodeTextNode():
    def __init__(self):
        super().__init__()
        
        self.model_file_name = "text_encoder.onnx"

        self.sess = rt.InferenceSession(self.model_file_name)

    def runonnx(self, texts):
        start_time = time.time()
        
        # 1. simple tokenizer
        text_np = clip.tokenize(texts)

        # 2. model inference 
        input_name_0 = self.sess.get_inputs()[0].name
        output_name_0 = self.sess.get_outputs()[0].name
        text_features_np = self.sess.run([output_name_0],{input_name_0:text_np})[0]

        return text_features_np

if __name__ == '__main__':
    node = YOLOEncodeTextNode()
    
    texts = []
    with open('class.list', 'r', encoding='utf-8') as file:
        texts = [line.strip() for line in file]

    data = {}
    for i, text in enumerate(texts):
        texts_input = []
        texts_input.append(texts[i])
        res = node.runonnx(texts_input)
        res = res.reshape(512)
        data[text] = res.tolist()
        if i % 10 == 0:
            print("Current Process: {}/{}".format(i + 1, len(texts)))

    # 将更新后的字典写入 JSON 文件
    with open('offline_vocabulary_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("finish!")