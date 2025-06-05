import dashscope
from http import HTTPStatus
import json
import os
import numpy as np
import sys




with open("./embeddings.json", "r", encoding="utf-8") as f:
    embeddings = json.load(f)

k = embeddings["size"]

query = sys.argv[1]

def encode_text(text):
    input = [{'text': text}]
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=input
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        print(f"文本编码请求失败: {resp.code} - {resp.message}")
        return None


text_embedding = encode_text(query)

similarity_list = []

import numpy as np
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# 将查询向量转换为numpy数组并重塑维度
query_vector = np.array(text_embedding)

# 遍历所有图像向量计算相似度
for i in range(k):
    image = embeddings["images"][i]
    if image["embedding"] == None:
        continue
    image_vector = np.array(image["embedding"])
   
    similarity = cosine_similarity(query_vector, image_vector)
    similarity_list.append({
        "image_path": image["image_path"],
        "similarity": float(similarity)
    })
  

# 按相似度降序排序
similarity_list.sort(key=lambda x: x["similarity"], reverse=True)

print("相似度计算完成,结果如下:")
for item in similarity_list:
    print(f"搜索{query}主题: {item['image_path']}, 相似度: {item['similarity']:.4f}")




