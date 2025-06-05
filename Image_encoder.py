import dashscope
from http import HTTPStatus
import json
import os

# def encode_text(text):
#     input = [{'text': text}]
#     resp = dashscope.MultiModalEmbedding.call(
#         model="multimodal-embedding-v1",
#         input=input
#     )
#     if resp.status_code == HTTPStatus.OK:
#         return resp.output['embeddings'][0]['embedding']
#     else:
#         print(f"文本编码请求失败: {resp.code} - {resp.message}")
#         return None

def encode_image(image_url):
    input = [{'image': image_url}]
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=input
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        print(f"图像编码请求失败: {resp.code} - {resp.message}")
        return None

# 初始化输出结构
# query = "颜色"
output_data = {
    "size": 20,
    # "query_embedding": None,
    "images": [
        {
            "image_path": f"./template/{i}.png",
            "embedding": None
        } for i in range(1, 21)
    ]
}

# 创建初始 JSON 文件
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
print("已创建 embeddings.json 并预留 20 个位置")

# 编码文本
# text_embedding = encode_text(query)
# if text_embedding:
#     print("文本编码成功")
#     output_data["query_embedding"] = text_embedding

# 编码图像并逐一写入
for i, item in enumerate(output_data["images"]):
    image_path = item["image_path"]
    image_embedding = encode_image(image_path)
    if image_embedding:
        print(f"图像 {i+1} 编码成功")
        output_data["images"][i]["embedding"] = image_embedding

    # 每一步保存当前状态（中断后可恢复）
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

print("所有嵌入向量已写入 embeddings.json")
