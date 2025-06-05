import dashscope
from http import HTTPStatus
import json

with open("./embeddings.json", "r", encoding="utf-8") as f:
    embeddings = json.load(f)
embeddings["size"] += 1


image_path = "./Template/22.png"


def encode_image(image_path):
    input = [{'image': image_path}]
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=input
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        print(f"图像编码请求失败: {resp.code} - {resp.message}")
        return None
    
image_embedding = encode_image(image_path)
embeddings["images"].append({"image_path":image_path,"embedding":image_embedding})

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=2)