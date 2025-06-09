import dashscope
from http import HTTPStatus
import json
import numpy as np
import os

def encode_image(image_url):
    input = [{'image': image_url}]
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=input
    )
    if resp.status_code == HTTPStatus.OK:
        result = resp.output['embeddings'][0]['embedding']
        return result
    
    else:
        print(f"图像编码请求失败: {resp.code} - {resp.message}")
        return None

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def search_similar_images(query_image_path):
    # 读取embeddings.json
    with open("embeddings.json", "r", encoding="utf-8") as f:
        embeddings = json.load(f)
    
    # 获取查询图片的embedding
    query_embedding = encode_image(query_image_path)
    if query_embedding is None:
        print("无法获取查询图片的embedding")
        return
    
    similarity_list = []
    
    # 计算与所有图片的相似度
    for image in embeddings["images"]:
        if image["embedding"] is None:
            continue
            
        image_vector = np.array(image["embedding"])
        similarity = cosine_similarity(query_embedding, image_vector)
        
        similarity_list.append({
            "image_path": image["image_path"],
            "similarity": float(similarity)
        })
    
    # 按相似度降序排序
    similarity_list.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 打印结果
    print(f"\n与图片 {query_image_path} 最相似的图片：")
    for item in similarity_list:
        print(f"图片: {item['image_path']}, 相似度: {item['similarity']:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python Search_similar.py <图片路径>")
        sys.exit(1)
        
    query_image_path = sys.argv[1]
    search_similar_images(query_image_path)