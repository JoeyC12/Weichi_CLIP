import dashscope
from http import HTTPStatus
import json
import numpy as np
import os

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

def describe_image(image_path):
    prompt = "你是一个网站风格分析助手。请只根据图像内容，从以下五个维度用逗号分隔给出简洁标签（不要写完整句）：【颜色（网站主要配色，而非图片颜色）、明暗、主题风格（如科技感、商务风、极简等）、视觉对比度（高对比或柔和）】输出格式必须严格如下（仅返回一句，不要写解释）：颜色，明暗，主题风格，内容类型，对比度，例如：白色为主，明亮，未来科技感，高对比"
    result = dashscope.MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=[{
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": prompt}
            ]
        }]
    )
    if result.status_code == HTTPStatus.OK:
        # 返回纯文本描述
        return result.output['choices'][0]['message']['content']
    else:
        print(f"描述失败: {result.message}")
        return None
    
def extract_json_from_markdown(text):
    # 去除 Markdown 代码块包裹
    if text.strip().startswith("```json"):
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
    return text

def image_to_json(image_path):
    prompt_text = None
    with open("image_to_json_prompt.txt", "r", encoding="utf-8") as f:
        prompt_text = f.read()
    prompt = prompt_text
    result = dashscope.MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=[{
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": prompt}
            ]
        }]
    )

        # 返回json
    try:
        raw = result.output['choices'][0]['message']['content'][0]['text']
        clean = extract_json_from_markdown(raw)
        return json.loads(clean)
    except Exception as e:
        print(f"⚠️ 无法解析为 JSON：{e}")
        print(f"⚠️ 原始返回内容为：{result.output['choices'][0]['message']['content']}")
        return None




# 初始化输出结构
# query = "颜色"
output_data = {
    "size": 22,
    # "query_embedding": None,
    "images": [
        {
            "image_path": f"./template/{i}.png",
            "embedding": None
        } for i in range(1, 23)
    ]
}

# 创建初始 JSON 文件
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
print("已创建 embeddings.json 并预留 22 个位置")

# 编码文本
# text_embedding = encode_text(query)
# if text_embedding:
#     print("文本编码成功")
#     output_data["query_embedding"] = text_embedding

# 编码图像并逐一写入
for i, item in enumerate(output_data["images"]):
    image_path = item["image_path"]
    image_description = describe_image(image_path)[0]['text']

    json_ = image_to_json(image_path)
    
    print(image_description)
    print(json_)
 
    image_embedding = 0.3 * np.array(encode_text(image_description)) + 0.7 * np.array(encode_image(image_path))
    if image_embedding is not None:
        
        output_data["images"][i]["embedding"] = image_embedding.tolist()
        output_data["images"][i]["json"] = json_
        print(f"图像 {i+1} 编码成功")
        
    # 每一步保存当前状态（中断后可恢复）
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

print("所有嵌入向量已写入 embeddings.json")
