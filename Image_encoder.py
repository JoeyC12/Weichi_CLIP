import dashscope
from http import HTTPStatus

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
        return resp.output['embeddings'][0]['embedding']
    else:
        print(f"图像编码请求失败: {resp.code} - {resp.message}")
        return None

# 示例使用
query = "明亮的网站模版"


vector_store = []

# 编码文本
text_embedding = encode_text(query)
if text_embedding:
    print("文本编码成功")

# 编码图像
for i in range(1,21):
    image_embedding = encode_image(f"./template/{i}.png")
    if image_embedding:
        print("图像编码成功")
        vector_store.append(image_embedding)
