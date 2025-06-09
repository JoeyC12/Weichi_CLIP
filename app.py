from flask import Flask, request, jsonify, send_from_directory
import json
import subprocess
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')

def ensure_list(val):
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    elif val is None:
        return []
    else:
        return [str(val).strip()]

# 读取所有模板信息
@app.route('/api/templates')
def get_templates():
    with open('embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 只返回图片和标签
    templates = []
    for img in data['images']:
        tpl = {
            'image_path': img['image_path'].replace('./', '/'),
            'industry': img.get('json', {}).get('industry', []),
            'colors': img.get('json', {}).get('colors', []),
            'styles': img.get('json', {}).get('styles', [])
        }
        templates.append(tpl)
    return jsonify({'templates': templates})

# 文字检索
@app.route('/api/inference', methods=['POST'])
def inference():
    query = request.json.get('query')
    result = subprocess.check_output(['python3', 'inference.py', query], encoding='utf-8')
    # 解析输出，返回图片路径和相似度
    lines = [l for l in result.split('\n') if '搜索' in l]
    templates = []
    for l in lines:
        try:
            parts = l.split(':')
            path = parts[1].split(',')[0].strip()
            sim = float(parts[2].strip())
            templates.append({'image_path': path.replace('./', '/'), 'similarity': sim})
        except:
            continue
    # 读取标签
    with open('embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    path2info = {img['image_path'].replace('./', '/'): img.get('json', {}) for img in data['images']}
    for tpl in templates:
        info = path2info.get(tpl['image_path'], {})
        tpl['industry'] = ensure_list(info.get('industry', []))
        tpl['colors'] = ensure_list(info.get('colors', []))
        tpl['styles'] = ensure_list(info.get('styles', []))
    return jsonify({'templates': templates})

# 图片检索
@app.route('/api/search_similar', methods=['POST'])
def search_similar():
    file = request.files['image']
    save_path = 'query.jpg'
    file.save(save_path)
    result = subprocess.check_output(['python3', 'Search_similar.py', save_path], encoding='utf-8')
    lines = [l for l in result.split('\n') if '图片:' in l]
    templates = []
    for l in lines:
        try:
            parts = l.split(':')
            path = parts[1].split(',')[0].strip()
            sim = float(parts[2].strip())
            templates.append({'image_path': path.replace('./', '/'), 'similarity': sim})
        except:
            continue
    # 读取标签
    with open('embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    path2info = {img['image_path'].replace('./', '/'): img.get('json', {}) for img in data['images']}
    for tpl in templates:
        info = path2info.get(tpl['image_path'], {})
        tpl['industry'] = ensure_list(info.get('industry', []))
        tpl['colors'] = ensure_list(info.get('colors', []))
        tpl['styles'] = ensure_list(info.get('styles', []))
    return jsonify({'templates': templates})

# 前端静态文件
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/template/<path:filename>')
def template_static(filename):
    return send_from_directory('template', filename)

if __name__ == '__main__':
    app.run(debug=True)