"""
Flask REST API 服务
License Plate Detection and Recognition API
"""

import os
import cv2
import numpy as np
import base64
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime


# 创建 Flask 应用
app = Flask(__name__, static_folder='../static')
CORS(app)

# 配置
UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保上传目录存在
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# 全局流水线实例
pipeline = None


def get_pipeline():
    """获取或创建流水线实例"""
    global pipeline
    if pipeline is None:
        from .pipeline import create_pipeline
        pipeline = create_pipeline(use_gpu=False)
    return pipeline


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image: np.ndarray) -> str:
    """将图像转换为 base64 字符串"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str: str) -> np.ndarray:
    """将 base64 字符串转换为图像"""
    # 移除可能的 data URL 前缀
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# ===================== API 路由 =====================

@app.route('/')
def index():
    """首页"""
    return send_from_directory('../static', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """静态文件服务"""
    return send_from_directory('../static', filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'License Plate Recognition API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/recognize', methods=['POST'])
def recognize_plate():
    """
    车牌识别 API
    
    支持两种输入方式：
    1. 文件上传 (multipart/form-data)
    2. Base64 图像 (application/json)
    """
    try:
        pipe = get_pipeline()
        image = None
        
        # 方式1：文件上传
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '未选择文件'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': '不支持的文件格式'}), 400
            
            # 读取文件为图像
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 方式2：Base64 图像
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': '缺少 image 字段'}), 400
            
            image = base64_to_image(data['image'])
        
        else:
            return jsonify({'error': '请上传图片文件或提供 base64 图像'}), 400
        
        if image is None:
            return jsonify({'error': '无法解析图像'}), 400
        
        # 处理图像
        result = pipe.process_image(image)
        
        # 准备响应
        response = {
            'success': True,
            'plates': result['plates'],
            'processing_time': result['processing_time'],
            'image_size': {
                'height': result['image_size'][0],
                'width': result['image_size'][1]
            }
        }
        
        # 添加标注图像（base64）
        if result['annotated_image'] is not None:
            response['annotated_image'] = image_to_base64(result['annotated_image'])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recognize/url', methods=['POST'])
def recognize_from_url():
    """从 URL 识别车牌"""
    try:
        if not request.is_json:
            return jsonify({'error': '需要 JSON 请求体'}), 400
        
        data = request.get_json()
        image_url = data.get('url')
        
        if not image_url:
            return jsonify({'error': '缺少 url 字段'}), 400
        
        # 下载图像
        import urllib.request
        with urllib.request.urlopen(image_url, timeout=10) as response:
            image_data = response.read()
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '无法从 URL 加载图像'}), 400
        
        # 处理图像
        pipe = get_pipeline()
        result = pipe.process_image(image)
        
        response = {
            'success': True,
            'plates': result['plates'],
            'processing_time': result['processing_time'],
            'source_url': image_url
        }
        
        if result['annotated_image'] is not None:
            response['annotated_image'] = image_to_base64(result['annotated_image'])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_recognize():
    """批量车牌识别"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': '未上传文件'}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': '文件列表为空'}), 400
        
        pipe = get_pipeline()
        results = []
        
        for file in files:
            if file.filename and allowed_file(file.filename):
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    result = pipe.process_image(image)
                    results.append({
                        'filename': file.filename,
                        'plates': result['plates'],
                        'processing_time': result['processing_time']
                    })
        
        return jsonify({
            'success': True,
            'total': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/demo', methods=['GET'])
def demo_recognition():
    """演示识别（使用示例图像）"""
    try:
        # 创建一个简单的测试图像
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        # 添加一些文字作为示例
        cv2.putText(test_image, 'Demo Image', (150, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 3)
        
        pipe = get_pipeline()
        result = pipe.process_image(test_image)
        
        return jsonify({
            'success': True,
            'message': '演示模式 - 实际使用请上传车辆图片',
            'plates': result['plates'],
            'processing_time': result['processing_time']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===================== 错误处理 =====================

@app.errorhandler(413)
def too_large(e):
    """文件过大错误"""
    return jsonify({
        'error': '文件太大，最大支持 16MB'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """404 错误"""
    return jsonify({
        'error': '接口不存在'
    }), 404


@app.errorhandler(500)
def server_error(e):
    """500 错误"""
    return jsonify({
        'error': '服务器内部错误'
    }), 500


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """启动服务器"""
    print(f"""
============================================================
    License Plate Detection and Recognition System
============================================================

    Server:  http://{host}:{port}
    Web UI:  http://localhost:{port}/
    API:     http://localhost:{port}/api/health

    Press Ctrl+C to stop
============================================================
    """)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
