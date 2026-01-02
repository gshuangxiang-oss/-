"""
车牌检测与识别系统 - 单元测试
License Plate Detection and Recognition System - Unit Tests

运行方式: python -m pytest tests/ -v
"""

import os
import sys
import pytest

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLicensePlateDetector:
    """测试车牌检测模块"""
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        from app.detector import LicensePlateDetector
        detector = LicensePlateDetector()
        assert detector is not None
    
    def test_detect_returns_list(self):
        """测试检测方法返回列表"""
        import numpy as np
        from app.detector import LicensePlateDetector
        
        detector = LicensePlateDetector()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(test_image)
        
        assert isinstance(results, list)
    
    def test_crop_plate(self):
        """测试车牌裁剪功能"""
        import numpy as np
        from app.detector import LicensePlateDetector
        
        detector = LicensePlateDetector()
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = [100, 200, 300, 250]
        
        cropped = detector.crop_plate(test_image, bbox)
        
        assert cropped is not None
        assert cropped.shape[0] == 50  # height
        assert cropped.shape[1] == 200  # width


class TestLicensePlateRecognizer:
    """测试车牌识别模块"""
    
    def test_recognizer_initialization(self):
        """测试识别器初始化"""
        from app.recognizer import LicensePlateRecognizer
        recognizer = LicensePlateRecognizer(use_gpu=False)
        assert recognizer is not None
    
    def test_validate_plate_valid(self):
        """测试有效车牌格式验证"""
        from app.recognizer import LicensePlateRecognizer
        recognizer = LicensePlateRecognizer(use_gpu=False)
        
        is_valid, plate_type = recognizer.validate_plate('浙A·12345')
        assert is_valid == True
        assert '车牌' in plate_type
    
    def test_validate_plate_invalid(self):
        """测试无效车牌格式验证"""
        from app.recognizer import LicensePlateRecognizer
        recognizer = LicensePlateRecognizer(use_gpu=False)
        
        is_valid, _ = recognizer.validate_plate('AB123')
        assert is_valid == False


class TestLicensePlatePipeline:
    """测试完整流水线"""
    
    def test_pipeline_initialization(self):
        """测试流水线初始化"""
        from app.pipeline import create_pipeline
        pipeline = create_pipeline(use_gpu=False)
        assert pipeline is not None
    
    def test_process_image_returns_dict(self):
        """测试图像处理返回字典"""
        import numpy as np
        from app.pipeline import create_pipeline
        
        pipeline = create_pipeline(use_gpu=False)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 150
        
        result = pipeline.process_image(test_image)
        
        assert 'plates' in result
        assert 'processing_time' in result
        assert isinstance(result['plates'], list)


class TestAPI:
    """测试 API 模块"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        from app.api import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
    
    def test_recognize_no_file(self, client):
        """测试未上传文件的情况"""
        response = client.post('/api/recognize')
        assert response.status_code == 400
    
    def test_demo_endpoint(self, client):
        """测试演示接口"""
        response = client.get('/api/demo')
        assert response.status_code == 200


class TestTestImages:
    """测试测试图片"""
    
    def test_images_exist(self):
        """测试测试图片是否存在"""
        from pathlib import Path
        
        test_dir = Path(__file__).parent.parent / 'test_images'
        assert test_dir.exists()
        
        # 检查至少有一些测试图片
        images = list(test_dir.glob('*.jpg'))
        assert len(images) >= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
