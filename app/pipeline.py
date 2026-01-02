"""
车牌检测与识别管道
License Plate Detection and Recognition Pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class PlateResult:
    """车牌识别结果"""
    plate_number: str
    confidence: float
    bbox: List[int]
    plate_image: np.ndarray
    plate_type: str
    is_valid: bool


class LicensePlatePipeline:
    """车牌检测与识别完整流水线"""
    
    def __init__(self, detector_model_path: Optional[str] = None, use_gpu: bool = False):
        """
        初始化流水线
        
        Args:
            detector_model_path: 检测模型路径
            use_gpu: 是否使用 GPU
        """
        from .detector import LicensePlateDetector
        from .recognizer import LicensePlateRecognizer
        
        self.detector = LicensePlateDetector(model_path=detector_model_path)
        self.recognizer = LicensePlateRecognizer(use_gpu=use_gpu)
        
        print("✓ 车牌识别流水线初始化完成")
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        处理单张图像
        
        Args:
            image: BGR 格式图像
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        
        results = {
            'plates': [],
            'annotated_image': None,
            'processing_time': 0,
            'image_size': image.shape[:2]
        }
        
        # 检测车牌
        detections = self.detector.detect(image)
        
        plate_texts = []
        
        for det in detections:
            # 裁剪车牌区域
            plate_img = self.detector.crop_plate(image, det['bbox'])
            
            if plate_img.size == 0:
                continue
            
            # 识别车牌字符
            plate_number, ocr_conf = self.recognizer.recognize(plate_img)
            
            # 验证车牌
            is_valid, plate_type = self.recognizer.validate_plate(plate_number)
            
            # 综合置信度
            combined_conf = (det['confidence'] + ocr_conf) / 2
            
            plate_result = PlateResult(
                plate_number=plate_number,
                confidence=combined_conf,
                bbox=det['bbox'],
                plate_image=plate_img,
                plate_type=plate_type,
                is_valid=is_valid
            )
            
            results['plates'].append({
                'plate_number': plate_result.plate_number,
                'confidence': round(plate_result.confidence, 3),
                'bbox': plate_result.bbox,
                'plate_type': plate_result.plate_type,
                'is_valid': plate_result.is_valid,
                'detection_confidence': round(det['confidence'], 3),
                'ocr_confidence': round(ocr_conf, 3)
            })
            
            plate_texts.append(plate_number)
        
        # 绘制标注图像
        results['annotated_image'] = self.detector.draw_detections(
            image, detections, plate_texts
        )
        
        # 计算处理时间
        results['processing_time'] = round(time.time() - start_time, 3)
        
        return results
    
    def process_image_file(self, image_path: str) -> Dict:
        """
        处理图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理结果字典
        """
        image = cv2.imread(image_path)
        
        if image is None:
            return {
                'error': f'无法读取图像: {image_path}',
                'plates': [],
                'processing_time': 0
            }
        
        return self.process_image(image)
    
    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        批量处理多张图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            处理结果列表
        """
        results = []
        
        for path in image_paths:
            result = self.process_image_file(path)
            result['image_path'] = path
            results.append(result)
        
        return results
    
    def save_result(self, result: Dict, output_path: str) -> bool:
        """
        保存处理结果
        
        Args:
            result: 处理结果
            output_path: 输出路径
            
        Returns:
            是否保存成功
        """
        try:
            if result.get('annotated_image') is not None:
                cv2.imwrite(output_path, result['annotated_image'])
                return True
            return False
        except Exception as e:
            print(f"保存失败: {e}")
            return False


def create_pipeline(use_gpu: bool = False) -> LicensePlatePipeline:
    """
    创建车牌识别流水线（工厂函数）
    
    Args:
        use_gpu: 是否使用 GPU
        
    Returns:
        LicensePlatePipeline 实例
    """
    return LicensePlatePipeline(use_gpu=use_gpu)

