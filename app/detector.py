"""
车牌检测模块 - 使用 YOLOv8 进行车牌定位
License Plate Detection using YOLOv8
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class LicensePlateDetector:
    """车牌检测器 - 基于 YOLOv8"""
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5):
        """
        初始化检测器
        
        Args:
            model_path: YOLO 模型路径，如果为 None 则使用预训练模型
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # 使用预训练的 YOLOv8n 模型（轻量级）
                self.model = YOLO('yolov8n.pt')
            
            print("✓ YOLO 模型加载成功")
        except Exception as e:
            print(f"⚠ YOLO 模型加载失败: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        检测图像中的车牌
        
        Args:
            image: BGR 格式的图像数组
            
        Returns:
            检测结果列表，每个元素包含 bbox, confidence, class_name
        """
        if self.model is None:
            return self._detect_traditional(image)
        
        try:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': result.names.get(cls, 'unknown')
                        })
            
            if not detections:
                detections = self._detect_traditional(image)
            
            return detections
            
        except Exception as e:
            print(f"YOLO 检测错误: {e}")
            return self._detect_traditional(image)
    
    def _detect_traditional(self, image: np.ndarray) -> List[dict]:
        """
        传统图像处理方法检测车牌（备选方案）
        """
        detections = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(blur, 30, 200)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
                
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    if 2.0 < aspect_ratio < 5.0 and w > 60 and h > 20:
                        detections.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.7,
                            'class_id': 0,
                            'class_name': 'license_plate'
                        })
                        break
            
        except Exception as e:
            print(f"传统检测方法错误: {e}")
        
        return detections
    
    def crop_plate(self, image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
        """根据边界框裁剪车牌区域"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def draw_detections(self, image: np.ndarray, detections: List[dict], 
                        plate_texts: Optional[List[str]] = None) -> np.ndarray:
        """在图像上绘制检测结果"""
        output = image.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            if plate_texts and i < len(plate_texts):
                label = f"{plate_texts[i]} ({conf:.2f})"
            else:
                label = f"Plate ({conf:.2f})"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(output, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return output
