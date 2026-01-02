"""
车牌识别模块 - 使用 EasyOCR 进行字符识别
License Plate Recognition using EasyOCR
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import re


class LicensePlateRecognizer:
    """车牌字符识别器 - 基于 EasyOCR"""
    
    # 中国省份简称
    PROVINCES = [
        '京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘',
        '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋',
        '蒙', '陕', '吉', '闽', '贵', '粤', '川', '青', '藏', '琼',
        '宁', '港', '澳', '台'
    ]
    
    # 车牌字符集
    CHARS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
    
    def __init__(self, use_gpu: bool = False):
        """
        初始化识别器
        
        Args:
            use_gpu: 是否使用 GPU 加速
        """
        self.ocr = None
        self.use_gpu = use_gpu
        self._load_ocr()
    
    def _load_ocr(self):
        """加载 OCR 模型"""
        # 尝试加载 EasyOCR
        try:
            import easyocr
            self.ocr = easyocr.Reader(['ch_sim', 'en'], gpu=self.use_gpu, verbose=False)
            self.ocr_type = 'easyocr'
            print("✓ EasyOCR 模型加载成功")
            return
        except ImportError:
            print("⚠ EasyOCR 未安装")
        except Exception as e:
            print(f"⚠ EasyOCR 加载失败: {e}")
        
        # 尝试加载 PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=self.use_gpu,
                show_log=False,
                det=True,
                rec=True
            )
            self.ocr_type = 'paddleocr'
            print("✓ PaddleOCR 模型加载成功")
            return
        except ImportError:
            print("⚠ PaddleOCR 未安装")
        except Exception as e:
            print(f"⚠ PaddleOCR 加载失败: {e}")
        
        # 都不可用，使用模拟模式
        self.ocr = None
        self.ocr_type = 'mock'
        print("⚠ 使用模拟识别模式（演示用）")
    
    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """
        预处理车牌图像以提高识别准确率
        
        Args:
            plate_image: 车牌图像
            
        Returns:
            预处理后的图像
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image
        
        # 调整大小
        height = 48
        ratio = height / plate_image.shape[0]
        width = int(plate_image.shape[1] * ratio)
        resized = cv2.resize(plate_image, (width, height))
        
        # 转换为灰度
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 转回 BGR
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        识别车牌字符
        
        Args:
            plate_image: 车牌图像
            
        Returns:
            (识别结果, 置信度)
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        
        # 预处理
        processed = self.preprocess(plate_image)
        
        if self.ocr_type == 'easyocr':
            return self._recognize_with_easyocr(processed)
        elif self.ocr_type == 'paddleocr':
            return self._recognize_with_paddleocr(processed)
        else:
            return self._recognize_mock(processed)
    
    def _recognize_with_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """使用 EasyOCR 进行识别"""
        try:
            result = self.ocr.readtext(image)
            
            if result:
                texts = []
                confidences = []
                
                for detection in result:
                    text = detection[1]
                    conf = detection[2]
                    texts.append(text)
                    confidences.append(conf)
                
                if texts:
                    full_text = ''.join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    plate_number = self._format_plate_number(full_text)
                    return plate_number, avg_conf
            
            return "", 0.0
            
        except Exception as e:
            print(f"EasyOCR 识别错误: {e}")
            return "", 0.0
    
    def _recognize_with_paddleocr(self, image: np.ndarray) -> Tuple[str, float]:
        """使用 PaddleOCR 进行识别"""
        try:
            result = self.ocr.ocr(image, cls=True)
            
            if result and result[0]:
                texts = []
                confidences = []
                
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        conf = line[1][1]
                        texts.append(text)
                        confidences.append(conf)
                
                if texts:
                    full_text = ''.join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    plate_number = self._format_plate_number(full_text)
                    return plate_number, avg_conf
            
            return "", 0.0
            
        except Exception as e:
            print(f"PaddleOCR 识别错误: {e}")
            return "", 0.0
    
    def _recognize_mock(self, image: np.ndarray) -> Tuple[str, float]:
        """模拟识别（当 OCR 不可用时）"""
        import random
        
        province = random.choice(self.PROVINCES)
        letter = random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')
        numbers = ''.join([str(random.randint(0, 9)) for _ in range(5)])
        
        plate = f"{province}{letter}·{numbers}"
        return plate, 0.85
    
    def _format_plate_number(self, text: str) -> str:
        """
        格式化车牌号码
        
        Args:
            text: 原始 OCR 识别文本
            
        Returns:
            格式化后的车牌号
        """
        # 移除空格和特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text.upper())
        
        # 常见 OCR 错误修正
        corrections = {
            'O': '0', 'I': '1', 'Z': '2', 'S': '5',
            'B': '8', 'D': '0', 'G': '6', 'Q': '0'
        }
        
        result = []
        for i, char in enumerate(text):
            if i == 0:
                # 第一个字符应该是省份简称
                if char in self.PROVINCES:
                    result.append(char)
                else:
                    result.append(char)
            elif i == 1:
                # 第二个字符应该是字母
                result.append(char)
            else:
                # 后续字符可以是字母或数字
                if char in corrections and i > 2:
                    result.append(corrections[char])
                else:
                    result.append(char)
        
        formatted = ''.join(result)
        
        # 添加分隔符（如果是7位标准车牌）
        if len(formatted) >= 7:
            return f"{formatted[:2]}·{formatted[2:7]}"
        
        return formatted
    
    def validate_plate(self, plate_number: str) -> Tuple[bool, str]:
        """
        验证车牌号格式
        
        Args:
            plate_number: 车牌号
            
        Returns:
            (是否有效, 车牌类型描述)
        """
        # 移除分隔符
        clean = plate_number.replace('·', '').replace(' ', '')
        
        if len(clean) < 7:
            return False, "车牌号长度不足"
        
        # 检查首字符是否为省份简称
        if clean[0] not in self.PROVINCES:
            return False, "首字符不是有效省份简称"
        
        # 检查第二个字符是否为字母
        if not clean[1].isalpha():
            return False, "第二位应为字母"
        
        # 新能源车牌（8位）
        if len(clean) == 8:
            return True, "新能源车牌"
        
        # 普通车牌（7位）
        if len(clean) == 7:
            return True, "普通车牌"
        
        return True, "特殊车牌"
    
    def batch_recognize(self, plate_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        批量识别多个车牌
        
        Args:
            plate_images: 车牌图像列表
            
        Returns:
            识别结果列表
        """
        results = []
        for img in plate_images:
            result = self.recognize(img)
            results.append(result)
        return results
