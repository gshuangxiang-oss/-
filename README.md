# 🚗 车牌检测与识别系统

> 基于深度学习的车牌自动检测与识别系统 | License Plate Detection and Recognition System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

本项目是一个完整的车牌检测与识别系统，采用深度学习技术实现车牌的自动定位和字符识别。

### ✨ 核心功能

- 🎯 **车牌检测**：使用 YOLOv8 目标检测算法精准定位车牌位置
- 📝 **字符识别**：使用 EasyOCR 进行高精度车牌字符识别
- 🌐 **Web 界面**：现代化响应式 Web 应用，支持拖拽上传
- 🔌 **REST API**：标准化 API 接口，便于系统集成

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web 前端界面                            │
│                   (HTML + CSS + JavaScript)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Flask REST API                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   处理流水线 Pipeline                        │
└─────────────────────────────────────────────────────────────┘
                    │                    │
                    ▼                    ▼
┌─────────────────────────┐  ┌─────────────────────────────┐
│     车牌检测模块         │  │      字符识别模块            │
│     (YOLOv8)            │  │      (EasyOCR)              │
└─────────────────────────┘  └─────────────────────────────┘
```

## 🔧 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 检测模型 | YOLOv8 | Ultralytics 目标检测 |
| 识别模型 | EasyOCR | 多语言 OCR 识别 |
| 深度学习框架 | PyTorch | 主流深度学习框架 |
| 后端框架 | Flask | Python Web 框架 |
| 前端技术 | HTML/CSS/JS | 原生前端技术栈 |
| 图像处理 | OpenCV | 计算机视觉库 |

## 📁 项目结构

```
project/
├── app/                          # 应用核心代码
│   ├── __init__.py              # 模块初始化
│   ├── detector.py              # 车牌检测模块 (YOLOv8)
│   ├── recognizer.py            # 字符识别模块 (EasyOCR)
│   ├── pipeline.py              # 处理流水线
│   └── api.py                   # Flask REST API
├── static/                       # 静态资源文件
│   ├── index.html               # Web 主页面
│   ├── styles.css               # 样式表
│   └── app.js                   # 前端交互逻辑
├── test_images/                  # 测试图片样例
│   ├── test_car_1~5.jpg         # 带车牌的车辆图片
│   ├── test_random_1~5.jpg      # 随机车牌图片
│   └── plate_only_1~3.jpg       # 纯车牌图片
├── docs/                         # 文档目录
│   └── TEST_REPORT.md           # 测试效果报告
├── tests/                        # 测试用例
│   └── test_pipeline.py         # 单元测试
├── requirements.txt              # Python 依赖
├── run.py                        # 启动入口
├── start.bat                     # Windows 一键启动脚本
├── .gitignore                    # Git 忽略配置
├── LICENSE                       # MIT 许可证
└── README.md                     # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.11 或 3.12（推荐）
- pip 包管理器
- Windows / Linux / macOS

### 方式一：一键启动（推荐）

**Windows 用户**：双击 `start.bat` 即可自动完成环境配置和启动

### 方式二：手动安装

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/license-plate-recognition.git
cd license-plate-recognition

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python run.py
```

### 访问系统

启动后打开浏览器访问：**http://localhost:5000**

## 📡 API 文档

### 健康检查
```http
GET /api/health
```

### 车牌识别
```http
POST /api/recognize
Content-Type: multipart/form-data

file: <image_file>
```

**响应示例：**
```json
{
    "success": true,
    "plates": [
        {
            "plate_number": "浙A·12345",
            "confidence": 0.95,
            "bbox": [100, 200, 300, 280],
            "plate_type": "普通车牌",
            "is_valid": true
        }
    ],
    "processing_time": 0.256,
    "annotated_image": "base64..."
}
```

## 🖼️ 测试图片

项目包含 13 张测试图片，位于 `test_images/` 目录：

| 文件 | 说明 |
|------|------|
| `test_car_1~5.jpg` | 固定车牌号的车辆图片 |
| `test_random_1~5.jpg` | 随机生成车牌的车辆图片 |
| `plate_only_1~3.jpg` | 纯车牌图片 |

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 检测准确率 | >95% |
| 识别准确率 | >90% |
| 处理速度 | <500ms (CPU) |
| 支持车牌 | 蓝牌、绿牌 |

## 🎨 界面预览

系统采用现代化暗色主题设计：

- 🌙 科技感暗色主题
- 📱 响应式布局
- ✨ 流畅动画效果
- 🖱️ 拖拽上传支持

## 📝 开发信息

- **作者**：郭双翔
- **学号**：23050420
- **课程**：深度学习
- **指导教师**：陈铭浩

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<p align="center">
  🚗 车牌检测与识别系统 | Made with ❤️ using Deep Learning
</p>
