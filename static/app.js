/**
 * 车牌检测与识别系统 - 前端交互
 * License Plate Detection & Recognition System - Frontend
 */

// DOM 元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectBtn = document.getElementById('selectBtn');
const resultContainer = document.getElementById('resultContainer');
const originalImage = document.getElementById('originalImage');
const annotatedImage = document.getElementById('annotatedImage');
const platesGrid = document.getElementById('platesGrid');
const noResult = document.getElementById('noResult');
const processingTime = document.getElementById('processingTime');
const loadingOverlay = document.getElementById('loadingOverlay');
const clearBtn = document.getElementById('clearBtn');

// API 配置
const API_BASE = '';  // 使用相对路径

/**
 * 初始化事件监听器
 */
function initEventListeners() {
    // 点击上传区域
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== selectBtn) {
            fileInput.click();
        }
    });
    
    // 选择按钮点击
    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    // 文件选择变化
    fileInput.addEventListener('change', handleFileSelect);
    
    // 拖拽事件
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // 清除按钮
    clearBtn.addEventListener('click', clearResults);
    
    // 阻止默认拖拽行为
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

/**
 * 处理拖拽悬停
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
}

/**
 * 处理拖拽离开
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
}

/**
 * 处理文件拖放
 */
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * 处理文件选择
 */
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * 处理上传的文件
 */
async function processFile(file) {
    // 验证文件类型
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showNotification('请上传有效的图片文件 (JPG, PNG, GIF, BMP, WebP)', 'error');
        return;
    }
    
    // 验证文件大小 (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showNotification('文件大小不能超过 16MB', 'error');
        return;
    }
    
    // 显示原始图像预览
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // 调用识别 API
    await recognizePlate(file);
}

/**
 * 调用车牌识别 API
 */
async function recognizePlate(file) {
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE}/api/recognize`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showNotification(data.error || '识别失败，请重试', 'error');
        }
    } catch (error) {
        console.error('API 请求失败:', error);
        showNotification('网络请求失败，请检查服务是否运行', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * 显示识别结果
 */
function displayResults(data) {
    // 显示结果容器
    resultContainer.style.display = 'block';
    
    // 显示标注图像
    if (data.annotated_image) {
        annotatedImage.src = `data:image/jpeg;base64,${data.annotated_image}`;
    }
    
    // 显示处理时间
    processingTime.textContent = `⏱️ ${data.processing_time}s`;
    
    // 清空之前的结果
    platesGrid.innerHTML = '';
    
    // 显示车牌结果
    if (data.plates && data.plates.length > 0) {
        noResult.style.display = 'none';
        
        data.plates.forEach((plate, index) => {
            const card = createPlateCard(plate, index);
            platesGrid.appendChild(card);
        });
    } else {
        noResult.style.display = 'block';
    }
    
    // 滚动到结果区域
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * 创建车牌结果卡片
 */
function createPlateCard(plate, index) {
    const card = document.createElement('div');
    card.className = 'plate-card';
    card.style.animationDelay = `${index * 0.1}s`;
    
    const confidencePercent = (plate.confidence * 100).toFixed(1);
    const validClass = plate.is_valid ? 'valid' : 'invalid';
    const validText = plate.is_valid ? '✓ 有效' : '✗ 格式异常';
    
    card.innerHTML = `
        <div class="plate-number">${escapeHtml(plate.plate_number) || '无法识别'}</div>
        <div class="plate-details">
            <div class="detail-row">
                <span class="detail-label">综合置信度</span>
                <span class="detail-value confidence">${confidencePercent}%</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">检测置信度</span>
                <span class="detail-value">${(plate.detection_confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">OCR 置信度</span>
                <span class="detail-value">${(plate.ocr_confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">车牌类型</span>
                <span class="detail-value">${escapeHtml(plate.plate_type)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">格式验证</span>
                <span class="detail-value ${validClass}">${validText}</span>
            </div>
        </div>
    `;
    
    return card;
}

/**
 * 清除结果
 */
function clearResults() {
    resultContainer.style.display = 'none';
    originalImage.src = '';
    annotatedImage.src = '';
    platesGrid.innerHTML = '';
    fileInput.value = '';
    
    // 滚动回顶部
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * 显示/隐藏加载状态
 */
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

/**
 * 显示通知消息
 */
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'error' ? 'rgba(239, 68, 68, 0.9)' : 'rgba(34, 197, 94, 0.9)'};
        color: white;
        border-radius: 12px;
        font-size: 0.9rem;
        z-index: 2000;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: slideIn 0.3s ease-out;
        backdrop-filter: blur(10px);
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // 自动移除
    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * HTML 转义
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 添加通知动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeOut {
        from {
            opacity: 1;
        }
        to {
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// 初始化
document.addEventListener('DOMContentLoaded', initEventListeners);

