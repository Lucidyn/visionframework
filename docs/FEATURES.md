# 功能特性

## 🆕 v0.4.0 — 新功能（2026-02-26）

### ROI 区域计数

```python
v = Vision(model="yolov8n.pt", track=True)
v.add_roi("entrance", [(100,100),(400,100),(400,400),(100,400)])
for frame, meta, result in v.run("video.mp4"):
    counts = result["counts"]  # {"entrance": {"inside": 3, "total_entered": 12, ...}}
```

### 批量图像处理

```python
results = v.process_batch([img1, img2, img3])
```

### 实例信息

```python
print(v.info())  # {"model": "yolov8n.pt", "device": "cpu", "rois": ["entrance"], ...}
```

### 热力图可视化

```python
from visionframework import Visualizer
vis = Visualizer()
heatmap = vis.draw_heatmap(frame, tracks, alpha=0.5, accumulate=True, _heat_state={})
```

### LoRA / QLoRA 微调

```python
from visionframework import FineTuningConfig, FineTuningStrategy, ModelFineTuner

cfg = FineTuningConfig(strategy=FineTuningStrategy.LORA, epochs=10)
tuner = ModelFineTuner(cfg)
```

### 统一导入风格

v0.4.0 起，所有组件均可直接从 `visionframework` 导入，无需记忆内部模块路径：

```python
# 旧写法（仍然有效）
from visionframework.utils.model_optimization.quantization import QuantizationConfig

# 新写法（推荐）
from visionframework import QuantizationConfig
```

---

## 🆕 v0.3.0 — 全新 Vision API (2026-02-07)

### API 极简化

整个框架只有一个入口类 `Vision`，两种创建方式：

```python
from visionframework import Vision

# 方式一：关键字参数
v = Vision(model="yolov8n.pt", track=True, pose=True)

# 方式二：配置文件 (JSON / YAML / dict)
v = Vision.from_config("config.json")

# 统一的 run() 方法处理一切
for frame, meta, result in v.run("video.mp4"):
    print(result["detections"], result["tracks"], result["poses"])
```

### 对比旧 API

| 操作 | 旧 API (v0.2.x) | 新 API (v0.3.0) |
|------|-----------------|-----------------|
| 检测 | `create_detector(model_path="yolov8n.pt")` | `Vision(model="yolov8n.pt")` |
| 跟踪 | `create_pipeline(detector_config=..., enable_tracking=True)` | `Vision(track=True)` |
| 姿态 | 手动创建 PoseEstimator + Pipeline | `Vision(pose=True)` |
| 处理 | `detector.detect_source(source)` / `pipeline.process_source(source)` / `process_image(source)` | `v.run(source)` |
| 配置 | `Config.load_from_file(...)` + 手动构建 | `Vision.from_config("config.json")` |

### v0.2.15 代码质量优化
- 共享工具模块 (`trackers/utils.py`)，减少 ~90 行重复代码
- 输入验证增强
- ByteTracker bug 修复
- ~250 行代码精简

## 核心功能

### 1. 目标检测
- **多模型支持**: YOLOv8、YOLOv9、YOLO26和自定义训练模型
- **特定类别置信度阈值**: 为每个检测类别设置不同的置信度水平
- **批量处理**: 高效处理多张图像
- **自动设备选择**: 根据可用性智能选择GPU/CPU

### 2. 目标跟踪
- **多跟踪器支持**: ByteTrack、IoUTracker、ReidTracker
- **跟踪持久性**: 在帧之间保持一致的ID
- **ReID跟踪增强**: 
  - 支持多种ReID模型（ResNet18/34/50/101/152）
  - FP16精度支持，提高推理速度
  - 改进的匹配算法，结合IoU和ReID特征
  - 自适应权重，根据轨迹年龄调整匹配策略
- **轨迹分析**: 
  - 速度和方向计算
  - 轨迹平滑和去噪
  - 未来位置预测
  - 轨迹长度和距离分析

### 3. 视频处理
- **视频文件支持**: 处理本地视频文件并保存输出
- **视频流支持**: RTSP、HTTP和其他流媒体协议
- **实时处理**: 低延迟视频处理管道
- **高性能后端**: 
  - PyAV集成（基于FFmpeg），提供比OpenCV更高的视频处理性能
  - 自动回退机制，当PyAV不可用时使用OpenCV
  - PyAV后端支持视频文件和RTSP/HTTP流，摄像机仍使用OpenCV

### 4. 自动标注
- **多种格式支持**: YOLO、COCO、Pascal VOC
- **批量标注**: 标注整个数据集
- **导出选项**: 以多种格式保存标注

### 5. 简化API
- **一个入口**: `Vision` 类是唯一需要的入口
- **两种创建**: 关键字参数 或 配置文件
- **统一处理**: `v.run(source)` 处理一切输入

#### 使用示例

```python
from visionframework import Vision

# 关键字创建
v = Vision(model="yolov8n.pt", track=True)

# 或从配置文件
v = Vision.from_config("config.json")

# 处理任意来源
for frame, meta, result in v.run("video.mp4"):
    print(result["detections"], result["tracks"])
```

### 6. 图像分割
- **SAM支持**: 集成Segment Anything Model (SAM)用于图像分割
- **多种分割模式**: 自动分割、交互式分割(点/框提示)
- **检测+分割联合推理**: 将检测结果与分割掩码关联
- **多种模型变体**: 支持vit_h, vit_l, vit_b模型
- **批量处理**: 高效处理多张图像

### 7. CLIP模型增强
- **多模型支持**: 扩展支持OpenAI CLIP、OpenCLIP和中文CLIP模型
- **图像-文本相似度**: 计算图像与文本的相似度
- **零样本分类**: 无需额外训练即可分类新类别
- **图像补丁提取**: 从检测结果中提取图像补丁进行处理

### 8. 姿态估计
- **多模型支持**: YOLO Pose和MediaPipe Pose
- **人体关键点检测**: 检测17-33个人体关键点
- **置信度过滤**: 基于阈值过滤低置信度关键点
- **骨架绘制**: 绘制人体骨架连接

### 9. 内存池管理
- **全局内存池**: 统一管理内存分配和释放
- **动态内存调整**: 根据需求自动调整内存池大小
- **内存块复用**: 减少内存碎片化，提高内存使用效率
- **多内存池支持**: 可创建多个内存池，用于不同大小的内存需求
- **内存优化**: 提供内存使用统计和优化建议
- **批处理优化**: 为批处理操作预分配内存，减少内存分配开销

### 10. 插件系统
- **组件注册**: 支持通过装饰器注册自定义组件
- **动态发现**: 自动发现和加载插件
- **统一接口**: 所有插件遵循统一的接口规范
- **模型注册**: 支持注册和管理自定义模型
- **灵活扩展**: 可扩展检测器、跟踪器、分割器等多种组件
- **插件发现机制**: 从默认位置和用户指定路径加载插件

### 11. 统一错误处理
- **一致的错误处理**: 提供统一的错误处理机制
- **错误包装**: 包装函数调用，捕获和处理错误
- **输入验证**: 验证输入参数的类型和值
- **错误消息格式化**: 生成一致的错误消息格式
- **上下文信息**: 错误消息包含详细的上下文信息
- **异常类型管理**: 支持自定义异常类型

### 12. 依赖管理优化
- **延迟加载**: 支持可选依赖的延迟加载，避免启动时加载所有依赖
- **依赖可用性检查**: 检查可选依赖是否可用
- **安装指导**: 提供缺失依赖的安装命令
- **依赖状态管理**: 跟踪所有依赖的状态
- **版本检查**: 检查依赖版本是否满足要求
- **模块导入管理**: 安全导入可选依赖，避免导入错误

### 13. 模型优化工具
- **量化 (Quantization)**: 支持动态量化、静态量化和感知训练量化，减少模型大小和提高推理速度
- **剪枝 (Pruning)**: 支持L1/L2非结构化剪枝、结构化剪枝等多种剪枝策略
- **知识蒸馏 (Distillation)**: 支持从教师模型到学生模型的知识蒸馏，提高小模型性能

### 14. 模型训练与微调
- **模型微调**: 支持全量微调、冻结微调、LoRA和QLoRA等多种微调策略
- **多任务支持**: 支持检测、分割、姿态估计、ReID、CLIP等多种模型类型的微调
- **训练配置**: 灵活的配置系统，支持自定义训练参数和策略

### 15. 模型转换与部署
- **格式转换**: 支持PyTorch、ONNX、TensorRT、OpenVINO、CoreML等多种模型格式转换
- **平台部署**: 支持多种部署平台，包括边缘设备、移动设备和云端
- **模型优化**: 自动优化模型以适应目标部署平台

### 16. 模型管理
- **自动模型选择**: 根据硬件配置和任务需求自动选择最合适的模型
- **硬件适配**: 支持高端、中端、低端、边缘和移动设备等不同硬件层级
- **性能评估**: 自动评估模型在不同硬件上的性能表现

### 17. 多模态融合
- **融合策略**: 支持拼接、相加、相乘、双线性、注意力、交叉注意力、门控、密集等多种融合方式
- **模态支持**: 支持视觉、语言、音频、文本、图像、视频、传感器等多种模态
- **CLIP融合**: 专门的CLIP模型融合实现
- **视觉-语言融合**: 专门的视觉-语言融合实现

### 18. 数据增强
- **增强类型**: 支持翻转、旋转、缩放、平移、亮度、对比度、饱和度、色调、模糊、噪声、Cutout、Mixup、CutMix、颜色抖动、随机擦除等多种增强方式
- **批量增强**: 支持批量图像增强处理
- **配置灵活**: 灵活的配置系统，支持自定义增强参数

### 19. 轨迹分析
- **速度计算**: 计算目标的速度（像素/帧或m/s）
- **方向分析**: 分析目标的运动方向
- **轨迹平滑**: 轨迹平滑和去噪
- **位置预测**: 预测目标的未来位置
- **距离分析**: 计算轨迹长度和总距离

## 技术特性

### 1. 模型管理
- **自动下载**: 从Ultralytics仓库下载模型
- **模型缓存**: 本地缓存模型以加快加载速度
- **哈希验证**: 确保模型完整性
- **自定义模型支持**: 加载自己训练的模型

### 2. 配置系统
- **Pydantic验证**: 强类型配置
- **YAML支持**: 易于阅读的配置文件
- **动态配置**: 运行时更新设置
- **默认配置**: 合理的默认值，便于快速设置

### 3. 性能优化
- **动态批量大小**: 根据输入大小自动调整批量大小，优化内存使用
- **FP16支持**: 半精度推理，加快处理速度，减少显存占用
- **并行处理**: 
  - 视频I/O的多线程处理
  - 批处理中的并行图像处理
  - 可配置的工作线程数
- **内存管理**: 
  - 内存池管理，减少内存碎片化
  - 针对大批量的高效内存使用
  - 内存块复用，提高内存使用效率
  - 内存使用统计和优化建议
  - 自动内存优化和垃圾回收
- **增强的性能监控**: 
  - 实时FPS和帧时间统计
  - 组件级性能分析（检测、跟踪、可视化等）
  - GPU内存和利用率监控
  - 内存使用统计
  - 磁盘I/O和网络I/O监控
  - 详细的性能报告生成
- **依赖管理优化**: 
  - 延迟加载可选依赖，减少启动时间
  - 智能依赖检查，避免不必要的依赖加载
  - 依赖状态缓存，提高依赖检查速度
- **批处理优化**:
  - 支持最大批处理大小限制
  - 自动分块处理大批量数据
  - 批处理间的内存优化
  - 支持并行批处理

### 4. 可扩展性
- **模块化设计**: 易于添加新的检测器、跟踪器或可视化工具
- **基类支持**: 定义良好的扩展接口
- **插件架构**: 支持自定义组件

### 5. 评估工具
- **检测指标**: mAP、精确率、召回率
- **跟踪指标**: MOTA、MOTP、IDF1
- **可视化评估**: 交互式结果可视化

## 应用场景

### 1. 监控系统
- 实时目标检测和跟踪
- 入侵检测
- 人群计数

### 2. 工业自动化
- 质量控制
- 目标计数
- 缺陷检测

### 3. 自动驾驶
- 障碍物检测
- 车道检测
- 交通标志识别

### 4. 零售分析
- 顾客计数
- 产品检测
- 货架监控

### 5. 医学成像
- 病变检测
- 器官分割
- 疾病诊断支持

### 6. 农业
- 作物监控
- 病虫害检测
- 产量估计

## 性能

### 模型性能
| 模型 | 大小 | 速度 (ms) | mAP50 |
|-------|------|------------|-------|
| YOLOv8n | 6.2MB | 2.5 | 67.3 |
| YOLOv8s | 18.5MB | 4.5 | 74.6 |
| YOLOv8m | 41.7MB | 8.9 | 79.0 |
| YOLOv8l | 78.9MB | 12.4 | 81.2 |
| YOLOv8x | 130.0MB | 19.4 | 82.5 |
| YOLO26n | 8.7MB | 3.1 | 69.5 |
| YOLO26s | 24.8MB | 5.2 | 76.3 |

### 硬件要求
- **最低配置**: 4GB RAM的CPU
- **推荐配置**: 8GB VRAM的GPU（用于实时视频处理）
- **最佳配置**: 16GB VRAM的GPU（用于大批量处理）

### 软件要求
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Ultralytics 8.0+
