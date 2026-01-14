# 高级功能文档

本文档介绍框架的高级功能模块。

## 0. CLIP 零样本分类

### 什么是 CLIP？

CLIP (Contrastive Language-Image Pre-training) 是 OpenAI 开发的视觉-语言预训练模型，能够进行零样本分类，无需训练新的分类器即可识别任意类别。

### 使用方法

```python
from visionframework import CLIPExtractor
from PIL import Image

# 初始化 CLIP 模型
clip = CLIPExtractor({
    "device": "cuda",          # 使用 GPU
    "use_fp16": True,          # 启用 FP16 加速（需要 GPU）
    "model_name": "openai/clip-vit-base-patch32"  # 或其他 CLIP 模型
})
clip.initialize()

# 加载图像
image = Image.open("cat.jpg")

# 零样本分类
candidates = ["a cat", "a dog", "a bird"]
scores = clip.zero_shot_classify(image, candidates)

for label, score in zip(candidates, scores):
    print(f"{label}: {score:.4f}")
```

### 高级用法

```python
# 获取单个图像的嵌入
image_emb = clip.encode_image(image)  # shape: (1, embedding_dim)

# 获取多个文本的嵌入
text_embs = clip.encode_text(candidates)  # shape: (num_texts, embedding_dim)

# 计算相似度矩阵
similarity = clip.image_text_similarity(image, candidates)  # shape: (1, num_texts)

# 批量处理多个图像
images = [Image.open(f"img_{i}.jpg") for i in range(3)]
batch_embeddings = clip.encode_image(images)  # shape: (3, embedding_dim)
```

### 集成到检测流程

```python
from visionframework import Detector, CLIPExtractor

detector = Detector({"model_path": "yolov8n.pt"})
detector.initialize()

clip = CLIPExtractor({"device": "cuda"})
clip.initialize()

# 检测对象
detections = detector.detect(image)

# 对每个检测到的对象进行 CLIP 分类
for detection in detections:
    bbox = detection.bbox
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    
    # 裁剪对象区域
    obj_img = Image.fromarray(image[y1:y2, x1:x2])
    
    # CLIP 分类
    candidates = ["person", "car", "dog", "cat"]
    scores = clip.zero_shot_classify(obj_img, candidates)
    
    # 输出结果
    best_label = candidates[scores.index(max(scores))]
    print(f"Detection: {best_label}")
```

## 1. 检测模型支持

### YOLO 实例分割

除了标准的目标检测，框架还支持 YOLO 实例分割模型，可以输出像素级的对象掩码。

**使用方法:**

```python
from visionframework import Detector, Visualizer

detector = Detector({
    "model_type": "yolo",
    "model_path": "yolov8n-seg.pt",  # 使用分割模型
    "enable_segmentation": True      # 开启分割功能
})
detector.initialize()

# 检测结果包含 mask
detections = detector.detect(image)
if detections[0].has_mask():
    print("Mask shape:", detections[0].mask.shape)
```

### DETR 检测

DETR (Detection Transformer) 是 Facebook 开发的基于 Transformer 的目标检测模型。

**使用方法:**

```python
from visionframework import Detector

# 初始化 DETR 检测器
detector = Detector({
    "model_type": "detr",
    "detr_model_name": "facebook/detr-resnet-50",  # 或 "facebook/detr-resnet-101"
    "conf_threshold": 0.5,
    "device": "cpu"
})
detector.initialize()

# 检测
detections = detector.detect(image)
```

**依赖安装:**
```bash
pip install transformers torch
```

### RF-DETR 检测

RF-DETR 是 Roboflow 开发的高性能实时目标检测模型，具有高精度和快速推理的特点。在 COCO 数据集上实现了超过 60 AP 的成绩，并在边缘设备上实现了每张图像 6 毫秒的推理速度。

**使用方法:**

```python
from visionframework import Detector, Visualizer
import cv2

# 初始化 RF-DETR 检测器
detector = Detector({
    "model_type": "rfdetr",
    "conf_threshold": 0.5,
    "device": "cuda"  # 可选: "cpu", "cuda", "mps"
})
detector.initialize()

# 检测
image = cv2.imread("your_image.jpg")
detections = detector.detect(image)

# 可视化
visualizer = Visualizer()
result_image = visualizer.draw_detections(image, detections)
cv2.imwrite("output.jpg", result_image)
```

**特点:**
- 高精度：在 COCO 数据集上超过 60 AP
- 快速推理：边缘设备上每张图像 6 毫秒
- 易于使用：简洁的 API，自动下载预训练模型
- 支持自定义训练：可在自定义数据集上训练

**依赖安装:**
```bash
pip install rfdetr supervision
```

**模型选择:**
RF-DETR 会自动使用默认的预训练模型。如果需要使用自定义训练的模型，可以通过 `rfdetr_model_name` 参数指定。

### 姿态估计

支持人体姿态估计，可以检测关键点并绘制骨架。

**使用方法:**

```python
from visionframework import PoseEstimator, Visualizer

# 初始化姿态估计器
pose_estimator = PoseEstimator({
    "model_path": "yolov8n-pose.pt",
    "conf_threshold": 0.25,
    "keypoint_threshold": 0.5
})
pose_estimator.initialize()

# 估计姿态
poses = pose_estimator.estimate(image)

# 可视化
visualizer = Visualizer()
result = visualizer.draw_poses(image, poses, draw_skeleton=True)
```

**关键点信息:**
- 支持 COCO 格式的 17 个关键点
- 包括：鼻子、眼睛、耳朵、肩膀、肘部、手腕、臀部、膝盖、脚踝等
- 可以绘制骨架连接

### ReID 特征跟踪

传统的跟踪算法（如 SORT/ByteTrack）主要依赖运动信息（IoU）。ReID 跟踪器引入外观特征提取（Re-Identification），结合运动和外观信息进行匹配，能有效处理遮挡和重找回问题。

**工作原理:**
1. 使用 ResNet50 提取每个检测框的外观特征向量（2048维）。
2. 计算检测框与轨迹之间的余弦距离。
3. 结合 IoU 距离和 ReID 距离进行关联匹配。
4. 更新匹配轨迹的外观特征（EMA 更新）。

**使用方法:**

```python
from visionframework import VisionPipeline

pipeline = VisionPipeline({
    "enable_tracking": True,
    "tracker_config": {
        "tracker_type": "reid",
        "reid_weight": 0.7,      # ReID 权重 (0~1)
        "reid_config": {
            "device": "cuda",    # 推荐使用 GPU
            "use_pretrained": True
        }
    }
})
```

## 1. 跟踪性能评估 (TrackingEvaluator)

### 功能概述

`TrackingEvaluator` 提供标准的 MOT (Multiple Object Tracking) 评估指标，用于定量评估跟踪算法性能。

### 支持的指标

- **MOTA** (Multiple Object Tracking Accuracy) - 跟踪准确度
  - 公式：`MOTA = 1 - (FN + FP + ID_switches) / GT_total`
  - 综合考虑漏检、误检和 ID 切换

- **MOTP** (Multiple Object Tracking Precision) - 跟踪精度
  - 平均位置误差（像素），仅计算匹配检测的距离
  - 越小越好

- **IDF1** (ID F1 Score) - ID 保持一致性评分
  - 公式：`IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)`
  - 评估跟踪 ID 的保持稳定性（1.0 为完美）

### 使用方法

```python
from visionframework import TrackingEvaluator

# 初始化评估器
evaluator = TrackingEvaluator(iou_threshold=0.5)

# 准备数据：每一帧的预测和真值轨迹
# 格式：List[List[{track_id, bbox}]]
pred_tracks = [
    # Frame 0
    [
        {"track_id": 1, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
        {"track_id": 2, "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}},
    ],
    # Frame 1
    [
        {"track_id": 1, "bbox": {"x1": 15, "y1": 15, "x2": 55, "y2": 55}},
        {"track_id": 2, "bbox": {"x1": 105, "y1": 105, "x2": 155, "y2": 155}},
    ]
]

gt_tracks = [...]  # 相同格式的真值数据

# 计算所有指标
results = evaluator.evaluate(pred_tracks, gt_tracks)
print(f"MOTA: {results['MOTA']:.4f}")
print(f"MOTP: {results['MOTP']:.4f} pixels")
print(f"IDF1: {results['IDF1']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")

# 或分别计算各项指标
mota = evaluator.calculate_mota(pred_tracks, gt_tracks)
motp = evaluator.calculate_motp(pred_tracks, gt_tracks)
idf1 = evaluator.calculate_idf1(pred_tracks, gt_tracks)
```

### 配置参数

- `iou_threshold` (float): IoU 阈值用于匹配检测框。默认 0.5。

### 返回值说明

`evaluate()` 方法返回包含以下键值的字典：
- `MOTA`: 跟踪准确度 (0-1 或更低)
- `MOTP`: 跟踪精度 (像素距离，越小越好)
- `IDF1`: ID F1 分数 (0-1)
- `precision`: 精准度 (0-1)
- `recall`: 召回率 (0-1)
- `details`: 各项指标的详细信息

## 2. 结果导出功能 (ResultExporter)

### 功能概述
将检测和跟踪结果导出为多种格式：JSON、CSV、COCO格式。

### 使用方法

```python
from visionframework import ResultExporter, VisionPipeline

# 初始化
pipeline = VisionPipeline()
pipeline.initialize()
exporter = ResultExporter()

# 处理图像
results = pipeline.process(image)

# 导出为JSON
exporter.export_detections_to_json(
    results["detections"],
    "output.json",
    metadata={"source": "image.jpg"}
)

# 导出为CSV
exporter.export_detections_to_csv(
    results["detections"],
    "output.csv"
)

# 导出为COCO格式
exporter.export_to_coco_format(
    results["detections"],
    image_id=0,
    image_info={"width": 640, "height": 480, "file_name": "image.jpg"},
    output_path="coco_output.json"
)
```

### 支持格式
- **JSON**: 完整的检测/跟踪信息，包含元数据
- **CSV**: 表格格式，便于数据分析
- **COCO**: 标准COCO格式，用于评估和训练

## 2. 性能分析工具 (PerformanceMonitor)

### 功能概述
实时监控和统计处理性能，包括FPS、处理时间等指标。

### 使用方法

```python
from visionframework import PerformanceMonitor, VisionPipeline

# 初始化
pipeline = VisionPipeline()
pipeline.initialize()
monitor = PerformanceMonitor(window_size=30)
monitor.start()

# 处理循环
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 记录检测时间
    import time
    start = time.time()
    results = pipeline.process(frame)
    detection_time = time.time() - start
    
    monitor.record_detection_time(detection_time)
    monitor.tick()
    
    # 获取当前FPS
    fps = monitor.get_current_fps()
    print(f"Current FPS: {fps:.2f}")

# 打印性能摘要
monitor.print_summary()

# 获取详细指标
metrics = monitor.get_metrics()
print(f"Average FPS: {metrics.avg_fps:.2f}")
print(f"Min FPS: {metrics.min_fps:.2f}")
print(f"Max FPS: {metrics.max_fps:.2f}")
```

### 性能指标
- **FPS**: 当前、平均、最小、最大帧率
- **处理时间**: 每帧平均、最小、最大处理时间
- **组件时间**: 检测、跟踪、可视化各组件耗时

## 3. 视频处理工具 (VideoProcessor, VideoWriter)

### 功能概述
提供便捷的视频读取、写入和处理功能。

### 使用方法

#### 基本视频读取

```python
from visionframework import VideoProcessor

# 打开视频
with VideoProcessor("video.mp4") as processor:
    info = processor.get_info()
    print(f"FPS: {info['fps']}, Size: {info['width']}x{info['height']}")
    
    # 读取帧
    while True:
        ret, frame = processor.read_frame()
        if not ret:
            break
        # 处理帧...
```

#### 视频写入

```python
from visionframework import VideoWriter

# 创建视频写入器
with VideoWriter("output.mp4", fps=30.0, frame_size=(640, 480)) as writer:
    writer.open()
    
    # 写入帧
    for frame in frames:
        writer.write(frame)
```

#### 批量处理视频

```python
from visionframework import process_video

def process_frame(frame, frame_num):
    # 处理帧
    return processed_frame

# 处理视频
process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    frame_callback=process_frame,
    start_frame=0,
    end_frame=100,
    skip_frames=0  # 不跳过帧
)
```

## 4. 区域检测 (ROIDetector)

### 功能概述
定义感兴趣区域(ROI)，过滤和分组检测/跟踪结果。

### 使用方法

```python
from visionframework import ROIDetector, VisionPipeline

# 定义ROI
roi_config = {
    "rois": [
        {
            "name": "zone1",
            "type": "rectangle",
            "points": [(100, 100), (400, 300)]
        },
        {
            "name": "zone2",
            "type": "polygon",
            "points": [(500, 200), (700, 200), (700, 400), (500, 400)]
        }
    ],
    "check_center": True  # 检查bbox中心点
}

# 初始化
roi_detector = ROIDetector(roi_config)
roi_detector.initialize()

# 处理
results = pipeline.process(image)
tracks = results["tracks"]

# 过滤
zone1_tracks = roi_detector.filter_tracks_by_roi(tracks, "zone1")

# 分组
grouped = roi_detector.get_detections_by_roi(results["detections"])
```

### ROI类型
- **rectangle**: 矩形区域，需要2个点 (左上, 右下)
- **polygon**: 多边形区域，需要多个点
- **circle**: 圆形区域，需要2个点 (中心, 半径点)

## 5. 计数功能 (Counter)

### 功能概述
统计进入、离开和停留在ROI内的对象数量。

### 使用方法

```python
from visionframework import Counter, VisionPipeline

# 创建计数器
counter = Counter({
    "roi_detector": {
        "rois": [{
            "name": "entrance",
            "type": "rectangle",
            "points": [(200, 100), (600, 400)]
        }]
    },
    "count_entering": True,
    "count_exiting": True,
    "count_inside": True
})
counter.initialize()

# 处理视频
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pipeline.process(frame)
    tracks = results["tracks"]
    
    # 计数
    counts = counter.count_tracks(tracks)
    
    # 获取统计信息
    for roi_name, info in counts.items():
        print(f"{roi_name}:")
        print(f"  Entering: {info['entering']}")
        print(f"  Exiting: {info['exiting']}")
        print(f"  Inside: {info['inside']}")
        print(f"  Total entered: {info['total_entered']}")

# 获取最终计数
final_counts = counter.get_counts()
```

### 计数功能
- **进入计数**: 对象进入ROI时计数
- **离开计数**: 对象离开ROI时计数
- **内部计数**: 当前在ROI内的对象数量
- **总计数**: 累计进入的对象总数

## 完整示例

查看 `examples/advanced_features.py` 获取完整的使用示例。

## 组合使用

这些功能可以灵活组合使用：

```python
# 完整的视频处理流程
from visionframework import (
    VisionPipeline, ROIDetector, Counter,
    PerformanceMonitor, ResultExporter, VideoProcessor
)

# 初始化所有组件
pipeline = VisionPipeline()
roi_detector = ROIDetector(roi_config)
counter = Counter({"roi_detector": roi_detector})
monitor = PerformanceMonitor()
exporter = ResultExporter()

# 处理视频
with VideoProcessor("input.mp4") as processor:
    monitor.start()
    video_results = []
    
    while True:
        ret, frame = processor.read_frame()
        if not ret:
            break
        
        # 处理
        results = pipeline.process(frame)
        tracks = results["tracks"]
        
        # ROI过滤
        filtered_tracks = roi_detector.filter_tracks_by_roi(tracks, "zone1")
        
        # 计数
        counts = counter.count_tracks(filtered_tracks)
        
        # 性能监控
        monitor.tick()
        
        # 保存结果
        video_results.append({
            "frame": processor.current_frame_num,
            "tracks": [t.to_dict() for t in tracks],
            "counts": counts
        })
    
    # 导出结果
    exporter.export_video_results_to_json(
        video_results,
        "output.json",
        video_info=processor.get_info()
    )
    
    # 打印性能摘要
    monitor.print_summary()
```

