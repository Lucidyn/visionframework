# 示例

所有示例都使用统一的 `Vision` API。运行前请激活 `frametest` conda 环境：

```bash
conda activate frametest
pip install -e .
```

## 基础示例

| 文件 | 功能 | 核心用法 |
|------|------|---------|
| `basic/00_basic_detection.py` | 基本目标检测 | `Vision()` + `v.run()` |
| `basic/01_detection_with_tracking.py` | 检测 + 跟踪 | `Vision(track=True)` |
| `basic/02_simplified_api.py` | 从配置文件加载 | `Vision.from_config()` |
| `basic/03_pose_estimation.py` | 姿态估计 | `Vision(pose=True)` |
| `basic/04_segmentation.py` | 实例分割 | `Vision(segment=True)` |
| `basic/05_video_processing.py` | 视频 / 摄像头处理 | `v.run()` + `v.draw()` |

## 进阶示例

| 文件 | 功能 |
|------|------|
| `advanced/06_clip_features.py` | CLIP 特征提取 / 零样本分类 |
| `advanced/08_model_tools_example.py` | 模型工具（量化、剪枝、增强等） |
| `advanced/09_multimodal_processing.py` | 多模态处理 |
| `advanced/10_batch_processing.py` | 批处理 |
| `advanced/11_custom_component.py` | 自定义组件与插件系统 |
| `advanced/12_result_export.py` | 结果导出（JSON / CSV） |
| `advanced/13_roi_counting.py` | ROI 区域计数（进出统计） |
| `advanced/14_heatmap.py` | 轨迹热力图可视化 |
| `advanced/15_batch_images.py` | `process_batch()` 批量图像推理 |
| `advanced/16_model_optimization.py` | 量化 + 剪枝工作流 |
| `advanced/17_performance_monitor.py` | 性能监控与报告 |

## 快速运行

```bash
# 基础检测
conda run -n frametest python examples/basic/00_basic_detection.py

# 检测 + 跟踪
conda run -n frametest python examples/basic/01_detection_with_tracking.py

# ROI 计数（需要 video.mp4）
conda run -n frametest python examples/advanced/13_roi_counting.py

# 热力图（需要 video.mp4）
conda run -n frametest python examples/advanced/14_heatmap.py

# 批量图像处理
conda run -n frametest python examples/advanced/15_batch_images.py

# 模型量化与剪枝
conda run -n frametest python examples/advanced/16_model_optimization.py

# 性能监控（需要 video.mp4）
conda run -n frametest python examples/advanced/17_performance_monitor.py
```

## API 速查

```python
from visionframework import Vision

# 方式一：直接创建
v = Vision(model="yolov8n.pt", track=True, pose=True)

# 方式二：从配置文件创建
v = Vision.from_config("config.json")

# 处理任意来源
for frame, meta, result in v.run(source):
    print(result["detections"])
    print(result["tracks"])
    print(result["poses"])
    print(result.get("counts"))   # 需先调用 v.add_roi(...)

# ROI 区域计数
v.add_roi("entrance", [(100,100),(400,100),(400,400),(100,400)])

# 批量处理
results = v.process_batch([img1, img2, img3])

# 实例信息
print(v.info())

# 热力图
from visionframework import Visualizer
vis = Visualizer()
heatmap = vis.draw_heatmap(frame, tracks)
```

`source` 支持：图片路径、视频路径、摄像头 (0)、RTSP 流、文件夹、路径列表、numpy 数组。
